"""Render a Markdown document to PDF with tables, code blocks and Unicode intact.

    python scripts/md_to_pdf.py SIH_MASTER_PLAN.md [-o out.pdf] [--title "..."]

Uses python-markdown for the HTML pass and xhtml2pdf/reportlab for layout. Windows
system fonts (Segoe UI, Consolas, Times New Roman) are registered explicitly, because
the reportlab built-ins are the base-14 fonts and cover very little beyond Latin-1.

Pass --plain for a black-and-white serif document with no fills or accent colours.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import markdown

FONT_DIR = Path("C:/Windows/Fonts")

# (css family name, regular, bold, italic, bold-italic)
FONTS = [
    ("BodyFont", "segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf", "segoeuiz.ttf"),
    ("MonoFont", "consola.ttf", "consolab.ttf", "consolai.ttf", "consolaz.ttf"),
    ("SerifFont", "times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"),
]


def register_fonts() -> None:
    """Register the TTFs straight into reportlab's registry.

    Going through CSS @font-face makes xhtml2pdf copy each file to a temp path and
    reopen it, which fails on Windows. Registering here means the CSS only has to
    name the family.
    """
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from xhtml2pdf.default import DEFAULT_FONT

    for family, regular, bold, italic, bolditalic in FONTS:
        faces = {}
        for suffix, path in (
            ("", regular), ("-Bold", bold), ("-Italic", italic), ("-BoldItalic", bolditalic),
        ):
            f = FONT_DIR / path
            if not f.exists():
                continue
            name = f"{family}{suffix}"
            pdfmetrics.registerFont(TTFont(name, str(f)))
            faces[suffix] = name

        if not faces:
            print(f"warning: no font files found for {family}", file=sys.stderr)
            continue

        base = faces.get("", next(iter(faces.values())))
        pdfmetrics.registerFontFamily(
            family,
            normal=base,
            bold=faces.get("-Bold", base),
            italic=faces.get("-Italic", base),
            boldItalic=faces.get("-BoldItalic", base),
        )

        # Registering with reportlab is not enough: xhtml2pdf resolves CSS font-family
        # through its own alias table, and anything missing there silently falls back
        # to Helvetica. Its @font-face path is not an option here -- it copies the file
        # to a temp path and reopens it, which fails on this Python/Windows build.
        DEFAULT_FONT[family.lower()] = base
        for suffix, key in (("-Bold", "bold"), ("-Italic", "italic"),
                            ("-BoldItalic", "bolditalic")):
            DEFAULT_FONT[f"{family.lower()}-{key}"] = faces.get(suffix, base)


CSS = """
@page {
    size: a4 portrait;
    /* bottom margin clears the footer frame, which ends at 1.9cm */
    margin: 1.6cm 1.5cm 2.3cm 1.5cm;
    @frame footer { -pdf-frame-content: footer; bottom: 0.9cm; height: 1cm;
                    left: 1.5cm; width: 18cm; }
}
body        { font-family: BodyFont; font-size: 9.2pt; line-height: 1.42; color: #1a1a1a; }
h1          { font-size: 20pt; color: #0b3c5d; margin: 0 0 4pt 0; -pdf-outline: true;
              -pdf-outline-level: 0; }
h2          { font-size: 14pt; color: #0b3c5d; margin: 16pt 0 5pt 0;
              border-bottom: 1.1pt solid #0b3c5d; padding-bottom: 2pt;
              -pdf-outline: true; -pdf-outline-level: 1; -pdf-keep-with-next: true; }
h3          { font-size: 11.5pt; color: #14557a; margin: 12pt 0 3pt 0;
              -pdf-outline: true; -pdf-outline-level: 2; -pdf-keep-with-next: true; }
h4          { font-size: 10pt; color: #333; margin: 9pt 0 3pt 0;
              -pdf-keep-with-next: true; }
p           { margin: 0 0 5pt 0; text-align: justify; }
ul, ol      { margin: 0 0 6pt 0; }
li          { margin-bottom: 2pt; }
a           { color: #14557a; text-decoration: none; }
strong      { font-weight: bold; }

code        { font-family: MonoFont; font-size: 8pt; background-color: #f2f4f6; }
.codeblock  { font-family: MonoFont; font-size: 7.4pt; line-height: 1.3;
              background-color: #f7f8fa; border: 0.6pt solid #d8dde2;
              padding: 6pt 7pt; margin: 5pt 0 9pt 0; }

blockquote  { margin: 6pt 0 7pt 14pt; padding: 4pt 8pt;
              border-left: 2.4pt solid #0b3c5d; background-color: #f4f7f9;
              color: #22303a; font-style: italic; }

table       { width: 100%; border: 0.6pt solid #c2cad1; margin: 5pt 0 9pt 0;
              -pdf-keep-in-frame-mode: shrink; }
th          { background-color: #0b3c5d; color: #ffffff; font-weight: bold;
              font-size: 7.6pt; padding: 3.2pt 4pt; text-align: left;
              border: 0.4pt solid #0b3c5d; }
td          { font-size: 7.6pt; padding: 3pt 4pt; border: 0.4pt solid #d3dae0;
              vertical-align: top; }

hr          { border: 0; border-top: 0.6pt solid #ccd3d9; margin: 10pt 0; }
.footer     { font-family: BodyFont; font-size: 7pt; color: #7a848c;
              text-align: center; }
.cover-sub  { font-size: 11pt; color: #33474f; margin-bottom: 14pt; }
"""


# Black-and-white document style: no fills, no accent colours, serif body.
PLAIN_CSS = """
@page {
    size: a4 portrait;
    margin: 2.0cm 2.0cm 2.2cm 2.0cm;
    @frame footer { -pdf-frame-content: footer; bottom: 1.0cm; height: 1cm;
                    left: 2.0cm; width: 17cm; }
}
body        { font-family: SerifFont; font-size: 10pt; line-height: 1.34; color: #000000; }
h1          { font-size: 14pt; margin: 0 0 2pt 0; font-weight: bold; }
h2          { font-size: 11pt; margin: 11pt 0 3pt 0; font-weight: bold;
              -pdf-keep-with-next: true; }
h3          { font-size: 10pt; margin: 8pt 0 2pt 0; font-weight: bold;
              -pdf-keep-with-next: true; }
h4          { font-size: 10pt; margin: 7pt 0 2pt 0; font-weight: bold; }
p           { margin: 0 0 5pt 0; text-align: justify; }
ul, ol      { margin: 0 0 5pt 0; }
li          { margin-bottom: 1pt; }
a           { color: #000000; text-decoration: none; }

code        { font-family: MonoFont; font-size: 9pt; }
.codeblock  { font-family: MonoFont; font-size: 8pt; line-height: 1.25;
              border: 0.5pt solid #000000; padding: 5pt; margin: 4pt 0 7pt 0; }

blockquote  { margin: 5pt 0 6pt 12pt; padding: 0 0 0 8pt;
              border-left: 0.75pt solid #000000; font-style: italic; }

table       { width: 100%; border: 0.5pt solid #000000; margin: 4pt 0 7pt 0;
              -pdf-keep-in-frame-mode: shrink; }
th          { font-weight: bold; font-size: 8.5pt; padding: 2.5pt 4pt;
              text-align: left; border: 0.5pt solid #000000; }
td          { font-size: 8.5pt; padding: 2.5pt 4pt; border: 0.5pt solid #000000;
              vertical-align: top; }

hr          { border: 0; border-top: 0.5pt solid #000000; margin: 8pt 0; }
.footer     { font-family: SerifFont; font-size: 8pt; color: #000000;
              text-align: center; }
"""

FOOTER = (
    '<div id="footer" class="footer">SHRAVAN &middot; '
    '<pdf:pagenumber> / <pdf:pagecount></div>'
)

PLAIN_FOOTER = '<div id="footer" class="footer"><pdf:pagenumber></div>'



def strip_anchor_links(md_text: str) -> str:
    """Turn '[Positioning](#1-positioning)' into plain text.

    xhtml2pdf renders in-document anchors as dead blue text, which reads as a broken
    link on paper. Bare URLs and real links are left alone.
    """
    return re.sub(r"\[([^\]]+)\]\(#[^)]*\)", r"\1", md_text)


# Neither Segoe UI nor Consolas carries these; reportlab has no fallback chain, so
# an unsubstituted glyph prints as a solid black box.
# The rupee sign is present in the Segoe UI cmap but maps to an empty glyph.
# Horizontal rules are mapped too: xhtml2pdf does not reliably apply a class-scoped
# font inside a div, so code blocks can fall back to the body face, which lacks them.
GLYPH_SUBS = {"★": "*", "▶": ">", "◀": "<", "▼": "v", "▲": "^",
              "₹": "Rs ", "─": "-", "│": "|"}

# Vertical and corner rules only: a block using horizontal rules as separators is
# ordinary text, not a diagram.
DIAGRAM_CHARS = re.compile(r"[│┌┐└┘├┤┬┴┼▼▲▶◀]")


def substitute_glyphs(text: str) -> str:
    for bad, good in GLYPH_SUBS.items():
        text = text.replace(bad, good)
    return text


def drop_ascii_diagrams(md_text: str) -> str:
    """Remove fenced blocks that are ASCII/box-drawing diagrams.

    Proportional-to-monospace layout in xhtml2pdf collapses their alignment into an
    unreadable jumble, so they are dropped rather than rendered wrongly. Blocks whose
    only box characters are horizontal rules (equations, config) are kept.
    """
    def repl(m: re.Match) -> str:
        return "" if DIAGRAM_CHARS.search(m.group(0)) else m.group(0)

    return re.sub(r"^```.*?^```\s*$", repl, md_text, flags=re.S | re.M)


def preserve_pre_whitespace(html: str) -> str:
    """Make <pre> blocks survive xhtml2pdf, which collapses runs of spaces.

    Leading indentation becomes non-breaking spaces and newlines become explicit
    breaks, so the block keeps its shape.
    """
    def repl(m: re.Match) -> str:
        inner = m.group(1)
        inner = re.sub(r"</?code[^>]*>", "", inner)
        lines = inner.split("\n")
        out = []
        for line in lines:
            stripped = line.lstrip(" ")
            out.append("&nbsp;" * (len(line) - len(stripped)) + stripped)
        return '<div class="codeblock">' + "<br/>".join(out) + "</div>"

    return re.sub(r"<pre>(.*?)</pre>", repl, html, flags=re.S)


def fill_empty_header_cells(html: str) -> str:
    """Put a non-breaking space in empty <th> cells.

    xhtml2pdf sizes columns from header content and an empty cell can resolve to a
    negative available width, which aborts the whole render rather than that table.
    """
    return re.sub(r"<th([^>]*)>\s*</th>", r"<th\1>&nbsp;</th>", html)


def build_html(md_text: str, title: str, plain: bool = False) -> str:
    md_text = substitute_glyphs(drop_ascii_diagrams(strip_anchor_links(md_text)))
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    body = fill_empty_header_cells(preserve_pre_whitespace(body))
    css = PLAIN_CSS if plain else CSS
    footer = PLAIN_FOOTER if plain else FOOTER
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>{css}</style></head>
<body>{footer}
{body}
</body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("-o", "--out")
    ap.add_argument("--title", default=None)
    ap.add_argument("--plain", action="store_true",
                    help="black-and-white document style: serif, no fills, no accent colours")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        print(f"not found: {src}", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else src.with_suffix(".pdf")
    title = args.title or src.stem.replace("_", " ").title()

    register_fonts()
    html = build_html(src.read_text(encoding="utf-8"), title, plain=args.plain)
    debug_html = out.with_suffix(".debug.html")
    debug_html.write_text(html, encoding="utf-8")

    from xhtml2pdf import pisa

    with open(out, "wb") as fh:
        result = pisa.CreatePDF(html, dest=fh, encoding="utf-8")

    if result.err:
        print(f"{result.err} error(s) during rendering; HTML kept at {debug_html}",
              file=sys.stderr)
        return 2

    debug_html.unlink(missing_ok=True)
    print(f"{out}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
