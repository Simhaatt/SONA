# Prompt For Claude Code: Build SONA Web Demo

You are working in this repo:

`D:\noise cancellation`

Build a complete local web demo for **SONA**, a neural audio enhancement engine.

You have full freedom to design and implement the app. Make technical and design decisions yourself. Use modern best practices. You may choose the stack, layout, animations, backend structure, tests, and UI details.

Hard rule:

- Do **not** delete, overwrite, or clean existing datasets, checkpoints, notebooks, result folders, or user files.
- Existing folders under `results/`, `external_data/`, checkpoints, zips, and notebooks must be preserved.
- You may add new files/folders and write new outputs to clearly named new paths.

## Project Context

SONA is a final-year project audio engine.

Layer A is already locked:

- Winner: `gtcrn_ch96_h96_b4_mw1p25_mr0p25`
- Best checkpoint:
  `D:\noise cancellation\results\kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07\extracted\ckpt_gtcrn_ch96_h96_b4_mw1p25_mr0p25\last.pt`

Layer B already exists:

- Controller package:
  `D:\noise cancellation\src\controller`
- Offline enhancement:
  `D:\noise cancellation\scripts\enhance_controlled.py`
- Scene classifier:
  `D:\noise cancellation\scripts\classify_scene.py`
- Preference memory:
  `D:\noise cancellation\scripts\manage_preferences.py`
- Layer A results:
  `D:\noise cancellation\LAYER_A_RESULTS.md`
- Living project status:
  `D:\noise cancellation\SONA_STATUS.md`

Use these existing pieces. Do not retrain models.

## Goal

Create a polished local web app that can:

1. Load a sample noisy audio file.
2. Accept uploaded audio.
3. Classify scene.
4. Select/adjust SONA mode/controller settings.
5. Run enhancement using the existing GTCRN checkpoint and controller pipeline.
6. Produce an enhanced `.wav`.
7. Play original/enhanced audio.
8. Show analytics and visualizations.
9. Store/restore scene preferences.

The app should feel like a real AI audio product, not a rough research script.

## Suggested Location

Put the web demo under:

`D:\noise cancellation\web_demo`

Suggested output folders:

- `D:\noise cancellation\web_demo\samples`
- `D:\noise cancellation\results\web_demo_outputs`
- `D:\noise cancellation\results\web_demo_uploads`
- `D:\noise cancellation\results\web_demo_preferences.json`

If you choose different new paths, document them clearly.

## Backend Requirements

Create a local backend. Choose the framework yourself, but FastAPI or Flask is fine.

Backend should support:

- health/status endpoint
- sample audio endpoint
- upload endpoint
- scene classification endpoint
- enhancement endpoint
- output download endpoint
- preference memory endpoint
- analytics endpoint or analytics returned with enhancement response

Backend should connect to existing repo code directly where possible:

- `src.controller.enhance_file`
- `src.controller.SceneClassifier`
- `src.controller.PreferenceMemory`

Calling existing scripts is acceptable if importing is awkward, but importing is preferred.

Do not retrain anything.

## Sample Audio Requirement

The app must work immediately after starting.

Add a sample noisy input file:

`D:\noise cancellation\web_demo\samples\sample_noisy.wav`

If a suitable sample already exists, copy it into the demo sample folder.

If not, generate a short synthetic noisy speech-like WAV.

When the app runs, the user should be able to click **Use Sample** and then **Enhance Audio**, producing an output WAV.

## Backend Security And Robustness

Add reasonable local-app backend security and safety:

- validate uploaded file type
- reject non-audio files
- use safe filenames
- avoid path traversal
- keep outputs inside allowed output folders
- set max upload size
- handle missing checkpoint with a clear error
- handle model/enhancement failure gracefully
- never expose arbitrary filesystem read/write
- do not allow user-supplied paths to escape the demo folders
- add CORS only as needed for the local frontend
- add basic request validation
- add clear JSON error responses
- avoid leaking long tracebacks to frontend
- log backend errors server-side
- ensure concurrent requests do not overwrite each other
- use unique job/output IDs
- ensure generated files have predictable safe extensions

Decide the exact implementation yourself.

## Frontend Requirements

Build the actual usable app as the first screen. Do not make a marketing landing page.

Design freedom:

- Full creative freedom.
- Make it modern, premium, polished, and product-ready.
- Refer to modern AI audio tools, DAW plugins, studio dashboards, denoising apps, and SaaS interfaces for inspiration.
- Create an original design.
- Add modern animations and micro-interactions.
- Add a modern loading screen.
- Add analytics and visualizations.

Brand:

- Name: `SONA`
- Feel: neural audio engine, intelligent, precise, futuristic, studio-quality
- Possible subtitle: `Neural Audio Engine`

Required UI:

1. Loading screen
   - SONA name/logo
   - animated waveform or spectrum
   - loading phrases like:
     - Loading GTCRN engine
     - Preparing controller
     - Calibrating scene memory
     - Ready

2. Audio input
   - drag/drop upload
   - normal upload button
   - use sample noisy audio button
   - selected file name
   - duration if available

3. Enhancement flow
   - Enhance Audio button
   - progress animation
   - show pipeline stages:
     - scene analysis
     - GTCRN enhancement
     - controller mode
     - EQ/compressor
     - export WAV

4. Audio playback
   - original/noisy player
   - enhanced player
   - A/B compare toggle
   - download enhanced WAV

5. Scene detection
   - detected scene:
     - speech
     - music
     - traffic
     - fan
     - crowd
     - mixed
     - unknown
   - confidence
   - auto-scene toggle

6. Mode presets
   - Speech
   - Music
   - Podcast
   - Gaming
   - Hearing
   - Studio

7. Controls
   - denoise strength slider
   - safety floor slider
   - EQ controls
   - compressor threshold
   - compressor ratio
   - limiter status

8. Preference memory
   - remember settings for this scene
   - restore remembered settings
   - reset scene memory
   - use:
     `D:\noise cancellation\results\web_demo_preferences.json`

9. Analytics
   Show useful analytics after classification/enhancement:
   - input duration
   - output duration
   - detected scene
   - confidence
   - selected mode
   - denoise strength
   - processing time
   - estimated realtime factor
   - input peak
   - output peak
   - approximate loudness change
   - output file path
   - checkpoint/model name

10. Visualizations
   - waveform for input
   - waveform for output
   - spectrum analyzer
   - before/after visualization
   - use real audio-derived data if practical; mocked visualization is acceptable only where needed

## Testing Requirement

Add tests. You decide the right level and framework.

At minimum, verify:

- backend health endpoint works
- sample audio exists or is generated
- upload validation works
- scene classification endpoint works
- enhancement endpoint can produce an output WAV from sample audio
- output download path is safe
- preference memory endpoint works

Also manually smoke-test the app:

- start backend
- open frontend
- use sample audio
- classify
- enhance
- play/download output

Document test/run commands.

## Documentation

Add a web demo README:

`D:\noise cancellation\web_demo\README.md`

Include:

- install steps
- run command
- local URL
- sample demo flow
- where outputs are saved
- where preferences are saved
- known limitations

Update:

- `D:\noise cancellation\SONA_STATUS.md`

Mention:

- web demo built
- backend/frontend paths
- sample audio path
- output path
- tests done
- next stage recommendation

Do not over-edit unrelated docs.

## Expected Final Behavior

After your work, a user should be able to run something like:

```powershell
cd "D:\noise cancellation\web_demo"
python app.py
```

Then open the printed local URL.

In the app:

1. Click **Use Sample**.
2. See waveform/spectrum.
3. Click **Classify Scene** or enable auto-scene.
4. Click **Enhance Audio**.
5. Wait through modern loading/progress UI.
6. Play original and enhanced audio.
7. Download enhanced WAV.
8. See analytics.
9. Save preference memory.

If your chosen stack requires a different command, make it simple and document it clearly.

## Creative Freedom

You have full freedom to make this excellent.

You may:

- choose FastAPI, Flask, React, Vite, plain HTML, Tailwind, CSS, canvas, SVG, Web Audio API, etc.
- add animations
- add modern dashboard layouts
- add waveform/spectrum canvases
- add useful backend helpers
- improve UX beyond the requirements
- add tests and scripts as you think appropriate

But:

- do not delete existing data
- do not retrain models
- do not break existing SONA code
- keep the app runnable locally
- verify the sample enhancement path works

## Important Project Direction

This is for a final-year project with 6-9 months left.

The demo should support the larger roadmap:

- trained scene classifier later
- controller ablation study
- RL governor later
- realtime mic pipeline
- edge deployment
- user study

So structure the app cleanly enough that those can be added later.
