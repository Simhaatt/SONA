# Kaggle VoiceBank-DEMAND run, 2026-07-03

Source artifact: `C:\Users\hp\Downloads\sona_artifacts.zip`.

This was a 300-clip VoiceBank-DEMAND test run using the earlier Kaggle notebook configuration:
classical baselines, supervised MaskGRU/MaskTCN/Attention, `rl_frozen`, and `rl_online`.

Key interpretation:

- Supervised frozen models are currently strongest on SI-SDR/SNR/STOI. `neural_mask_gru` has the
  best SI-SDRi: 9.382 dB.
- `rl_online` improves perceptual metrics versus `rl_frozen` on this run: PESQ 2.573 vs 2.429 and
  LSD 8.135 vs 8.520.
- `rl_online` loses reconstruction/intelligibility metrics versus `rl_frozen`: SI-SDRi 3.538 vs
  6.142, SNRi 2.193 vs 6.169, STOI 0.923 vs 0.928.
- Runtime is not edge-ready for the online loop in this configuration: RTF 0.532 for `rl_online`
  vs ~0.002 for frozen neural models.

Conclusion: the first real-data SQUIM online run is promising for PESQ but does not pass the thesis
gate yet. The next Kaggle run should use the newer repo notebook/runner defaults that evaluate
anchored supervised-online variants such as `attention_online` and `mask_gru_online`.

