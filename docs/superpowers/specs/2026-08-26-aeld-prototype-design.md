# AELD Prototype — Design Spec

**Date:** 2026-08-26
**Status:** Approved

## Goal

A functional digital prototype of an **Acoustic Emission Leak Detection (AELD)** system: given two acoustic sensors at either end of a pipe, locate a leak by measuring the Time Difference of Arrival (TDoA) of the leak's acoustic signature using GCC-PHAT. Must be explainable live during a demonstration.

## Scope Decisions

Two decisions narrowed the original brief:

1. **Lean stack.** The live demo path is Streamlit + NumPy/SciPy + a mock PyTorch CNN. WNTR, FastAPI, and paho-mqtt are listed in `requirements.txt` (stack completeness) but **not wired into the demo** — they add install fragility and failure surface without improving the demonstrated result.
2. **Modular package.** Core math lives in `src/aeld/` with unit tests; `app.py` is a thin UI layer. Testable, and each module is small enough to explain on its own.

The clean-slate deletion of the pre-existing directory contents was explicitly authorized by the user after being shown what would be lost.

## Architecture

```
requirements.txt          # full stack listed; core deps pinned
setup.py                  # installs deps; MPS-first PyTorch, CPU/CUDA fallback
claude.md                 # original prompt + project goals
context.md                # progress / file structure / completed steps
app.py                    # Streamlit UI, 2 pages via sidebar nav
src/aeld/
  config.py               # SimulationConfig dataclass + defaults
  signals.py              # leak source model + per-sensor A/B signal generation
  gccphat.py              # GCC-PHAT -> TDoA -> leak position
  cnn_filter.py           # mock PyTorch CNN noise filter (MPS/CUDA/CPU)
  detector.py             # orchestration + timing instrumentation
tests/
  test_signals.py         # shape / attenuation / noise sanity
  test_gccphat.py         # known injected delay -> recovered within tolerance
```

## Physics & Math

**Signal model.** The leak source is band-limited turbulent noise: white noise passed through a Butterworth bandpass (~100–2000 Hz), which is a reasonable stand-in for the broadband hiss of a pressurized leak.

Each sensor receives a delayed, attenuated copy:

- `t_A = d / v`, `t_B = (L - d) / v` where `d` = leak distance from Sensor A, `L` = pipe length, `v` = wave velocity
- Delay is applied with **fractional-sample** (FFT phase-shift) precision — integer-sample delay would quantize the answer and cap achievable accuracy
- Amplitude scaled by `exp(-alpha * distance)` for distance-dependent attenuation
- Independent Gaussian noise added per sensor, scaled by the UI noise slider

Signal duration auto-scales to `max(0.1 s, 3 * L / v)` so the true delay always fits comfortably inside the analysis window.

**GCC-PHAT.** Generalized Cross-Correlation with Phase Transform:

1. FFT both sensor signals
2. Cross-power spectrum `X_A * conj(X_B)`
3. **PHAT whitening**: divide by `|X_A * conj(X_B)| + eps` — discards magnitude, keeps only phase, which makes the peak sharp and robust to the coloration of the source spectrum
4. IFFT back to the lag domain
5. Peak pick, refined by **parabolic sub-sample interpolation** for accuracy beyond one sample period

TDoA to position:

```
d_leak = (L + v * dt) / 2      clamped to [0, L]
```

Derivation: `d/v - (L-d)/v = dt` → `2d = L + v*dt`.

**Mock CNN.** A small `nn.Module` (conv → pool → fc) consuming a spectrogram feature, with device auto-selection **MPS → CUDA → CPU**. Per the brief it returns a high fixed leak probability (~0.97) to simulate a successful noise-filter pass. Explicitly a stub, labelled as such in the UI — it is not a trained model and makes no real classification claim.

**Detector.** Orchestrates generate → filter → locate and returns a result object carrying true vs. calculated position, absolute error, CNN confidence, per-stage and total latency (ms), and the raw arrays needed for plotting.

## UI

**Page 1 — Configuration & Simulation.** Sliders for Pipe Length (m), Wave Velocity (m/s), Background Noise Level. A "Trigger Random Leak" button selects a random position along the pipe, runs the full detection pipeline, and stores results in `st.session_state`. Optional manual leak position for repeatable demos.

**Page 2 — Detection Results & Inner Workings.** Total processing time; calculated distance from Sensor A alongside ground truth and error. Then the visual explanation: Sensor A/B waveforms (showing the delay and attenuation the algorithm exploits) and the GCC-PHAT correlation curve with the peak clearly marked as the mathematical evidence for the reported location.

## Testing

`pytest` suite verifying GCC-PHAT recovers a known injected delay within tolerance, and that signal generation produces correct shapes, monotonic attenuation, and noise scaling. The math is verifiably correct rather than merely plausible.

## Non-Goals

- Real hydraulic simulation (WNTR/EPANET)
- Real backend or telemetry transport (FastAPI, MQTT)
- A trained leak-classification model
- Multi-pipe networks, junctions, or reflections
