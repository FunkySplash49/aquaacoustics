# context.md — Progress, Structure & Completed Steps

Working log for the AELD prototype. See `claude.md` for the original brief.

**Status:** Complete and verified. 38/38 tests passing; Streamlit app boots
clean (HTTP 200, no errors or warnings).
**Last updated:** 2026-08-26

**Latest change:** the PyTorch CNN stage was removed from the pipeline and
archived to `archive/ml_model/`, and the UI was rewritten in plain language for
visitors with no signal-processing background. End-to-end time dropped from
75.9 ms to 2.3 ms (33x) and PyTorch is no longer a dependency.

---

## Quick start

```bash
python setup.py            # install deps (numpy/scipy/streamlit/matplotlib)
python -m pytest           # 38 tests, ~2 s
streamlit run app.py       # launch the demo
```

Everything needed was already present in this environment (Python 3.13.5,
numpy 1.26.4, scipy 1.17.1, streamlit 1.50.0, matplotlib 3.11.1, pytest 9.1.1),
so `setup.py` was not required here. It is written and platform-correct for a
fresh machine.

The prototype needs **no machine-learning dependency**. `python setup.py
--with-ml` installs PyTorch only if you are reviving the archived CNN.

---

## File structure

```
.
├── claude.md                  # original prompt + project goals
├── context.md                 # this file
├── requirements.txt           # core deps; optional stack commented out
├── setup.py                   # installer; --with-ml adds MPS-aware torch
├── pytest.ini                 # puts src/ on sys.path; testpaths=tests
├── EXPLANATION.txt            # long-form plain-text walkthrough
├── app.py                     # Streamlit UI, 2 pages (plain language)
├── src/aeld/
│   ├── __init__.py            # public API re-exports
│   ├── config.py              # SimulationConfig + derived props
│   ├── signals.py             # forward model: 2 sensor signals
│   ├── gccphat.py             # GCC-PHAT -> TDoA -> position
│   └── detector.py            # orchestration + timing
├── tests/
│   ├── test_gccphat.py        # delay recovery, confidence gate
│   └── test_signals.py        # forward-model physics
├── archive/ml_model/
│   ├── cnn_filter.py          # the removed CNN, intact and unmodified
│   └── README.md              # why it went, how to bring it back, how to train
└── docs/superpowers/specs/
    └── 2026-08-26-aeld-prototype-design.md   # approved design spec
```

`src/aeld/` imports only numpy and scipy. No torch anywhere in the active path
(verified: `'torch' in sys.modules` is False after `run_detection`).

## Data flow

```
SimulationConfig  (sliders on page 1)
        |
        v
signals.generate_sensor_signals(leak_position, config)
        |   band-limited noise -> fractional delay -> attenuate -> + noise
        v
   sensor_a, sensor_b   <-- the ONLY thing downstream stages see
        |
        +--> gccphat.gcc_phat()           Stage 2: measure TDoA
                    |
                    v
             gccphat.locate_leak(tau)     Stage 3: tau -> metres
                    |
                    v
            DetectionResult  --> st.session_state --> page 2 plots
```

The information boundary matters: only `signals` ever sees the true leak
position. The localisation stages receive nothing but the two waveforms,
exactly as they would from real hardware. Ground truth is carried through
solely to score the result afterwards.

---

## Completed steps

### Step 0 — Clean slate
The directory contained a complete same-day AELD prototype (39 KB `app.py`,
plus `src/`, `tests/`, `models/`, `data/`, `configs/`). That was listed for the
user **before** anything was deleted; they confirmed with full knowledge, and
the wipe (12 entries → 0) then ran as the first step of the approved plan.

### Step 1 — Project tracking
`requirements.txt`, `setup.py`, `claude.md`, `context.md`, `pytest.ini`, and an
approved design spec under `docs/superpowers/specs/`.

`setup.py` installs via `sys.executable -m pip` so packages land in the
interpreter that runs it, and finishes with an environment report separating
required from optional packages. Its platform-aware PyTorch wheel selection is
retained behind `--with-ml` (Apple Silicon → default PyPI wheel shipping MPS;
NVIDIA → CUDA 12.1 index; otherwise → CPU-only index) for reviving the archived
CNN.

### Step 2 — Core physics & maths
- **`signals.py`** — leak modelled as white noise through a 4th-order
  Butterworth bandpass (100–2000 Hz), applied with `sosfiltfilt` for zero phase
  distortion (a filter that shifted the signal in time would inject a fake
  delay). Per-sensor fractional-sample delay via FFT phase ramp, exponential
  distance attenuation, and independent Gaussian noise per channel.
- **`gccphat.py`** — cross-power spectrum, PHAT whitening, band-limiting,
  spectral upsampling, inverse FFT, peak pick with parabolic sub-sample
  refinement, then `d = (L + v·tau)/2`.
- **`detector.py`** — orders the stages and times each one with
  `perf_counter`.
- **`cnn_filter.py`** — was built as specified (a real 2-D CNN running a
  genuine forward pass on MPS, returning a fixed 0.97) and has since been
  archived. See Step 5.

### Step 3 — Streamlit UI
Two pages via sidebar radio. Page 1: the three required sliders (pipe length,
wave velocity, noise level) plus advanced settings, derived-parameter metrics,
and a trigger button (random or manual position) writing to `st.session_state`.
Page 2: total processing time, calculated location vs. truth, per-stage timing,
the two sensor waveforms, the GCC-PHAT correlation curve with the peak marked
(plus a zoom showing the individual lag samples the parabola is fitted to), the
substituted arithmetic, and a to-scale pipe diagram.

### Step 4 — Code quality
Line-by-line comments throughout, written to explain *why* rather than restate
the code. Every non-obvious numerical choice carries its rationale.

### Step 5 — ML removal and UI rewrite (requested after the first build)

**The CNN was archived, not deleted.** `src/aeld/cnn_filter.py` moved intact to
`archive/ml_model/cnn_filter.py`, accompanied by a README covering why it was
removed, exactly how to reinstate it (file restore, `__init__` re-export,
`detector.py` stage insertion, the five `DetectionResult` fields to restore,
`setup.py --with-ml`), and a four-point checklist for training it into something
real.

Removed from the active path: the `classify_leak` import and stage in
`detector.py`; the `leak_probability`, `cnn_device`, `cnn_raw_logit`,
`time_cnn_ms` and `spectrogram` fields on `DetectionResult`; the CNN re-exports
in `__init__.py`; one assertion in `test_gccphat.py`; and all CNN UI in
`app.py`. `torch` moved to a commented-out optional block in
`requirements.txt`, and behind `--with-ml` in `setup.py`.

Measured effect: end-to-end detection went from **75.9 ms → 2.3 ms (33x
faster)**, because the CNN stage was almost entirely first-call Metal device
setup. PyTorch is no longer imported at all.

**The UI was rewritten for a non-technical reader.** The previous version led
with terms like TDoA, GCC-PHAT and peak sharpness. The new version leads with
plain sentences and defers every technical term to opt-in expanders:

| before | after |
|---|---|
| "Configuration & Simulation" | "Find a hidden leak" |
| "Trigger Random Leak" | "Start the leak" |
| "Wave Velocity (m/s)" | "How fast does sound travel in it?" |
| "Background Noise Level: 0.8" | "Noisy — traffic passing overhead" |
| "GCC-PHAT cross-correlation" | "Slide one recording against the other until they match" |
| "Peak sharpness 40.1 (threshold 10)" | "Spike height above background: 40.1x" |

Three additions aimed at making the result land rather than just be reported:

1. **Human-scale framing.** `human_scale_comparison()` converts the error into
   a precision ratio and maps it onto a familiar distance — a 0.134 mm error on
   a 200 m pipe becomes "1 part in 1,492,537 — the equivalent of measuring the
   344 km from London to Paris to within 23 centimetres." Computed from the
   actual result, so it stays truthful at any accuracy.
2. **A three-step narrative** replacing the flat list of plots: two microphones
   hear the same hiss → one hears it sooner → slide them until they match.
   Anchored to the lightning-and-thunder intuition.
3. **Failure framed as a feature.** When the confidence gate trips, the page
   explains that a tool admitting uncertainty beats one that sends a crew to
   dig up the wrong road, and invites the visitor to reproduce it by pushing the
   noise slider to 1.5.

The "why this is remarkable" callout is shown **only when the result is
trustworthy**; a low-confidence run gets the honest failure message instead, so
the framing can never oversell a bad answer.

---

## Bugs found and fixed

The test suite caught four real defects. Recording them because three were
invisible to inspection and would have surfaced mid-demo.

**1. MPS crash in adaptive pooling.** *(in the now-archived CNN; the fix is
preserved in `archive/ml_model/cnn_filter.py`)*
`RuntimeError: Adaptive pool MPS: input sizes must be divisible by output
sizes.` Spectrogram frame counts vary with pipe length, so the input was
usually non-divisible. Fixed by resampling every spectrogram to a fixed 64×64
via `F.interpolate` inside `forward()`, making 64→32→16→4 divide cleanly at
every stage. A CPU fallback now wraps the forward pass as well, so no future
Metal gap can take a demo down.

**2. Energy loss in the fractional delay.**
For an even-length real signal the last `rfft` bin is Nyquist, which has no
conjugate partner and must stay real. The phase ramp made it complex, and
`irfft` silently discarded the imaginary part — distorting the waveform and
leaking ~0.5% of the energy. Fixed by zeroing the Nyquist bin (the standard
treatment); residual loss dropped to ~7e-5.

**3. Sub-sample bias: a 33.25-sample delay read as 33.14.**
The whitened correlation peak is sinc-shaped, not parabolic, so fitting a
parabola across whole samples is systematically biased — worst near ±0.5
fractional offset. Fixed by zero-padding the spectrum before the inverse FFT
(`interp=4`), which is exact sinc interpolation onto a 4× finer lag grid where
a parabola *is* a good local fit.

**4. `app.py` referenced a non-existent field.**
Page 2 used `result.peak_index`, which `DetectionResult` did not carry — a
guaranteed `AttributeError` on the results page. Caught by exercising the plot
code paths headlessly rather than by reading the file. Added `peak_index` and
`n_bins_used` to the dataclass.

---

## Two engineering findings worth knowing

### Band-limiting the correlation (a 3.3× robustness win)
PHAT whitening forces every frequency bin to unit magnitude. That is what
sharpens the peak — but it also means a bin containing *only noise* is
amplified to exactly the same weight as a bin full of clean signal, and then
votes for a random lag. The leak radiates only 100–2000 Hz while sensor noise
is broadband, so above 2 kHz roughly half the spectrum was voting pure garbage.

Restricting the correlation to the leak's band cut the error in a
heavy-noise case from **136 m to 41 m**. The peak widens slightly (a narrower
band means a broader sinc), trading a little precision for a lot of robustness.

### GCC-PHAT does not fail gracefully — hence the confidence gate
Measured accuracy at 200 m / 1200 m/s, 60 seeds per noise level:

| noise | median error | within 2 m | mean sharpness |
|-------|--------------|------------|----------------|
| 0.0   | 0.00 m       | 100%       | 69.7 |
| 0.05  | 0.00 m       | 100%       | 61.1 |
| 0.2   | 0.00 m       | 100%       | 27.3 |
| 0.5   | 0.01 m       |  97%       |  8.4 |
| 1.0   | 65.35 m      |  13%       |  4.6 |
| 2.0   | 64.21 m      |   2%       |  4.6 |

That is a **cliff, not a slope**. Past ~0.5 the estimator stops drifting and
starts jumping to essentially random lags. A system that reported those with
full confidence would send crews to dig in the wrong place — worse than
reporting nothing.

So `peak_sharpness` (peak height ÷ mean level) gates the result, exposed as
`DetectionResult.is_confident`.

**Calibrating the threshold, including getting it wrong first.** An initial
sweep of 320 runs suggested 6.0 gave 100% precision. Broader sweeping found a
counterexample at sharpness 6.25 with 26.6 m error — the first calibration was
overfit to too narrow a sample. Re-calibrated across **1400 runs** spanning
noise 0.0–2.0, four pipe geometries (100–500 m, 1000–1500 m/s) and random
positions: the worst inaccurate estimate scored **7.98**. The threshold is set
to **10.0**, leaving ~25% margin while still accepting 62% of runs.

This is an empirical bound from one simulator, not a proof. A real deployment
must re-derive it from field data.

---

## Verification

```
38 passed in 2.04s
```

The suite's most important test is `test_never_confidently_wrong`, which sweeps
105 runs across noise 0.0–2.0 and asserts the actual safety invariant:
*whenever the system declares itself confident, it must be accurate.* Being
inaccurate is permitted only when also flagged unreliable. It includes a guard
against passing trivially by never being confident.

Supporting coverage: known-delay recovery including fractional delays, sign
convention, peak sharpness for correlated vs. uncorrelated input, lag-range
bounding, the three `locate_leak` boundary cases plus clamping, forward-model
geometry and attenuation, noise scaling, seed reproducibility, and config
derived properties.

Accuracy at default settings (noise 0.15) is ~0.13 mm — the localisation is
essentially exact when the signal supports it — and one full detection now
takes ~2.3 ms end to end.

Also verified after the ML removal: `setup.py --check` reports all required
packages present and correctly lists torch as optional; the app boots headless
with HTTP 200 and no errors or warnings; and `'torch' in sys.modules` is False
after a detection, confirming nothing in the active path pulls it in.

---

## Honest limitations

- **No leak/non-leak classification at all.** The system assumes what it hears
  is a leak and locates it. Distinguishing a leak from a passing truck or a
  running tap needs the archived CNN, trained. Until then, `is_confident`
  answers only "can this location be trusted?", never "is this really a leak?".
- **Single pipe, no reflections.** No junctions, branches, or dispersive
  propagation. Real pipe networks produce multipath that would complicate the
  correlation considerably.
- **Wave velocity is assumed known and constant.** In practice it varies with
  pipe material, diameter, and temperature, and the position estimate scales
  directly with it — a 5% velocity error is a 5% position error.
- **Attenuation is a single exponential coefficient**, not frequency-dependent.
- **The confidence threshold is calibrated on synthetic data only.**

## Possible next steps

1. Train a real classifier on labelled leak/non-leak recordings and reinstate it
   as Stage 2 — see `archive/ml_model/README.md` for the restore steps and the
   training checklist. This is the largest remaining gap to a product.
2. Wire WNTR for genuine hydraulic pressure modelling feeding signal amplitude.
3. Expose the detector over FastAPI and stream sensor frames over MQTT.
4. Extend to 3+ sensors, which allows over-determined localisation and
   self-consistency checks.
5. Replace strict PHAT with coherence-weighted (Hannan–Thomson) GCC, which
   degrades far better at low SNR than the cliff documented above.
