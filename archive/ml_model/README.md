# Archived: PyTorch CNN Noise Filter

**Status:** Removed from the active pipeline on 2026-08-26. Preserved here,
intact and unmodified, for later use.

**Why it was removed:** the CNN was an untrained stub returning a hard-coded
0.97 leak probability. It contributed nothing to the actual result while adding
a heavy dependency (PyTorch), the slowest stage in the pipeline (~73 ms of
first-call device setup vs. 0.36 ms for the real localisation), and a
confusing "97% confident" number in the UI that did not mean anything. Taking
it out made the working system faster, simpler to install, and easier to
explain honestly.

**Nothing was lost.** The file below is the complete original.

---

## What this module does

`cnn_filter.py` is the intended **Stage 2** of the pipeline: an acoustic event
classifier that answers "is this even a leak?" before the localiser spends
effort on it. In a real deployment sensors trigger constantly on non-leak
events — passing trucks, pumps starting, valves closing, ordinary water use —
and localising every one would bury operators in false alarms.

### What is real in it

- A working 2-D CNN in PyTorch:
  `Conv2d(1→8) → ReLU → MaxPool → Conv2d(8→16) → ReLU → MaxPool →
   AdaptiveAvgPool(4×4) → Flatten → Linear(256→32) → Linear(32→1)`
- A real log-magnitude spectrogram feature (via `scipy.signal.spectrogram`, so
  it never hard-depends on librosa)
- A real forward pass that genuinely executes on the GPU
- Device auto-selection: **MPS → CUDA → CPU**, with a CPU fallback wrapped
  around the forward pass so a Metal backend gap cannot crash the caller
- Fixed 64×64 input resampling via `F.interpolate`, which exists for a real
  reason: Apple's MPS backend raises a hard `RuntimeError` on adaptive pooling
  when the input size is not an exact multiple of the output size, and
  spectrogram frame counts vary with pipe length. 64→32→16→4 divides cleanly.

### What is not real in it

**The network is untrained — its weights are random, and the returned
probability is the hard-coded constant `MOCK_LEAK_PROBABILITY = 0.97`.** The
code computes a genuine `raw_logit` and then deliberately ignores it. This was
scoped as a stub, and it must be described as one.

---

## How to bring it back

### 1. Restore the file

```bash
cp archive/ml_model/cnn_filter.py src/aeld/cnn_filter.py
```

### 2. Re-export it from the package

In `src/aeld/__init__.py`:

```python
from .cnn_filter import classify_leak, get_device
```

and add `"classify_leak"`, `"get_device"` to `__all__`.

### 3. Re-insert the stage in `detector.py`

Add the import:

```python
from .cnn_filter import classify_leak
```

Then between acquisition and localisation in `run_detection()`:

```python
t0 = perf_counter()
cnn = classify_leak(sensor_a, sensor_b, config.sample_rate_hz)
time_cnn_ms = (perf_counter() - t0) * 1000.0
```

Add these fields back to `DetectionResult` and populate them from `cnn`:

| field | source |
|---|---|
| `leak_probability` | `cnn["leak_probability"]` |
| `cnn_device` | `cnn["device"]` |
| `cnn_raw_logit` | `cnn["raw_logit"]` |
| `time_cnn_ms` | measured above |
| `spectrogram` | `cnn["spectrogram"]` |

Include `time_cnn_ms` in the total, and gate localisation on
`cnn["is_leak"]` if you want the filter to actually filter.

### 4. Reinstall PyTorch

```bash
python setup.py --with-ml
```

`setup.py` retains the full platform-aware wheel selection (Apple Silicon →
default PyPI wheel with MPS; NVIDIA → CUDA index; otherwise → CPU-only index).
Nothing about that logic was removed.

---

## Before trusting it: train it

Restoring the file gives you working plumbing, **not** a working classifier.
To make it real:

1. **Collect labelled audio** — leak and non-leak acoustic events from real
   pipes, ideally across pipe materials, diameters and background conditions.
   This is the hard part and the main cost.
2. **Replace the mock return.** In `classify_leak()`, delete the
   `MOCK_LEAK_PROBABILITY` assignment and return
   `float(torch.sigmoid(logit).item())` instead.
3. **Add a training script** with a held-out validation split, and persist
   weights (`torch.save` / `load_state_dict`) instead of constructing a fresh
   random network on every call — which is what the current code does, and is
   only acceptable because the output is ignored.
4. **Report honest metrics** — precision/recall on held-out data. Until those
   exist, the module makes no classification claim.

## A note on confidence

While this stub was in the pipeline, the UI showed "97% leak probability",
which was meaningless. The number that actually indicates whether a result can
be trusted is the **GCC-PHAT peak sharpness** (`DetectionResult.is_confident`),
calibrated over 1400 measured runs. If this module is restored, keep that
distinction clear in the UI: a trained classifier answers *"is it a leak?"*,
while peak sharpness answers *"can we trust the location?"* — two different
questions.
