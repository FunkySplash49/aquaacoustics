# AquaAcoustics

Acoustic-emission leak detection demo and prototype. It simulates two pipe-mounted sensors, measures the inter-sensor time difference using GCC-PHAT, and converts that time difference into a distance along the pipe. Includes a Streamlit demo (map + detailed view), a small, well-documented physics + DSP library (src/aeld/) and unit tests.

Highlights
- Accurate, documented GCC‑PHAT implementation with sub-sample peak refinement.
- Reproducible simulator for pipe geometry, attenuation, and sensor noise.
- Streamlit UI to run the simulation/detection and visualise diagnostics.
- Clean separation between simulation, estimator, and orchestration (detector).
- An archived, untrained CNN classifier is available in archive/ml_model/ with instructions for reinstating it if you train it.

Status
- Prototype / demo. The core DSP and simulation are unit-tested and usable for demonstration and experimentation. Sites and presets in the UI are illustrative; the role selector is cosmetic (not a security boundary).

Table of contents
- Features
- Quick start
- Run the Streamlit demo
- Use the library from Python
- Project layout
- Implementation notes
- Tests
- Archival ML model
- Contributing
- License

Features
- GCC‑PHAT based time-difference-of-arrival estimator with:
  - PHAT whitening and optional frequency band-limiting for robustness.
  - Zero‑padding / sinc interpolation (upsampling) and parabolic sub-sample refinement.
  - A simple confidence metric (peak sharpness).
- Simulation of acoustic leak source, distance attenuation, fractional delays and independent sensor noise.
- Orchestrator that times pipeline stages and returns a single DetectionResult dataclass that contains diagnostics and raw arrays for plotting.

Quick start

1. Clone
```bash
git clone https://github.com/FunkySplash49/aquaacoustics.git
cd aquaacoustics
```

2. Create a virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate    # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
# or (for editable/development install)
# pip install -e .
```

Run the Streamlit demo
```bash
streamlit run app.py
```
Open the URL printed by Streamlit (usually http://localhost:8501). The app provides:
- Survey Map — pick a site and trigger detection.
- Leak Detection Detail — per-run diagnostics, plots and honest confidence reporting.

Use the library from Python
The core library lives in src/aeld and exposes a small public API re-exported from aeld.__init__.

Example:
```python
from aeld import SimulationConfig, run_detection

# create a config (defaults are sensible for demo)
config = SimulationConfig(pipe_length_m=200.0, sample_rate_hz=8000, random_seed=42)

# run a simulation for a leak 50 m from Sensor A
result = run_detection(50.0, config)

print(f"Estimated position: {result.estimated_position_m:.2f} m")
print(f"True position:      {result.true_position_m:.2f} m")
print(f"Error:              {result.error_m:.2f} m ({result.error_percent:.2f} % of pipe length)")
print(f"Peak sharpness:     {result.peak_sharpness:.2f}")
print(f"Is confident?       {result.is_confident}")
```

Project layout
(Top-level entries)
- .gitignore
- PROGRESS.md
- app.py                         — Streamlit application entry point (demo)
- archive/                       — archived/experimental modules (includes ml_model/)
- claude.md, context.md          — notes / context files
- docs/                          — documentation and assets
- pytest.ini
- requirements.txt               — Python dependencies (used by the demo & tests)
- setup.py                       — packaging / install helper
- src/
  - aeld/                        — acoustic-emission leak-detection library (simulation + GCC‑PHAT + detector)
    - __init__.py                 — public API re-exports (SimulationConfig, run_detection, etc.)
    - config.py                   — SimulationConfig dataclass and derived properties
    - signals.py                  — leak source generator, fractional delay, sensor simulation
    - gccphat.py                  — GCC‑PHAT estimator and locate_leak conversion
    - detector.py                 — pipeline orchestration, DetectionResult dataclass
  - aquaacoustics/                — Streamlit pages and map integration (UI code)
- tests/                          — pytest unit tests

How it fits together
- The Streamlit app (app.py) routes to UI pages which call into src/aeld/ for physics and DSP.
- src/aeld/signals.py synthesises sensor recordings for a configured leak position.
- src/aeld/gccphat.py estimates the TDoA using PHAT whitening and returns diagnostic arrays.
- src/aeld/detector.py orchestrates acquisition → localisation → conversion and returns a DetectionResult with timings and raw arrays for plotting.

Implementation notes & design decisions
- PHAT whitening is used to sharpen the correlation peak, but band-limiting is important to avoid noise-dominated bins voting equally; the code exposes a leak_band_hz config for this.
- Fractional delays are implemented via FFT-phase ramps to avoid quantisation error from integer-sample shifts.
- Peak refinement uses parabolic interpolation over an upsampled correlation to reach sub-sample accuracy.
- The pipeline is intentionally modular: acquisition (signals), localisation (gccphat), and conversion (locate_leak) are separate and independently testable.
- The code deliberately avoids machine learning dependencies in the main pipeline; an untrained CNN used previously is archived to avoid heavy runtime/install time.

Tests
Run the unit tests with:
```bash
pytest
```
The tests exercise the simulator, GCC‑PHAT implementation and numerical helpers.

Archived ML model
Under archive/ml_model/ there is a preserved PyTorch CNN stub and a README describing why it was removed from the active pipeline (it was untrained and returned a fixed probability). That README also contains step-by-step instructions for restoring and training the model if you choose to add a learned classifier back into the pipeline. See archive/ml_model/README.md.

Contributing
- Open an issue or submit a pull request.
- The code is structured to make adding new signal models, noise models, or UI pages straightforward.
- If you restore or introduce a trained classifier, document training datasets and held-out metrics clearly — the repository keeps the deterministic simulation and honest confidence metrics separate from any learned components.

License
- No license file present. Please add a LICENSE if you intend to make the project permissively available or to specify terms.

Acknowledgements
- Core DSP algorithms implemented and documented with emphasis on numerical correctness and reproducibility.
- Designed for clarity so the demo can teach the measurement principles as well as show results.

If you want, I can:
- Draft a ready-to-commit README.md file in this repository format (copyable).
- Add quick-start setup instructions tailored to pinned dependency versions from requirements.txt.
- Extract and include screenshots or examples from the Streamlit UI into the README.
