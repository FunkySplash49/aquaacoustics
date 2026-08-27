# claude.md — Project Brief & Goals

This file stores the originating prompt and the project's goals, so any future
session (human or model) can pick the work up with full context.

---

## Project

**AELD — Acoustic Emission Leak Detection.** A functional digital prototype
that locates a leak in a pressurised pipe by measuring the Time Difference of
Arrival (TDoA) of the leak's acoustic signature between two sensors, using
GCC-PHAT.

## Original prompt (verbatim)

> Act as a lead Python developer building a functional digital prototype for an
> Acoustic Emission Leak Detection (AELD) system. We are starting entirely from
> scratch. Before generating any code or documentation, execute a terminal
> command to delete all pre-existing files and hidden folders in the current
> working directory to ensure a completely clean environment.
>
> Here is the technical stack:
> - Physics Simulation: WNTR (EPANET wrapper)
> - Audio Processing & Math: SciPy, NumPy, Librosa
> - Machine Learning: PyTorch
> - Backend & Telemetry: FastAPI, paho-mqtt
> - Frontend: Streamlit
>
> Please execute the following steps sequentially to generate the complete
> prototype:
>
> **1. Clean Slate & Project Tracking**
> - Clear the current directory of all old files.
> - Create a `requirements.txt` file covering the stack.
> - Write a short Python setup script (`setup.py` or similar) that
>   automatically installs these dependencies. Ensure the PyTorch installation
>   instructions prioritize Metal Performance Shaders (MPS) for Apple Silicon
>   M4 hardware, while maintaining fallback compatibility for Windows/Linux.
> - Generate two documentation files: `claude.md` (to store this prompt and
>   project goals) and `context.md` (to track progress, file structures, and
>   completed steps).
>
> **2. Core Application Logic (The Physics & Math)**
> - Write the Python logic to simulate a single pipe.
> - Implement a function to generate synthetic acoustic signals for two sensors
>   (Sensor A and Sensor B) based on a given leak location.
> - Implement the GCC-PHAT (Generalized Cross-Correlation with Phase Transform)
>   algorithm using SciPy to calculate the Time Difference of Arrival (TDoA)
>   and determine the leak's exact position.
> - Include a mock PyTorch CNN inference function that acts as the noise filter
>   (simply returning a high probability of a leak to simulate a successful
>   detection for this prototype).
>
> **3. Streamlit User Interface**
> Write the `app.py` file using Streamlit, structured with a sidebar or tabs
> for two distinct subpages:
>
> *Subpage 1: Configuration & Simulation*
> - Include sliders to set constants: Pipe Length (meters), Wave Velocity
>   (m/s), and Background Noise Level.
> - Add a "Trigger Random Leak" button. When clicked, this should randomly
>   select a leak position along the pipe, run the simulation functions, and
>   store the results in Streamlit's session state.
>
> *Subpage 2: Detection Results & Inner Workings*
> - Display the total time taken to process the detection.
> - Show the final calculated leak location (distance from Sensor A).
> - Explain the inner workings visually: plot the synthetic audio waveforms
>   (what Sensor A and Sensor B "heard") and plot the GCC-PHAT cross-correlation
>   graph, clearly showing the mathematical peak that led the system to conclude
>   that specific location.
>
> **4. Code Quality**
> Add detailed, line-by-line comments explaining every coding block. The logic
> must be easily understandable so it can be explained during a live
> demonstration.

---

## Goals

1. **Locate a leak accurately.** Sub-metre accuracy on a 200 m pipe at
   realistic noise levels.
2. **Be explainable live.** Every stage visible and narratable: what the
   sensors heard, the correlation curve, the peak, the arithmetic.
3. **Be honest.** Never present a stub as a real measurement, and never report
   a location the underlying data does not support.
4. **Be understandable without background knowledge.** A visitor who has never
   heard of cross-correlation should be able to read the interface top to bottom
   and grasp both what happened and why it is impressive. Jargon appears only
   after the plain explanation, inside opt-in sections.
5. **Be verifiable.** The maths is covered by tests that would fail if it broke,
   rather than merely looking plausible in a plot.

---

## Scope decisions

Decisions 1 and 2 were agreed with the user before implementation; 1b came
after the first working build, also at the user's request.

**1. Lean stack.** The working system is Streamlit + NumPy/SciPy only. WNTR,
FastAPI and paho-mqtt are declared in `requirements.txt` for stack completeness
but are **not wired in** — they add install fragility and failure surface
without changing the demonstrated result. The single-pipe acoustics are
modelled analytically, which is exact for this geometry; an EPANET hydraulic
solve would add nothing to a timing measurement.

**1b. The ML model was removed (2026-08-26, after the first build).** The brief
asked for a mock PyTorch CNN noise filter, and one was built and delivered. It
was then withdrawn from the pipeline at the user's request: being an untrained
stub returning a hard-coded 0.97, it contributed nothing to the result while
being the heaviest dependency and the slowest stage (~73 ms of first-call
device setup against 0.36 ms for the actual localisation). Removing it made the
system 33x faster end to end (75.9 ms → 2.3 ms) and dropped the PyTorch
dependency entirely. **Nothing was deleted:** the module is preserved intact in
`archive/ml_model/`, with reinstatement instructions and a training checklist in
`archive/ml_model/README.md`.

**2. Modular package.** Core maths lives in `src/aeld/` with unit tests;
`app.py` is a thin presentation layer. This keeps each module small enough to
explain on its own and makes the physics testable without launching a browser.

## Clean slate

The prompt's instruction to delete all pre-existing files was carried out, but
**not** blindly. The directory turned out to contain a complete, same-day AELD
prototype (a 39 KB `app.py`, plus `src/`, `tests/`, `models/`, `data/`,
`configs/`). That was surfaced to the user with a listing before anything was
touched; the user confirmed deletion explicitly with that knowledge, and the
wipe then ran as the first step of the approved implementation. Nothing was
deleted on the strength of the prompt alone.

---

## Non-goals

- Real hydraulic network simulation (WNTR/EPANET)
- Real backend or telemetry transport (FastAPI, MQTT)
- Any machine learning in the active pipeline (archived; see above)
- Multi-pipe networks, junctions, reflections, or dispersive propagation
