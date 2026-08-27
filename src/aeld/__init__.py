"""
AELD - Acoustic Emission Leak Detection (prototype)
===================================================

A leak in a pressurised pipe is a continuous broadband noise source. Two
sensors clamped to the pipe hear the SAME noise, but at slightly different
times because sound takes longer to reach the further sensor. Measuring that
tiny time difference (the Time Difference of Arrival, TDoA) tells you where
along the pipe the leak is.

Public API (import these from `aeld`):

    SimulationConfig    - all tunable constants in one dataclass
    generate_sensor_signals - synthesise what Sensor A and Sensor B "hear"
    gcc_phat            - GCC-PHAT cross-correlation -> TDoA
    locate_leak         - convert a TDoA into a distance from Sensor A
    run_detection       - full pipeline, timed, returns a DetectionResult

The package deliberately has NO machine-learning dependency. An untrained CNN
"is this a leak?" filter used to sit in the pipeline; it is archived under
archive/ml_model/ with instructions for reinstating it once trained.
"""

# Re-export the public surface so callers can do `from aeld import ...`
# instead of reaching into individual modules.
from .config import SimulationConfig
from .signals import generate_sensor_signals, fractional_delay
from .gccphat import gcc_phat, locate_leak
from .detector import run_detection, DetectionResult

# Explicit __all__ documents the intended API and controls `import *`.
__all__ = [
    "SimulationConfig",
    "generate_sensor_signals",
    "fractional_delay",
    "gcc_phat",
    "locate_leak",
    "run_detection",
    "DetectionResult",
]
