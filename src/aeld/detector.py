"""
detector.py - the end-to-end detection pipeline, with timing.
=============================================================

This module is the orchestrator. It owns the ORDER of operations and the
stopwatch; it deliberately contains no physics or DSP of its own, so each
stage stays independently testable.

Pipeline:

    1. ACQUIRE  - synthesise the two sensor recordings   (signals.py)
    2. LOCALISE - GCC-PHAT measures the TDoA             (gccphat.py)
    3. CONVERT  - TDoA -> distance from Sensor A         (gccphat.py)

Every stage is timed separately so the UI can show where the milliseconds go.

NOTE: an earlier version had a PyTorch CNN "is this even a leak?" filter
between acquisition and localisation. It was an untrained stub returning a
fixed probability, so it added a heavy dependency and the slowest stage in the
pipeline while contributing nothing to the result. It now lives in
archive/ml_model/ with instructions for reinstating it once it is trained.
"""

# perf_counter is a monotonic high-resolution clock - the correct tool for
# measuring short durations (time.time() can jump if the system clock changes).
from time import perf_counter
from dataclasses import dataclass, field

import numpy as np

from .config import SimulationConfig
from .signals import generate_sensor_signals
from .gccphat import gcc_phat, locate_leak


# ============================================================================
# Result container
# ============================================================================

@dataclass
class DetectionResult:
    """
    Everything one detection run produced: the answer, the diagnostics, and
    the raw arrays the UI needs for its "inner workings" plots.

    Bundled into a dataclass so it can be dropped straight into Streamlit's
    session_state and read back on another page as one coherent object.
    """

    # ---- The answer ----------------------------------------------------
    estimated_position_m: float      # detector's output: distance from Sensor A
    true_position_m: float           # ground truth (simulation only)
    error_m: float                   # |estimated - true|

    # ---- The measurement behind the answer -----------------------------
    estimated_tau_s: float           # TDoA measured by GCC-PHAT
    true_tau_s: float                # TDoA the physics actually produced
    peak_sharpness: float            # correlation peak height / mean level
    peak_index: int                  # index of the peak within `correlation`
    n_bins_used: int                 # frequency bins that survived band-limiting

    # ---- Timing (milliseconds) ------------------------------------------
    time_acquire_ms: float
    time_gccphat_ms: float
    time_total_ms: float

    # ---- Raw arrays for plotting ---------------------------------------
    # field(repr=False) keeps these big arrays out of the printed repr, so
    # debugging output stays readable.
    time_s: np.ndarray = field(repr=False, default=None)
    sensor_a: np.ndarray = field(repr=False, default=None)
    sensor_b: np.ndarray = field(repr=False, default=None)
    lags_s: np.ndarray = field(repr=False, default=None)
    correlation: np.ndarray = field(repr=False, default=None)

    # ---- Echo of the configuration used --------------------------------
    config: SimulationConfig = None

    @property
    def error_percent(self) -> float:
        """Localisation error as a percentage of total pipe length."""
        if self.config is None or self.config.pipe_length_m == 0:
            return 0.0
        return 100.0 * self.error_m / self.config.pipe_length_m

    @property
    def is_confident(self) -> bool:
        """
        Whether the correlation peak is strong enough to trust the location.

        This is the system's self-assessment, computed WITHOUT any knowledge of
        the true answer - so it is exactly what a real deployment would have.
        Below the threshold the reported position should be treated as "leak
        possibly detected, location unreliable" rather than as a coordinate to
        go dig at. See SimulationConfig.confidence_threshold.
        """
        if self.config is None:
            return False
        return self.peak_sharpness >= self.config.confidence_threshold


# ============================================================================
# The pipeline
# ============================================================================

def run_detection(leak_position_m: float,
                  config: SimulationConfig) -> DetectionResult:
    """
    Run the full detection pipeline for a leak at `leak_position_m`.

    Note the information boundary: this function is HANDED the true position
    because it must simulate the physics. But only `signals` ever sees it -
    the localisation stages receive nothing but the two waveforms, exactly
    as they would from real hardware. The true value is carried through purely
    to score the result afterwards.
    """
    # Master stopwatch for the whole pipeline.
    t_start = perf_counter()

    # ---- STAGE 1: ACQUIRE ----------------------------------------------
    # Synthesise what the two sensors hear. On real hardware this is where
    # you would instead read a buffer off the ADC.
    t0 = perf_counter()
    signals = generate_sensor_signals(leak_position_m, config)
    time_acquire_ms = (perf_counter() - t0) * 1000.0

    sensor_a = signals["sensor_a"]
    sensor_b = signals["sensor_b"]

    # ---- STAGE 2: LOCALISE (GCC-PHAT) ----------------------------------
    # Two physics-derived constraints are handed to the estimator:
    #   max_tau_s    - restricts the search to physically reachable lags.
    #   freq_band_hz - restricts it to the band where a leak actually radiates,
    #                  so noise-only bins cannot vote. This is what keeps the
    #                  estimate stable at low signal-to-noise ratios.
    t0 = perf_counter()
    gcc = gcc_phat(sensor_a, sensor_b,
                   sample_rate_hz=config.sample_rate_hz,
                   max_tau_s=config.max_tau_s,
                   freq_band_hz=config.leak_band_hz)
    time_gccphat_ms = (perf_counter() - t0) * 1000.0

    # ---- STAGE 3: CONVERT TDoA -> POSITION -----------------------------
    estimated_position_m = locate_leak(gcc["tau_s"], config)

    # Stop the master clock.
    time_total_ms = (perf_counter() - t_start) * 1000.0

    # ---- Score the result ----------------------------------------------
    # Absolute error against ground truth. Available only in simulation; on
    # real hardware you'd confirm by excavating.
    true_position_m = float(np.clip(leak_position_m, 0.0, config.pipe_length_m))
    error_m = abs(estimated_position_m - true_position_m)

    return DetectionResult(
        estimated_position_m=estimated_position_m,
        true_position_m=true_position_m,
        error_m=error_m,
        estimated_tau_s=gcc["tau_s"],
        true_tau_s=signals["true_delay_s"],
        peak_sharpness=gcc["sharpness"],
        peak_index=gcc["peak_index"],
        n_bins_used=gcc["n_bins_used"],
        time_acquire_ms=time_acquire_ms,
        time_gccphat_ms=time_gccphat_ms,
        time_total_ms=time_total_ms,
        time_s=signals["time_s"],
        sensor_a=sensor_a,
        sensor_b=sensor_b,
        lags_s=gcc["lags_s"],
        correlation=gcc["correlation"],
        config=config,
    )
