"""
config.py - every tunable constant for the AELD simulation, in one place.
========================================================================

Keeping the knobs in a single frozen dataclass means:
  * the Streamlit sliders build ONE object and pass it down the pipeline,
  * the unit tests can construct a known configuration deterministically,
  * nothing downstream can silently mutate the settings mid-run (frozen=True).
"""

# `dataclass` removes the boilerplate __init__/__repr__ for a settings object.
from dataclasses import dataclass


# frozen=True makes instances immutable: any attempt to reassign a field raises.
# This is deliberate - the config is a snapshot of one simulation run.
@dataclass(frozen=True)
class SimulationConfig:
    """Physical + numerical parameters describing one simulated pipe run."""

    # ---------------- Pipe geometry & wave physics -----------------------
    pipe_length_m: float = 200.0
    """Distance between Sensor A (at 0 m) and Sensor B (at pipe_length_m)."""

    wave_velocity_ms: float = 1200.0
    """
    Speed of the acoustic wave along the pipe wall / water column, in m/s.
    Real values depend on pipe material and fluid: ~1200-1500 m/s in water
    filled plastic pipe, higher in steel. This is the single most important
    constant - the location estimate scales directly with it.
    """

    attenuation_per_m: float = 0.004
    """
    Exponential amplitude decay coefficient (1/metres). The signal reaching a
    sensor is scaled by exp(-attenuation_per_m * distance). This is why the
    nearer sensor records a visibly louder trace than the further one.
    """

    # ---------------- Sampling & signal content --------------------------
    sample_rate_hz: int = 8000
    """
    Samples per second. 8 kHz gives 0.125 ms resolution per sample; combined
    with sub-sample peak interpolation that is ample for metre-level accuracy
    at ~1200 m/s (one sample period == 0.15 m of path difference).
    """

    leak_band_hz: tuple = (100.0, 2000.0)
    """
    Bandpass limits of the leak's acoustic signature. A real leak radiates
    broadband turbulent noise concentrated in the low kHz range; band-limiting
    white noise is a faithful and cheap stand-in.
    """

    noise_level: float = 0.15
    """
    Standard deviation of the independent Gaussian sensor noise, relative to
    the unit-amplitude leak source. 0.0 = perfectly clean, 1.0 = noise as
    strong as the signal. Driven by the UI slider.
    """

    # ---------------- Numerical detail -----------------------------------
    min_duration_s: float = 0.10
    """Floor on the capture window so very short pipes still give enough data."""

    duration_margin: float = 3.0
    """
    The capture window is duration_margin * (pipe_length / velocity), so the
    full end-to-end travel time always fits comfortably inside the window.
    """

    random_seed: int = None
    """Optional RNG seed. Set it for reproducible demos, leave None for fresh
    noise on every run."""

    confidence_threshold: float = 10.0
    """
    Minimum GCC-PHAT peak sharpness (peak height / mean level) for the result
    to be reported as trustworthy.

    WHY A THRESHOLD IS NEEDED AT ALL
    --------------------------------
    GCC-PHAT does not degrade gracefully. Past roughly noise_level 0.5 it stops
    drifting and starts jumping to essentially random lags, because PHAT
    whitening gives noise-dominated frequency bins exactly the same vote as
    signal-dominated ones. A confident wrong answer sends a crew to dig in the
    wrong place, which is worse than reporting nothing - so a magnitude check
    on the correlation peak is a load-bearing safety feature, not a nicety.

    WHERE 10.0 COMES FROM: measured, not guessed. Across 1400 runs spanning
    noise levels 0.0-2.0, four pipe geometries (100-500 m, 1000-1500 m/s) and
    random leak positions, the worst INACCURATE estimate (error > 1% of pipe
    length) scored 7.98. Setting the bar at 10.0 leaves ~25% margin above that
    while still accepting 62% of all runs.

    An earlier calibration on a narrower sample suggested 6.0; broader
    sweeping found counterexamples at 6.25. The lesson is worth keeping in
    mind: this is an EMPIRICAL bound from one simulator, not a proof. A real
    deployment must re-derive it from field data.
    """

    # ---------------- Derived quantities ---------------------------------
    # These are computed properties, not stored fields, so they can never
    # drift out of sync with the values above.

    @property
    def max_travel_time_s(self) -> float:
        """Longest possible one-way travel time: the full length of the pipe."""
        return self.pipe_length_m / self.wave_velocity_ms

    @property
    def duration_s(self) -> float:
        """
        Length of the simulated capture window, in seconds.

        We need the window to be comfortably longer than the largest possible
        delay, otherwise the correlation peak would fall outside the data.
        """
        return max(self.min_duration_s,
                   self.duration_margin * self.max_travel_time_s)

    @property
    def n_samples(self) -> int:
        """Number of samples in the capture window."""
        return int(round(self.duration_s * self.sample_rate_hz))

    @property
    def max_tau_s(self) -> float:
        """
        Physically possible bound on |TDoA|.

        A leak somewhere between the two sensors can never produce a time
        difference larger than the full end-to-end travel time. Restricting
        the correlation search to this window rejects spurious far-lag peaks
        caused by noise - a cheap, physics-based robustness win.
        """
        return self.max_travel_time_s
