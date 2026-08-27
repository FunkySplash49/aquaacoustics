"""
test_gccphat.py - prove the delay estimation and localisation maths are right.
==============================================================================

These are the tests that matter. Everything else in the prototype is
presentation; if GCC-PHAT recovers the wrong delay, the system is simply
wrong. So each test injects a KNOWN delay and asserts the algorithm gets it
back within tolerance.
"""

import numpy as np
import pytest

from aeld import SimulationConfig, gcc_phat, locate_leak, run_detection
from aeld.signals import fractional_delay


# ============================================================================
# The fundamental test: known delay in, same delay out
# ============================================================================

@pytest.mark.parametrize("true_delay_samples", [0.0, 1.0, 5.0, -5.0,
                                                12.5, -12.5, 33.25])
def test_recovers_known_delay(true_delay_samples):
    """
    Build a signal, delay a copy by a known amount, and check GCC-PHAT
    measures that amount back. Includes fractional delays (12.5, 33.25) to
    verify the sub-sample parabolic interpolation actually earns its keep.
    """
    sample_rate = 8000
    n = 4096

    # A reproducible broadband source. Broadband matters: GCC-PHAT needs
    # energy across many frequencies to localise the peak sharply.
    rng = np.random.default_rng(42)
    source = rng.standard_normal(n)

    # Reference channel is the source itself; the other is a delayed copy.
    # Guard padding + trimming avoids circular wrap contaminating the result.
    guard = 128
    padded = np.concatenate([np.zeros(guard), source, np.zeros(guard)])
    sig_b = padded[guard:guard + n]
    sig_a = fractional_delay(padded, true_delay_samples)[guard:guard + n]

    result = gcc_phat(sig_a, sig_b, sample_rate_hz=sample_rate)

    # Convert the measured delay back into samples for comparison.
    measured_samples = result["tau_s"] * sample_rate

    # Tolerance of 0.1 samples. Plain integer peak-picking could only ever
    # manage 0.5, so passing this proves the interpolation works.
    assert measured_samples == pytest.approx(true_delay_samples, abs=0.1)


def test_sign_convention():
    """
    Pin down the sign convention, because getting it backwards would mirror
    every leak location to the wrong end of the pipe.

    Contract: a POSITIVE tau means sig_a arrived LATER than sig_b.
    """
    sample_rate = 8000
    n = 2048
    rng = np.random.default_rng(7)
    source = rng.standard_normal(n + 256)

    # Delay channel A by 20 samples; leave B undelayed.
    sig_a = fractional_delay(source, 20.0)[128:128 + n]
    sig_b = source[128:128 + n]

    result = gcc_phat(sig_a, sig_b, sample_rate_hz=sample_rate)

    # A is later, so tau must be positive.
    assert result["tau_s"] > 0
    assert result["tau_s"] * sample_rate == pytest.approx(20.0, abs=0.1)


def test_peak_is_sharp_for_correlated_signals():
    """
    A genuine shared source should give a peak standing well clear of the
    correlation floor - that's the 'confidence' signal the UI reports.
    """
    rng = np.random.default_rng(3)
    n = 4096
    source = rng.standard_normal(n + 256)

    sig_a = fractional_delay(source, 10.0)[128:128 + n]
    sig_b = source[128:128 + n]

    result = gcc_phat(sig_a, sig_b, sample_rate_hz=8000)

    # For two clean copies of one source, the peak towers over the mean.
    assert result["sharpness"] > 10.0


def test_uncorrelated_signals_give_flat_correlation():
    """
    Two INDEPENDENT noise signals share no source, so there is no true delay
    and no dominant peak. Sharpness should stay low - this is what stops the
    system confidently reporting a location from pure noise.
    """
    rng = np.random.default_rng(11)
    n = 4096
    sig_a = rng.standard_normal(n)
    sig_b = rng.standard_normal(n)   # independent draw: no shared content

    result = gcc_phat(sig_a, sig_b, sample_rate_hz=8000)

    # Far below the >10 seen for genuinely correlated input.
    assert result["sharpness"] < 8.0


def test_max_tau_restricts_lag_range():
    """The physical bound should actually narrow the returned lag axis."""
    rng = np.random.default_rng(5)
    n = 8192
    sig = rng.standard_normal(n)

    unbounded = gcc_phat(sig, sig, sample_rate_hz=8000)
    bounded = gcc_phat(sig, sig, sample_rate_hz=8000, max_tau_s=0.01)

    assert bounded["lags_s"].size < unbounded["lags_s"].size
    # 0.01 s at 8 kHz = 80 samples, plus 2 samples of slack each side.
    assert np.max(np.abs(bounded["lags_s"])) <= 0.01 + 3 / 8000


# ============================================================================
# TDoA -> position conversion
# ============================================================================

def test_locate_leak_midpoint():
    """Zero TDoA means equidistant, i.e. the exact middle of the pipe."""
    config = SimulationConfig(pipe_length_m=200.0, wave_velocity_ms=1000.0)
    assert locate_leak(0.0, config) == pytest.approx(100.0)


def test_locate_leak_at_sensor_a():
    """
    A leak at Sensor A: sound arrives at A instantly and at B after L/v,
    so tau = -L/v and the position must come out as 0 m.
    """
    config = SimulationConfig(pipe_length_m=200.0, wave_velocity_ms=1000.0)
    tau = -200.0 / 1000.0
    assert locate_leak(tau, config) == pytest.approx(0.0)


def test_locate_leak_at_sensor_b():
    """Mirror image of the above: tau = +L/v puts the leak at the far end."""
    config = SimulationConfig(pipe_length_m=200.0, wave_velocity_ms=1000.0)
    tau = 200.0 / 1000.0
    assert locate_leak(tau, config) == pytest.approx(200.0)


def test_locate_leak_clamps_to_pipe():
    """
    Noise can push tau past its physical limit. The result must still be a
    point ON the pipe, never off the end.
    """
    config = SimulationConfig(pipe_length_m=200.0, wave_velocity_ms=1000.0)
    assert locate_leak(10.0, config) == pytest.approx(200.0)    # clamped high
    assert locate_leak(-10.0, config) == pytest.approx(0.0)     # clamped low


# ============================================================================
# End-to-end: the whole pipeline on a clean signal
# ============================================================================

@pytest.mark.parametrize("leak_position", [25.0, 50.0, 100.0, 150.0, 175.0])
def test_end_to_end_localisation_accuracy(leak_position):
    """
    The headline claim: with low noise, the pipeline locates a leak to within
    a couple of metres on a 200 m pipe. Seeded for determinism so this test
    cannot flake.
    """
    config = SimulationConfig(
        pipe_length_m=200.0,
        wave_velocity_ms=1200.0,
        noise_level=0.05,      # low but non-zero: still a realistic test
        random_seed=123,
    )

    result = run_detection(leak_position, config)

    # 2 m on a 200 m pipe = 1% of pipe length.
    assert result.error_m < 2.0
    assert result.true_position_m == pytest.approx(leak_position)


def test_accuracy_holds_up_to_moderate_noise():
    """
    Characterises the usable operating range. Up to noise_level 0.5 the
    localiser stays essentially exact, which is the honest claim to make about
    this system.
    """
    for noise in (0.0, 0.1, 0.3, 0.5):
        errors = []
        for seed in range(12):
            config = SimulationConfig(pipe_length_m=200.0,
                                      wave_velocity_ms=1200.0,
                                      noise_level=noise,
                                      random_seed=seed)
            errors.append(run_detection(70.0, config).error_m)

        # Median rather than max: at 0.5 an occasional outlier is expected,
        # and the confidence metric is what catches those (test below).
        assert np.median(errors) < 1.0, f"noise={noise}"


def test_never_confidently_wrong():
    """
    THE MOST IMPORTANT TEST IN THE SUITE.

    GCC-PHAT does not degrade gracefully: past a certain noise level it stops
    drifting and starts jumping to essentially random lags. A system that
    reported those with full confidence would send crews to dig in the wrong
    place - far worse than reporting nothing.

    So the invariant is not "always accurate". It is: whenever the system
    declares itself CONFIDENT, it must be right. Inaccurate answers are
    acceptable only if they are also flagged as unreliable.

    Swept across the full noise range, including levels where localisation is
    known to fail outright.
    """
    confident_count = 0

    for noise in (0.0, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0):
        for seed in range(15):
            config = SimulationConfig(pipe_length_m=200.0,
                                      wave_velocity_ms=1200.0,
                                      noise_level=noise,
                                      random_seed=seed)
            # Vary the leak position too, so this isn't one lucky geometry.
            position = 20.0 + (seed * 11.0) % 160.0
            result = run_detection(position, config)

            if result.is_confident:
                confident_count += 1
                # The actual invariant: confident => accurate.
                assert result.error_m < 2.0, (
                    f"CONFIDENTLY WRONG: noise={noise} seed={seed} "
                    f"error={result.error_m:.1f}m "
                    f"sharpness={result.peak_sharpness:.2f}")

    # Guard against the test passing trivially by simply never being
    # confident. The sweep above is deliberately weighted towards punishing
    # noise (4 of its 7 levels are >= 0.8, where localisation is expected to
    # fail), so only around a third of the 105 runs should qualify. This bound
    # just has to be high enough to prove the detector still commits to an
    # answer when the signal genuinely supports one.
    assert confident_count >= 25, (
        f"only {confident_count} confident results; threshold may be too strict")


def test_high_noise_is_flagged_unconfident():
    """
    The complement of the above: at punishing noise the system must actually
    lower its hand rather than quietly reporting a number.
    """
    unconfident = 0
    trials = 20

    for seed in range(trials):
        config = SimulationConfig(pipe_length_m=200.0,
                                  wave_velocity_ms=1200.0,
                                  noise_level=2.0,   # noise 2x the signal
                                  random_seed=seed)
        if not run_detection(50.0, config).is_confident:
            unconfident += 1

    # At this noise level localisation essentially never works, so the system
    # should be declining to commit in the overwhelming majority of runs.
    assert unconfident >= 0.8 * trials


def test_clean_signal_is_always_confident():
    """With no noise the peak is unmistakable; confidence must reflect that."""
    config = SimulationConfig(noise_level=0.0, random_seed=4)
    result = run_detection(120.0, config)

    assert result.is_confident
    assert result.error_m < 0.5


def test_result_reports_timing_and_confidence():
    """The UI depends on these fields being populated and sane."""
    config = SimulationConfig(random_seed=1)
    result = run_detection(80.0, config)

    # Every stage must record a positive duration ...
    assert result.time_acquire_ms > 0
    assert result.time_gccphat_ms > 0
    # ... and the total must cover the sum of its parts.
    assert result.time_total_ms >= result.time_gccphat_ms

    # Plotting arrays must be present and correctly paired.
    assert result.sensor_a.size == result.sensor_b.size
    assert result.lags_s.size == result.correlation.size
