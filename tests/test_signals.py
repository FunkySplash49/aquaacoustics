"""
test_signals.py - verify the forward (simulation) model behaves physically.
===========================================================================

If the synthetic signals are wrong, GCC-PHAT would be solving the wrong
problem perfectly. These tests check the physics the generator claims to
implement: correct travel times, correct attenuation, and correct noise.
"""

import numpy as np
import pytest

from aeld import SimulationConfig, generate_sensor_signals
from aeld.signals import fractional_delay


# ============================================================================
# Fractional delay helper
# ============================================================================

def test_fractional_delay_integer_shift():
    """An integer delay should behave like a plain array shift."""
    x = np.zeros(64)
    x[10] = 1.0                       # a single impulse at index 10

    delayed = fractional_delay(x, 5.0)

    # The impulse should now peak at index 15.
    assert int(np.argmax(delayed)) == 15


def test_fractional_delay_half_sample():
    """
    A half-sample delay has no single peak - the energy splits between the two
    neighbouring samples. That split is the evidence sub-sample delay works.
    """
    x = np.zeros(64)
    x[20] = 1.0

    delayed = fractional_delay(x, 0.5)

    # Samples 20 and 21 should be near-equal and clearly the two largest.
    assert delayed[20] == pytest.approx(delayed[21], rel=0.05)
    top_two = np.argsort(np.abs(delayed))[-2:]
    assert set(top_two.tolist()) == {20, 21}


def test_fractional_delay_preserves_length_and_energy():
    """
    A pure phase shift must not change length, and must very nearly preserve
    energy (Parseval: a unit-modulus multiplier is energy-preserving).

    The tolerance is 1% rather than exact because `fractional_delay`
    deliberately zeroes the Nyquist bin - that bin cannot carry a fractional
    phase shift in a real-valued signal. For white noise Nyquist holds roughly
    1/n of the total energy, so a 256-sample signal loses ~0.4%.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(256)

    delayed = fractional_delay(x, 7.3)

    assert delayed.size == x.size
    assert np.sum(delayed ** 2) == pytest.approx(np.sum(x ** 2), rel=0.01)


def test_fractional_delay_energy_loss_shrinks_with_length():
    """
    Confirms the diagnosis above: the only energy lost is the single Nyquist
    bin, so the relative loss should fall as the signal gets longer.

    Averaged over many random draws, deliberately. A single draw is useless
    here - the Nyquist bin is one chi-squared(1) sample, whose relative
    variance is enormous, so individual trials routinely run backwards. The
    1/n trend only appears in the mean.
    """
    def mean_relative_loss(n, trials=40):
        losses = []
        for seed in range(trials):
            # A fresh, independent generator per trial keeps the two sample
            # sizes statistically comparable.
            x = np.random.default_rng(seed).standard_normal(n)
            delayed = fractional_delay(x, 3.7)
            losses.append(
                abs(np.sum(delayed ** 2) - np.sum(x ** 2)) / np.sum(x ** 2))
        return float(np.mean(losses))

    # 16x more samples -> the Nyquist bin is a far smaller share of the total.
    assert mean_relative_loss(4096) < mean_relative_loss(256)


# ============================================================================
# Sensor signal generation
# ============================================================================

def test_true_delay_matches_geometry():
    """
    The reported ground-truth TDoA must equal (d_A - d_B)/v computed by hand.
    This is the definition the localiser inverts, so it has to be exact.
    """
    config = SimulationConfig(pipe_length_m=300.0, wave_velocity_ms=1500.0,
                              random_seed=1)
    leak = 100.0

    out = generate_sensor_signals(leak, config)

    # By hand: to A = 100/1500, to B = 200/1500, difference = -100/1500.
    expected = (100.0 - 200.0) / 1500.0
    assert out["true_delay_s"] == pytest.approx(expected)


def test_midpoint_leak_has_zero_delay():
    """A leak dead centre is equidistant, so the TDoA must be exactly zero."""
    config = SimulationConfig(pipe_length_m=200.0, random_seed=1)
    out = generate_sensor_signals(100.0, config)
    assert out["true_delay_s"] == pytest.approx(0.0)


def test_nearer_sensor_is_louder():
    """
    Attenuation is distance-dependent, so a leak close to Sensor A must give
    A a larger amplitude factor than B. This is the visible asymmetry in the
    waveform plots.
    """
    config = SimulationConfig(pipe_length_m=200.0, attenuation_per_m=0.01,
                              random_seed=1)

    out = generate_sensor_signals(20.0, config)      # much closer to A

    assert out["amp_a"] > out["amp_b"]
    # And quantitatively: exp(-0.01 * 20) vs exp(-0.01 * 180).
    assert out["amp_a"] == pytest.approx(np.exp(-0.01 * 20.0))
    assert out["amp_b"] == pytest.approx(np.exp(-0.01 * 180.0))


def test_signal_shapes_are_consistent():
    """All returned arrays must share one length, or plotting breaks."""
    config = SimulationConfig(random_seed=1)
    out = generate_sensor_signals(50.0, config)

    n = config.n_samples
    assert out["sensor_a"].size == n
    assert out["sensor_b"].size == n
    assert out["time_s"].size == n


def test_noise_level_increases_variance():
    """Turning the noise slider up must actually add energy to the traces."""
    quiet = generate_sensor_signals(
        50.0, SimulationConfig(noise_level=0.0, random_seed=5))
    loud = generate_sensor_signals(
        50.0, SimulationConfig(noise_level=0.8, random_seed=5))

    assert np.var(loud["sensor_a"]) > np.var(quiet["sensor_a"])


def test_seed_makes_output_reproducible():
    """Same seed, same signal - required for repeatable demos and tests."""
    config = SimulationConfig(random_seed=777)
    first = generate_sensor_signals(60.0, config)
    second = generate_sensor_signals(60.0, config)

    np.testing.assert_allclose(first["sensor_a"], second["sensor_a"])


def test_leak_position_is_clamped_to_pipe():
    """Out-of-range input must be pulled back onto the pipe, not error out."""
    config = SimulationConfig(pipe_length_m=100.0, random_seed=1)

    beyond = generate_sensor_signals(500.0, config)
    assert beyond["distance_a"] == pytest.approx(100.0)
    assert beyond["distance_b"] == pytest.approx(0.0)

    negative = generate_sensor_signals(-50.0, config)
    assert negative["distance_a"] == pytest.approx(0.0)


# ============================================================================
# Configuration derived properties
# ============================================================================

def test_duration_scales_with_pipe_length():
    """
    The capture window must always be long enough to contain the full
    end-to-end travel time, or the correlation peak would fall outside it.
    """
    config = SimulationConfig(pipe_length_m=1000.0, wave_velocity_ms=1000.0)

    # Travel time is 1.0 s; the window should be several times that.
    assert config.duration_s >= config.max_travel_time_s
    assert config.duration_s == pytest.approx(3.0)


def test_short_pipe_uses_minimum_duration():
    """Very short pipes fall back to the floor so there is enough data."""
    config = SimulationConfig(pipe_length_m=10.0, wave_velocity_ms=1200.0)
    assert config.duration_s == pytest.approx(config.min_duration_s)
