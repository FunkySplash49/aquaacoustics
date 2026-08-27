"""
signals.py - synthesise what the two acoustic sensors "hear".
=============================================================

The forward model, in one paragraph:

A leak at distance `d` from Sensor A radiates broadband turbulent noise. That
same noise reaches Sensor A after d/v seconds and Sensor B after (L-d)/v
seconds, each attenuated by how far it travelled. Every sensor also adds its
own independent electrical/environmental noise. So the two recordings are
DELAYED, SCALED copies of one common source, plus uncorrelated noise.

That structure is exactly what GCC-PHAT is built to exploit.
"""

import numpy as np                 # arrays, FFTs, random numbers
from scipy import signal as sps    # Butterworth filter design + filtering

from .config import SimulationConfig


# ============================================================================
# Helper: fractional-sample delay via FFT phase shift
# ============================================================================

def fractional_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    """
    Delay a signal by an arbitrary (possibly non-integer) number of samples.

    WHY THIS MATTERS
    ----------------
    The naive way to delay a signal is to shift the array by a whole number of
    samples (np.roll). But a leak can sit anywhere along the pipe, so the true
    delay is almost never a whole number of samples. Rounding it would quantise
    every answer to the nearest sample period - at 8 kHz and 1200 m/s that is a
    0.15 m floor on the path difference, i.e. a built-in error we don't need.

    HOW IT WORKS
    ------------
    A pure time shift is a linear phase ramp in the frequency domain:

        x(t - tau)   <->   X(f) * exp(-2*pi*i*f*tau)

    So we FFT, multiply by that complex exponential, and inverse-FFT. Because
    the exponent is continuous in tau, fractional delays come for free.

    CAVEAT: the FFT treats the signal as periodic, so this is a CIRCULAR shift
    - samples pushed off the end wrap around to the front. The caller handles
    that by generating extra guard samples and trimming them off afterwards.

    Parameters
    ----------
    x : 1-D array, the signal to delay.
    delay_samples : float, delay in samples. Positive shifts LATER in time.

    Returns
    -------
    Array the same length as `x`, delayed by `delay_samples`.
    """
    n = x.size

    # rfft: real-input FFT. Half the spectrum (the rest is the conjugate
    # mirror), so it's ~2x faster and uses half the memory of a full fft.
    spectrum = np.fft.rfft(x)

    # Frequencies of each bin in CYCLES PER SAMPLE (d=1.0 is the default),
    # which is the unit that pairs with a delay measured in samples.
    freqs = np.fft.rfftfreq(n, d=1.0)

    # Apply the linear phase ramp. Negative exponent == delay (shift right).
    spectrum = spectrum * np.exp(-2j * np.pi * freqs * delay_samples)

    # NYQUIST CORRECTION (subtle but real):
    # For an even-length real signal, the last rfft bin is Nyquist (0.5
    # cycles/sample). An inverse real FFT REQUIRES that bin to be real -
    # it has no conjugate partner to cancel an imaginary part against.
    # Our phase ramp multiplies it by exp(-i*pi*tau), which is complex for
    # any non-integer tau. irfft would then silently discard the imaginary
    # part, distorting the waveform and leaking a little energy.
    # Zeroing the bin is the standard fix: Nyquist carries ~1/n of the total
    # energy (negligible), and discarding it keeps the filter well-behaved.
    if n % 2 == 0:
        spectrum[-1] = 0.0

    # Back to the time domain. n= is passed explicitly because rfft loses the
    # parity of the original length (odd-length signals would come back wrong).
    return np.fft.irfft(spectrum, n=n)


# ============================================================================
# Helper: the leak's acoustic source signature
# ============================================================================

def _generate_leak_source(n_samples: int,
                          config: SimulationConfig,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Build the raw acoustic signature radiated by the leak itself.

    A pressurised leak is a jet of fluid escaping through a small orifice. The
    turbulence produces a continuous hiss: broadband, no clean tones, energy
    concentrated in the low kHz. Band-limited white noise captures that well.

    Steps: white noise -> Butterworth bandpass -> normalise to unit amplitude.
    """
    # 1) Start with Gaussian white noise: flat power at every frequency.
    white = rng.standard_normal(n_samples)

    # 2) Band-limit it to the leak's frequency range.
    low_hz, high_hz = config.leak_band_hz

    # Nyquist = half the sample rate: the highest representable frequency.
    nyquist = 0.5 * config.sample_rate_hz

    # sps.butter with fs= given takes cutoffs in Hz directly. Clip the upper
    # edge just below Nyquist, since a cutoff at or above it is undefined.
    high_hz = min(high_hz, nyquist * 0.99)

    # 4th-order bandpass as second-order sections (SOS). SOS is numerically
    # far more stable than transfer-function ("ba") form for higher orders.
    sos = sps.butter(N=4, Wn=(low_hz, high_hz), btype="bandpass",
                     fs=config.sample_rate_hz, output="sos")

    # sosfiltfilt filters forwards then backwards: zero phase distortion.
    # Critical here - a filter that shifted the signal in time would inject a
    # fake delay and corrupt the very quantity we're trying to measure.
    filtered = sps.sosfiltfilt(sos, white)

    # 3) Normalise to peak amplitude 1.0 so the noise_level slider is a clean
    # signal-to-noise ratio, independent of the filter's arbitrary gain.
    peak = np.max(np.abs(filtered))
    if peak > 0:
        filtered = filtered / peak

    return filtered


# ============================================================================
# Main entry point: generate both sensor recordings
# ============================================================================

def generate_sensor_signals(leak_position_m: float,
                            config: SimulationConfig) -> dict:
    """
    Simulate the two sensor recordings for a leak at `leak_position_m`.

    Parameters
    ----------
    leak_position_m : distance of the leak from Sensor A, in metres.
                      Sensor A sits at 0 m, Sensor B at config.pipe_length_m.
    config : the SimulationConfig for this run.

    Returns
    -------
    dict with:
        time_s      -> time axis (seconds) for plotting
        sensor_a    -> Sensor A's recording
        sensor_b    -> Sensor B's recording
        true_delay_s-> the ground-truth TDoA (t_A - t_B), for scoring only
        distance_a  -> path length leak->A, metres
        distance_b  -> path length leak->B, metres
        amp_a, amp_b-> the attenuation factors actually applied
    """
    # ---- 0) Reproducibility -------------------------------------------
    # A seeded Generator gives identical noise every run (handy for tests and
    # for a demo you want to repeat); seed=None draws fresh entropy.
    rng = np.random.default_rng(config.random_seed)

    # ---- 1) Geometry ---------------------------------------------------
    # Keep the leak physically on the pipe, whatever the caller passed in.
    leak_position_m = float(np.clip(leak_position_m, 0.0, config.pipe_length_m))

    distance_a = leak_position_m                             # leak -> Sensor A
    distance_b = config.pipe_length_m - leak_position_m      # leak -> Sensor B

    # ---- 2) Travel times ----------------------------------------------
    # Time = distance / speed. These are the physical truths the detector is
    # trying to recover; it never gets to see them.
    time_a_s = distance_a / config.wave_velocity_ms
    time_b_s = distance_b / config.wave_velocity_ms

    # The TDoA is the DIFFERENCE. Sign convention (used consistently by
    # gccphat.locate_leak): positive means the wave reached A later than B,
    # i.e. the leak is nearer Sensor B.
    true_delay_s = time_a_s - time_b_s

    # ---- 3) Convert delays to samples ----------------------------------
    delay_a_samples = time_a_s * config.sample_rate_hz
    delay_b_samples = time_b_s * config.sample_rate_hz

    # ---- 4) Build the source with guard padding ------------------------
    n_out = config.n_samples   # how many samples we will finally return

    # `fractional_delay` shifts circularly, so samples pushed past the end
    # reappear at the front as garbage. We dodge that by generating a longer
    # source, shifting it, then cutting a clean window out of the MIDDLE. The
    # guard must exceed the largest delay we will apply.
    guard = int(np.ceil(max(delay_a_samples, delay_b_samples))) + 64
    n_source = n_out + 2 * guard

    source = _generate_leak_source(n_source, config, rng)

    # ---- 5) Delay, then trim to the clean window -----------------------
    # Both sensors are shifted from the SAME source array, so the RELATIVE
    # delay between them - the only thing the algorithm measures - is exact.
    delayed_a = fractional_delay(source, delay_a_samples)[guard:guard + n_out]
    delayed_b = fractional_delay(source, delay_b_samples)[guard:guard + n_out]

    # ---- 6) Distance attenuation ---------------------------------------
    # Amplitude decays exponentially with distance travelled. This is why the
    # nearer sensor's waveform is visibly taller in the UI plots.
    amp_a = float(np.exp(-config.attenuation_per_m * distance_a))
    amp_b = float(np.exp(-config.attenuation_per_m * distance_b))

    sensor_a = amp_a * delayed_a
    sensor_b = amp_b * delayed_b

    # ---- 7) Independent sensor noise -----------------------------------
    # Each sensor gets its OWN noise draw. That independence is the whole
    # point: noise does not correlate between channels, so cross-correlation
    # suppresses it while reinforcing the shared leak signal.
    if config.noise_level > 0:
        sensor_a = sensor_a + config.noise_level * rng.standard_normal(n_out)
        sensor_b = sensor_b + config.noise_level * rng.standard_normal(n_out)

    # ---- 8) Time axis for plotting -------------------------------------
    time_s = np.arange(n_out) / config.sample_rate_hz

    return {
        "time_s": time_s,
        "sensor_a": sensor_a,
        "sensor_b": sensor_b,
        "true_delay_s": true_delay_s,
        "distance_a": distance_a,
        "distance_b": distance_b,
        "amp_a": amp_a,
        "amp_b": amp_b,
    }
