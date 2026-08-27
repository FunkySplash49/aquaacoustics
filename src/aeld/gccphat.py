"""
gccphat.py - GCC-PHAT time-delay estimation and leak localisation.
==================================================================

THE CORE IDEA (this is the module to explain in the demo)
---------------------------------------------------------
We have two recordings of the same leak, one delayed relative to the other.
We want that delay.

Plain cross-correlation would work, but it has a weakness: its peak is only as
sharp as the source's autocorrelation. Our leak noise is band-limited, so plain
correlation gives a broad, rounded hump - and a broad peak means an imprecise
delay. Any loud low-frequency rumble also dominates the result purely because
it is loud, not because it is informative.

GCC-PHAT ("Phase Transform") fixes both problems by throwing away magnitude
and keeping only PHASE. Every frequency bin then contributes equally, and the
correlation collapses to a sharp spike at the true delay.

THE MATH
--------
1. Cross-power spectrum:      R(f) = X_A(f) * conj(X_B(f))
2. PHAT whitening:            R'(f) = R(f) / (|R(f)| + eps)
3. Correlation in lag domain: r(tau) = IFFT{ R'(f) }
4. The delay is argmax r(tau), refined to sub-sample precision.

Why step 2 works: if sensor A's signal is sensor B's delayed by `d`, then
X_A = X_B * exp(-2*pi*i*f*d), so R(f) = |X_B|^2 * exp(-2*pi*i*f*d). Dividing
by |R| leaves exactly exp(-2*pi*i*f*d) - a pure phase ramp whose inverse FFT
is a perfect delta function at lag d. Magnitude was never carrying the delay
information; only phase was.
"""

import numpy as np

from .config import SimulationConfig


# ============================================================================
# Sub-sample peak refinement
# ============================================================================

def _parabolic_interpolation(y_prev: float, y_peak: float,
                             y_next: float) -> float:
    """
    Estimate the true peak's offset from the discrete maximum, in samples.

    The correlation is only sampled at integer lags, but the real peak almost
    always sits between two of them. Fitting a parabola through the highest
    sample and its two neighbours and solving for the vertex recovers that
    fraction - typically an order-of-magnitude accuracy gain for three lines
    of arithmetic.

    Standard vertex formula for three equally spaced points:

        offset = 0.5 * (y_prev - y_next) / (y_prev - 2*y_peak + y_next)

    Returns a value in roughly (-0.5, +0.5) to ADD to the integer peak index.
    """
    # The denominator is the discrete second derivative (curvature). If it is
    # ~0 the three points are collinear, there is no well-defined vertex, and
    # dividing would explode - so bail out with no correction.
    denominator = y_prev - 2.0 * y_peak + y_next
    if abs(denominator) < 1e-20:
        return 0.0

    offset = 0.5 * (y_prev - y_next) / denominator

    # Guard against a pathological fit (e.g. a noise spike next to the peak)
    # dragging the estimate more than half a sample, which would be nonsense.
    if not np.isfinite(offset) or abs(offset) > 1.0:
        return 0.0

    return float(offset)


# ============================================================================
# GCC-PHAT
# ============================================================================

def gcc_phat(sig_a: np.ndarray,
             sig_b: np.ndarray,
             sample_rate_hz: int,
             max_tau_s: float = None,
             interp: int = 4,
             freq_band_hz: tuple = None,
             eps: float = 1e-12) -> dict:
    """
    Estimate the time delay between two signals using GCC-PHAT.

    Parameters
    ----------
    sig_a, sig_b   : the two sensor recordings (1-D arrays, equal length).
    sample_rate_hz : samples per second.
    max_tau_s      : optional physical bound on |delay|, in seconds. Lags
                     beyond it are discarded before peak picking, which
                     rejects noise-driven far-lag artefacts.
    interp         : correlation upsampling factor (see below). 1 disables it.
    freq_band_hz   : optional (low, high) band in Hz. Bins outside it are
                     discarded (see below). Strongly recommended.
    eps            : small constant preventing division by zero in whitening.

    WHY `freq_band_hz` MATTERS (the big robustness win)
    ---------------------------------------------------
    PHAT whitening is a double-edged sword. Forcing every bin to unit
    magnitude is what sharpens the peak - but it also means a bin containing
    NOTHING BUT NOISE gets amplified to exactly the same weight as a bin full
    of clean leak signal. Its phase is then random, and it votes for a random
    lag.

    Our leak only radiates between ~100-2000 Hz, while sensor noise is
    broadband. So above 2000 Hz there is no signal at all - yet unrestricted
    PHAT gives those bins full voting rights. At low SNR they outvote the real
    ones and the estimate collapses to an essentially random lag.

    Restricting the correlation to the band where the leak actually has energy
    removes those votes entirely. The peak gets slightly wider (a narrower
    band means a broader sinc), but that costs a little precision to buy a
    large amount of robustness - a trade worth making.

    WHY `interp` EXISTS
    -------------------
    After whitening, the ideal correlation peak is not a parabola - it is a
    sinc-like (Dirichlet) kernel. Fitting a parabola to a sinc therefore has a
    SYSTEMATIC bias, worst when the true peak sits near half-way between two
    samples (measured empirically at ~0.1 samples of error).

    Zero-padding the spectrum before the inverse FFT is exact sinc
    interpolation in the time domain: it evaluates the same continuous
    correlation function on a finer lag grid. The parabola then only has to
    fit a much narrower slice of the sinc, where a parabola is a good local
    approximation, and the residual bias falls by roughly interp^2.

    Returns
    -------
    dict with:
        tau_s       -> estimated delay of sig_a relative to sig_b (seconds).
                       POSITIVE means sig_a arrived LATER than sig_b.
        n_bins_used -> how many frequency bins contributed after band-limiting.
        lags_s      -> lag axis (seconds), for plotting.
        correlation -> the GCC-PHAT curve over lags_s, for plotting.
        peak_index  -> index into `correlation` of the discrete maximum.
        peak_value  -> correlation height at that peak.
        sharpness   -> peak height / mean level; a rough confidence measure.
    """
    # ---- 1) Zero-pad to avoid circular wrap-around ---------------------
    # The FFT correlation theorem gives CIRCULAR correlation. Padding to at
    # least len(a)+len(b) makes the circular result equal the linear one, so
    # energy from one end cannot masquerade as a correlation at the other.
    n = sig_a.size + sig_b.size

    # ---- 2) Forward transforms ----------------------------------------
    # rfft is the real-input FFT: half the bins, same information.
    spec_a = np.fft.rfft(sig_a, n=n)
    spec_b = np.fft.rfft(sig_b, n=n)

    # ---- 3) Cross-power spectrum --------------------------------------
    # Multiplying by the conjugate of B correlates A against B. The phase of
    # each bin now encodes the delay at that frequency.
    cross_spectrum = spec_a * np.conj(spec_b)

    # ---- 4) THE PHASE TRANSFORM (the "PHAT" in GCC-PHAT) --------------
    # Divide out the magnitude, keeping only the unit-modulus phase term.
    # Every frequency now votes with equal weight, so a loud narrow band can
    # no longer dominate, and the resulting peak is sharp instead of rounded.
    cross_spectrum = cross_spectrum / (np.abs(cross_spectrum) + eps)

    # ---- 4b) Band-limiting: silence the bins with no signal in them ----
    # See the docstring. Whitening just gave every bin an equal vote; here we
    # revoke the vote of every bin outside the leak's actual frequency range,
    # because those bins contain only noise and would vote at random.
    if freq_band_hz is not None:
        # Bin centre frequencies in Hz. d=1/fs makes the output Hz rather
        # than cycles-per-sample.
        freqs_hz = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)

        low_hz, high_hz = freq_band_hz
        in_band = (freqs_hz >= low_hz) & (freqs_hz <= high_hz)

        # Guard: if the band excluded everything (a nonsense band), keep all
        # bins rather than returning a correlation of pure zeros.
        if np.any(in_band):
            cross_spectrum = cross_spectrum * in_band

    # ---- 5) Back to the lag domain, on a finer grid --------------------
    # Requesting an output length of n*interp implicitly zero-pads the
    # spectrum, which sinc-interpolates the correlation onto a lag grid
    # `interp` times denser. Lag units below are therefore "upsampled
    # samples", each worth 1/interp of an original sample period.
    interp = max(1, int(interp))
    n_up = n * interp
    correlation_full = np.fft.irfft(cross_spectrum, n=n_up)

    # ---- 6) Reorder so lag 0 sits in the middle ------------------------
    # irfft returns lags in FFT order: 0, 1, 2, ... then the NEGATIVE lags
    # wrapped onto the tail. fftshift-style reordering gives a natural,
    # plottable axis running from -max_shift to +max_shift.
    max_shift = n_up // 2

    # Optionally narrow the search to physically possible lags only.
    if max_tau_s is not None:
        # Convert the bound into upsampled units, plus a couple of original
        # samples of slack so a delay right at the physical limit isn't
        # clipped by rounding.
        physical_limit = int(np.ceil(max_tau_s * sample_rate_hz * interp)) \
            + 2 * interp
        max_shift = min(max_shift, physical_limit)

    # Negative lags (from the tail) followed by zero and the positive lags.
    correlation = np.concatenate((correlation_full[-max_shift:],
                                  correlation_full[:max_shift + 1]))

    # Matching lag axis in UPSAMPLED samples, then converted to seconds.
    lags_upsampled = np.arange(-max_shift, max_shift + 1)
    lags_s = lags_upsampled / float(sample_rate_hz * interp)

    # ---- 7) Pick the peak ---------------------------------------------
    # argmax on the raw (signed) correlation, NOT its absolute value: the
    # true delay produces a POSITIVE spike, and using abs() could latch onto
    # a negative trough and also break the parabolic fit below.
    peak_index = int(np.argmax(correlation))
    peak_value = float(correlation[peak_index])

    # ---- 8) Refine to sub-sample precision ----------------------------
    # Only possible when the peak has neighbours on both sides.
    offset = 0.0
    if 0 < peak_index < correlation.size - 1:
        offset = _parabolic_interpolation(correlation[peak_index - 1],
                                          correlation[peak_index],
                                          correlation[peak_index + 1])

    # Refined lag = integer lag + fractional correction, in upsampled units...
    refined_lag = lags_upsampled[peak_index] + offset
    # ... then converted to seconds. This is the TDoA estimate.
    tau_s = float(refined_lag / (sample_rate_hz * interp))

    # ---- 9) A crude confidence metric ---------------------------------
    # How far the peak stands above the typical correlation level. A tall,
    # isolated spike means a confident measurement; a peak barely above the
    # floor means the delay estimate should not be trusted.
    mean_level = float(np.mean(np.abs(correlation)))
    sharpness = (abs(peak_value) / mean_level) if mean_level > 0 else 0.0

    return {
        "tau_s": tau_s,
        "lags_s": lags_s,
        "correlation": correlation,
        "peak_index": peak_index,
        "peak_value": peak_value,
        "sharpness": sharpness,
        "n_bins_used": int(np.count_nonzero(cross_spectrum)),
    }


# ============================================================================
# TDoA -> physical position
# ============================================================================

def locate_leak(tau_s: float, config: SimulationConfig) -> float:
    """
    Convert a measured TDoA into a distance from Sensor A, in metres.

    DERIVATION
    ----------
    Let d = distance from Sensor A, L = pipe length, v = wave velocity.

        time to A  = d / v
        time to B  = (L - d) / v
        tau        = time to A - time to B = (2d - L) / v

    Rearranging for d:

        d = (L + v * tau) / 2

    Sanity checks on that formula:
      * leak exactly in the middle -> tau = 0     -> d = L/2       (correct)
      * leak at Sensor A           -> tau = -L/v  -> d = 0         (correct)
      * leak at Sensor B           -> tau = +L/v  -> d = L         (correct)

    The result is clipped to [0, L]: noise can push the estimate slightly past
    an endpoint, and a leak off the end of the pipe is not a physical answer.
    """
    position_m = (config.pipe_length_m + config.wave_velocity_ms * tau_s) / 2.0
    return float(np.clip(position_m, 0.0, config.pipe_length_m))
