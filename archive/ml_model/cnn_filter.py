"""
cnn_filter.py - mock PyTorch CNN acting as the acoustic noise filter.
=====================================================================

WHAT THIS IS FOR
----------------
In a deployed AELD system, sensors trigger constantly on things that are NOT
leaks: a passing truck, a pump starting, a valve closing, someone using water.
Localising every one of those would flood operators with false alarms. So a
classifier sits in front of the localiser and answers one question first:
"is this acoustic event actually a leak?" Only then is GCC-PHAT run.

WHAT THIS IS *NOT*
------------------
An honest disclaimer, because it matters when demonstrating this: the network
below is UNTRAINED. Its weights are random. It performs a real forward pass on
a real spectrogram through a real PyTorch graph - so the plumbing, the tensor
shapes, and the MPS/CUDA device acceleration are all genuine and measurable -
but the returned probability is a FIXED HIGH CONSTANT, not a learned decision.

It is a stub standing in for a trained model, exactly as scoped for this
prototype. It makes no classification claim about any real signal. Training a
genuine classifier needs a labelled corpus of leak and non-leak recordings.
"""

import numpy as np
from scipy import signal as sps    # spectrogram, so librosa stays optional

# PyTorch is imported defensively. If the wheel failed to install, the rest of
# the prototype (which is where the actual maths lives) must still run.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:                # pragma: no cover - environment dependent
    TORCH_AVAILABLE = False

# Every spectrogram is resampled to this fixed (freq, time) shape before it
# enters the network. Two reasons:
#   1. The number of spectrogram time frames depends on the capture duration,
#      which changes with pipe length. A fixed input keeps the graph static.
#   2. Apple's MPS backend cannot do adaptive pooling when the input size is
#      not an exact multiple of the output size. 64 -> 32 -> 16 -> 4 divides
#      cleanly at every stage, so the network stays MPS-compatible.
CNN_INPUT_SHAPE = (64, 64)


# The fixed probability the stub reports. Named as a constant so nobody
# mistakes it for something the network computed.
MOCK_LEAK_PROBABILITY = 0.97


# ============================================================================
# Device selection: MPS -> CUDA -> CPU
# ============================================================================

def get_device():
    """
    Pick the best available torch device, preferring Apple Silicon's GPU.

    Priority order, matching the setup script:
      1. "mps"  - Metal Performance Shaders: the GPU on M1-M4 Macs.
      2. "cuda" - an NVIDIA GPU on Linux/Windows.
      3. "cpu"  - always available fallback.

    Returns a torch.device, or the string "unavailable" if torch is missing.
    """
    if not TORCH_AVAILABLE:
        return "unavailable"

    # is_built() checks the wheel was COMPILED with MPS support;
    # is_available() checks this machine can actually run it. Both required.
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================================
# The network
# ============================================================================

if TORCH_AVAILABLE:

    class LeakClassifierCNN(nn.Module):
        """
        A small 2-D CNN over a spectrogram patch: the standard shape for
        acoustic event classification.

        Architecture rationale (what a trained version would be doing):
          * Conv2d layers learn local time-frequency textures. A leak's hiss
            is a steady broadband band; a truck is a rising/falling sweep.
            Those look different in a spectrogram, and convolutions are good
            at spotting that difference.
          * MaxPool2d downsamples, giving tolerance to exactly where in the
            window the event happened.
          * AdaptiveAvgPool2d collapses to a fixed size so variable-length
            recordings all reach the classifier head with the same shape.
          * A Linear head maps the learned features to one logit; sigmoid
            turns that into a probability.
        """

        def __init__(self):
            super().__init__()

            # --- Feature extractor -------------------------------------
            self.features = nn.Sequential(
                # Block 1: 1 input channel (mono spectrogram) -> 8 maps.
                # padding=1 with a 3x3 kernel preserves the spatial size.
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),                      # non-linearity
                nn.MaxPool2d(2),                # halve both dimensions

                # Block 2: 8 -> 16 maps, capturing more complex patterns.
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # Force a fixed 4x4 output regardless of input size, so the
                # linear layer below always sees the same number of features.
                nn.AdaptiveAvgPool2d((4, 4)),
            )

            # --- Classifier head ---------------------------------------
            self.classifier = nn.Sequential(
                nn.Flatten(),                   # 16 maps * 4 * 4 = 256
                nn.Linear(16 * 4 * 4, 32),
                nn.ReLU(),
                nn.Linear(32, 1),               # single logit: leak vs not
            )

        def forward(self, x):
            """x: (batch, 1, freq_bins, time_frames) -> (batch, 1) logits."""
            # Resample the arbitrary-sized spectrogram onto the fixed grid.
            # Bilinear interpolation is the 2-D analogue of resizing an image,
            # and is supported on MPS. This guarantees every pooling stage
            # downstream divides evenly (see CNN_INPUT_SHAPE).
            x = F.interpolate(x, size=CNN_INPUT_SHAPE,
                              mode="bilinear", align_corners=False)
            x = self.features(x)
            return self.classifier(x)

else:
    # Placeholder so the name always exists and error messages stay clear.
    LeakClassifierCNN = None


# ============================================================================
# Feature extraction
# ============================================================================

def _compute_spectrogram(sig: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    """
    Turn a 1-D waveform into a 2-D log-magnitude spectrogram.

    A CNN needs a 2-D image-like input. The spectrogram is the natural choice
    for audio: one axis frequency, one axis time, brightness = energy.

    scipy.signal.spectrogram is used rather than librosa so the demo never
    hard-depends on librosa being installed (it is heavy, and an import error
    seconds before a live demo is not a good time).
    """
    # nperseg: samples per FFT window. 256 balances frequency resolution
    # against time resolution. Capped at the signal length for short inputs.
    nperseg = min(256, sig.size)

    _freqs, _times, magnitude = sps.spectrogram(
        sig,
        fs=sample_rate_hz,
        nperseg=nperseg,
        noverlap=nperseg // 2,      # 50% overlap: smoother time axis
        mode="magnitude",
    )

    # Log scaling compresses the huge dynamic range of acoustic power into
    # something a neural net can train on. +1e-10 avoids log(0) = -inf.
    log_magnitude = np.log10(magnitude + 1e-10)

    # Per-sample normalisation to roughly zero mean / unit variance, so the
    # network sees a consistent input scale regardless of absolute loudness.
    mean = log_magnitude.mean()
    std = log_magnitude.std()
    if std > 0:
        log_magnitude = (log_magnitude - mean) / std

    return log_magnitude.astype(np.float32)


# ============================================================================
# Public inference entry point
# ============================================================================

def classify_leak(sensor_a: np.ndarray,
                  sensor_b: np.ndarray,
                  sample_rate_hz: int) -> dict:
    """
    Run the mock CNN noise filter over the two sensor recordings.

    Returns
    -------
    dict with:
        leak_probability -> float in [0,1]. STUBBED to a fixed high value.
        is_leak          -> bool, probability > 0.5.
        device           -> which torch device ran the forward pass.
        spectrogram      -> the feature array, for optional display.
        raw_logit        -> the untrained network's actual output (shown to
                            make it obvious the forward pass really happened,
                            and that it is NOT what drives the decision).
        is_mock          -> always True. Explicit so the UI can label it.
    """
    # Feature extraction on Sensor A's trace (the nearer/louder channel in a
    # real deployment would be chosen; either works for the stub).
    spectrogram = _compute_spectrogram(sensor_a, sample_rate_hz)

    # --- Graceful degradation if torch is absent ------------------------
    if not TORCH_AVAILABLE:
        return {
            "leak_probability": MOCK_LEAK_PROBABILITY,
            "is_leak": True,
            "device": "unavailable (torch not installed)",
            "spectrogram": spectrogram,
            "raw_logit": None,
            "is_mock": True,
        }

    def _forward(device):
        """Build the model on `device`, run one forward pass, return the logit."""
        # Instantiate the network and move it onto the chosen device.
        model = LeakClassifierCNN().to(device)

        # eval() switches off training-only behaviour (dropout, batchnorm
        # stats). Always required before inference, even for an untrained model.
        model.eval()

        # Shape the input to what Conv2d expects:
        # (batch, channels, height, width) = (1, 1, freq_bins, time_frames).
        # Two unsqueeze(0) calls add the leading singleton dimensions.
        tensor = torch.from_numpy(spectrogram) \
            .unsqueeze(0).unsqueeze(0).to(device)

        # no_grad() skips building the autograd graph: faster, less memory,
        # and correct - we are not training.
        with torch.no_grad():
            return float(model(tensor).item())

    device = get_device()

    # Try the accelerated device first, but NEVER let a backend quirk take the
    # demo down. MPS in particular still has unimplemented operator/shape
    # combinations, so we fall back to CPU (which always works) and report
    # which device actually ran.
    try:
        raw_logit = _forward(device)
    except (RuntimeError, NotImplementedError) as exc:
        fallback = torch.device("cpu")
        raw_logit = _forward(fallback)
        device = f"cpu (fell back from {device}: {type(exc).__name__})"

    # ---------------------------------------------------------------
    # THE MOCK: we deliberately IGNORE raw_logit and return a fixed high
    # probability, so the prototype always proceeds to localisation. A trained
    # model would instead return torch.sigmoid(logit) here.
    # ---------------------------------------------------------------
    leak_probability = MOCK_LEAK_PROBABILITY

    return {
        "leak_probability": leak_probability,
        "is_leak": leak_probability > 0.5,
        "device": str(device),
        "spectrogram": spectrogram,
        "raw_logit": raw_logit,
        "is_mock": True,
    }
