#!/usr/bin/env python3
"""
============================================================================
 AELD Prototype - Environment Setup / Dependency Installer
============================================================================
This is NOT a setuptools/distutils packaging script. It is a small, dependency
free installer you run once to build the environment:

    python setup.py              # install everything the demo needs
    python setup.py --with-ml    # also install PyTorch (for the archived CNN)
    python setup.py --full       # also install WNTR / FastAPI / MQTT / librosa
    python setup.py --check      # don't install; just report what's present

NOTE: PyTorch is NO LONGER REQUIRED. The CNN noise filter that used it was an
untrained stub and has been moved to archive/ml_model/. The platform-aware
wheel selection below is kept intact for when that module is trained and
reinstated - use --with-ml to install it.

Why a custom script instead of a plain `pip install -r requirements.txt`?
Because PyTorch needs a DIFFERENT wheel index depending on the platform:

  * Apple Silicon (M1-M4)  -> default PyPI wheel, which ships MPS (Metal
                              Performance Shaders) GPU acceleration built in.
  * Linux/Windows + NVIDIA -> the CUDA index gives GPU acceleration.
  * Anything else          -> the CPU-only index is smaller and always works.

This script detects the platform and picks the right one automatically.
============================================================================
"""

# --- Standard library only: this script must run BEFORE anything is installed
import argparse      # parse the --full / --check command line flags
import platform      # detect OS ("Darwin"/"Linux"/"Windows") and CPU arch
import shutil        # locate external executables (used to find nvidia-smi)
import subprocess    # shell out to pip
import sys           # the current Python interpreter path + exit codes


# ============================================================================
# SECTION 1: Dependency groups
# ============================================================================
# Split into groups so the heavy/fragile optional packages can be skipped.
# CORE is genuinely all the working system needs.

CORE_PACKAGES = [
    "numpy>=1.26,<3.0",   # array math + FFT + RNG
    "scipy>=1.11",        # Butterworth bandpass filter
    "streamlit>=1.32",    # the web UI
    "matplotlib>=3.8",    # plotting waveforms and the correlation curve
    "pytest>=8.0",        # unit tests that prove the GCC-PHAT math is right
    "streamlit-folium>=0.23",  # interactive Leaflet/OpenStreetMap map widget
    "folium>=0.16",        # underlying map-building library (OSM tiles, no API key)
]

# Optional extras: declared in the stack, but nothing in src/aeld/ or app.py
# imports them. Separated because wntr (EPANET) and the MQTT/HTTP stack are the
# most likely things to fail on a fresh machine right before a demo.
OPTIONAL_PACKAGES = [
    "librosa>=0.10",      # richer audio features (MFCC etc.)
    "wntr>=1.0",          # EPANET hydraulic network simulation
    "fastapi>=0.110",     # HTTP backend (future work)
    "uvicorn>=0.27",      # ASGI server for FastAPI
    "paho-mqtt>=2.0",     # MQTT telemetry transport (future work)
]


# ============================================================================
# SECTION 2: Platform detection
# ============================================================================

def detect_platform() -> dict:
    """
    Inspect the machine and return a small dict describing it.

    Returns keys:
      os         -> "Darwin" | "Linux" | "Windows"
      machine    -> e.g. "arm64" (Apple Silicon) or "x86_64"
      is_apple_silicon -> True on M1/M2/M3/M4 Macs
      has_nvidia -> True if an NVIDIA driver appears to be installed
    """
    os_name = platform.system()          # "Darwin" on macOS
    machine = platform.machine()         # "arm64" on Apple Silicon

    # Apple Silicon = macOS running on an ARM64 chip. This is the M4 case.
    is_apple_silicon = (os_name == "Darwin" and machine in ("arm64", "aarch64"))

    # Heuristic for NVIDIA: the `nvidia-smi` binary only exists with the driver.
    # shutil.which() returns None when the executable isn't on PATH.
    has_nvidia = shutil.which("nvidia-smi") is not None

    return {
        "os": os_name,
        "machine": machine,
        "is_apple_silicon": is_apple_silicon,
        "has_nvidia": has_nvidia,
    }


def torch_install_args(info: dict) -> list:
    """
    Decide which pip arguments install the best PyTorch build for this machine.

    Returns a list of pip arguments (package spec plus, if needed, an
    --index-url pointing at a platform-specific wheel index).
    """
    if info["is_apple_silicon"]:
        # PRIORITY CASE (Apple Silicon M4):
        # The standard PyPI wheel for macOS/arm64 already contains the MPS
        # backend, so no special index is required. MPS lets torch run tensor
        # ops on the Mac's integrated GPU via Metal.
        return ["torch>=2.2"]

    if info["has_nvidia"]:
        # Linux/Windows with an NVIDIA GPU -> use the CUDA 12.1 wheel index
        # so torch is built against CUDA and can use the discrete GPU.
        return ["torch>=2.2", "--index-url",
                "https://download.pytorch.org/whl/cu121"]

    # FALLBACK: no GPU we can target. The CPU-only wheel is much smaller and
    # works everywhere (including CI containers and older Intel Macs).
    return ["torch>=2.2", "--index-url",
            "https://download.pytorch.org/whl/cpu"]


# ============================================================================
# SECTION 3: pip helpers
# ============================================================================

def pip_install(args: list) -> bool:
    """
    Run `python -m pip install <args>` and return True on success.

    We invoke pip via `sys.executable -m pip` rather than a bare `pip` so the
    packages always land in the SAME interpreter that is running this script.
    That avoids the classic "installed it but the import still fails" trap.
    """
    cmd = [sys.executable, "-m", "pip", "install", *args]
    print(f"\n  $ {' '.join(cmd)}")
    # check=False: we handle the failure ourselves so one bad optional package
    # doesn't abort the whole setup with a traceback.
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def report_environment() -> None:
    """
    Print which key packages are importable, and which torch device is live.
    Used by --check and printed at the end of a normal install.
    """
    print("\n--- Environment report ---")

    # Probe each package by importing it. Import is the real test; a package
    # can be "pip installed" and still fail to load (bad wheel, missing libs).
    print("  Required by the prototype:")
    required_ok = True
    for module in ("numpy", "scipy", "streamlit", "matplotlib", "pytest",
                   "folium", "streamlit_folium"):
        try:
            mod = __import__(module)
            # Not every package exposes __version__; fall back to "?".
            version = getattr(mod, "__version__", "?")
            print(f"    [ok]      {module:<14} {version}")
        except ImportError:
            print(f"    [MISSING] {module}")
            required_ok = False

    # Optional packages: absence here is completely fine and expected.
    print("\n  Optional (not used by the prototype):")
    for module in ("torch", "librosa", "wntr", "fastapi", "paho.mqtt"):
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "?")
            print(f"    [ok]      {module:<14} {version}")
        except ImportError:
            print(f"    [absent]  {module:<14} (fine - not needed)")

    # If torch happens to be present, report the device it would use. Useful
    # when reviving the archived CNN; irrelevant otherwise.
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps  (Apple Silicon GPU via Metal)"
        elif torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name(0)})"
        else:
            device = "cpu"
        print(f"\n  torch is present; it would use -> {device}")
        print("  (Only relevant if you revive archive/ml_model/.)")
    except ImportError:
        pass

    if required_ok:
        print("\n  All required packages present.")


# ============================================================================
# SECTION 4: Entry point
# ============================================================================

def main() -> int:
    """Parse flags, install the right things, then report. Returns exit code."""
    parser = argparse.ArgumentParser(
        description="Set up the AELD prototype environment.")
    parser.add_argument("--with-ml", action="store_true", dest="with_ml",
                        help="also install PyTorch (only needed to revive the "
                             "archived CNN in archive/ml_model/)")
    parser.add_argument("--full", action="store_true",
                        help="also install optional WNTR/FastAPI/MQTT/librosa")
    parser.add_argument("--check", action="store_true",
                        help="only report the environment; install nothing")
    args = parser.parse_args()

    # Step 1: figure out what machine we're on.
    info = detect_platform()
    print("=" * 72)
    print(" AELD Prototype - environment setup")
    print("=" * 72)
    print(f"  OS            : {info['os']} ({info['machine']})")
    print(f"  Apple Silicon : {info['is_apple_silicon']}")
    print(f"  NVIDIA driver : {info['has_nvidia']}")
    print(f"  Interpreter   : {sys.executable}")

    # --check short-circuits: report and leave without touching the env.
    if args.check:
        report_environment()
        return 0

    # Step 2: make sure pip itself is modern. Old pip versions pick wrong
    # wheels (or none at all) for arm64 macOS.
    print("\n[1/2] Upgrading pip ...")
    pip_install(["--upgrade", "pip"])

    # Step 3: install the packages the demo cannot run without.
    print("\n[2/2] Installing core packages ...")
    if not pip_install(CORE_PACKAGES):
        print("\nERROR: core packages failed to install. Stopping.")
        return 1

    # Step 4: PyTorch, ONLY if explicitly requested. It is no longer part of
    # the working system - see archive/ml_model/README.md. The platform-aware
    # wheel selection is preserved for when that module is trained.
    if args.with_ml:
        print("\n[ml] Installing PyTorch (for the archived CNN) ...")
        torch_args = torch_install_args(info)
        if info["is_apple_silicon"]:
            print("      -> Apple Silicon detected: using the default PyPI")
            print("         wheel, which ships the MPS (Metal) GPU backend.")
        elif info["has_nvidia"]:
            print("      -> NVIDIA GPU detected: using the CUDA 12.1 index.")
        else:
            print("      -> No supported GPU: using the CPU-only wheel index.")
        if not pip_install(torch_args):
            print("\nWARNING: PyTorch install failed. This does not affect the")
            print("         prototype, which does not use it.")
    else:
        print("\n[ml] Skipping PyTorch - the current system does not need it.")
        print("     Use 'python setup.py --with-ml' to install it for the")
        print("     archived CNN in archive/ml_model/.")

    # Step 5: optional extras, only when explicitly requested. Each is
    # installed individually so one failure doesn't block the others.
    if args.full:
        print("\n[extra] Installing optional packages ...")
        for package in OPTIONAL_PACKAGES:
            if not pip_install([package]):
                print(f"      WARNING: {package} failed (safe to ignore).")
    else:
        print("\n[extra] Skipping optional packages (WNTR/FastAPI/MQTT/librosa).")
        print("        Run 'python setup.py --full' to install them.")

    # Step 6: final verification so the user knows what actually landed.
    report_environment()

    print("\n" + "=" * 72)
    print(" Setup complete. Launch the demo with:")
    print("     streamlit run app.py")
    print("=" * 72)
    return 0


# Only run main() when executed directly, never on import.
if __name__ == "__main__":
    sys.exit(main())
