"""
============================================================================
 app.py - AELD: Finding a hidden water leak with two microphones
============================================================================
Run with:
    streamlit run app.py

DESIGN GOAL FOR THIS FILE
-------------------------
A visitor with no signal-processingst background should be able to read this
interface top to bottom and understand both WHAT happened and WHY it is
impressive. So:

  * Every heading is a plain-English sentence, not a technical term.
  * Jargon appears only AFTER the plain explanation, and only inside
    collapsible "for the curious" sections.
  * The result is put in human scale (millimetres, comparisons, ratios)
    rather than left as a bare number.

All maths lives in src/aeld/. This file is presentation only.
============================================================================
"""

# --- Standard library -------------------------------------------------------
import sys
from pathlib import Path

# --- Third party ------------------------------------------------------------
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# --- Make `src/` importable -------------------------------------------------
# The core package lives in src/aeld/ but isn't pip-installed, so add src/ to
# the import path. Path(__file__).parent keeps this correct regardless of the
# directory streamlit was launched from.
SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aeld import SimulationConfig, run_detection    # noqa: E402


# ============================================================================
# Page setup and shared styling
# ============================================================================

st.set_page_config(
    page_title="Leak Finder",
    page_icon="~",
    layout="wide",
    initial_sidebar_state="expanded",
)

# One consistent palette, referenced by every plot.
COLOR_A = "#2E86DE"        # Sensor A / microphone 1 - blue
COLOR_B = "#EE5A24"        # Sensor B / microphone 2 - orange
COLOR_CORR = "#8854D0"     # the matching curve - purple
COLOR_TRUE = "#20BF6B"     # ground truth - green


def style_axes(ax):
    """Apply one consistent, low-clutter look to a matplotlib axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    return ax


def describe_noise(level: float) -> str:
    """
    Translate the abstract noise number into something physically meaningful.

    A slider reading "0.8" means nothing to a visitor. "Noisy - traffic
    overhead" does.
    """
    if level <= 0.05:
        return "Silent - a perfect laboratory"
    if level <= 0.25:
        return "Quiet - a rural pipe at night"
    if level <= 0.5:
        return "Normal - a typical residential street"
    if level <= 0.8:
        return "Noisy - traffic passing overhead"
    return "Very noisy - this will probably defeat the system"


def human_scale_comparison(error_m: float, pipe_length_m: float) -> str:
    """
    Put the accuracy in a form a person can picture.

    A "0.0001 m error" is impressive but unreadable. Expressing it as a ratio
    and mapping that ratio onto a familiar distance makes it land.

    Uses the London-Paris rail distance (~344 km) as the reference, because it
    is a distance most people can hold in their head.
    """
    # Avoid dividing by zero on a flawless run.
    if error_m <= 0:
        return "The estimate landed on the exact answer."

    ratio = pipe_length_m / error_m          # e.g. 1 part in 1,500,000

    # Scale that same relative precision onto a 344 km journey.
    reference_km = 344.0
    equivalent_m = reference_km * 1000.0 / ratio

    # Choose sensible units for whatever came out.
    if equivalent_m < 0.01:
        equivalent = f"{equivalent_m * 1000:.1f} millimetres"
    elif equivalent_m < 1.0:
        equivalent = f"{equivalent_m * 100:.1f} centimetres"
    elif equivalent_m < 1000:
        equivalent = f"{equivalent_m:.0f} metres"
    else:
        equivalent = f"{equivalent_m / 1000:.1f} kilometres"

    return (f"That is an accuracy of **1 part in {ratio:,.0f}** - the "
            f"equivalent of measuring the {reference_km:,.0f} km from London "
            f"to Paris to within **{equivalent}**.")


def format_distance(metres: float) -> str:
    """Pick readable units for an error that may span 8 orders of magnitude."""
    if metres < 0.001:
        return f"{metres * 1000:.3f} mm"
    if metres < 1.0:
        return f"{metres * 100:.2f} cm"
    return f"{metres:.2f} m"


# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.title("Leak Finder")
st.sidebar.caption("Locating a buried pipe leak using sound")

page = st.sidebar.radio(
    "Go to",
    ["1. Set up and run", "2. See the result"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown(
    """
**The idea in three lines**

A leak hisses continuously.

A microphone at each end of the pipe hears that hiss - but the nearer one
hears it a fraction of a second sooner.

That fraction tells you where the leak is.
    """
)


# ============================================================================
# PAGE 1: Set up and run
# ============================================================================

def render_setup_page():
    """Plain-language configuration, then a single button that does the work."""

    st.title("Find a hidden leak")

    # ---- Frame the problem before showing any controls -------------------
    st.markdown(
        """
    A water pipe is buried underground. Somewhere along it, there is a leak.
    You cannot see it, and digging up the whole road to look would cost a
    fortune.

    **This tool finds it using nothing but sound.** One microphone is clamped
    to each end of the pipe. Both listen. Below, you will hide a leak at a
    random spot, and the system will try to tell you where it is.
    """
    )

    st.divider()

    # ---- The three controls, in plain words ------------------------------
    st.subheader("Describe the pipe")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**How long is the pipe?**")
        pipe_length_m = st.slider(
            "Pipe length", min_value=20.0, max_value=1000.0, value=200.0,
            step=10.0, label_visibility="collapsed",
        )
        st.caption(f"{pipe_length_m:.0f} metres between the two microphones.")

    with col2:
        st.markdown("**How fast does sound travel in it?**")
        wave_velocity_ms = st.slider(
            "Wave velocity", min_value=300.0, max_value=2000.0, value=1200.0,
            step=10.0, label_visibility="collapsed",
        )
        st.caption(
            f"{wave_velocity_ms:.0f} m/s. Sound moves much faster through "
            f"water and pipe walls than through air (~340 m/s)."
        )

    with col3:
        st.markdown("**How noisy is the environment?**")
        noise_level = st.slider(
            "Noise level", min_value=0.0, max_value=2.0, value=0.15,
            step=0.05, label_visibility="collapsed",
        )
        st.caption(describe_noise(noise_level))

    # ---- Everything else, hidden by default -----------------------------
    with st.expander("Advanced settings (safe to ignore)"):
        col_a, col_b = st.columns(2)

        with col_a:
            attenuation_per_m = st.slider(
                "How quickly sound fades with distance (1/m)",
                min_value=0.0, max_value=0.02, value=0.004, step=0.001,
                format="%.3f",
                help="Higher means the far microphone hears a fainter hiss.",
            )
            sample_rate_hz = st.select_slider(
                "Microphone sample rate (Hz)",
                options=[4000, 8000, 16000, 32000], value=8000,
                help="How many times per second each microphone is read.",
            )

        with col_b:
            band_low, band_high = st.slider(
                "Frequency range of the leak's hiss (Hz)",
                min_value=50, max_value=4000, value=(100, 2000), step=50,
                help=("Real leaks hiss mostly in the low kHz range. The "
                      "system listens only here, and ignores everything "
                      "else as noise."),
            )
            use_seed = st.checkbox(
                "Repeatable run (fixed random seed)", value=False,
                help="Turn on to get the identical result every time.")
            seed_value = st.number_input(
                "Seed", value=42, step=1, disabled=not use_seed)

    # ---- Build the config ------------------------------------------------
    # Guard: the frequency band must stay below the Nyquist limit (half the
    # sample rate), otherwise the filter design is mathematically undefined.
    # Clamp rather than crash.
    nyquist = sample_rate_hz / 2.0
    safe_high = min(float(band_high), nyquist * 0.98)

    config = SimulationConfig(
        pipe_length_m=pipe_length_m,
        wave_velocity_ms=wave_velocity_ms,
        noise_level=noise_level,
        attenuation_per_m=attenuation_per_m,
        sample_rate_hz=int(sample_rate_hz),
        leak_band_hz=(float(band_low), safe_high),
        random_seed=int(seed_value) if use_seed else None,
    )

    st.divider()

    # ---- Run it ----------------------------------------------------------
    st.subheader("Hide a leak, then find it")

    mode_col, btn_col = st.columns([2, 1])

    with mode_col:
        position_mode = st.radio(
            "Where should the leak be?",
            ["Surprise me", "Let me choose"],
            horizontal=True,
        )
        manual_position = None
        if position_mode == "Let me choose":
            manual_position = st.slider(
                "Distance from microphone 1 (m)",
                min_value=0.0, max_value=float(pipe_length_m),
                value=float(pipe_length_m) / 3.0, step=1.0,
            )
        else:
            st.caption(
                "A position will be picked at random and kept hidden from "
                "the detector."
            )

    with btn_col:
        st.write("")
        st.write("")
        triggered = st.button("Start the leak", type="primary",
                              use_container_width=True)

    if triggered:
        if position_mode == "Surprise me":
            # Keep the leak slightly away from the very ends - a leak sitting
            # exactly on a microphone is a degenerate edge case, not a useful
            # demonstration.
            leak_position = float(np.random.default_rng().uniform(
                0.05 * pipe_length_m, 0.95 * pipe_length_m))
        else:
            leak_position = float(manual_position)

        with st.spinner("Listening ..."):
            result = run_detection(leak_position, config)

        # Streamlit re-runs this whole script on every interaction, so
        # session_state is the only way results survive a page switch.
        st.session_state["result"] = result
        st.session_state["config"] = config

        st.success("Done. Open **2. See the result** in the sidebar.")

        # A short preview so the button feels like it did something.
        if result.is_confident:
            st.metric("Leak located at",
                      f"{result.estimated_position_m:.2f} m",
                      delta=f"{format_distance(result.error_m)} off",
                      delta_color="off")
        else:
            st.warning(
                "A leak was detected, but it was too noisy to pin down "
                "where. See the result page for why."
            )


# ============================================================================
# PAGE 2: See the result
# ============================================================================

def render_result_page():
    """The answer in human terms, then the story of how it was reached."""

    st.title("The result")

    if "result" not in st.session_state:
        st.info(
            "Nothing has been run yet. Go to **1. Set up and run** and press "
            "**Start the leak**."
        )
        return

    result = st.session_state["result"]
    config = st.session_state["config"]

    # ------------------------------------------------------------------
    # 1. THE ANSWER, in plain language
    # ------------------------------------------------------------------
    if result.is_confident:
        st.markdown(
            f"## The leak is {result.estimated_position_m:.2f} metres "
            f"from microphone 1."
        )
        st.markdown(
            f"It was actually at **{result.true_position_m:.2f} m**, so the "
            f"system was off by **{format_distance(result.error_m)}** - "
            f"found in **{result.time_total_ms:.1f} milliseconds**, using "
            f"nothing but two microphones and no digging."
        )
    else:
        st.markdown("## Something is leaking, but I cannot tell you where.")
        st.markdown(
            f"The background noise was too high to pick the leak's hiss out "
            f"reliably. The system's best guess would have been "
            f"**{result.estimated_position_m:.2f} m** (the leak was really at "
            f"**{result.true_position_m:.2f} m**), but it has flagged that "
            f"answer as untrustworthy rather than sending you to dig there."
        )

    # ---- Three headline numbers -----------------------------------------
    m1, m2, m3 = st.columns(3)
    m1.metric("Where the system said", f"{result.estimated_position_m:.2f} m")
    m2.metric("Where it actually was", f"{result.true_position_m:.2f} m")
    m3.metric("How far off", format_distance(result.error_m))

    # ---- Why that is remarkable (only when the answer is trustworthy) ---
    if result.is_confident:
        st.info(
            f"**Why this is remarkable.** "
            f"{human_scale_comparison(result.error_m, config.pipe_length_m)} "
            f"And it took {result.time_total_ms:.1f} milliseconds - roughly "
            f"{max(1, int(150 / max(result.time_total_ms, 0.01)))}x faster "
            f"than a blink of an eye."
        )
    else:
        st.error(
            "**The system knew it was unsure - and said so.** That matters "
            "more than it sounds. A tool that confidently reports a wrong "
            "location sends a crew to dig up the wrong road. One that admits "
            "uncertainty tells them to come back with better equipment."
        )

    st.divider()

    # ------------------------------------------------------------------
    # 2. HOW IT WORKS - three plain steps
    # ------------------------------------------------------------------
    st.header("How it figured that out, in three steps")

    # ---------------- STEP 1 ----------------
    st.subheader("Step 1: Two microphones listen to the same hiss")

    st.markdown(
        """
    A leak is not silent - escaping water makes a constant hissing sound.
    Both microphones pick up **the very same hiss**, because there is only one
    leak making it.

    Below is what each one recorded. To the eye, they are just two patches of
    noise. Notice only this: the top trace is slightly **taller**, because
    that microphone is closer to the leak and sound fades as it travels.
    """
    )

    zoom_ms = st.slider(
        "How much of the recording to show (milliseconds)",
        min_value=2, max_value=100, value=20, step=1,
    )
    n_show = min(int(config.sample_rate_hz * zoom_ms / 1000.0),
                 result.time_s.size)

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    ax_a.plot(result.time_s[:n_show] * 1000, result.sensor_a[:n_show],
              color=COLOR_A, linewidth=0.9)
    ax_a.set_ylabel("Loudness")
    ax_a.set_title(
        f"Microphone 1  -  {result.true_position_m:.0f} m from the leak",
        loc="left", fontsize=11, fontweight="bold")
    style_axes(ax_a)

    distance_b = config.pipe_length_m - result.true_position_m
    ax_b.plot(result.time_s[:n_show] * 1000, result.sensor_b[:n_show],
              color=COLOR_B, linewidth=0.9)
    ax_b.set_ylabel("Loudness")
    ax_b.set_xlabel("Time (milliseconds)")
    ax_b.set_title(
        f"Microphone 2  -  {distance_b:.0f} m from the leak",
        loc="left", fontsize=11, fontweight="bold")
    style_axes(ax_b)

    # Force one shared y-scale so the loudness difference is a fair visual
    # comparison rather than an artefact of independent autoscaling.
    limit = max(np.abs(result.sensor_a[:n_show]).max(),
                np.abs(result.sensor_b[:n_show]).max()) * 1.1
    ax_a.set_ylim(-limit, limit)
    ax_b.set_ylim(-limit, limit)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ---------------- STEP 2 ----------------
    st.subheader("Step 2: One of them heard it a fraction sooner")

    delay_ms = abs(result.true_tau_s) * 1000.0
    nearer = "1" if result.true_tau_s < 0 else "2"

    st.markdown(
        f"""
    Sound takes time to travel. The leak is closer to **microphone {nearer}**,
    so that microphone heard the hiss **{delay_ms:.1f} milliseconds** before
    the other one.

    That gap is the entire secret. Measure it accurately and you know the
    position, because sound covers a known distance in a known time - the same
    reason counting the seconds between lightning and thunder tells you how
    far away a storm is.

    The catch: **{delay_ms:.1f} ms is far too small to spot by looking at
    those two traces.** You cannot eyeball it. So the computer does something
    cleverer.
    """
    )

    # ---------------- STEP 3 ----------------
    st.subheader("Step 3: Slide one recording against the other until they "
                 "match")

    st.markdown(
        """
    Take the second recording and slide it back and forth against the first.
    At most positions the two disagree - noise against noise. But at **one**
    position, the shared hiss lines up, and suddenly they agree strongly.

    The graph below scores that agreement at every possible slide. The tall
    spike is the moment they matched. **That spike is the measurement** -
    everything else is arithmetic.
    """
    )

    fig2, ax = plt.subplots(figsize=(12, 4.5))
    lags_ms = result.lags_s * 1000.0

    ax.plot(lags_ms, result.correlation, color=COLOR_CORR, linewidth=1.0,
            label="How well the two recordings agree")

    peak_lag_ms = result.estimated_tau_s * 1000.0
    ax.axvline(peak_lag_ms, color=COLOR_CORR, linewidth=1.8, alpha=0.9,
               label=f"Best match at {peak_lag_ms:+.3f} ms")
    ax.plot([peak_lag_ms], [result.correlation[result.peak_index]],
            marker="v", markersize=11, color=COLOR_CORR, zorder=5)

    true_lag_ms = result.true_tau_s * 1000.0
    ax.axvline(true_lag_ms, color=COLOR_TRUE, linestyle="--", linewidth=1.8,
               label=f"The true answer: {true_lag_ms:+.3f} ms")

    ax.axvline(0.0, color="grey", linestyle=":", linewidth=1.2, alpha=0.7,
               label="No slide (leak would be dead centre)")

    ax.set_xlabel("How far the second recording was slid (milliseconds)")
    ax.set_ylabel("Agreement")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    style_axes(ax)

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    if result.is_confident:
        st.caption(
            "The purple line (what the system worked out) sits on top of the "
            "green dashed line (the truth). That overlap is the whole result."
        )
    else:
        st.caption(
            "Notice there is no single clear spike here - the curve is a mess "
            "of similar-sized bumps. The system picked the tallest, but it "
            "was not meaningfully taller than the rest, which is exactly why "
            "the answer was flagged as untrustworthy."
        )

    st.divider()

    # ------------------------------------------------------------------
    # 3. THE ANSWER ON THE PIPE
    # ------------------------------------------------------------------
    st.header("Where that puts the leak")

    fig4, ax4 = plt.subplots(figsize=(12, 2.4))

    # The pipe.
    ax4.plot([0, config.pipe_length_m], [0, 0], color="#34495E", linewidth=9,
             solid_capstyle="butt", zorder=1)

    # Microphones at each end.
    ax4.plot([0], [0], marker="s", markersize=15, color=COLOR_A, zorder=3)
    ax4.plot([config.pipe_length_m], [0], marker="s", markersize=15,
             color=COLOR_B, zorder=3)
    ax4.annotate("Microphone 1\n0 m", (0, 0), textcoords="offset points",
                 xytext=(0, -40), ha="center", fontsize=9, color=COLOR_A,
                 fontweight="bold")
    ax4.annotate(f"Microphone 2\n{config.pipe_length_m:.0f} m",
                 (config.pipe_length_m, 0), textcoords="offset points",
                 xytext=(0, -40), ha="center", fontsize=9, color=COLOR_B,
                 fontweight="bold")

    # Truth above the pipe.
    ax4.plot([result.true_position_m], [0], marker="v", markersize=14,
             color=COLOR_TRUE, zorder=4)
    ax4.annotate(f"Real leak\n{result.true_position_m:.1f} m",
                 (result.true_position_m, 0), textcoords="offset points",
                 xytext=(0, 20), ha="center", fontsize=9, color=COLOR_TRUE,
                 fontweight="bold")

    # Estimate below, so labels never collide.
    ax4.plot([result.estimated_position_m], [0], marker="^", markersize=14,
             color=COLOR_CORR, zorder=4)
    ax4.annotate(f"Found here\n{result.estimated_position_m:.1f} m",
                 (result.estimated_position_m, 0),
                 textcoords="offset points", xytext=(0, -36), ha="center",
                 fontsize=9, color=COLOR_CORR, fontweight="bold")

    ax4.set_xlim(-0.06 * config.pipe_length_m, 1.06 * config.pipe_length_m)
    ax4.set_ylim(-1, 1)
    ax4.axis("off")       # a diagram, not a chart

    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    if result.is_confident and result.error_m < config.pipe_length_m / 500:
        st.caption(
            "The two markers are drawn at the same place because, at this "
            "scale, they are the same place."
        )

    st.divider()

    # ------------------------------------------------------------------
    # 4. CAN WE TRUST IT?
    # ------------------------------------------------------------------
    st.header("Can we trust this answer?")

    st.markdown(
        """
    This is the question that decides whether anyone picks up a shovel, so the
    system answers it explicitly rather than leaving you to guess.

    It measures **how much taller the matching spike was than the general
    background** of that graph. A tall, lonely spike means a real match. A
    spike barely above the clutter means the computer essentially picked at
    random - and it says so.
    """
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Spike height above background",
                  f"{result.peak_sharpness:.1f}x",
                  delta=f"needs {config.confidence_threshold:.0f}x or more",
                  delta_color="off")
    with c2:
        if result.is_confident:
            st.success(
                f"**Trustworthy.** The spike was "
                f"{result.peak_sharpness:.1f} times the background level, "
                f"comfortably past the {config.confidence_threshold:.0f}x bar. "
                f"Go dig at {result.estimated_position_m:.1f} m."
            )
        else:
            st.error(
                f"**Not trustworthy.** The spike was only "
                f"{result.peak_sharpness:.1f} times the background, below the "
                f"{config.confidence_threshold:.0f}x bar. Do not dig on this "
                f"reading."
            )

    st.markdown(
        """
    **Try this:** go back to page 1, drag the noise slider up to 1.5, and run
    it again. You will watch the system change its own mind and refuse to
    answer. A tool that knows when it does not know is worth far more than one
    that is occasionally, confidently, expensively wrong.
    """
    )

    # ------------------------------------------------------------------
    # 5. TECHNICAL DETAIL - opt-in only
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("For the curious")

    with st.expander("How long each part took"):
        t1, t2, t3 = st.columns(3)
        t1.metric("Generating the recordings",
                  f"{result.time_acquire_ms:.2f} ms")
        t2.metric("Finding the match", f"{result.time_gccphat_ms:.2f} ms")
        t3.metric("Total", f"{result.time_total_ms:.2f} ms")
        st.caption(
            "On real equipment the first step is replaced by simply reading "
            "the microphones. The actual detection work - the second number - "
            "is a fraction of a millisecond, which is why this can run on a "
            "small battery-powered device sitting in a roadside chamber."
        )

    with st.expander("The arithmetic that turns a time into a distance"):
        st.markdown(
            "Once the time gap is known, the position is straightforward "
            "geometry. With `L` the pipe length, `v` the speed of sound in "
            "the pipe, and `t` the measured time gap:"
        )
        st.latex(r"\text{distance from microphone 1} = \frac{L + v \cdot t}{2}")
        st.code(
            f"t = {result.estimated_tau_s:+.8f} s   (measured from the spike)\n"
            f"L = {config.pipe_length_m:.2f} m      (pipe length)\n"
            f"v = {config.wave_velocity_ms:.2f} m/s (speed of sound in pipe)\n"
            f"\n"
            f"distance = ({config.pipe_length_m:.2f} + "
            f"{config.wave_velocity_ms:.2f} x {result.estimated_tau_s:+.8f}) / 2\n"
            f"         = {result.estimated_position_m:.4f} m\n"
            f"\n"
            f"Real answer: {result.true_position_m:.4f} m\n"
            f"Off by:      {result.error_m:.6f} m",
            language="text",
        )
        st.markdown(
            "Sanity-check the formula at the points you already know: a leak "
            "dead centre gives a time gap of zero, so the formula returns "
            "`L/2`. A leak sitting on microphone 1 gives `t = -L/v`, which "
            "returns 0. Both are unit-tested."
        )

    with st.expander("The name of the technique, and why it works"):
        st.markdown(
            """
        The method is called **GCC-PHAT** - Generalised Cross-Correlation with
        Phase Transform. It is the standard approach for this class of problem
        and also underpins how smart speakers work out which direction your
        voice came from.

        **Why not plain sliding-and-comparing?** Because a hiss compared
        against itself gives a broad, rounded hump rather than a sharp spike,
        and a broad hump means you cannot tell exactly where the top is. Loud
        frequencies also dominate the answer simply for being loud, not for
        being informative.

        **The trick** is to throw away *how loud* each frequency is and keep
        only *when* it arrived. Mathematically, if one recording is the other
        delayed by `d`, then after dividing out the loudness you are left with
        a pure timing pattern whose transform is a perfect spike at `d`. The
        loudness was never carrying the timing information - only the phase
        was. Discarding it loses nothing and sharpens the spike enormously.

        **Three refinements** in this implementation:

        1. **Reading between the samples.** The graph is only calculated at
           discrete slide positions, but the true peak usually falls between
           two of them. Fitting a small curve through the highest point and
           its neighbours finds the true top.
        2. **A finer grid.** That peak is not actually curve-shaped, so fitting
           a curve across widely spaced points is slightly biased. Padding the
           data before the final transform evaluates the same graph on a 4x
           finer grid, where the fit is accurate.
        3. **Listening only where leaks hiss.** This is the big one. The
           loudness-discarding trick has a weakness: a frequency band
           containing *only noise* gets amplified to the same importance as
           one containing real signal, and then votes for a random answer.
           Leaks only hiss between about 100 and 2000 Hz, so everything above
           that is ignored. In testing this alone cut the error in a noisy
           case from 136 m down to 41 m.
        """
        )

    with st.expander("Honest limitations"):
        st.markdown(
            f"""
        - **This is a simulation.** The physics is modelled faithfully -
          travel times, fading with distance, independent microphone noise -
          but no real pipe was recorded.
        - **One straight pipe.** No junctions, branches or bends. Real
          networks produce echoes that make the matching graph much messier.
        - **The speed of sound is assumed known.** The answer scales directly
          with it, so a 5% error in that number is a 5% error in position. In
          the field this is the main thing you would need to calibrate.
        - **It fails abruptly, not gradually.** Past roughly 0.5 on the noise
          slider the method does not drift - it jumps to essentially random
          answers. That is why the trust check exists. The
          {config.confidence_threshold:.0f}x bar was calibrated over 1400 test
          runs, and it is an empirical bound from this simulator, not a
          guarantee.
        - **One leak at a time.** Two simultaneous leaks would produce two
          spikes and the code simply takes the taller one.
        """
        )


# ============================================================================
# Router
# ============================================================================

# Streamlit re-runs this file top to bottom on every interaction, so routing
# is just calling the function for whichever page is selected.
if page.startswith("1"):
    render_setup_page()
else:
    render_result_page()

st.sidebar.divider()
st.sidebar.caption(
    "Prototype. The physics simulation and the leak-locating maths are real "
    "and unit-tested; no real pipe was recorded."
)
