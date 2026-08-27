"""
detection_view.py - the Leak Detection Detail page for one site.
====================================================================
This is the AELD prototype's existing "how it figured that out" narrative
(waveforms, the GCC-PHAT correlation spike, the arithmetic, the honesty
checks), unchanged in substance, now scoped to whichever site was last
triggered on the Survey Map page.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from aeld import SimulationConfig, run_detection
from .sites import get_site
from .map_view import pick_random_leak_position


COLOR_A = "#2E86DE"
COLOR_B = "#EE5A24"
COLOR_CORR = "#8854D0"
COLOR_TRUE = "#20BF6B"


def style_axes(ax):
    """Apply one consistent, low-clutter look to a matplotlib axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    return ax


def describe_noise(level: float) -> str:
    """Translate the abstract noise number into something physically meaningful."""
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
    """Put the accuracy in a form a person can picture."""
    if error_m <= 0:
        return "The estimate landed on the exact answer."

    ratio = pipe_length_m / error_m
    reference_km = 344.0
    equivalent_m = reference_km * 1000.0 / ratio

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
    """Pick readable units for an error that may span many orders of magnitude."""
    if metres < 0.001:
        return f"{metres * 1000:.3f} mm"
    if metres < 1.0:
        return f"{metres * 100:.2f} cm"
    return f"{metres:.2f} m"


def build_override_config(base_result, pipe_length_m: float,
                          wave_velocity_ms: float,
                          noise_level: float) -> SimulationConfig:
    """
    Build a re-run config for the "Advanced: override this site's pipe"
    control. Changes only the three exposed sliders; every other setting
    (attenuation, sample rate, frequency band, confidence threshold) carries
    over unchanged from the previous run at this site.
    """
    base_config = base_result.config
    return SimulationConfig(
        pipe_length_m=pipe_length_m,
        wave_velocity_ms=wave_velocity_ms,
        noise_level=noise_level,
        attenuation_per_m=base_config.attenuation_per_m,
        sample_rate_hz=base_config.sample_rate_hz,
        leak_band_hz=base_config.leak_band_hz,
        confidence_threshold=base_config.confidence_threshold,
    )


def render_detection_page() -> None:
    """The answer in plain language, then the story of how it was reached,
    for whichever site was last selected on the Survey Map page."""

    st.title("Leak Detection Detail")

    selected_name = st.session_state.get("selected_site")
    site_results = st.session_state.get("site_results", {})

    if not selected_name or selected_name not in site_results:
        st.info(
            "Nothing has been triggered yet. Go to **Survey Map**, pick a "
            "site, and press **Trigger Detection**."
        )
        return

    site = get_site(selected_name)
    result = site_results[selected_name]
    config = result.config

    st.caption(f"{site.name}, {site.state}")

    # ------------------------------------------------------------------
    # 1. THE ANSWER, in plain language
    # ------------------------------------------------------------------
    if result.is_confident:
        st.markdown(
            f"## The leak is {result.estimated_position_m:.2f} metres "
            f"from Sensor A."
        )
        st.markdown(
            f"It was actually at **{result.true_position_m:.2f} m**, so the "
            f"system was off by **{format_distance(result.error_m)}** - "
            f"found in **{result.time_total_ms:.1f} milliseconds**, using "
            f"nothing but two sensors and no digging."
        )
    else:
        st.markdown("## Something is leaking, but I cannot tell you where.")
        st.markdown(
            f"The background noise was too high to pick the leak's hiss "
            f"out reliably. The system's best guess would have been "
            f"**{result.estimated_position_m:.2f} m** (the leak was really "
            f"at **{result.true_position_m:.2f} m**), but it has flagged "
            f"that answer as untrustworthy rather than sending you to dig "
            f"there."
        )

    m1, m2, m3 = st.columns(3)
    m1.metric("Where the system said", f"{result.estimated_position_m:.2f} m")
    m2.metric("Where it actually was", f"{result.true_position_m:.2f} m")
    m3.metric("How far off", format_distance(result.error_m))

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
            "**The system knew it was unsure - and said so.** A tool that "
            "confidently reports a wrong location sends a crew to dig up "
            "the wrong road. One that admits uncertainty tells them to come "
            "back with better equipment."
        )

    st.divider()

    # ------------------------------------------------------------------
    # 2. HOW IT WORKS
    # ------------------------------------------------------------------
    st.header("How it figured that out, in three steps")

    st.subheader("Step 1: Two sensors listen to the same hiss")
    st.markdown(
        """
    A leak is not silent - escaping water makes a constant hissing sound.
    Both sensors pick up **the very same hiss**, because there is only one
    leak making it. Notice only this: the closer sensor's trace is
    **taller**, because sound fades as it travels.
    """
    )

    zoom_ms = st.slider(
        "How much of the recording to show (milliseconds)",
        min_value=2, max_value=100, value=20, step=1,
        key="detail_zoom_ms",
    )
    n_show = min(int(config.sample_rate_hz * zoom_ms / 1000.0),
                 result.time_s.size)

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax_a.plot(result.time_s[:n_show] * 1000, result.sensor_a[:n_show],
              color=COLOR_A, linewidth=0.9)
    ax_a.set_ylabel("Loudness")
    ax_a.set_title(
        f"Sensor A  -  {result.true_position_m:.0f} m from the leak",
        loc="left", fontsize=11, fontweight="bold")
    style_axes(ax_a)

    distance_b = config.pipe_length_m - result.true_position_m
    ax_b.plot(result.time_s[:n_show] * 1000, result.sensor_b[:n_show],
              color=COLOR_B, linewidth=0.9)
    ax_b.set_ylabel("Loudness")
    ax_b.set_xlabel("Time (milliseconds)")
    ax_b.set_title(
        f"Sensor B  -  {distance_b:.0f} m from the leak",
        loc="left", fontsize=11, fontweight="bold")
    style_axes(ax_b)

    limit = max(np.abs(result.sensor_a[:n_show]).max(),
                np.abs(result.sensor_b[:n_show]).max()) * 1.1
    ax_a.set_ylim(-limit, limit)
    ax_b.set_ylim(-limit, limit)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Step 2: One of them heard it a fraction sooner")
    delay_ms = abs(result.true_tau_s) * 1000.0
    nearer = "A" if result.true_tau_s < 0 else "B"
    st.markdown(
        f"""
    Sound takes time to travel. The leak is closer to **Sensor {nearer}**,
    so that sensor heard the hiss **{delay_ms:.1f} milliseconds** before
    the other one. That gap is the entire secret - measure it accurately
    and you know the position.

    The catch: **{delay_ms:.1f} ms is far too small to spot by eye.** So
    the computer does something cleverer.
    """
    )

    st.subheader(
        "Step 3: Slide one recording against the other until they match"
    )
    st.markdown(
        """
    Slide the second recording back and forth against the first. At most
    positions the two disagree - noise against noise. But at **one**
    position, the shared hiss lines up and they agree strongly. The tall
    spike below is that moment. **That spike is the measurement.**
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

    st.divider()

    # ------------------------------------------------------------------
    # 3. CAN WE TRUST IT?
    # ------------------------------------------------------------------
    st.header("Can we trust this answer?")
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
                f"comfortably past the {config.confidence_threshold:.0f}x "
                f"bar."
            )
        else:
            st.error(
                f"**Not trustworthy.** The spike was only "
                f"{result.peak_sharpness:.1f} times the background, below "
                f"the {config.confidence_threshold:.0f}x bar. Do not dig on "
                f"this reading."
            )

    st.divider()

    # ------------------------------------------------------------------
    # 4. TRY A DIFFERENT SCENARIO FOR THIS SITE
    # ------------------------------------------------------------------
    with st.expander("Advanced: override this site's pipe and re-run"):
        st.caption(
            f"{site.name}'s preset is {site.pipe_length_m:.0f} m, "
            f"{site.wave_velocity_ms:.0f} m/s, noise "
            f"{site.noise_level:.2f}. Adjust below and re-run to try a "
            f"different scenario at this same site."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            override_length = st.slider(
                "Pipe length (m)", min_value=20.0, max_value=1000.0,
                value=float(config.pipe_length_m), step=10.0)
        with col2:
            override_velocity = st.slider(
                "Wave velocity (m/s)", min_value=300.0, max_value=2000.0,
                value=float(config.wave_velocity_ms), step=10.0)
        with col3:
            override_noise = st.slider(
                "Noise level", min_value=0.0, max_value=2.0,
                value=float(config.noise_level), step=0.05)
            st.caption(describe_noise(override_noise))

        if st.button("Re-run detection with these settings"):
            new_config = build_override_config(
                result, override_length, override_velocity, override_noise)
            rng = np.random.default_rng()
            leak_position = pick_random_leak_position(override_length, rng)
            new_result = run_detection(leak_position, new_config)
            st.session_state["site_results"][selected_name] = new_result
            st.rerun()

    # ------------------------------------------------------------------
    # 5. FOR THE CURIOUS
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("For the curious")

    with st.expander("How long each part took"):
        t1, t2, t3 = st.columns(3)
        t1.metric("Generating the recordings", f"{result.time_acquire_ms:.2f} ms")
        t2.metric("Finding the match", f"{result.time_gccphat_ms:.2f} ms")
        t3.metric("Total", f"{result.time_total_ms:.2f} ms")

    with st.expander("The arithmetic that turns a time into a distance"):
        st.markdown(
            "With `L` the pipe length, `v` the speed of sound in the pipe, "
            "and `t` the measured time gap:"
        )
        st.latex(r"\text{distance from Sensor A} = \frac{L + v \cdot t}{2}")
        st.code(
            f"t = {result.estimated_tau_s:+.8f} s\n"
            f"L = {config.pipe_length_m:.2f} m\n"
            f"v = {config.wave_velocity_ms:.2f} m/s\n\n"
            f"distance = ({config.pipe_length_m:.2f} + "
            f"{config.wave_velocity_ms:.2f} x {result.estimated_tau_s:+.8f}) / 2\n"
            f"         = {result.estimated_position_m:.4f} m\n\n"
            f"Real answer: {result.true_position_m:.4f} m\n"
            f"Off by:      {result.error_m:.6f} m",
            language="text",
        )

    with st.expander("Honest limitations"):
        st.markdown(
            f"""
        - **This is a simulation.** No real pipe was recorded.
        - **One straight pipe per site.** No junctions, branches or bends.
        - **The speed of sound is assumed known.** The answer scales
          directly with it.
        - **The {config.confidence_threshold:.0f}x trust bar was calibrated
          on this simulator, not a proof.**
        - **The map's on-screen sensor spacing is illustrative, not to
          scale.** This site's real pipe length is
          {site.pipe_length_m:.0f} m; only the leak marker's *position along
          the line* is real, proportional to the detector's actual result.
        """
        )
