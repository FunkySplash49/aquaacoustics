"""
map_view.py - the interactive Survey Map page.
==================================================
Shows every preset monitoring site on a Folium/OpenStreetMap map of India.
Admin can select a site and trigger detection, which runs the REAL aeld
GCC-PHAT pipeline (src/aeld/detector.py) against that site's preset pipe and
draws the result - never a random point - on the map. Field Staff see the
same map read-only.

See docs/superpowers/specs/2026-08-27-aquaacoustics-integration-design.md
for the full design and the reasoning behind the "honest marker" rule.
"""

from datetime import datetime, timezone

import folium
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

from aeld import SimulationConfig, run_detection
from .sites import (
    SITES, get_site, sensor_coordinates, interpolate_position, polygon_corners,
)


INDIA_CENTER = (22.9734, 78.6569)     # geographic centre, initial map view
INDIA_DEFAULT_ZOOM = 5
SITE_ZOOM = 17


# ============================================================================
# Pure helpers (unit tested in tests/test_map_view.py)
# ============================================================================

def pick_random_leak_position(pipe_length_m: float,
                              rng: np.random.Generator) -> float:
    """
    Pick an unknown-to-the-detector leak position, same rule the original
    AELD app used: stay slightly inside the ends, since a leak sitting
    exactly on a sensor is a degenerate edge case, not a useful demo.
    """
    return float(rng.uniform(0.05 * pipe_length_m, 0.95 * pipe_length_m))


def build_history_entry(site, result, timestamp: datetime) -> dict:
    """Shape one row of the session's trigger history for display."""
    return {
        "site": site.name,
        "state": site.state,
        "estimated_position_m": result.estimated_position_m,
        "true_position_m": result.true_position_m,
        "error_m": result.error_m,
        "is_confident": result.is_confident,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


# ============================================================================
# Session state
# ============================================================================

def _ensure_session_state():
    if "site_results" not in st.session_state:
        st.session_state["site_results"] = {}
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "selected_site" not in st.session_state:
        st.session_state["selected_site"] = SITES[0].name


def _trigger_detection(site) -> None:
    """Run the real detector on `site`'s preset pipe and record the result."""
    rng = np.random.default_rng()
    leak_position = pick_random_leak_position(site.pipe_length_m, rng)

    config = SimulationConfig(
        pipe_length_m=site.pipe_length_m,
        wave_velocity_ms=site.wave_velocity_ms,
        noise_level=site.noise_level,
    )

    with st.spinner(f"Listening at {site.name}..."):
        result = run_detection(leak_position, config)

    st.session_state["site_results"][site.name] = result

    timestamp = datetime.now(timezone.utc)
    st.session_state["history"].append(build_history_entry(site, result, timestamp))


# ============================================================================
# The page
# ============================================================================

def render_map_page(role: str) -> None:
    """Render the Survey Map page. `role` is 'Admin' or 'Field Staff'."""
    _ensure_session_state()

    st.title("Survey Map")
    st.caption("Pick a monitoring site to see or run its leak detection.")

    site_names = [site.name for site in SITES]
    selected_name = st.selectbox(
        "Monitoring site",
        site_names,
        index=site_names.index(st.session_state["selected_site"]),
    )
    st.session_state["selected_site"] = selected_name
    site = get_site(selected_name)

    result = st.session_state["site_results"].get(site.name)

    # Once a detection has run for this site, describe the pipe that was
    # ACTUALLY used for that run (result.config), not the site's own preset -
    # an "Advanced: override" re-run on the Leak Detection Detail page can use
    # a different pipe length than the preset, and the preset is no longer
    # what was actually run. Before anything has run, the preset is the only
    # meaningful number, so fall back to it.
    if result is not None:
        described_length = result.config.pipe_length_m
        described_velocity = result.config.wave_velocity_ms
        described_noise = result.config.noise_level
    else:
        described_length = site.pipe_length_m
        described_velocity = site.wave_velocity_ms
        described_noise = site.noise_level

    st.caption(f"{site.state} - preset pipe: {described_length:.0f} m, "
               f"{described_velocity:.0f} m/s, "
               f"noise level {described_noise:.2f}")

    if role == "Admin":
        if st.button("Trigger Detection", type="primary"):
            _trigger_detection(site)
    else:
        st.caption(
            "Field Staff view: read-only. Ask an Admin to trigger detection."
        )

    if result is not None:
        map_center = (site.lat, site.lng)
        zoom = SITE_ZOOM
    else:
        map_center = INDIA_CENTER
        zoom = INDIA_DEFAULT_ZOOM

    fmap = folium.Map(location=map_center, zoom_start=zoom, tiles="OpenStreetMap")

    for other_site in SITES:
        folium.CircleMarker(
            location=(other_site.lat, other_site.lng),
            radius=6,
            color="#34495E",
            fill=True,
            fill_opacity=0.8,
            tooltip=f"{other_site.name} ({other_site.state})",
        ).add_to(fmap)

    if result is not None:
        folium.Polygon(
            locations=polygon_corners(site),
            color="#34495E",
            weight=2,
            fill=True,
            fill_opacity=0.08,
            tooltip="Surveyed area",
        ).add_to(fmap)

        sensor_a, sensor_b = sensor_coordinates(site)
        folium.Marker(
            location=sensor_a, tooltip="Sensor A (0 m)",
            icon=folium.Icon(color="blue", icon="microphone", prefix="fa"),
        ).add_to(fmap)
        folium.Marker(
            location=sensor_b,
            tooltip=f"Sensor B ({described_length:.0f} m)",
            icon=folium.Icon(color="orange", icon="microphone", prefix="fa"),
        ).add_to(fmap)

        # The honest marker: always the detector's real computed position,
        # never a random point. See the design doc for why this matters.
        # Divide by the REAL run's pipe length (result.config.pipe_length_m),
        # not the site's preset - an override re-run can use a different
        # length, and dividing by the wrong one places the marker incorrectly.
        leak_point = interpolate_position(
            site, result.estimated_position_m, result.config.pipe_length_m)
        marker_color = "green" if result.is_confident else "gray"
        folium.Marker(
            location=leak_point,
            tooltip=(f"Detected leak: {result.estimated_position_m:.1f} m "
                     f"from Sensor A"
                     + ("" if result.is_confident else " (low confidence)")),
            icon=folium.Icon(color=marker_color, icon="exclamation-triangle",
                             prefix="fa"),
        ).add_to(fmap)

    st_folium(fmap, width=None, height=520, returned_objects=[])

    if result is not None:
        confidence_note = ("" if result.is_confident else
                           " (flagged low-confidence - see Leak Detection Detail)")
        st.info(
            f"Last result for **{site.name}**: leak found "
            f"{result.estimated_position_m:.1f} m from Sensor A"
            f"{confidence_note}. Open **Leak Detection Detail** in the "
            f"sidebar for the full explanation."
        )
    else:
        st.info("No detection has been triggered for this site yet.")

    st.divider()
    st.subheader("Session history")
    history = st.session_state["history"]
    if not history:
        st.caption("Nothing triggered yet this session.")
    else:
        st.dataframe(
            [
                {
                    "Site": row["site"],
                    "State": row["state"],
                    "Found at (m)": round(row["estimated_position_m"], 2),
                    "Actual (m)": round(row["true_position_m"], 2),
                    "Error (m)": round(row["error_m"], 3),
                    "Confident?": "Yes" if row["is_confident"] else "No",
                    "When": row["timestamp"],
                }
                for row in reversed(history)
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "In-memory only for this session - resets if the server "
            "restarts. This is a prototype, not a durable log."
        )
