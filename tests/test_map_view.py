"""
test_map_view.py - pure helper functions behind the survey map page.
========================================================================
The Streamlit rendering itself (the folium map, buttons, layout) is covered
by the end-to-end AppTest in tests/test_app_flow.py, matching how the rest
of this project tests maths and logic directly and leaves UI layout to be
exercised by running the app. These tests cover the two testable pieces of
logic map_view.py owns: picking a leak position, and shaping a history row.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from aeld import SimulationConfig, run_detection
from aquaacoustics.map_view import pick_random_leak_position, build_history_entry
from aquaacoustics.sites import SITES


def test_pick_random_leak_position_stays_within_the_safe_range():
    rng = np.random.default_rng(7)
    pipe_length_m = 200.0
    for _ in range(200):
        position = pick_random_leak_position(pipe_length_m, rng)
        assert 0.05 * pipe_length_m <= position <= 0.95 * pipe_length_m


def test_pick_random_leak_position_is_reproducible_with_a_seeded_rng():
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    assert (pick_random_leak_position(200.0, rng_a)
            == pick_random_leak_position(200.0, rng_b))


def test_build_history_entry_has_the_expected_shape():
    site = SITES[0]
    config = SimulationConfig(
        pipe_length_m=site.pipe_length_m,
        wave_velocity_ms=site.wave_velocity_ms,
        noise_level=site.noise_level,
        random_seed=42,
    )
    result = run_detection(site.pipe_length_m / 2.0, config)
    timestamp = datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc)

    entry = build_history_entry(site, result, timestamp)

    assert entry["site"] == site.name
    assert entry["state"] == site.state
    assert entry["estimated_position_m"] == pytest.approx(result.estimated_position_m)
    assert entry["true_position_m"] == pytest.approx(result.true_position_m)
    assert entry["error_m"] == pytest.approx(result.error_m)
    assert entry["is_confident"] == result.is_confident
    assert entry["timestamp"] == "2026-08-27 12:00:00 UTC"
