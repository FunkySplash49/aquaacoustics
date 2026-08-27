"""
test_sites.py - the preset site registry and its map geometry helpers.
========================================================================
These tests don't touch aeld's physics at all - they prove two separate
things: (1) every preset site is well-formed, and (2) the geometry that
turns a 1-D detector result into a 2-D map point is correct at its three
checkable landmarks: the two sensor endpoints, and the midpoint.
"""

from math import radians, cos, sqrt

import pytest

from aquaacoustics.sites import (
    SITES, get_site, sensor_coordinates, interpolate_position,
    polygon_corners, ON_MAP_SENSOR_SEPARATION_M,
)


# ============================================================================
# Registry sanity
# ============================================================================

def test_six_sites_registered():
    assert len(SITES) == 6


def test_site_names_are_unique():
    names = [site.name for site in SITES]
    assert len(names) == len(set(names))


def test_site_states_are_distinct():
    states = [site.state for site in SITES]
    assert len(states) == len(set(states))


@pytest.mark.parametrize("site", SITES, ids=lambda s: s.name)
def test_site_has_valid_coordinates(site):
    assert -90.0 <= site.lat <= 90.0
    assert -180.0 <= site.lng <= 180.0


@pytest.mark.parametrize("site", SITES, ids=lambda s: s.name)
def test_site_has_positive_pipe_parameters(site):
    assert site.pipe_length_m > 0
    assert site.wave_velocity_ms > 0
    assert site.noise_level >= 0


def test_get_site_returns_the_right_site():
    site = get_site("Jaipur Ring Main")
    assert site.name == "Jaipur Ring Main"
    assert site.state == "Rajasthan"


def test_get_site_unknown_name_raises():
    with pytest.raises(KeyError):
        get_site("Nonexistent Site")


# ============================================================================
# Map geometry: the three checkable landmarks
# ============================================================================

@pytest.mark.parametrize("site", SITES, ids=lambda s: s.name)
def test_interpolate_at_zero_matches_sensor_a(site):
    sensor_a, _sensor_b = sensor_coordinates(site)
    result = interpolate_position(site, 0.0)
    assert result[0] == pytest.approx(sensor_a[0], abs=1e-9)
    assert result[1] == pytest.approx(sensor_a[1], abs=1e-9)


@pytest.mark.parametrize("site", SITES, ids=lambda s: s.name)
def test_interpolate_at_full_length_matches_sensor_b(site):
    _sensor_a, sensor_b = sensor_coordinates(site)
    result = interpolate_position(site, site.pipe_length_m)
    assert result[0] == pytest.approx(sensor_b[0], abs=1e-9)
    assert result[1] == pytest.approx(sensor_b[1], abs=1e-9)


@pytest.mark.parametrize("site", SITES, ids=lambda s: s.name)
def test_interpolate_at_half_length_is_the_midpoint(site):
    sensor_a, sensor_b = sensor_coordinates(site)
    result = interpolate_position(site, site.pipe_length_m / 2.0)
    expected_lat = (sensor_a[0] + sensor_b[0]) / 2.0
    expected_lng = (sensor_a[1] + sensor_b[1]) / 2.0
    assert result[0] == pytest.approx(expected_lat, abs=1e-9)
    assert result[1] == pytest.approx(expected_lng, abs=1e-9)


def test_interpolate_clamps_beyond_pipe_length():
    site = SITES[0]
    beyond = interpolate_position(site, site.pipe_length_m * 2.0)
    _sensor_a, sensor_b = sensor_coordinates(site)
    assert beyond[0] == pytest.approx(sensor_b[0], abs=1e-9)
    assert beyond[1] == pytest.approx(sensor_b[1], abs=1e-9)


def test_interpolate_clamps_below_zero():
    site = SITES[0]
    below = interpolate_position(site, -50.0)
    sensor_a, _sensor_b = sensor_coordinates(site)
    assert below[0] == pytest.approx(sensor_a[0], abs=1e-9)
    assert below[1] == pytest.approx(sensor_a[1], abs=1e-9)


def test_sensor_separation_matches_constant():
    """Sensor A and B should be ON_MAP_SENSOR_SEPARATION_M apart on the map,
    regardless of the site's real (much larger) pipe_length_m."""
    site = SITES[0]
    sensor_a, sensor_b = sensor_coordinates(site)

    d_lat_m = (sensor_b[0] - sensor_a[0]) * 111_320.0
    d_lng_m = (sensor_b[1] - sensor_a[1]) * 111_320.0 * cos(radians(site.lat))
    separation_m = sqrt(d_lat_m ** 2 + d_lng_m ** 2)

    assert separation_m == pytest.approx(ON_MAP_SENSOR_SEPARATION_M, rel=1e-6)


@pytest.mark.parametrize("override_length", [1000.0, 500.0, 33.0])
def test_interpolate_with_override_length_uses_override_not_preset(override_length):
    """
    Regression test for the "override re-run" bug: when a run used a
    DIFFERENT pipe length than the site's preset (e.g. the Leak Detection
    Detail page's "Advanced: override this site's pipe and re-run" control),
    interpolate_position must divide by that override length, not the site's
    preset pipe_length_m - otherwise the fraction (and therefore the marker
    position) is wrong whenever override_length != site.pipe_length_m.
    """
    site = SITES[0]
    assert override_length != site.pipe_length_m   # the whole point of this test

    sensor_a, sensor_b = sensor_coordinates(site)
    result = interpolate_position(site, override_length / 2.0,
                                  pipe_length_m=override_length)
    expected_lat = (sensor_a[0] + sensor_b[0]) / 2.0
    expected_lng = (sensor_a[1] + sensor_b[1]) / 2.0
    assert result[0] == pytest.approx(expected_lat, abs=1e-9)
    assert result[1] == pytest.approx(expected_lng, abs=1e-9)


def test_interpolate_without_pipe_length_arg_still_uses_site_preset():
    """The optional third argument must not change default behaviour when
    omitted - existing callers (and the landmark tests above) keep working
    unchanged."""
    site = SITES[0]
    sensor_a, sensor_b = sensor_coordinates(site)
    result = interpolate_position(site, site.pipe_length_m / 2.0)
    expected_lat = (sensor_a[0] + sensor_b[0]) / 2.0
    expected_lng = (sensor_a[1] + sensor_b[1]) / 2.0
    assert result[0] == pytest.approx(expected_lat, abs=1e-9)
    assert result[1] == pytest.approx(expected_lng, abs=1e-9)


def test_polygon_corners_returns_four_points_centred_on_site():
    site = SITES[0]
    corners = polygon_corners(site)
    assert len(corners) == 4

    avg_lat = sum(c[0] for c in corners) / 4.0
    avg_lng = sum(c[1] for c in corners) / 4.0
    assert avg_lat == pytest.approx(site.lat, abs=1e-9)
    assert avg_lng == pytest.approx(site.lng, abs=1e-9)
