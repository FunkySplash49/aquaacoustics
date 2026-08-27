"""
test_detection_view.py - pure helper behind the Leak Detection Detail page's
"Advanced: override this site's pipe" re-run control.
==============================================================================
Rendering itself is covered by the end-to-end AppTest in
tests/test_app_flow.py. This test covers the one piece of standalone logic
detection_view.py owns: building a re-run config that changes only the three
exposed sliders and carries every other setting over from the previous run.
"""

import pytest

from aeld import SimulationConfig, run_detection
from aquaacoustics.detection_view import build_override_config
from aquaacoustics.sites import SITES


def test_build_override_config_changes_only_the_three_exposed_fields():
    site = SITES[0]
    base_config = SimulationConfig(
        pipe_length_m=site.pipe_length_m,
        wave_velocity_ms=site.wave_velocity_ms,
        noise_level=site.noise_level,
        attenuation_per_m=0.007,
        sample_rate_hz=16000,
    )
    base_result = run_detection(site.pipe_length_m / 2.0, base_config)

    new_config = build_override_config(base_result, 999.0, 1500.0, 0.9)

    assert new_config.pipe_length_m == 999.0
    assert new_config.wave_velocity_ms == 1500.0
    assert new_config.noise_level == 0.9
    assert new_config.attenuation_per_m == 0.007
    assert new_config.sample_rate_hz == 16000
    assert new_config.leak_band_hz == base_config.leak_band_hz
    assert new_config.confidence_threshold == base_config.confidence_threshold
