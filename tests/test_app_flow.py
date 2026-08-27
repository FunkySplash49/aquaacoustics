"""
test_app_flow.py - end-to-end wiring, run without a browser.
================================================================
Uses Streamlit's official AppTest framework to drive the real app.py: role
selection, site selection, triggering detection, and switching pages -
proving the pieces built in isolation (sites.py, map_view.py,
detection_view.py) are actually wired together correctly. This does not
replace tests/test_gccphat.py etc. - it proves the INTEGRATION, not the
maths (which is already proven elsewhere).
"""

from streamlit.testing.v1 import AppTest


def _run_app():
    # AppTest.from_file resolves a relative path against the file that CALLS
    # it (this test file, in tests/), not the pytest working directory - so
    # the app lives one directory up.
    at = AppTest.from_file("../app.py")
    at.run(timeout=30)
    assert not at.exception
    return at


def test_app_starts_on_survey_map_with_no_result_yet():
    at = _run_app()
    assert any("Survey Map" in t.value for t in at.title)
    assert at.session_state["role"] == "Admin"


def test_admin_can_trigger_detection_for_the_selected_site():
    at = _run_app()

    assert at.session_state["role"] == "Admin"
    selected_site = at.session_state["selected_site"]

    at.button[0].click().run(timeout=30)
    assert not at.exception

    assert selected_site in at.session_state["site_results"]
    result = at.session_state["site_results"][selected_site]
    assert result.estimated_position_m >= 0.0
    assert len(at.session_state["history"]) == 1


def test_field_staff_sees_no_trigger_button():
    at = _run_app()
    at.sidebar.radio[0].set_value("Field Staff").run(timeout=30)
    assert not at.exception
    assert at.session_state["role"] == "Field Staff"
    assert len(at.button) == 0


def test_field_staff_sees_no_buttons_on_detection_detail_page_either():
    """
    Regression test: the Leak Detection Detail page's "Advanced: override
    this site's pipe and re-run" control has its own button, separate from
    the Survey Map's "Trigger Detection" button. Field Staff must not see
    ANY button on that page either, or they could still trigger a real
    detection run despite being read-only everywhere else.
    """
    at = _run_app()

    # Trigger as Admin first, so there IS a result to view on the Detail page.
    assert at.session_state["role"] == "Admin"
    at.button[0].click().run(timeout=30)
    assert not at.exception

    # Now switch to Field Staff.
    at.sidebar.radio[0].set_value("Field Staff").run(timeout=30)
    assert not at.exception
    assert at.session_state["role"] == "Field Staff"

    # And switch to the Leak Detection Detail page.
    at.sidebar.radio[1].set_value("Leak Detection Detail").run(timeout=30)
    assert not at.exception
    assert any("Leak Detection Detail" in t.value for t in at.title)
    assert len(at.button) == 0


def test_detection_detail_page_renders_the_triggered_result():
    at = _run_app()
    at.button[0].click().run(timeout=30)
    assert not at.exception

    at.sidebar.radio[1].set_value("Leak Detection Detail").run(timeout=30)
    assert not at.exception
    assert any("Leak Detection Detail" in t.value for t in at.title)
