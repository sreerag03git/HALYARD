"""UI tests for the loading splash and the mobile / compact-layout feature.

These pin the deploy-critical first-load behaviour: a once-per-session boot splash, the
global CSS injection, and a sidebar toggle that forces the compact layout and swaps the
section nav for a dropdown. They must not regress the section-nav design (only the
SELECTED section renders) that keeps the app inside the free-tier RAM budget.

The CSS/splash are injected with st.html, whose string content AppTest does not expose
(it appears as an untyped "html" element), so these assert on behaviour and element
counts. The actual rendered CSS (columns stacking, splash overlay) is verified live in a
browser during development, not here.
"""
from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _run():
    return AppTest.from_file("streamlit_app.py", default_timeout=120).run()


def _n_html(at) -> int:
    return len(at.get("html"))


def test_boot_splash_renders_once_per_session():
    at = _run()
    assert not at.exception
    assert "_booted" in at.session_state and at.session_state["_booted"]
    n1 = _n_html(at)          # inject_css + boot_splash
    at.run()                  # a plain rerun
    assert not at.exception
    n2 = _n_html(at)          # inject_css only — splash must be gone
    assert n2 == n1 - 1, f"splash should emit exactly once per session (n1={n1}, n2={n2})"


def test_global_css_injected_without_error():
    at = _run()
    assert not at.exception
    assert _n_html(at) >= 1   # the global <style> block from inject_css


def test_compact_toggle_swaps_nav_and_forces_css():
    at = _run()
    assert not at.exception
    # Desktop default: the section nav is a horizontal radio, not a dropdown.
    assert any(r.label == "Section" for r in at.radio)
    assert not any(s.label == "Section" for s in at.selectbox)
    # Turn on the compact / mobile layout.
    at.session_state["_mobile_view"] = True
    at.run()
    assert not at.exception
    assert any(s.label == "Section" for s in at.selectbox)   # nav is now a dropdown
    # inject_css + inject_mobile_css both emit an html <style> element on this (booted) run.
    assert _n_html(at) >= 2


def test_section_choice_preserved_across_toggle():
    at = _run()
    assert not at.exception
    nav = next(r for r in at.radio if r.label == "Section")
    nav.set_value("High-fidelity ingest")
    at.run()
    assert not at.exception
    assert at.session_state["_nav"] == "High-fidelity ingest"
    # Flip to compact: the dropdown must open on the SAME section (persisted via _nav).
    at.session_state["_mobile_view"] = True
    at.run()
    assert not at.exception
    sel = next(s for s in at.selectbox if s.label == "Section")
    assert sel.value == "High-fidelity ingest"


def test_only_selected_section_renders():
    """The compact nav must preserve the free-tier guarantee: switching sections does not
    render the others. 'High-fidelity ingest' with no upload shows an uploader and none of
    the heavy-simulation headings from other sections."""
    at = _run()
    nav = next(r for r in at.radio if r.label == "Section")
    nav.set_value("High-fidelity ingest")
    at.run()
    assert not at.exception
    body = " ".join(m.value for m in at.markdown)
    assert "Grid frequency, support and recovery" not in body
    assert "Platform motion and the dynamic cable" not in body
