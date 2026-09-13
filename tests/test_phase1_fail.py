"""The Phase-1 FAIL branch MUST be reachable — the app can honestly say "negligible".

This guarantees the honesty mechanism (§6, §13): a stiff-grid / low-deviation input returns
NEGLIGIBLE, and a weak-grid / calm-sea input returns PASS.
"""
import pytest

from analysis.config_io import load_case
from analysis.phase1 import run_phase1
from physics.cable import build_quasistatic_family


@pytest.fixture(scope="module")
def family():
    return build_quasistatic_family()


def test_stiff_grid_is_negligible(family):
    c = load_case("stiff_grid")
    r = run_phase1(c.sim, thresholds=c.thresholds, family=family,
                   fatigue_params=c.fatigue, daf=c.daf, hangoff_stick=c.hangoff_stick)
    assert not r.passed
    assert r.verdict == "NEGLIGIBLE"
    # It fails specifically on the added-damage criterion (the effect is not there).
    assert r.reasons["added_damage"] is False


def test_default_case_passes(family):
    c = load_case("default")
    r = run_phase1(c.sim, thresholds=c.thresholds, family=family,
                   fatigue_params=c.fatigue, daf=c.daf, hangoff_stick=c.hangoff_stick)
    assert r.passed
    assert r.verdict == "PASS"


def test_gate_returns_evidence(family):
    c = load_case("default")
    r = run_phase1(c.sim, thresholds=c.thresholds, family=family)
    assert r.psd_freq.size > 0
    assert r.psd_with_support.size > 0
    assert set(r.modal_bands_hz) == {"surge", "pitch"}
