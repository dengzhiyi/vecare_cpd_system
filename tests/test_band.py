"""Bandwidth-limited hardware: carrier planning, health gating, safe degradation.

The paper assumes 10.5-23.5 kHz. Real drivers are often 18 kHz and sometimes
16 kHz, and the failure is not graceful: a carrier above the rolloff has almost
no probe power, so spectral division returns noise divided by noise. Those
carriers do not merely add error -- they inject a wideband random amplitude
sequence that residual energy reads as motion and MRC weights read as
correlation. Measured on an empty cabin with the paper band and an 18 kHz
driver, that produced a confirmed presence at 17.6 s.

These tests pin down both halves of the answer: plan the band from the hardware
so the dead carriers never exist, and keep a runtime check for when the
datasheet is wrong.
"""
import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from core.audio import (
    carrier_bins,
    carrier_health,
    make_probe,
    probe_condition_db,
    read_wav,
    write_wav,
)
from core.config import (
    CARRIER_MIN_SPACING_HZ,
    F_HIGH,
    F_LOW,
    HW_GUARD_HZ,
    plan_band,
)
from core.offline import run_offline
from core.replay import export_probe, render_wav, run_wav


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def test_full_band_is_unchanged_when_no_ceiling_is_given():
    band = plan_band(None)
    assert band.f_hi == F_HIGH
    assert band.n_carriers == 24
    assert band.f_lo == F_LOW


def test_low_edge_never_moves_with_the_ceiling():
    """Every driver here is a 10 kHz high-pass: only the top of the band is planned.

    A band that shrank from both ends would change which physical paths the
    average covers between two runs, and the breathing ACF calibration assumes a
    fixed noise floor. So the low edge is pinned, and it must stay clear of the
    10 kHz high-pass corner.
    """
    for hw in (None, 20_000, 18_000, 16_000, 14_000):
        band = plan_band(hw)
        assert band.f_lo == F_LOW, f"{hw}: low edge moved to {band.f_lo}"
        assert band.f_lo > 10_000, "must sit above the 10 kHz high-pass corner"
        assert band.sweep_lo < band.f_lo, "sweep must overhang the low edge too"


def test_declared_ceiling_is_never_sounded():
    """Carriers must stay HW_GUARD_HZ below the ceiling, not on it."""
    for hw in (20_000, 18_000, 16_000):
        band = plan_band(hw)
        assert band.f_hi <= hw - HW_GUARD_HZ + 1e-9
        assert band.sweep_hi > band.f_hi, "sweep must overhang the carrier band"


def test_narrow_band_keeps_diversity_and_drops_carrier_count():
    """Trade count for spacing, not spacing for count."""
    wide, mid, narrow = plan_band(None), plan_band(18_000), plan_band(16_000)
    assert wide.n_carriers == 24 and mid.n_carriers == 24
    assert narrow.n_carriers < mid.n_carriers
    for band in (wide, mid, narrow):
        assert band.spacing_hz >= CARRIER_MIN_SPACING_HZ - 1e-9
        # fringes must rotate between neighbours, or they are not independent
        assert band.diversity >= 1.0, band.describe()


def test_impossible_band_is_rejected_loudly():
    with pytest.raises(ValueError):
        plan_band(12_000)


def test_carriers_snap_onto_the_probe_grid_for_every_ceiling():
    """Off-grid carriers interpolate a quantity that oscillates: 10 % error."""
    for hw in (None, 18_000, 16_000):
        probe = make_probe(hw_f_max=hw)
        carriers = carrier_bins(probe)
        df = probe.fs / probe.length
        residual = np.abs(carriers / df - np.round(carriers / df))
        assert residual.max() < 1e-9
        # and the probe must actually have power there
        assert probe_condition_db(probe, carriers).min() > -12.0


def test_probe_stops_at_the_hardware_ceiling():
    """Sounding above the driver wastes period energy on unheard frequencies."""
    wide, narrow = make_probe(), make_probe(hw_f_max=16_000)
    spec_w = np.abs(np.fft.fft(wide.samples))
    spec_n = np.abs(np.fft.fft(narrow.samples))
    df = wide.fs / wide.length
    k18 = int(round(18_000 / df))
    # the wide probe still excites 18 kHz; the narrow one is 50 dB down there
    assert spec_w[k18] > 0.5 * spec_w.max()
    assert spec_n[k18] < 1e-2 * spec_n.max()
    # and the narrow probe must be strong everywhere it does place carriers
    carriers = carrier_bins(narrow)
    assert probe_condition_db(narrow, carriers).min() > -3.0


# ---------------------------------------------------------------------------
# runtime health gating
# ---------------------------------------------------------------------------

def _bandlimited(path_in, path_out, fc, order=6):
    x, fs = read_wav(path_in)
    sos = butter(order, fc, btype="low", fs=fs, output="sos")
    write_wav(path_out, sosfiltfilt(sos, x), fs=fs)
    return path_out


def test_health_flags_dead_carriers_and_keeps_live_ones(tmp_path):
    probe = make_probe()
    src = tmp_path / "empty.wav"
    render_wav(src, "empty", duration_s=16.0, snr_db=35.0)
    limited = _bandlimited(src, tmp_path / "lp18.wav", 18_000)

    from core.replay import load_csi

    amp, _, meta = load_csi(limited)
    alive, rel, threshold = carrier_health(amp[:200])
    freqs = carrier_bins(probe)

    # well below the cutoff every carrier survives; well above it none do.
    # (Between them, a carrier can also die from a multipath null -- a real
    # condition, not a bandwidth artefact, so that range is left unasserted.)
    below = freqs < 12_000
    above = freqs > 20_500   # a 6th-order 18 kHz rolloff is only -3 dB at 18k
    assert alive[below].all(), f"false kill below cutoff: {rel[below]}"
    assert not alive[above].any(), f"dead carriers kept: {rel[above]}"
    assert 0.0 < threshold < 1.0


def test_pruning_survives_half_the_band_being_dead():
    """The baseline must not be the median -- once half the band dies the
    median has already moved into the dead carriers and prunes nothing."""
    rng = np.random.default_rng(0)
    live = np.abs(1.0 + 0.02 * rng.standard_normal((400, 12)))
    dead = np.abs(0.01 * rng.standard_normal((400, 12)))   # noise/noise
    alive, rel, threshold = carrier_health(np.hstack([live, dead]))
    assert alive[:12].sum() >= 10
    assert alive[12:].sum() == 0, f"threshold {threshold:.3f}, dead rel {rel[12:].round(3)}"


def test_all_dead_band_is_not_silently_pruned_to_nothing():
    rng = np.random.default_rng(1)
    dead = np.abs(0.01 * rng.standard_normal((400, 20)))
    alive, _, _ = carrier_health(dead)
    assert alive.all(), "pruning everything would hide the hardware fault"


# ---------------------------------------------------------------------------
# end-to-end degradation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hw", [18_000, 16_000])
def test_narrow_hardware_keeps_a_quiet_cabin_quiet(tmp_path, hw):
    """The whole point: a door slam must not become a child."""
    path = tmp_path / "stress.wav"
    render_wav(path, "stress", snr_db=35.0, hw_f_max=hw)
    metrics = run_wav(path, hw_f_max=hw)["metrics"]
    assert not metrics["detected"], f"false presence at {metrics['presence_at']} s"
    assert metrics["alarm_at"] is None


@pytest.mark.parametrize("hw", [18_000, 16_000])
def test_narrow_hardware_still_sees_a_breathing_child(tmp_path, hw):
    path = tmp_path / "sleeping.wav"
    render_wav(path, "sleeping", snr_db=35.0, hw_f_max=hw)
    metrics = run_wav(path, hw_f_max=hw)["metrics"]
    assert metrics["detected"]
    assert metrics["bpm_median"] == pytest.approx(18.0, abs=1.5)
    # the response budget starts at the first evidence, not at confirmation
    assert 9.5 <= metrics["response_time_s"] <= 11.0


def test_unplanned_band_is_caught_by_runtime_pruning(tmp_path):
    """A full-band probe behind an 18 kHz driver: the design is wrong, but the
    front end must not turn a quiet cabin into an alarm on its own."""
    src = tmp_path / "empty.wav"
    render_wav(src, "empty", duration_s=26.0, snr_db=35.0)
    limited = _bandlimited(src, tmp_path / "lp18.wav", 18_000)
    result = run_wav(limited)
    assert result["source"]["carriers_used"] < result["source"]["carriers"]
    assert result["source"]["carriers_pruned"], "pruned carriers should be named"
    assert result["metrics"]["alarm_at"] is None


def test_offline_runner_reports_the_band_it_used():
    wide = run_offline("empty", hw_f_max=None)["metrics"]
    narrow = run_offline("empty", hw_f_max=16_000)["metrics"]
    assert wide["carriers_planned"] == 24
    assert narrow["carriers_planned"] < wide["carriers_planned"]
    assert "carriers" in narrow["band"]


def test_exported_probe_matches_the_declared_ceiling(tmp_path):
    info = export_probe(tmp_path / "p.wav", seconds=0.2, hw_f_max=16_000)
    assert info["sweep_hz"][1] < 16_000
    assert info["carriers"] == plan_band(16_000).n_carriers


def test_ui_offers_exactly_the_bands_the_planner_accepts():
    """The chips must never be able to request a band plan_band would reject.

    The UI reads the carrier counts from this payload rather than re-deriving
    anything in the browser, so a drift here would put the numbers on screen out
    of step with the numbers the DSP actually uses.
    """
    from server import BAND_CHOICES, band_options

    opts = band_options()
    assert [o["hw_f_max"] for o in opts] == list(BAND_CHOICES)
    for opt, hw in zip(opts, BAND_CHOICES):
        band = plan_band(hw)
        assert opt["n_carriers"] == band.n_carriers
        assert opt["f_lo_hz"] == band.f_lo
        assert opt["f_hi_hz"] == band.f_hi
        assert opt["label"], "every chip needs a short label"
        assert opt["describe"] == band.describe()
