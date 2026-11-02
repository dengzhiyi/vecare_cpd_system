"""Replay a recorded WAV through the SAS pipeline.

This is the bridge between a real capture and the detector. The simulator hands
``SASPipeline`` 24 complex channel coefficients every 20 ms; a recording only
gives a microphone waveform. The chain is:

    probe -> (speaker, cabin, mic) -> WAV -> frame sync -> |H(f)| -> pipeline

``render_wav`` closes the loop in the other direction: it turns a simulated
scenario into a WAV, so the whole audio path can be checked end to end without
any hardware. That check is the only honest way to trust a real capture's
numbers -- if the rendered WAV does not reproduce the simulator's own verdict,
the front end is at fault, not the cabin.

Known modelling assumption: the probe repeats every 20 ms, so the estimate
assumes the cabin impulse response is shorter than 20 ms (6.9 m of path). A
large gym-hall reverb would wrap around and alias; a car cabin does not.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from .audio import (
    Probe,
    carrier_bins,
    estimate_csi,
    frame_offsets,
    make_probe,
    probe_condition_db,
    read_wav,
    write_wav,
)
from .channel import AcousticChannel
from .config import (
    AUDIO_FS,
    CSI_FS,
    F_HIGH,
    MRC_WEIGHT_FLOOR,
    SWEEP_HI,
    SWEEP_LO,
    DetectorConfig,
    plan_band,
)
from .pipeline import SASPipeline
from .scenarios import get_scenario


# ---------------------------------------------------------------------------
# probe export
# ---------------------------------------------------------------------------

def export_probe(
    path: str | Path,
    seconds: float = 120.0,
    fs: int = AUDIO_FS,
    hw_f_max: float | None = None,
) -> dict:
    """Write the sounding signal as a WAV you can play during a capture.

    The probe has to be playing while you record; without it there is no CSI to
    estimate, only speech and road noise.

    ``hw_f_max`` is the measured -3 dB ceiling of speaker + microphone. Sounding
    above it is not free: the sweep spends period energy on frequencies the
    driver cannot reproduce, so every carrier that *is* used comes out noisier.
    Pass the real ceiling and the sweep shrinks to fit.
    """
    probe = make_probe(fs=fs, hw_f_max=hw_f_max)
    reps = max(1, int(round(seconds * probe.csi_fs)))
    write_wav(path, np.tile(probe.samples, reps), fs=fs)
    band = probe.band
    return {
        "path": str(path),
        "seconds": round(reps * probe.period_s, 2),
        "fs": fs,
        "period_ms": round(probe.period_s * 1000, 3),
        "sweep_hz": [round(band.sweep_lo, 1), round(band.sweep_hi, 1)],
        "carrier_hz": [round(band.f_lo, 1), round(band.f_hi, 1)],
        "carriers": band.n_carriers,
        "band": band.describe(),
        "samples_per_period": probe.length,
    }


# ---------------------------------------------------------------------------
# WAV -> CSI
# ---------------------------------------------------------------------------

def load_csi(
    path: str | Path,
    fs: int | None = None,
    probe: Probe | None = None,
    carriers: np.ndarray | None = None,
    min_peak_ratio: float = 0.2,
    hw_f_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a WAV and return ``(amplitude, times, meta)``.

    ``amplitude`` has shape ``(n_frames, n_carriers)``; ``times`` are true
    positions in the file, so a dropped frame shows up as a real gap in time
    rather than silently stretching the series.

    ``hw_f_max`` plans the carrier band around the hardware's usable ceiling
    instead of the paper's 23.5 kHz. Without it, a 16 kHz driver is asked for
    carriers it cannot emit -- see ``config.plan_band``.
    """
    x, native_fs = read_wav(path, target_fs=fs)
    band = plan_band(hw_f_max)
    if 2 * band.sweep_hi >= native_fs:
        raise RuntimeError(
            f"{Path(path).name}: sample rate is {native_fs} Hz, but the "
            f"{band.sweep_hi/1000:.1f} kHz sweep needs at least "
            f"{2*band.sweep_hi/1000:.1f} kHz of Nyquist bandwidth. Record at 48 kHz."
        )
    probe = probe or make_probe(fs=native_fs, hw_f_max=hw_f_max)

    probe = probe or make_probe(fs=native_fs)
    starts, sync = frame_offsets(x, probe, min_ratio=min_peak_ratio)
    if starts.size < 2 or sync["match_ratio"] < 8.0:
        raise RuntimeError(
            f"{Path(path).name}: no probe repetition found "
            f"(correlation peak/floor {sync['match_ratio']:.1f}, need >= 8; a "
            f"recording that contains the probe scores around 24). The probe "
            f"must be playing during the recording, at its original "
            f"{probe.period_s*1000:.1f} ms period -- no resampling, no tempo "
            f"change, no noise suppression. Run "
            f"`python replay_audio.py --emit-probe probe.wav` for the file, and "
            f"see README「实采录音接入」for the recording setup."
        )

    carriers = carrier_bins(probe) if carriers is None else np.asarray(carriers, dtype=float)
    cfr, used = estimate_csi(x, probe, carriers, starts=starts)
    if used.size == 0:
        raise RuntimeError(f"{Path(path).name}: frames detected but none complete")

    amp = np.abs(cfr)
    times = used / float(probe.fs)

    meta = {
        "path": str(path),
        "fs": native_fs,
        "frames": int(used.size),
        "carriers": int(carriers.size),
        "carrier_span_hz": [round(float(carriers.min()), 1), round(float(carriers.max()), 1)],
        "carrier_spacing_hz": (round(float(np.mean(np.diff(carriers))), 1)
                               if carriers.size > 1 else None),
        "band": band.describe(),
        "hw_f_max_hz": round(float(band.hw_f_max), 1),
        "sweep_hz": [round(band.sweep_lo, 1), round(band.sweep_hi, 1)],
        "csi_fs": round(float(probe.csi_fs), 3),
        "period_samples": int(sync["period_samples"]),
        "duration_s": round(float(times[-1]), 2),
        "sync": sync,
        "probe_db_min": round(float(probe_condition_db(probe, carriers).min()), 2),
        # crude level check: a clipped or near-silent capture is the most common
        # reason a recording "does nothing"
        "peak": round(float(np.max(np.abs(x))), 4),
        "rms": round(float(np.sqrt(np.mean(x**2))), 5),
        "clipped_samples": int(np.sum(np.abs(x) >= 0.999)),
        # handed to run_wav so a pruned carrier can be named by frequency;
        # popped before the meta reaches a caller.
        "_carriers": [round(float(f), 1) for f in carriers],
    }
    return amp, times, meta


# ---------------------------------------------------------------------------
# WAV -> decisions
# ---------------------------------------------------------------------------

def run_wav(
    path: str | Path,
    sensitivity: float = 1.0,
    fs: int | None = None,
    probe: Probe | None = None,
    carriers: np.ndarray | None = None,
    cfg: DetectorConfig | None = None,
    hw_f_max: float | None = None,
) -> dict:
    """Run a recorded WAV through the full detection chain."""
    amp, times, meta = load_csi(
        path, fs=fs, probe=probe, carriers=carriers, hw_f_max=hw_f_max
    )
    cfg = cfg or DetectorConfig(sensitivity=sensitivity)
    pipe = SASPipeline(cfg, fs=float(meta["csi_fs"]))

    frames: list[dict] = []
    for t, a in zip(times, amp):
        frame = pipe.step(float(t), a)
        if frame is not None:
            frames.append(frame)

    if pipe.carrier_mask is not None:
        meta["carriers_planned"] = int(pipe.carrier_mask.size)
        meta["carriers_used"] = int(pipe.carrier_mask.sum())
        meta["carriers_pruned"] = [
            round(float(f), 1)
            for f, ok in zip(meta.get("_carriers", []), pipe.carrier_mask)
            if not ok
        ]
        meta["prune_threshold"] = round(float(pipe.prune_threshold), 4)
    meta.pop("_carriers", None)

    return {
        "source": meta,
        "frames": frames,
        "events": pipe.detector.events,
        "metrics": summarise(frames, meta),
        "detector_config": cfg.to_dict(),
    }


def summarise(frames: list[dict], meta: dict | None = None) -> dict:
    """Metrics that do not need ground truth.

    A real cabin has no labelled BPM, so unlike ``evaluate_frames`` this reports
    what the detector decided and when -- not accuracy.
    """
    if not frames:
        return {"frames": 0, "detected": False}

    presence_frame = next((f for f in frames if f["state"]["presence"]), None)
    alarm_frame = next((f for f in frames if f["state"]["alarm"]), None)
    apnea_frame = next((f for f in frames if f["state"]["alarm_reason"] == "APNEA"), None)
    bpm_frame = next((f for f in frames if f["state"]["bpm"] is not None), None)

    bpms = [f["state"]["bpm"] for f in frames if f["state"]["bpm"] is not None]
    first_evidence = None if presence_frame is None else presence_frame["state"].get("first_evidence_t")
    alarm_at = None if alarm_frame is None else alarm_frame["t"]

    neff = []
    for f in frames:
        g = np.asarray(f["features"].get("gains") or [], dtype=float)
        if g.size == 0:
            continue
        w = np.maximum(g, 0.0) + MRC_WEIGHT_FLOOR
        total = w.sum()
        if total > 0:
            w = w / total
            neff.append(float(1.0 / np.sum(w ** 2)))

    out = {
        "frames": len(frames),
        "duration_s": frames[-1]["t"] if frames else 0.0,
        "detected": presence_frame is not None,
        "presence_at": None if presence_frame is None else presence_frame["t"],
        "confirmed_by": None if presence_frame is None else presence_frame["state"].get("breath_source"),
        "alarm_at": alarm_at,
        "alarm_reason": None if alarm_frame is None else alarm_frame["state"]["alarm_reason"],
        "response_time_s": None
        if alarm_at is None or first_evidence is None
        else round(alarm_at - first_evidence, 2),
        "apnea_escalated_at": None if apnea_frame is None else apnea_frame["t"],
        "breathing_at": None if bpm_frame is None else bpm_frame["t"],
        "bpm_median": None if not bpms else round(float(np.median(bpms)), 1),
        "bpm_range": None if not bpms else [round(float(min(bpms)), 1), round(float(max(bpms)), 1)],
        "bpm_samples": len(bpms),
        "motion_evidence_frames": sum(1 for f in frames if f["state"].get("motion_detected")),
        "interference_frames": sum(1 for f in frames if f["features"].get("interference")),
        "max_motion_ratio": round(
            float(max(f["state"].get("motion_ratio") or 0.0 for f in frames)), 2
        ),
        "mrc_neff": None if not neff else round(float(np.mean(neff)), 2),
    }
    if meta is not None:
        out["source"] = {
            "fs": meta.get("fs"),
            "csi_fs": meta.get("csi_fs"),
            "carriers": meta.get("carriers"),
            "duration_s": meta.get("duration_s"),
            "sync_filled": (meta.get("sync") or {}).get("filled"),
            "sync_holes": (meta.get("sync") or {}).get("holes"),
        }
    return out


# ---------------------------------------------------------------------------
# self-check: scenario -> WAV
# ---------------------------------------------------------------------------

def _default_seed(key: str) -> int:
    return 20231102 + zlib.crc32(key.encode("utf-8")) % 997


def render_wav(
    path: str | Path,
    key: str = "sleeping",
    duration_s: float | None = None,
    fs: int = AUDIO_FS,
    snr_db: float = 40.0,
    seed: int | None = None,
    hw_f_max: float | None = None,
) -> dict:
    """Render a simulated scenario into a WAV that a real front end can read.

    The probe is convolved with the continuous-frequency CIR of the scene and
    only the positive-frequency band is kept, which is exactly the analytic
    channel model the simulator uses for its 24 carriers. Feeding the result
    back through ``run_wav`` therefore tests the front end, not the scene.

    ``snr_db`` is in-band SNR relative to the rendered signal RMS. The
    simulator's own CSI noise is about 35 dB, so ~40 dB here is comparable.

    ``hw_f_max`` renders the capture as a band-limited driver would produce it:
    the probe itself is narrowed, so no energy is wasted above the ceiling. That
    is the *configured correctly* case. The *not configured* case -- a full-band
    probe behind a low-passing driver -- is a different failure mode and worth
    testing separately, because the dead carriers there inject noise rather than
    simply vanish; ``SASPipeline`` catches it at runtime by pruning.
    """
    scenario = get_scenario(key)
    band = plan_band(hw_f_max)
    ch = AcousticChannel(
        scenario,
        seed=seed if seed is not None else _default_seed(key),
        band=band,
    )
    probe = make_probe(fs=fs, hw_f_max=hw_f_max)
    L = probe.length
    spectrum = np.fft.fft(probe.samples)
    freqs = np.fft.fftfreq(L, 1.0 / fs)

    n = int(round((duration_s or scenario.duration_s) * CSI_FS))
    out = np.empty(n * L)
    truth: list[dict] = []
    for i in range(n):
        delays, gains, tr = ch.sounding_step(1.0 / CSI_FS)
        H = (
            gains[:, None] * np.exp(-1j * 2 * np.pi * freqs[None, :] * delays[:, None])
        ).sum(axis=0)
        # keep the positive band only and mirror it, so the waveform is real and
        # its CFR on f > 0 is H exactly (a complex path gain is not a real
        # signal, so re-symmetrising is what makes it playable)
        full = np.zeros(L, dtype=complex)
        full[: L // 2 + 1] = (spectrum * H)[: L // 2 + 1]
        full[L // 2 + 1:] = np.conj(full[L // 2 - 1: 0: -1])
        out[i * L: (i + 1) * L] = np.fft.ifft(full).real
        truth.append(tr)

    rng = np.random.default_rng(seed if seed is not None else _default_seed(key))
    if snr_db is not None and np.isfinite(snr_db):
        rms = float(np.sqrt(np.mean(out ** 2)))
        gain = np.repeat([t["noise_gain"] for t in truth], L)[: out.size]
        out = out + rng.standard_normal(out.size) * rms * 10 ** (-snr_db / 20.0) * gain

    peak = float(np.max(np.abs(out)))
    if peak > 0.98:
        out *= 0.98 / peak
    write_wav(path, out, fs=fs)

    return {
        "path": str(path),
        "scenario": key,
        "seconds": round(n / CSI_FS, 2),
        "fs": fs,
        "snr_db": snr_db,
        "peak": round(float(np.max(np.abs(out))), 3),
        "band": band.describe(),
        "carriers": band.n_carriers,
        "sweep_hz": [round(band.sweep_lo, 1), round(band.sweep_hi, 1)],
    }
