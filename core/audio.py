"""Real-audio front end: probe design, WAV I/O and CSI estimation.

The simulator hands the pipeline a complex CFR sampled on 24 carriers. A real
capture only gives a microphone waveform, so this module closes the gap:

    probe -> (speaker, cabin, mic) -> waveform -> frame sync -> CIR
          -> DFT at the 24 carrier frequencies -> |H(f)| -> pipeline.step()

Why the numbers line up: the probe repeats every 20 ms, and 48 000 Hz / 50 Hz
= 960 samples, so a 960-sample probe period produces exactly 50 CSI frames per
second -- the rate the SAS model is defined at. 960 is not a Kasami length
(those are 2^n - 1), so the default probe is a band-limited linear chirp; an
MLS probe is available when the user picks a matching period.

Note on frame alignment: because the probe is periodic, cutting an L-sample
window at any offset yields the same period cyclically rotated. A rotation of
delta samples multiplies H(f) by exp(-j 2 pi f delta / fs), i.e. it only adds a
phase ramp and leaves |H(f)| -- the only quantity the SAS model uses --
untouched. Frame sync therefore only has to be good enough to keep the window
inside one period, not sample-exact.
"""
from __future__ import annotations

import wave
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, find_peaks, resample_poly

from .config import (
    AUDIO_FS,
    CSI_FS,
    F_HIGH,
    F_LOW,
    N_SUBCARRIERS,
    PROBE_TAPER_S,
    SWEEP_HI,
    SWEEP_LO,
    AcousticBand,
    plan_band,
)

DEFAULT_PERIOD_S = 1.0 / CSI_FS


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

_MLS_TAPS = {5: [5, 3], 6: [6, 5], 7: [7, 6], 8: [8, 6, 5, 4],
             9: [9, 5], 10: [10, 7], 11: [11, 9], 12: [12, 11, 10, 4],
             13: [13, 12, 11, 8], 14: [14, 13, 12, 2], 15: [15, 14],
             16: [16, 15, 13, 4]}


@dataclass(frozen=True)
class Probe:
    """The periodic sounding signal that was (or will be) played."""

    samples: np.ndarray
    fs: int
    kind: str
    band: AcousticBand | None = None

    @property
    def length(self) -> int:
        return int(self.samples.size)

    @property
    def period_s(self) -> float:
        return self.length / self.fs

    @property
    def csi_fs(self) -> float:
        """CSI frame rate implied by this probe period."""
        return self.fs / self.length

    @property
    def energy(self) -> float:
        return float(np.dot(self.samples, self.samples))


def make_probe(
    fs: int = AUDIO_FS,
    period_s: float = DEFAULT_PERIOD_S,
    kind: str = "chirp",
    taper_s: float = PROBE_TAPER_S,
    hw_f_max: float | None = None,
) -> Probe:
    """Build the periodic probe.

    ``chirp`` (default) is a linear sweep covering the carrier band with margin
    on both sides. ``mls`` is a maximal-length sequence, the ingredient Kasami
    codes are built from -- only usable when fs * period_s happens to equal
    2^n - 1; it is not, for the 20 ms period this pipeline needs (960 samples at
    48 kHz), so a sweep is the only option that keeps the 50 Hz CSI rate.

    The sweep range matters twice over. A sweep that *starts and ends* on the
    carrier band puts the two taper ramps right on the outermost carriers,
    which drops their probe power by 23 dB and makes them 200x noisier than the
    rest; 700 Hz of overhang keeps the in-band spread under ~3 dB. And when the
    hardware ceiling is below the paper's 23.5 kHz, the sweep must stop there
    too -- sounding a band the driver cannot reproduce just spends period
    energy on frequencies nobody measures, which lowers the SNR of every
    carrier that is actually used.

    ``hw_f_max`` is the measured -3 dB ceiling of speaker + microphone. Pass it
    and the band shrinks to fit; leave it and the paper's 23.5 kHz is assumed.
    """
    band = plan_band(hw_f_max)
    length = int(round(fs * period_s))
    if length < 32:
        raise ValueError("probe period too short for this sample rate")

    if kind == "chirp":
        if band.sweep_hi >= fs / 2:
            raise ValueError(
                f"probe sweep reaches {band.sweep_hi:.0f} Hz but the Nyquist "
                f"limit at fs={fs} is {fs/2:.0f} Hz; record at 48 kHz or higher"
            )
        n = np.arange(length)
        u = n / max(1, length - 1)
        inst_f = band.sweep_lo + (band.sweep_hi - band.sweep_lo) * u
        s = np.sin(2 * np.pi * np.cumsum(inst_f) / fs)
    elif kind == "mls":
        order = int(round(np.log2(length + 1)))
        if 2 ** order - 1 != length or order not in _MLS_TAPS:
            raise ValueError(
                f"MLS probe needs a period of 2^n-1 samples; {length} is not. "
                f"Use a chirp, or pick period_s = {(2 ** 10 - 1) / fs:.6f} "
                f"(MLS order 10) and account for the resulting "
                f"{fs / (2 ** 10 - 1):.3f} Hz CSI rate."
            )
        taps = _MLS_TAPS[order]
        reg = np.ones(order, dtype=np.int8)
        out = np.empty(length, dtype=np.int8)
        for i in range(length):
            out[i] = reg[-1]
            fb = 0
            for t in taps:
                fb ^= reg[-t]
            reg[1:] = reg[:-1]
            reg[0] = fb
        s = out.astype(np.float64) * 2.0 - 1.0
    else:
        raise ValueError(f"unknown probe kind: {kind!r}")

    n_taper = max(1, int(round(taper_s * fs)))
    if n_taper * 2 < length:
        w = np.ones(length)
        edge = 0.5 - 0.5 * np.cos(np.pi * np.arange(n_taper) / n_taper)
        w[:n_taper] = edge
        w[-n_taper:] = edge[::-1]
        s = s * w

    peak = float(np.max(np.abs(s)))
    if peak > 0:
        s = s / peak
    return Probe(samples=s.astype(np.float64), fs=int(fs), kind=kind, band=band)


# ---------------------------------------------------------------------------
# WAV I/O (standard library only)
# ---------------------------------------------------------------------------

def read_wav(path: str | Path, target_fs: int | None = None) -> tuple[np.ndarray, int]:
    """Read a WAV file as mono float64 in [-1, 1].

    Supports 8/16/24/32-bit PCM. Float WAVs are rejected with a conversion
    hint because the stdlib ``wave`` module cannot parse them.
    """
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as w:
            fs = w.getframerate()
            nch = w.getnchannels()
            sw = w.getsampwidth()
            raw = w.readframes(w.getnframes())
    except wave.Error as exc:
        raise RuntimeError(
            f"{path.name}: unsupported WAV encoding ({exc}). "
            "Convert to 16-bit PCM first, e.g. "
            f"ffmpeg -i \"{path.name}\" -ac 1 -ar {AUDIO_FS} -c:a pcm_s16le out.wav"
        ) from exc

    if sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
        data = ints.astype(np.float64) / float(1 << 23)
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / float(1 << 31)
    else:
        raise RuntimeError(f"{path.name}: unsupported sample width {sw} bytes")

    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)

    if target_fs is not None and target_fs != fs:
        g = gcd(int(target_fs), int(fs))
        data = resample_poly(data, target_fs // g, fs // g)
        fs = int(target_fs)
    return np.ascontiguousarray(data), int(fs)


def write_wav(path: str | Path, samples: np.ndarray, fs: int = AUDIO_FS) -> Path:
    """Write mono 16-bit PCM, clipping to [-1, 1]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes(pcm.tobytes())
    return path


# ---------------------------------------------------------------------------
# frame sync
# ---------------------------------------------------------------------------

def matched_filter(x: np.ndarray, probe: Probe) -> np.ndarray:
    """Envelope of the probe correlation, one value per possible frame start."""
    return np.abs(fftconvolve(x, probe.samples[::-1], mode="valid"))


def match_quality(mf: np.ndarray) -> float:
    """Peak-to-RMS of the matched filter, the probe-detection statistic.

    Correlating ~960 samples of a real probe gives a peak about
    ``sqrt(length / correlation_width)`` above the floor -- measured at ~24.
    Correlating wideband noise gives a filtered noise process with no such gain,
    i.e. ~4. The gap is wide enough to use as a hard gate: it is what tells
    "the probe is in the recording" apart from "somebody recorded a cabin".
    """
    if mf.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(mf ** 2)))
    return float(np.max(mf)) / rms if rms > 0 else 0.0


def frame_starts(x: np.ndarray, probe: Probe, min_ratio: float = 0.2) -> np.ndarray:
    """Locate probe repetitions by matched filtering."""
    mf = matched_filter(x, probe)
    if mf.size == 0:
        return np.zeros(0, dtype=int)
    height = min_ratio * float(np.max(mf))
    peaks, _ = find_peaks(mf, height=height, distance=max(1, int(0.8 * probe.length)))
    return peaks.astype(int)


def frame_offsets(
    x: np.ndarray,
    probe: Probe,
    min_ratio: float = 0.2,
    max_fill: int = 4,
) -> tuple[np.ndarray, dict]:
    """Frame start samples, with single missed periods interpolated.

    A periodic probe means every frame can be cut independently, so a missed
    matched-filter peak is just a gap. Small gaps are filled on the measured
    period (which also absorbs playback/microphone clock drift, because the
    anchor stays on detected peaks); large gaps are left alone and reported,
    since time-stretching the amplitude series would corrupt the breathing lag.
    """
    mf = matched_filter(x, probe)
    found = frame_starts(x, probe, min_ratio)
    stats = {
        "match_ratio": round(match_quality(mf), 2),
        "detected_periods": int(found.size),
        "nominal_period": probe.length,
        "nominal_csi_fs": probe.csi_fs,
        "filled": 0,
        "holes": 0,
        "period_samples": probe.length,
        "csi_fs": probe.csi_fs,
        "coverage": 0.0,
        "longest_gap_s": 0.0,
    }
    if found.size < 2:
        return found, stats

    d = np.diff(found)
    med = float(np.median(d))
    near = d[np.abs(d - med) <= 0.12 * med]
    period = int(round(float(np.median(near)) if near.size else med))
    period = max(1, period)
    stats["period_samples"] = period
    stats["csi_fs"] = probe.fs / period

    starts = [int(found[0])]
    filled = 0
    holes = 0
    for a, b in zip(found[:-1], found[1:]):
        gap = int(b - a)
        n = int(round(gap / period))
        if 2 <= n <= max_fill:
            for k in range(1, n):
                starts.append(int(round(a + k * period)))
            filled += n - 1
        elif n > max_fill:
            holes += 1
            stats["longest_gap_s"] = max(stats["longest_gap_s"], gap / probe.fs)
        starts.append(int(b))

    starts = np.unique(np.asarray(starts, dtype=int))
    stats["filled"] = filled
    stats["holes"] = holes
    stats["coverage"] = float(
        min(1.0, starts.size / max(1.0, len(x) / period))
    )
    return starts, stats


# ---------------------------------------------------------------------------
# CSI estimation
# ---------------------------------------------------------------------------

def carrier_bins(
    probe: Probe,
    f_lo: float | None = None,
    f_hi: float | None = None,
    n: int | None = None,
) -> np.ndarray:
    """Carrier frequencies snapped onto the probe's own DFT grid.

    A real channel sounder can only measure H on its subcarrier grid. Asking for
    10 500 Hz when the grid is 50 Hz wide and the multipath fringes are ~150 Hz
    apart means interpolating a quantity that oscillates between samples -- that
    is a 10 % amplitude error even with zero noise. Snapping the carriers to
    bins makes the estimate exact.

    Defaults come from ``probe.band``, i.e. from the hardware ceiling the probe
    was built for, so a 16 kHz driver gets a 4 kHz span instead of asking for
    carriers it cannot reproduce.
    """
    band = probe.band or plan_band(None)
    f_lo = band.f_lo if f_lo is None else f_lo
    f_hi = band.f_hi if f_hi is None else f_hi
    n = band.n_carriers if n is None else n

    df = probe.fs / probe.length
    k0 = int(round(f_lo / df))
    k1 = int(round(f_hi / df))
    ks = np.unique(np.round(np.linspace(k0, k1, n)).astype(int))
    ks = ks[(ks > 0) & (ks < probe.length // 2)]
    if ks.size < 4:
        raise ValueError(
            f"carrier band {f_lo}-{f_hi} Hz holds only {ks.size} grid points at "
            f"{df:.1f} Hz spacing; lengthen the probe period"
        )
    return ks * df


def carrier_health(
    amplitudes: np.ndarray,
    rel_floor: float = 3.0,
    abs_floor: float = 0.15,
    baseline_percentile: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Which carriers are actually carrying signal, judged from a quiet window.

    A carrier that the speaker cannot reach, or the microphone cannot hear, has
    almost no probe power at its bin. Spectral division then divides noise by
    that near-zero, and the recovered ``|H|`` becomes a wideband random
    sequence -- measured at 0.20-0.48 relative jitter versus 0.013-0.029 for a
    healthy carrier. That sequence is not a small error on top of the signal,
    it *is* the signal at that frequency, so residual energy sees motion and
    the MRC weights see correlation.

    The baseline is a *low* percentile, not the median, and that matters: once
    half the band is dead the median has already moved into the dead carriers,
    so ``3 x median`` sits above them and prunes nothing. Measured at an 18 kHz
    ceiling, the median lands at 0.11 (threshold 0.34) while the 25th
    percentile stays at 0.028 (threshold 0.084) -- the difference between two
    carriers surviving and twelve.

    The test is relative on purpose: absolute jitter would flag every carrier
    at once whenever the cabin is loud, while a dead carrier stays an order of
    magnitude above the healthy ones. ``abs_floor`` guards the case where every
    carrier is dead, and if fewer than half would survive the function keeps
    them all and lets the caller fail loudly rather than running on three
    carriers.
    """
    A = np.asarray(amplitudes, dtype=float)
    if A.ndim != 2 or A.shape[0] < 8:
        n = A.shape[-1] if A.ndim == 2 else 0
        return np.ones(n, dtype=bool), np.zeros(n), 0.0
    mu = np.abs(A).mean(axis=0)
    sd = A.std(axis=0)
    rel = sd / np.maximum(mu, 1e-12)
    baseline = float(np.percentile(rel, baseline_percentile))
    threshold = max(abs_floor, rel_floor * baseline)
    alive = rel <= threshold
    if alive.sum() < max(4, A.shape[1] // 2):
        alive[:] = True
    return alive, rel, threshold


def probe_condition_db(probe: Probe, carriers: np.ndarray) -> np.ndarray:
    """Probe power at each carrier, in dB relative to the strongest carrier.

    Read this as the noise penalty the front end pays: a carrier at -20 dB is
    estimated 10x noisier than one at 0 dB. It is a *design* check, not a
    runtime check -- the value is fixed by the probe waveform.
    """
    spectrum = np.fft.fft(probe.samples)
    power = np.abs(spectrum) ** 2
    pos = (np.asarray(carriers, dtype=float) * probe.length / probe.fs) % probe.length
    bins = np.round(pos).astype(int) % probe.length
    p = power[bins]
    peak = float(p.max()) or 1.0
    return 10.0 * np.log10(np.maximum(p, peak * 1e-12) / peak)


def estimate_csi(
    x: np.ndarray,
    probe: Probe,
    carriers: np.ndarray | None = None,
    starts: np.ndarray | None = None,
    rel_floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the complex CFR on ``carriers`` for every frame.

    The probe repeats with period L and is longer than any cabin path, so the
    recording is the *cyclic* convolution of the probe with the cabin CIR:
    on the L-point DFT grid, ``R[k] = S[k] * H[k]``. Regularised spectral
    division therefore recovers H exactly at every bin the probe excites.

    A time-domain correlation estimate -- ``ifft(fft(block) * conj(fft(probe)))``
    -- is *not* equivalent, and that was the earlier bug: its CIR is smeared by
    the probe's autocorrelation main lobe (~4 samples at 13 kHz of bandwidth, an
    83 us boxcar), and sinc(83 us x 23.5 kHz) has already passed its first null,
    so the top carriers came out tens of percent low.

    If ``carriers`` does not land on the grid, H is evaluated from the estimated
    CIR instead (band-limited interpolation), which is exact for a sparse short
    cabin CIR but clearly worse than just picking grid frequencies.

    Returns ``(cfr, starts)`` where ``cfr`` has shape ``(n_frames, n_carriers)``.
    """
    L, fs = probe.length, probe.fs
    carriers = carrier_bins(probe) if carriers is None else np.asarray(carriers, dtype=float)
    if starts is None:
        starts, _ = frame_offsets(x, probe)

    spectrum = np.fft.fft(probe.samples)
    power = np.abs(spectrum) ** 2
    inverse = np.conj(spectrum) / (power + rel_floor * float(power.max()))

    pos = carriers * L / fs
    on_grid = bool(np.all(np.abs(pos - np.round(pos)) < 1e-6))
    if on_grid:
        pick = np.round(pos).astype(int) % L
        kernel = None
    else:
        pick = None
        kernel = np.exp(-1j * 2 * np.pi * np.outer(carriers, np.arange(L)) / fs)

    cfr = np.empty((len(starts), len(carriers)), dtype=np.complex128)
    used = 0
    for start in starts:
        block = x[start:start + L]
        if block.size < L:
            break
        hk = np.fft.fft(block) * inverse
        cfr[used] = hk[pick] if kernel is None else kernel @ np.fft.ifft(hk)
        used += 1
    return cfr[:used], np.asarray(starts)[:used]


def csi_amplitude(x: np.ndarray, probe: Probe, carriers: np.ndarray | None = None):
    """Convenience wrapper returning ``(|CFR|, frame_times, cfr)``."""
    cfr, starts = estimate_csi(x, probe, carriers)
    times = starts / float(probe.fs)
    return np.abs(cfr), times, cfr
