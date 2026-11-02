"""Cabin acoustic multipath simulator.

Implements the signal model of Eq. (1) in the VeCare paper:

    H(f, t) = sum_r a_r(t) exp(-j 2 pi f tau_r(t)) + N(f, t)

A static cabin multipath profile (direct path, seats, windows, roof) is mixed
with a dynamic reflection from a breathing / moving infant.  The weak body
reflection interferes with the static paths, producing the frequency-selective
sensitivity exploited by the SAS model.  Noise bursts and cabin vibrations let
us test false-alarm rejection.
"""
from __future__ import annotations

import numpy as np

from .config import CSI_FS, F_HIGH, F_LOW, N_SUBCARRIERS, SOUND_SPEED, AcousticBand, plan_band
from .scenarios import Scenario, TargetSpec


class AcousticChannel:
    def __init__(
        self,
        scenario: Scenario,
        seed: int = 20231102,
        band: AcousticBand | None = None,
    ):
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        # Carrier set follows the hardware ceiling, exactly as the audio front
        # end does -- otherwise the simulated ground truth would be measured on
        # a band the real driver cannot excite.
        self.band = band or plan_band(None)
        self.carriers = np.linspace(
            self.band.f_lo, self.band.f_hi, self.band.n_carriers
        )
        self.t = 0.0
        self._phase = self.rng.uniform(0, 2 * np.pi)
        self._last_breath_t = 0.0
        self._present_since = None

        # Static cabin multipath. Delays are round-trip path delays in seconds.
        self.static_delays = np.array(
            [0.00058, 0.0017, 0.0028, 0.0039, 0.0051, 0.0064, 0.0079, 0.0092]
        )
        base_gain = np.array([1.0, 0.58, 0.44, 0.35, 0.30, 0.24, 0.19, 0.15])
        self.static_gains = base_gain * np.exp(
            1j * self.rng.uniform(-np.pi, np.pi, len(base_gain))
        )
        self._target = scenario.target
        self._vib_phase = self.rng.uniform(0, 2 * np.pi, len(base_gain))

    def _target_visible(self, t: float) -> bool:
        tg = self._target
        if tg is None:
            return False
        return t >= tg.enter_t and (tg.exit_t is None or t < tg.exit_t)

    def _breathing_active(self, t: float) -> bool:
        tg = self._target
        if tg is None:
            return False
        if not tg.breathing_windows:
            return True
        return any(a <= t < b for a, b in tg.breathing_windows)

    def _motion_offset(self, t: float) -> np.ndarray:
        tg = self._target
        if tg is None or not self._target_visible(t):
            return 0.0

        offset = 0.0
        for ev in tg.motion_events:
            start, dur = ev["t"], ev["duration"]
            if start <= t < start + dur:
                u = (t - start) / dur
                offset += ev["amplitude_m"] * 0.5 * (1 - np.cos(2 * np.pi * u))

        if tg.awake_jitter_mm > 0:
            # Slow postural drift of an awake body (<0.15 Hz). It must stay
            # well below the 10-66 BPM breathing band: a coherent 0.2-0.4 Hz
            # wobble is a second breathing rhythm and makes the breathing rate
            # genuinely ambiguous, which is a modelling artefact rather than a
            # sensing challenge.
            j = tg.awake_jitter_mm / 1000.0
            offset += j * (0.6 * np.sin(2 * np.pi * 0.06 * t + 0.4)
                           + 0.4 * np.sin(2 * np.pi * 0.11 * t + 1.7))
        return offset

    def _noise_sigma(self, t: float) -> float:
        sigma = self.scenario.noise_sigma
        for b in self.scenario.noise_bursts:
            if b["t"] <= t < b["t"] + b["duration"]:
                u = (t - b["t"]) / b["duration"]
                shape = np.sin(np.pi * u) ** 2
                sigma += self.scenario.noise_sigma * (b["multiplier"] - 1) * shape
        return sigma

    def _paths(self, t: float, dt: float, advance: bool = True) -> dict:
        """Resolve the delay/gain profile of one 20 ms frame.

        Shared by ``step`` (which advances the breathing phase) and
        ``impulse_response`` (which must not disturb the timeline).
        """
        tg = self._target
        present = self._target_visible(t)

        breath_amp_m = 0.0
        breathing = False
        if present and tg is not None:
            if self._breathing_active(t):
                if advance:
                    drifting_bpm = tg.bpm * (1 + 0.02 * np.sin(2 * np.pi * 0.013 * t))
                    self._phase += 2 * np.pi * drifting_bpm / 60.0 * dt
                    self._last_breath_t = t
                # chest displacement (peak), decays smoothly if breathing stops
                breath_amp_m = tg.chest_mm / 1000.0 * np.sin(self._phase)
                breathing = True

        motion = self._motion_offset(t)

        # Target reflection: weak, range dependent. Ramp in after entry.
        a_target = 0.0
        if present and tg is not None:
            age = t - tg.enter_t
            ramp = min(1.0, age / 0.25)
            a_target = tg.reflectivity * ramp * (1.4 / max(1.0, tg.distance_m))

        # Round trip path length. Breathing and body motion modulate the delay.
        d = 0.0
        if tg is not None:
            d = tg.distance_m + motion + breath_amp_m
        tau_target = 2 * d / SOUND_SPEED if present else 0.0

        # Cabin vibration during slam / passing-car events: differential
        # path-length jitter decorrelates the static channel.
        delays = self.static_delays.copy()
        vib_amp = 0.0
        for b in self.scenario.vibration_bursts:
            if b["t"] <= t < b["t"] + b["duration"]:
                u = (t - b["t"]) / b["duration"]
                vib_amp = b["amplitude_mm"] / 1000.0 * np.sin(np.pi * u) ** 2
        if vib_amp > 0:
            # vib_amp is a *displacement* in metres; delays are in seconds, so
            # it has to be divided by the speed of sound. Adding metres to
            # seconds inflated a 0.05 mm panel vibration into 50 us of delay
            # jitter (17 mm of apparent motion) and made every door slam look
            # like a moving body.
            delays = delays + (vib_amp / SOUND_SPEED) * np.sin(
                2 * np.pi * 120.0 * t + self._vib_phase
            )

        return {
            "present": present,
            "breathing": breathing,
            "breath_amp_m": breath_amp_m,
            "motion": motion,
            "a_target": a_target,
            "tau_target": tau_target,
            "delays": delays,
            "vib_amp": vib_amp,
        }

    def sounding_step(self, dt: float) -> tuple[np.ndarray, np.ndarray, dict]:
        """Advance the scene and return the frame's (delays, gains, truth).

        Unlike ``step`` this does not evaluate the 24-carrier CSI nor draw noise:
        the audio renderer needs the continuous-frequency CIR, so that the probe
        can be convolved with it at every audio bin.
        """
        t = self.t
        p = self._paths(t, dt=dt, advance=True)
        delays = list(p["delays"])
        gains = list(self.static_gains)
        if p["present"] and p["a_target"] > 0:
            delays.append(p["tau_target"])
            gains.append(complex(p["a_target"]))

        tg = self._target
        truth = {
            "present": bool(p["present"]),
            "breathing": bool(p["breathing"]),
            "bpm": float(tg.bpm) if tg else None,
            "distance_m": float(tg.distance_m + p["motion"] + p["breath_amp_m"])
            if tg
            else None,
            "motion_offset_m": float(p["motion"]),
            "target_amplitude": float(p["a_target"]),
            "vibration": float(p["vib_amp"]),
            "noise_sigma": float(self.scenario.noise_sigma),
            "noise_gain": float(self._noise_sigma(t) / max(1e-9, self.scenario.noise_sigma)),
        }
        self.t += dt
        return np.asarray(delays, dtype=float), np.asarray(gains, dtype=complex), truth

    def step(self, dt: float) -> tuple[np.ndarray, np.ndarray, dict]:
        """Advance the scene by dt seconds and return (CSI, amplitude, truth)."""
        t = self.t
        tg = self._target
        p = self._paths(t, dt=dt, advance=True)
        present = p["present"]
        breathing = p["breathing"]
        breath_amp_m = p["breath_amp_m"]
        motion = p["motion"]
        a_target = p["a_target"]
        tau_target = p["tau_target"]
        delays = p["delays"]
        vib_amp = p["vib_amp"]

        # Common phase offset models speaker/mic asynchronization; it cancels
        # when only CSI amplitude is used.
        common_phase = self.rng.uniform(-np.pi, np.pi)
        static = self.static_gains[:, None] * np.exp(
            -1j * 2 * np.pi * self.carriers[None, :] * delays[:, None]
        )
        h = static.sum(axis=0) * np.exp(1j * common_phase)

        if present and a_target > 0:
            h = h + a_target * np.exp(
                -1j * 2 * np.pi * self.carriers * tau_target
            ) * np.exp(1j * common_phase)

        sigma = self._noise_sigma(t)
        n_carriers = self.carriers.size
        noise = sigma / np.sqrt(2) * (
            self.rng.standard_normal(n_carriers)
            + 1j * self.rng.standard_normal(n_carriers)
        )
        h = h + noise

        true_distance = float(tg.distance_m + motion + breath_amp_m) if tg else None
        truth = {
            "present": bool(present),
            "breathing": bool(breathing),
            "bpm": float(tg.bpm) if tg else None,
            "distance_m": true_distance,
            "motion_offset_m": float(motion),
            "noise_sigma": float(sigma),
            "target_amplitude": float(a_target),
            "vibration": float(vib_amp),
        }
        self.t += dt
        return h, np.abs(h), truth
