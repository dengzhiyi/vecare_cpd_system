"""VeCare prototype configuration.

Values follow the SAS / VeCare pipeline described in the NSDI'23 paper.
The prototype uses a simulated acoustic front-end (one cabin speaker + one
microphone) at a 48 kHz sound sampling rate, compressed into 50 Hz CSI frames.

Two channels share the same CSI but measure different things:

* motion  -- 1 s window (paper §4.3 "e.g., 1s"), raw amplitude, no high-pass.
  The decision statistic is the detrended residual energy normalised by the
  calibrated background, gated by inter-subcarrier coherence so that
  uncorrelated ambient noise bursts cannot masquerade as a moving body.
* breathing -- 8 s window, per-subcarrier ACF combined by MRC with weights
  g(f) = rho(f, 1/Fs), as in Eq. (4) and Eq. (6).
"""
from dataclasses import dataclass, asdict

SOUND_SPEED = 343.0          # m/s, in-cabin air
AUDIO_FS = 48000             # Hz, commodity audio sampling rate
CSI_FS = 50                  # Hz, channel sounding rate (Kasami period = 20 ms)
DECISION_FS = 10             # Hz, CPD decision rate
N_SUBCARRIERS = 24           # ceiling, not a promise: narrow bands get fewer
# The low edge is fixed, not planned: every driver we care about is specified as
# a 10 kHz high-pass, so 10.5 kHz -- the paper's own lower edge -- is already
# 500 Hz of guard into the high-pass corner and stays put for every hardware
# ceiling. Only the top of the band tracks the hardware. If a future device
# raises its high-pass too, this constant is the one thing that has to change.
F_LOW = 10_500               # Hz, 10 kHz high-pass + 500 Hz guard, never planned
F_HIGH = 23_500              # Hz, design ceiling when the hardware allows it

# The probe sweeps deliberately wider than the carrier band, so the raised-cosine
# edges (which necessarily lose energy) sit outside it. With the sweep starting
# and ending at the carrier band itself, the two extreme carriers got -23 dB of
# probe power, i.e. 200x more noise than the others.
SWEEP_LO = 9_800             # Hz
SWEEP_HI = 23_900            # Hz, must stay below Nyquist for fs = 48 kHz
PROBE_TAPER_S = 0.0006       # 0.6 ms; 2 ms edges cost 23 dB at the band ends

# --- hardware band ----------------------------------------------------------
# Speaker and microphone response is never flat to 23.5 kHz. A driver spec of
# "18 kHz" means -3 dB there and a steep rolloff above it, and a carrier that
# sits on that rolloff has almost no probe power: the spectral-division
# estimate degrades into noise/noise. Those carriers do not merely lose
# accuracy, they inject a wideband random amplitude sequence that the residual
# energy and the MRC weights cannot tell apart from a moving body -- measured
# on an empty cabin, they fired a false presence at 17.6 s.
#
# So the band has to be planned from the hardware, not from the paper:
HW_F_MAX = None              # measured -3 dB ceiling; None = assume F_HIGH
HW_GUARD_HZ = 1_500          # keep carriers this far below the hardware ceiling
CARRIER_MIN_SPACING_HZ = 250  # frequency-diversity floor, see plan_band docstring
SWEEP_MARGIN_HZ = 700        # sweep overhang past the carrier band on each side


@dataclass(frozen=True)
class AcousticBand:
    """A carrier set plus the sweep that excites it, for one hardware ceiling."""
    f_lo: float
    f_hi: float
    n_carriers: int
    sweep_lo: float
    sweep_hi: float
    hw_f_max: float

    @property
    def span_hz(self) -> float:
        return self.f_hi - self.f_lo

    @property
    def spacing_hz(self) -> float:
        return self.span_hz / max(1, self.n_carriers - 1)

    @property
    def diversity(self) -> float:
        """``spacing x longest path delay``: fringes rotated between neighbours.

        Below ~1 the neighbouring carriers see nearly the same phase and stop
        being independent observations, which costs MRC diversity and makes the
        inter-subcarrier coherence test optimistic.
        """
        return self.spacing_hz * 9.2e-3

    def describe(self) -> str:
        return (
            f"{self.n_carriers} carriers @ {self.f_lo/1e3:.1f}-{self.f_hi/1e3:.1f} kHz "
            f"({self.spacing_hz:.0f} Hz spacing, diversity {self.diversity:.2f}), "
            f"sweep {self.sweep_lo/1e3:.1f}-{self.sweep_hi/1e3:.1f} kHz"
        )


def plan_band(hw_f_max: float | None = None) -> AcousticBand:
    """Derive the carrier set from the hardware's usable bandwidth.

    Three constraints fight each other when the top of the band disappears:

    1. **Stay below the rolloff.** Carriers within ``HW_GUARD_HZ`` of the
       hardware ceiling lose probe power and turn into noise meters.
    2. **Keep enough spacing.** Frequency diversity needs
       ``spacing x delay_spread >~ 1``; with a 9.2 ms cabin spread that is
       roughly 110 Hz, and 250 Hz leaves a comfortable margin.
    3. **Keep enough carriers.** MRC combines them, and the coherence test
       needs a population to estimate a rank-1 share from.

    With 18 kHz you can still hold 24 carriers (6 kHz span at 261 Hz). With
    16 kHz the span is 4 kHz, so the count drops to 17 at 250 Hz -- trading
    diversity order for spacing is the right side of that trade, because 17
    carriers still average down the noise while 24 carriers at 174 Hz would
    start sharing fringes.

    ``hw_f_max=None`` means "no measured limit": the design band is used as-is,
    with no guard, because there is no rolloff to stay away from.

    Only the **top** of the band is planned; ``f_lo`` is always ``F_LOW``. A
    10 kHz high-pass corner is already covered by the 500 Hz between it and
    ``F_LOW``, so lowering the ceiling shrinks the band from above and never
    moves the low edge. Keeping the low edge fixed matters for more than
    tidiness: the breathing channel's per-carrier ACF gains are calibrated
    against a fixed noise floor, and sliding the lower carriers would change
    which physical paths the band averages over between two runs.

    Measured: widening the band back by shrinking the guard buys nothing. At a
    16 kHz ceiling, guards of 1500 / 800 / 400 Hz gave sleeping-scenario BPM
    MAE of 0.249 / 0.268 / 0.290 and motion-evidence frames of 0 / 1 / 2 --
    more carriers, slightly worse BPM, and the first still-cabin motion
    misfires. The residual limit is the coherence time of the breathing signal,
    not the carrier count, so the conservative guard stays.
    """
    if hw_f_max:
        f_hi = min(float(F_HIGH), float(hw_f_max) - HW_GUARD_HZ)
        ceiling = float(hw_f_max)
    else:
        f_hi = float(F_HIGH)
        ceiling = float(F_HIGH)
    if f_hi <= F_LOW + CARRIER_MIN_SPACING_HZ * 3:
        raise ValueError(
            f"hardware ceiling {ceiling/1e3:.1f} kHz leaves only "
            f"{f_hi/1e3:.1f} kHz of usable band above {F_LOW/1e3:.1f} kHz; "
            f"need at least 4 carriers at {CARRIER_MIN_SPACING_HZ:.0f} Hz spacing. "
            f"Lower F_LOW or use a wider driver."
        )
    span = f_hi - F_LOW
    n = min(N_SUBCARRIERS, int(span // CARRIER_MIN_SPACING_HZ) + 1)
    return AcousticBand(
        f_lo=float(F_LOW),
        f_hi=float(f_hi),
        n_carriers=int(n),
        sweep_lo=max(1_000.0, F_LOW - SWEEP_MARGIN_HZ),
        sweep_hi=min(AUDIO_FS / 2 - 100.0, f_hi + SWEEP_MARGIN_HZ),
        hw_f_max=ceiling,
    )

# --- motion channel ---------------------------------------------------------
MOTION_WINDOW_S = 1.0        # paper §4.3: short window for responsiveness
MOTION_LAG_S = 1.0 / CSI_FS  # Eq. (4): g(f) = rho(f, tau = 1/Fs) = 20 ms
MOTION_COHERENCE_WINDOW_S = 2.0  # longer window stabilises the coherence estimate
MOTION_ENERGY_RATIO = 1.60   # residual energy / calibrated background
MOTION_COHERENCE_MIN = 0.115  # first-singular-value share across subcarriers
MOTION_RATIO_FLOOR = 1.25    # floor applied at sensitivity = 1.0

# --- breathing channel ------------------------------------------------------
BREATH_WINDOW_S = 8.0
BREATH_LAG_MIN_S = 0.80      # 75 BPM, headroom above the newborn band
BREATH_LAG_MAX_S = 6.00      # 10 BPM
BREATH_MAX_OVERLAP = 0.75    # unbiased ACF dies below ~25 % overlap
BREATH_MIN_HEIGHT = 0.10
BREATH_MIN_PROMINENCE = 0.05
BREATH_CONF_THRESHOLD = 0.50

# --- decision ---------------------------------------------------------------
EVIDENCE_WINDOW_S = 5.0
PRESENCE_RATIO = 0.30
RESPONSE_BUDGET_S = 10.0     # total CPD response budget, counted from the
                             # first evidence sample (not from confirmation)
APNEA_TIMEOUT_S = 15.0
CLEAR_TIMEOUT_S = 25.0
CALIBRATION_S = 5.0
EVIDENCE_GAP_RESET_S = 5.0   # a new evidence episode restarts the budget
MRC_WEIGHT_FLOOR = 0.02      # keeps zero/negative gains from killing a subcarrier
INTERFERENCE_DEPTH = 0.35    # V-shaped ACF depth for impulse-noise rejection


@dataclass
class DetectorConfig:
    motion_window_s: float = MOTION_WINDOW_S
    motion_lag_s: float = MOTION_LAG_S
    motion_coherence_window_s: float = MOTION_COHERENCE_WINDOW_S
    motion_energy_ratio: float = MOTION_ENERGY_RATIO
    motion_coherence_min: float = MOTION_COHERENCE_MIN
    breath_window_s: float = BREATH_WINDOW_S
    breath_lag_min_s: float = BREATH_LAG_MIN_S
    breath_lag_max_s: float = BREATH_LAG_MAX_S
    breath_max_overlap: float = BREATH_MAX_OVERLAP
    breath_min_height: float = BREATH_MIN_HEIGHT
    breath_min_prominence: float = BREATH_MIN_PROMINENCE
    breath_conf_threshold: float = BREATH_CONF_THRESHOLD
    evidence_window_s: float = EVIDENCE_WINDOW_S
    presence_ratio: float = PRESENCE_RATIO
    response_budget_s: float = RESPONSE_BUDGET_S
    apnea_timeout_s: float = APNEA_TIMEOUT_S
    clear_timeout_s: float = CLEAR_TIMEOUT_S
    calibration_s: float = CALIBRATION_S
    evidence_gap_reset_s: float = EVIDENCE_GAP_RESET_S
    sensitivity: float = 1.0

    def motion_ratio_floor(self) -> float:
        # sensitivity > 1 raises sensitivity (lowers the ratio threshold)
        return max(MOTION_RATIO_FLOOR - 0.1, self.motion_energy_ratio / max(0.3, self.sensitivity))

    def to_dict(self) -> dict:
        return asdict(self)
