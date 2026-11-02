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
N_SUBCARRIERS = 24
F_LOW = 10_500               # Hz, 10 kHz high-pass passband used in the paper
F_HIGH = 23_500              # Hz

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
