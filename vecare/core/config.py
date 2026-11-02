"""VeCare prototype configuration.

Values follow the SAS / VeCare pipeline described in the NSDI'23 paper.
The prototype uses a simulated acoustic front-end (one cabin speaker + one
microphone) at a 48 kHz sound sampling rate, compressed into 50 Hz CSI frames.
"""
from dataclasses import dataclass, asdict

SOUND_SPEED = 343.0          # m/s, in-cabin air
AUDIO_FS = 48000             # Hz, commodity audio sampling rate
CSI_FS = 50                  # Hz, channel sounding rate (Kasami period = 20 ms)
DECISION_FS = 10             # Hz, CPD decision rate
N_SUBCARRIERS = 24
F_LOW = 10_500               # Hz, 10 kHz high-pass passband used in the paper
F_HIGH = 23_500              # Hz

MOTION_WINDOW_S = 2.0
MOTION_LAG_S = 0.10
BREATH_WINDOW_S = 8.0
BREATH_LAG_MIN_S = 1.0       # 60 BPM
BREATH_LAG_MAX_S = 5.0       # 12 BPM (8 s window limits reliable lags)
BREATH_MIN_HEIGHT = 0.16
BREATH_MIN_PROMINENCE = 0.09
BREATH_CONF_THRESHOLD = 0.50

EVIDENCE_WINDOW_S = 5.0
PRESENCE_RATIO = 0.30
ALARM_DELAY_S = 10.0         # CPD response requirement from the paper
APNEA_TIMEOUT_S = 15.0
CLEAR_TIMEOUT_S = 25.0
CALIBRATION_S = 5.0

DEFAULT_MIN_MOTION_THRESHOLD = 0.15
MOTION_HIGHPASS_HZ = 1.0
MRC_WEIGHT_FLOOR = 0.35


@dataclass
class DetectorConfig:
    motion_window_s: float = MOTION_WINDOW_S
    motion_lag_s: float = MOTION_LAG_S
    breath_window_s: float = BREATH_WINDOW_S
    breath_lag_min_s: float = BREATH_LAG_MIN_S
    breath_lag_max_s: float = BREATH_LAG_MAX_S
    breath_min_height: float = BREATH_MIN_HEIGHT
    breath_min_prominence: float = BREATH_MIN_PROMINENCE
    breath_conf_threshold: float = BREATH_CONF_THRESHOLD
    evidence_window_s: float = EVIDENCE_WINDOW_S
    presence_ratio: float = PRESENCE_RATIO
    alarm_delay_s: float = ALARM_DELAY_S
    apnea_timeout_s: float = APNEA_TIMEOUT_S
    clear_timeout_s: float = CLEAR_TIMEOUT_S
    calibration_s: float = CALIBRATION_S
    min_motion_threshold: float = DEFAULT_MIN_MOTION_THRESHOLD
    sensitivity: float = 1.0

    def motion_threshold_floor(self) -> float:
        # sensitivity > 1 lowers the threshold (more sensitive)
        return max(0.08, self.min_motion_threshold / max(0.3, self.sensitivity))

    def to_dict(self) -> dict:
        return asdict(self)
