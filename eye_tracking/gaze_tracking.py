"""Nose-primary eye cursor and blink detection logic."""

from __future__ import annotations

from statistics import mean

import config
from utils import CooldownTimer, ExponentialSmoother, clamp, distance_2d, mean_point


class GazeTracker:
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    LEFT_H = (33, 133)
    RIGHT_H = (362, 263)
    LEFT_V = (159, 145)
    RIGHT_V = (386, 374)

    LEFT_EAR = [33, 160, 158, 133, 153, 144]
    RIGHT_EAR = [362, 385, 387, 263, 373, 380]

    def __init__(self, calibration) -> None:
        self.calibration = calibration
        self.blink_counter = 0
        self.blink_cooldown = CooldownTimer(config.BLINK_COOLDOWN_SECONDS)
        self.smoother = ExponentialSmoother(config.EYE_NOSE_SMOOTHING_ALPHA)

    @staticmethod
    def _ratio(value: float, start: float, end: float) -> float:
        low = min(start, end)
        high = max(start, end)
        span = high - low
        if span < 1e-9:
            return 0.5
        return clamp((value - low) / span, 0.0, 1.0)

    def _ear(self, landmarks, ids) -> float:
        p1 = landmarks[ids[0]][:2]
        p2 = landmarks[ids[1]][:2]
        p3 = landmarks[ids[2]][:2]
        p4 = landmarks[ids[3]][:2]
        p5 = landmarks[ids[4]][:2]
        p6 = landmarks[ids[5]][:2]
        denom = 2.0 * distance_2d(p1, p4)
        if denom < 1e-9:
            return 0.0
        return (distance_2d(p2, p6) + distance_2d(p3, p5)) / denom

    def _estimate_gaze_norm(self, landmarks):
        """Estimate a normalized gaze point; return None when iris points are unavailable."""
        try:
            left_iris = mean_point([landmarks[idx][:2] for idx in self.LEFT_IRIS])
            right_iris = mean_point([landmarks[idx][:2] for idx in self.RIGHT_IRIS])

            left_h = self._ratio(
                left_iris[0], landmarks[self.LEFT_H[0]][0], landmarks[self.LEFT_H[1]][0]
            )
            right_h = self._ratio(
                right_iris[0], landmarks[self.RIGHT_H[0]][0], landmarks[self.RIGHT_H[1]][0]
            )

            left_v = self._ratio(
                left_iris[1], landmarks[self.LEFT_V[0]][1], landmarks[self.LEFT_V[1]][1]
            )
            right_v = self._ratio(
                right_iris[1], landmarks[self.RIGHT_V[0]][1], landmarks[self.RIGHT_V[1]][1]
            )
        except (IndexError, TypeError):
            return None

        raw_h = mean([left_h, right_h])
        raw_v = mean([left_v, right_v])
        return clamp(raw_h, 0.0, 1.0), clamp(raw_v, 0.0, 1.0)

    def estimate(self, face):
        landmarks = face["landmarks_norm"]

        # Primary cursor anchor: nose tip.
        nose_x_raw = float(landmarks[config.EYE_NOSE_LANDMARK_ID][0])
        nose_y_raw = float(landmarks[config.EYE_NOSE_LANDMARK_ID][1])

        # C-key stores this same calibration key from main.py; reuse it as neutral nose center.
        center_x, center_y = self.calibration.get("gaze_center")
        deadzone = float(self.calibration.get("nose_deadzone") or config.EYE_NOSE_DEADZONE)

        nose_x = center_x if abs(nose_x_raw - center_x) < deadzone else nose_x_raw
        nose_y = center_y if abs(nose_y_raw - center_y) < deadzone else nose_y_raw

        # Optional micro-adjust from gaze with low influence.
        gaze_norm = self._estimate_gaze_norm(landmarks)
        nose_weight = float(self.calibration.get("nose_weight") or config.EYE_NOSE_WEIGHT)
        gaze_weight = float(self.calibration.get("gaze_micro_weight") or config.EYE_GAZE_MICRO_WEIGHT)
        nose_weight = clamp(nose_weight, 0.0, 1.0)
        gaze_weight = clamp(gaze_weight, 0.0, 1.0)

        blend_total = nose_weight + gaze_weight
        if blend_total <= 1e-9:
            nose_weight, gaze_weight = 1.0, 0.0
        else:
            nose_weight /= blend_total
            gaze_weight /= blend_total

        target_x = nose_x
        target_y = nose_y
        gaze_offset = None
        if config.ENABLE_EYE_GAZE_MICRO_ADJUST and gaze_norm is not None:
            gaze_offset_x = gaze_norm[0] - center_x
            gaze_offset_y = gaze_norm[1] - center_y
            gaze_offset = (gaze_offset_x, gaze_offset_y)

            # -------- CONTROL REGION (faster cursor movement) --------
            # Horizontal movement (slightly larger range)
            min_x, max_x = 0.45, 0.55   # left/right

            # Vertical movement (much larger range)
            min_y, max_y = 0.49, 0.51   # up/down

            # Normalize nose position inside smaller region
            norm_x = (nose_x - min_x) / (max_x - min_x)
            norm_y = (nose_y - min_y) / (max_y - min_y)

            norm_x = clamp(norm_x, 0.0, 1.0)
            norm_y = clamp(norm_y, 0.0, 1.0)

            # Hybrid: nose primary, gaze micro-adjust
            target_x = (norm_x * nose_weight) + ((norm_x + gaze_offset_x) * gaze_weight)
            target_y = (norm_y * nose_weight) + ((norm_y + gaze_offset_y) * gaze_weight)

            target_x = clamp(target_x, 0.0, 1.0)
            target_y = clamp(target_y, 0.0, 1.0)

        # Adaptive smoothing: smooth = prev*0.7 + new*0.3 (alpha=0.3)
        smooth_alpha = float(
            self.calibration.get("nose_smoothing_alpha") or config.EYE_NOSE_SMOOTHING_ALPHA
        )
        smooth_alpha = clamp(smooth_alpha, 0.01, 1.0)
        norm_x, norm_y = self.smoother.update(target_x, target_y, alpha=smooth_alpha)

        ear_left = self._ear(landmarks, self.LEFT_EAR)
        ear_right = self._ear(landmarks, self.RIGHT_EAR)
        ear = (ear_left + ear_right) * 0.5

        blink_click = False
        blink_threshold = self.calibration.get("blink_ear_threshold")
        if ear < blink_threshold:
            self.blink_counter += 1
            if self.blink_counter >= config.BLINK_CONSEC_FRAMES and self.blink_cooldown.ready():
                blink_click = True
                self.blink_counter = 0
                self.blink_cooldown.trigger()
        else:
            self.blink_counter = 0

        label = "Blink Click" if blink_click else "Nose Move"
        return {
            "cursor_norm": (clamp(norm_x, 0.0, 1.0), clamp(norm_y, 0.0, 1.0)),
            "blink_click": blink_click,
            "ear": ear,
            # Keep this key for main.py C-key calibration compatibility.
            "raw_gaze": (nose_x_raw, nose_y_raw),
            "gesture_label": label,
            "nose_norm": (nose_x_raw, nose_y_raw),
            "cursor_target_norm": (target_x, target_y),
            "gaze_hint": gaze_norm,
            "gaze_offset": gaze_offset,
        }

    def training_metrics(self, face):
        landmarks = face["landmarks_norm"]
        ear_left = self._ear(landmarks, self.LEFT_EAR)
        ear_right = self._ear(landmarks, self.RIGHT_EAR)
        return {"ear": (ear_left + ear_right) * 0.5}
