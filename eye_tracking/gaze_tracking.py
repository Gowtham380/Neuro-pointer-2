"""Nose-primary eye cursor and face-action detection logic."""

from __future__ import annotations

import time
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
        self.blink_frames = 0
        self.left_wink_frames = 0
        self.right_wink_frames = 0
        self.scroll_frames = 0
        self.drag_active = False
        self.scroll_residual = 0.0
        self.prev_scroll_y = None
        self.last_blink_time = 0.0
        self.left_wink_latched = False
        self.right_wink_latched = False
        self.click_cooldown = CooldownTimer(config.BLINK_COOLDOWN_SECONDS)
        self.right_click_cooldown = CooldownTimer(config.ACTION_COOLDOWN_SECONDS)
        self.mode_toggle_cooldown = CooldownTimer(config.EYE_MODE_TOGGLE_COOLDOWN_SECONDS)
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

        nose_x_raw = float(landmarks[config.EYE_NOSE_LANDMARK_ID][0])
        nose_y_raw = float(landmarks[config.EYE_NOSE_LANDMARK_ID][1])
        center_x, center_y = self.calibration.get("gaze_center")
        deadzone = float(self.calibration.get("nose_deadzone") or config.EYE_NOSE_DEADZONE)

        nose_x = center_x if abs(nose_x_raw - center_x) < deadzone else nose_x_raw
        nose_y = center_y if abs(nose_y_raw - center_y) < deadzone else nose_y_raw

        min_x, max_x = 0.45, 0.55
        min_y, max_y = 0.49, 0.51
        norm_x = clamp((nose_x - min_x) / max(max_x - min_x, 1e-6), 0.0, 1.0)
        norm_y = clamp((nose_y - min_y) / max(max_y - min_y, 1e-6), 0.0, 1.0)

        gaze_norm = self._estimate_gaze_norm(landmarks)
        nose_weight = float(self.calibration.get("nose_weight") or config.EYE_NOSE_WEIGHT)
        gaze_weight = float(self.calibration.get("gaze_micro_weight") or config.EYE_GAZE_MICRO_WEIGHT)
        nose_weight = clamp(nose_weight, 0.0, 1.0)
        gaze_weight = clamp(gaze_weight, 0.0, 1.0)

        target_x = norm_x
        target_y = norm_y
        if config.ENABLE_EYE_GAZE_MICRO_ADJUST and gaze_norm is not None:
            gaze_offset_x = gaze_norm[0] - center_x
            gaze_offset_y = gaze_norm[1] - center_y
            target_x = (norm_x * nose_weight) + ((norm_x + gaze_offset_x) * gaze_weight)
            target_y = (norm_y * nose_weight) + ((norm_y + gaze_offset_y) * gaze_weight)
            target_x = clamp(target_x, 0.0, 1.0)
            target_y = clamp(target_y, 0.0, 1.0)

        smooth_alpha = float(
            self.calibration.get("nose_smoothing_alpha") or config.EYE_NOSE_SMOOTHING_ALPHA
        )
        smooth_alpha = clamp(smooth_alpha, 0.01, 1.0)
        norm_x, norm_y = self.smoother.update(target_x, target_y, alpha=smooth_alpha)

        ear_left = self._ear(landmarks, self.LEFT_EAR)
        ear_right = self._ear(landmarks, self.RIGHT_EAR)
        ear = (ear_left + ear_right) * 0.5

        blink_threshold = self.calibration.get("blink_ear_threshold")
        open_threshold = blink_threshold * config.EYE_OPEN_RATIO
        squint_threshold = blink_threshold * config.EYE_SQUINT_RATIO

        both_closed = ear_left < blink_threshold and ear_right < blink_threshold
        left_wink = ear_left < blink_threshold and ear_right > open_threshold
        right_wink = ear_right < blink_threshold and ear_left > open_threshold
        scroll_pose = (
            not both_closed
            and not left_wink
            and not right_wink
            and ear_left < squint_threshold
            and ear_right < squint_threshold
        )

        blink_click = False
        double_click = False
        right_click = False
        drag_toggled = False
        scroll_amount = 0
        diagnostics = []

        if both_closed:
            self.blink_frames += 1
        else:
            if self.blink_frames >= config.BLINK_CONSEC_FRAMES and self.click_cooldown.ready():
                now = time.perf_counter()
                if (now - self.last_blink_time) <= config.DOUBLE_CLICK_WINDOW_SECONDS:
                    double_click = True
                    self.last_blink_time = 0.0
                else:
                    blink_click = True
                    self.last_blink_time = now
                self.click_cooldown.trigger()
            self.blink_frames = 0

        if left_wink:
            self.left_wink_frames += 1
        else:
            self.left_wink_frames = 0
            self.left_wink_latched = False

        if (
            self.left_wink_frames >= config.EYE_WINK_CONFIRM_FRAMES
            and not self.left_wink_latched
            and self.right_click_cooldown.ready()
        ):
            right_click = True
            self.left_wink_latched = True
            self.right_click_cooldown.trigger()

        if right_wink:
            self.right_wink_frames += 1
        else:
            self.right_wink_frames = 0
            self.right_wink_latched = False

        if (
            self.right_wink_frames >= config.EYE_DRAG_TOGGLE_FRAMES
            and not self.right_wink_latched
            and self.mode_toggle_cooldown.ready()
        ):
            self.drag_active = not self.drag_active
            drag_toggled = True
            self.right_wink_latched = True
            self.mode_toggle_cooldown.trigger()

        if scroll_pose:
            self.scroll_frames += 1
        else:
            self.scroll_frames = 0
            self.prev_scroll_y = None
            self.scroll_residual = 0.0

        if self.drag_active:
            scroll_pose = False
            self.scroll_frames = 0
            self.prev_scroll_y = None
            self.scroll_residual = 0.0

        scroll_active = self.scroll_frames >= config.EYE_SCROLL_CONFIRM_FRAMES
        last_scroll_delta = 0.0
        if scroll_active:
            if self.prev_scroll_y is not None:
                scroll_delta = self.prev_scroll_y - norm_y
                last_scroll_delta = scroll_delta
                motion_threshold = max(config.EYE_SCROLL_MOTION_RATIO, 1e-6)
                if abs(scroll_delta) > motion_threshold:
                    self.scroll_residual += (
                        scroll_delta / motion_threshold
                    ) * self.calibration.get("eye_scroll_sensitivity") * config.EYE_SCROLL_GAIN
                    emitted = int(clamp(self.scroll_residual, -4.0, 4.0))
                    if emitted != 0:
                        scroll_amount = emitted
                        self.scroll_residual -= emitted
            self.prev_scroll_y = norm_y

        if self.drag_active and not drag_toggled:
            blink_click = False
            double_click = False
            right_click = False
            scroll_amount = 0

        if left_wink and right_wink:
            diagnostics.append("wink conflict")
        if both_closed and scroll_active:
            diagnostics.append("blink/scroll overlap")

        eye_states = {
            "left_closed": ear_left < blink_threshold,
            "right_closed": ear_right < blink_threshold,
            "left_wink": left_wink,
            "right_wink": right_wink,
            "scroll_pose": scroll_pose,
            "drag_active": self.drag_active,
        }

        left_score = 0.55 + 0.45 * clamp((blink_threshold - ear) / max(blink_threshold, 1e-6), 0.0, 1.0)
        right_score = 0.55 + 0.45 * clamp(
            (blink_threshold - min(ear_left, ear_right)) / max(blink_threshold, 1e-6),
            0.0,
            1.0,
        )
        scroll_score = 0.45 + 0.30 * clamp(
            abs(last_scroll_delta) / max(config.EYE_SCROLL_MOTION_RATIO, 1e-6),
            0.0,
            1.0,
        ) if scroll_active else 0.0

        if drag_toggled:
            gesture_label = "Drag On" if self.drag_active else "Drag Off"
            gesture_confidence = 0.85
        elif self.drag_active:
            gesture_label = "Drag"
            gesture_confidence = 0.80
        elif double_click:
            gesture_label = "Double Click"
            gesture_confidence = left_score
        elif blink_click:
            gesture_label = "Blink Click"
            gesture_confidence = left_score
        elif right_click:
            gesture_label = "Right Click"
            gesture_confidence = right_score
        elif scroll_amount != 0:
            gesture_label = "Scroll Up" if scroll_amount > 0 else "Scroll Down"
            gesture_confidence = clamp(scroll_score, 0.0, 1.0)
        elif scroll_active:
            gesture_label = "Scroll Hold"
            gesture_confidence = clamp(max(scroll_score, 0.60), 0.0, 1.0)
        else:
            gesture_label = "Nose Move"
            gesture_confidence = 0.55

        return {
            "cursor_norm": (clamp(norm_x, 0.0, 1.0), clamp(norm_y, 0.0, 1.0)),
            "move_cursor": not scroll_active,
            "blink_click": blink_click,
            "double_click": double_click,
            "right_click": right_click,
            "scroll_amount": scroll_amount,
            "drag_active": self.drag_active,
            "drag_toggled": drag_toggled,
            "ear": ear,
            "ear_left": ear_left,
            "ear_right": ear_right,
            "raw_gaze": (nose_x_raw, nose_y_raw),
            "gesture_label": gesture_label,
            "gesture_confidence": gesture_confidence,
            "nose_norm": (nose_x_raw, nose_y_raw),
            "cursor_target_norm": (target_x, target_y),
            "gaze_hint": gaze_norm,
            "diagnostics": diagnostics,
            "eye_states": eye_states,
        }

    def training_metrics(self, face):
        landmarks = face["landmarks_norm"]
        ear_left = self._ear(landmarks, self.LEFT_EAR)
        ear_right = self._ear(landmarks, self.RIGHT_EAR)
        return {
            "ear": (ear_left + ear_right) * 0.5,
            "ear_left": ear_left,
            "ear_right": ear_right,
        }
