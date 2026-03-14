"""Gesture logic for hand-mode interactions with diagnostics and stabilization."""

from __future__ import annotations

import time
from statistics import mean

import config
from utils import CooldownTimer, clamp, distance_2d, mean_point, remap


class HandGestureRecognizer:
    """Recognize robust hand gestures using hand-size-normalized metrics."""

    def __init__(self, calibration) -> None:
        self.calibration = calibration
        self.left_click_cooldown = CooldownTimer(config.ACTION_COOLDOWN_SECONDS)
        self.right_click_cooldown = CooldownTimer(config.ACTION_COOLDOWN_SECONDS)
        self.prev_scroll_y = None
        self.left_click_latched = False
        self.right_click_latched = False
        self.hand_scale_ema = None
        self.prev_keypoints = None
        self.last_print_time = 0.0
        self.pose_frames = {
            "left_click": 0,
            "right_click": 0,
            "scroll": 0,
            "drag": 0,
            "volume": 0,
        }

    @staticmethod
    def _normalize_axis(vector):
        length = max((vector[0] ** 2 + vector[1] ** 2) ** 0.5, 1e-6)
        return vector[0] / length, vector[1] / length

    @staticmethod
    def _project(point, origin, axis) -> float:
        return ((point[0] - origin[0]) * axis[0]) + ((point[1] - origin[1]) * axis[1])

    @staticmethod
    def _ratio(value: float, scale: float) -> float:
        return value / max(scale, 1e-6)

    def reset_tracking(self) -> None:
        self.prev_scroll_y = None
        self.left_click_latched = False
        self.right_click_latched = False
        self.prev_keypoints = None
        for key in self.pose_frames:
            self.pose_frames[key] = 0

    def _update_pose_frame(self, name: str, active: bool) -> int:
        self.pose_frames[name] = self.pose_frames[name] + 1 if active else 0
        return self.pose_frames[name]

    def _compute_hand_scale(self, landmarks) -> float:
        palm_width = distance_2d(landmarks[5][:2], landmarks[17][:2])
        palm_height = distance_2d(landmarks[0][:2], landmarks[9][:2])
        middle_length = distance_2d(landmarks[9][:2], landmarks[12][:2])
        return max((palm_width + palm_height + middle_length) / 3.0, 1e-4)

    def _dynamic_thresholds(self, hand_scale: float) -> dict[str, float]:
        if self.hand_scale_ema is None:
            self.hand_scale_ema = hand_scale
        else:
            alpha = config.HAND_SCALE_EMA_ALPHA
            self.hand_scale_ema = (1.0 - alpha) * self.hand_scale_ema + alpha * hand_scale

        scale_ratio = hand_scale / max(self.hand_scale_ema, 1e-6)
        pinch_threshold = self.calibration.get("pinch_click_threshold") * scale_ratio
        right_threshold = self.calibration.get("right_click_threshold") * scale_ratio
        fist_threshold = self.calibration.get("fist_threshold") * scale_ratio

        return {
            "pinch_click_threshold": max(0.018, pinch_threshold),
            "right_click_threshold": max(0.015, right_threshold),
            "scroll_separation_threshold": max(right_threshold * 1.55, 0.028),
            "fist_threshold": max(0.080, fist_threshold),
            "volume_pinch_threshold": self.calibration.get("volume_pinch_threshold") * scale_ratio,
            "scroll_motion_threshold": max(config.SCROLL_DEADZONE, hand_scale * config.HAND_SCROLL_MOTION_RATIO),
        }

    def _finger_is_extended(self, landmarks, wrist, palm_center, axis, tip_id, pip_id, mcp_id, hand_scale) -> bool:
        tip = landmarks[tip_id][:2]
        pip = landmarks[pip_id][:2]
        mcp = landmarks[mcp_id][:2]

        tip_proj = self._project(tip, wrist, axis)
        pip_proj = self._project(pip, wrist, axis)
        mcp_proj = self._project(mcp, wrist, axis)

        extension_margin = hand_scale * 0.015
        spread_margin = hand_scale * 0.010

        projection_ok = tip_proj > (pip_proj + extension_margin) and pip_proj > (mcp_proj - extension_margin)
        spread_ok = distance_2d(tip, palm_center) > (distance_2d(pip, palm_center) + spread_margin)
        return projection_ok and spread_ok

    def _finger_states(self, landmarks, hand_label: str, hand_scale: float) -> dict[str, bool]:
        wrist = landmarks[0][:2]
        palm_center = mean_point([landmarks[idx][:2] for idx in (0, 5, 9, 13, 17)])
        forward_axis = self._normalize_axis((landmarks[9][0] - wrist[0], landmarks[9][1] - wrist[1]))

        thumb_tip = landmarks[4][:2]
        thumb_ip = landmarks[3][:2]
        index_mcp = landmarks[5][:2]

        thumb_open = (
            distance_2d(thumb_tip, index_mcp) > (hand_scale * 0.36)
            and distance_2d(thumb_tip, palm_center) > distance_2d(thumb_ip, palm_center)
        )

        # MediaPipe handedness can swap with mirrored cameras, so keep the thumb rule geometric.
        if hand_label not in {"Left", "Right"}:
            hand_label = "Right"

        return {
            "thumb": thumb_open,
            "index": self._finger_is_extended(landmarks, wrist, palm_center, forward_axis, 8, 6, 5, hand_scale),
            "middle": self._finger_is_extended(landmarks, wrist, palm_center, forward_axis, 12, 10, 9, hand_scale),
            "ring": self._finger_is_extended(landmarks, wrist, palm_center, forward_axis, 16, 14, 13, hand_scale),
            "pinky": self._finger_is_extended(landmarks, wrist, palm_center, forward_axis, 20, 18, 17, hand_scale),
        }

    def _compute_metrics(self, landmarks, hand_scale: float):
        wrist = landmarks[0][:2]
        pinch_distance = distance_2d(landmarks[4][:2], landmarks[8][:2])
        index_middle_distance = distance_2d(landmarks[8][:2], landmarks[12][:2])
        thumb_pinky_distance = distance_2d(landmarks[4][:2], landmarks[20][:2])
        fist_ratio = mean(distance_2d(wrist, landmarks[idx][:2]) for idx in (8, 12, 16, 20))

        return {
            "hand_scale": hand_scale,
            "pinch_distance": pinch_distance,
            "pinch_ratio": self._ratio(pinch_distance, hand_scale),
            "index_middle_distance": index_middle_distance,
            "index_middle_ratio": self._ratio(index_middle_distance, hand_scale),
            "thumb_pinky_distance": thumb_pinky_distance,
            "thumb_pinky_ratio": self._ratio(thumb_pinky_distance, hand_scale),
            "fist_ratio": fist_ratio,
            "fist_ratio_norm": self._ratio(fist_ratio, hand_scale),
        }

    def _gesture_scores(self, fingers, metrics, thresholds, poses, scroll_delta) -> dict[str, float]:
        left_score = 0.0
        if poses["left_click"]:
            left_score = 0.55 + 0.45 * clamp(
                (thresholds["pinch_click_threshold"] - metrics["pinch_distance"])
                / max(thresholds["pinch_click_threshold"], 1e-6),
                0.0,
                1.0,
            )

        right_score = 0.0
        if poses["right_click"]:
            right_score = 0.55 + 0.45 * clamp(
                (thresholds["right_click_threshold"] - metrics["index_middle_distance"])
                / max(thresholds["right_click_threshold"], 1e-6),
                0.0,
                1.0,
            )

        drag_score = 0.0
        if poses["drag"]:
            drag_score = 0.55 + 0.45 * clamp(
                (thresholds["fist_threshold"] - metrics["fist_ratio"])
                / max(thresholds["fist_threshold"], 1e-6),
                0.0,
                1.0,
            )

        scroll_score = 0.0
        if poses["scroll"]:
            motion_score = clamp(abs(scroll_delta) / max(thresholds["scroll_motion_threshold"], 1e-6), 0.0, 1.0)
            separation_score = clamp(
                (metrics["index_middle_distance"] - thresholds["scroll_separation_threshold"])
                / max(thresholds["scroll_separation_threshold"], 1e-6),
                0.0,
                1.0,
            )
            scroll_score = 0.45 + 0.30 * motion_score + 0.25 * separation_score

        volume_score = 0.0
        if poses["volume"]:
            volume_score = 0.50 + 0.50 * clamp(
                (thresholds["volume_pinch_threshold"] - metrics["thumb_pinky_distance"])
                / max(thresholds["volume_pinch_threshold"], 1e-6),
                0.0,
                1.0,
            )

        move_score = 0.25 + 0.15 * int(fingers["index"])
        return {
            "Move": move_score,
            "Left Click": left_score,
            "Right Click": right_score,
            "Scroll": scroll_score,
            "Drag": drag_score,
            "Volume": volume_score,
        }

    def _diagnostics(self, metrics, thresholds, poses, keypoints) -> list[str]:
        diagnostics = []

        if thresholds["scroll_separation_threshold"] <= thresholds["right_click_threshold"] * 1.20:
            diagnostics.append("right/scroll overlap")

        if self.prev_keypoints is not None:
            jitter = mean(
                distance_2d(keypoints[idx], self.prev_keypoints[idx]) for idx in keypoints
            ) / max(metrics["hand_scale"], 1e-6)
            if jitter > config.HAND_JITTER_RATIO_THRESHOLD and not any(poses.values()):
                diagnostics.append("unstable landmarks")

        active_poses = [name for name, active in poses.items() if active]
        if len(active_poses) > 1:
            diagnostics.append("gesture conflict")

        pinch_threshold_ratio = thresholds["pinch_click_threshold"] / max(metrics["hand_scale"], 1e-6)
        right_threshold_ratio = thresholds["right_click_threshold"] / max(metrics["hand_scale"], 1e-6)
        if (
            pinch_threshold_ratio < 0.10
            or pinch_threshold_ratio > 0.55
            or right_threshold_ratio < 0.08
            or right_threshold_ratio > 0.40
        ):
            diagnostics.append("threshold misconfiguration")

        self.prev_keypoints = dict(keypoints)
        return diagnostics

    def _maybe_print_debug(self, fingers, metrics, gesture_label, confidence, diagnostics):
        now = time.perf_counter()
        if not config.SHOW_DEBUG or (now - self.last_print_time) < config.HAND_DIAGNOSTIC_PRINT_SECONDS:
            return

        self.last_print_time = now
        print(
            "[hand-debug]",
            f"gesture={gesture_label}",
            f"confidence={confidence:.2f}",
            f"fingers=T{int(fingers['thumb'])} I{int(fingers['index'])} M{int(fingers['middle'])} "
            f"R{int(fingers['ring'])} P{int(fingers['pinky'])}",
            f"pinch={metrics['pinch_ratio']:.2f}",
            f"im={metrics['index_middle_ratio']:.2f}",
            f"fist={metrics['fist_ratio_norm']:.2f}",
            f"diag={','.join(diagnostics) if diagnostics else 'ok'}",
        )

    def recognize(self, hand):
        landmarks = hand["landmarks_norm"]
        hand_scale = self._compute_hand_scale(landmarks)
        thresholds = self._dynamic_thresholds(hand_scale)
        metrics = self._compute_metrics(landmarks, hand_scale)
        fingers = self._finger_states(landmarks, hand.get("label", "Right"), hand_scale)

        is_pinched = metrics["pinch_distance"] < thresholds["pinch_click_threshold"]
        right_fingers = fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]
        scroll_fingers = right_fingers
        closed_fist = not fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]

        left_pose = (
            is_pinched
            and fingers["index"]
            and not fingers["middle"]
            and not fingers["ring"]
            and not fingers["pinky"]
        )
        right_pose = (
            right_fingers
            and not is_pinched
            and metrics["index_middle_distance"] < thresholds["right_click_threshold"]
        )
        scroll_pose = (
            scroll_fingers
            and not is_pinched
            and metrics["index_middle_distance"] > thresholds["scroll_separation_threshold"]
        )
        drag_pose = (
            closed_fist
            and not is_pinched
            and metrics["fist_ratio"] < thresholds["fist_threshold"]
            and metrics["pinch_distance"] > thresholds["pinch_click_threshold"] * 1.15
        )
        volume_pose = (
            config.ENABLE_VOLUME_GESTURE
            and fingers["thumb"]
            and fingers["index"]
            and fingers["middle"]
            and fingers["ring"]
            and fingers["pinky"]
            and metrics["thumb_pinky_distance"] < thresholds["volume_pinch_threshold"]
        )

        self._update_pose_frame("left_click", left_pose)
        self._update_pose_frame("right_click", right_pose)
        self._update_pose_frame("scroll", scroll_pose)
        self._update_pose_frame("drag", drag_pose)
        self._update_pose_frame("volume", volume_pose)

        left_click = False
        left_confirmed = self.pose_frames["left_click"] >= config.CLICK_CONFIRM_FRAMES
        if left_confirmed and not self.left_click_latched and self.left_click_cooldown.ready():
            left_click = True
            self.left_click_latched = True
            self.left_click_cooldown.trigger()
        elif not left_pose:
            self.left_click_latched = False

        right_click = False
        right_confirmed = self.pose_frames["right_click"] >= config.CLICK_CONFIRM_FRAMES
        if right_confirmed and not self.right_click_latched and self.right_click_cooldown.ready():
            right_click = True
            self.right_click_latched = True
            self.right_click_cooldown.trigger()
        elif not right_pose:
            self.right_click_latched = False

        drag = self.pose_frames["drag"] >= config.DRAG_HOLD_FRAMES
        volume_active = self.pose_frames["volume"] >= config.CLICK_CONFIRM_FRAMES
        scroll_confirmed = self.pose_frames["scroll"] >= config.SCROLL_CONFIRM_FRAMES

        scroll_anchor_y = mean((landmarks[8][1], landmarks[12][1]))
        scroll_delta = 0.0
        scroll_amount = 0
        if scroll_pose:
            if self.prev_scroll_y is not None:
                scroll_delta = self.prev_scroll_y - scroll_anchor_y
                if scroll_confirmed and abs(scroll_delta) > thresholds["scroll_motion_threshold"]:
                    speed = clamp(
                        scroll_delta / max(thresholds["scroll_motion_threshold"], 1e-6),
                        -3.0,
                        3.0,
                    )
                    scroll_amount = int(
                        round(speed * self.calibration.get("scroll_sensitivity"))
                    )
            self.prev_scroll_y = scroll_anchor_y
        else:
            self.prev_scroll_y = None

        volume_level = None
        if volume_active:
            volume_level = clamp(remap(metrics["pinch_distance"], 0.02, 0.24, 0.0, 1.0), 0.0, 1.0)

        poses = {
            "left_click": left_pose,
            "right_click": right_pose,
            "scroll": scroll_pose,
            "drag": drag_pose,
            "volume": volume_pose,
        }
        keypoints = {idx: landmarks[idx][:2] for idx in (0, 5, 8, 9, 12, 17)}
        diagnostics = self._diagnostics(metrics, thresholds, poses, keypoints)
        scores = self._gesture_scores(fingers, metrics, thresholds, poses, scroll_delta)

        if left_click:
            gesture_label = "Left Click"
        elif right_click:
            gesture_label = "Right Click"
        elif drag:
            gesture_label = "Drag"
        elif scroll_amount != 0:
            gesture_label = "Scroll Up" if scroll_amount > 0 else "Scroll Down"
        elif volume_active:
            gesture_label = "Volume"
        elif left_pose and not left_confirmed:
            gesture_label = "Left Ready"
        elif right_pose and not right_confirmed:
            gesture_label = "Right Ready"
        elif drag_pose and not drag:
            gesture_label = "Drag Ready"
        elif scroll_pose:
            gesture_label = "Scroll Hold" if scroll_confirmed else "Scroll Ready"
        elif fingers["index"]:
            gesture_label = "Move"
        else:
            gesture_label = "Idle"

        if drag:
            cursor_norm = mean_point([landmarks[idx][:2] for idx in (5, 9, 13, 17)])
        else:
            cursor_norm = landmarks[8][:2]

        gesture_confidence = clamp(
            scores.get(
                "Scroll" if gesture_label.startswith("Scroll") else gesture_label,
                scores["Move"],
            ),
            0.0,
            1.0,
        )

        self._maybe_print_debug(fingers, metrics, gesture_label, gesture_confidence, diagnostics)

        return {
            "cursor_norm": cursor_norm,
            "left_click": left_click,
            "right_click": right_click,
            "scroll_amount": scroll_amount,
            "drag": drag,
            "volume_active": volume_active,
            "volume_level": volume_level,
            "gesture_label": gesture_label,
            "gesture_confidence": gesture_confidence,
            "metrics": metrics,
            "dynamic_thresholds": thresholds,
            "finger_states": fingers,
            "diagnostics": diagnostics,
            "gesture_scores": scores,
        }

    def training_metrics(self, hand):
        landmarks = hand["landmarks_norm"]
        return self._compute_metrics(landmarks, self._compute_hand_scale(landmarks))
