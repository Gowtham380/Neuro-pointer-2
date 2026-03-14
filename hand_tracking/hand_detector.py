"""MediaPipe-based hand detector."""

from __future__ import annotations

import cv2
import mediapipe as mp

import config


def _resolve_mediapipe_solutions():
    """Return MediaPipe legacy solutions module or raise a clear compatibility error."""
    if hasattr(mp, "solutions"):
        return mp.solutions
    try:
        # Some MediaPipe builds expose `solutions` as a direct submodule.
        from mediapipe import solutions as mp_solutions  # type: ignore

        return mp_solutions
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Installed mediapipe package does not provide the legacy `solutions` API "
            "(required by this project). Install a compatible build, e.g.:\n"
            "pip install \"mediapipe==0.10.14\""
        ) from exc


class HandDetector:
    def __init__(self) -> None:
        mp_solutions = _resolve_mediapipe_solutions()
        self._mp_hands = mp_solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.HAND_MAX_HANDS,
            model_complexity=0,
            min_detection_confidence=config.HAND_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.HAND_MIN_TRACKING_CONFIDENCE,
        )
        self._drawer = mp_solutions.drawing_utils

    def detect(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        detections = []
        if not results.multi_hand_landmarks:
            return detections

        height, width = frame_bgr.shape[:2]
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = "Right"
            if results.multi_handedness and idx < len(results.multi_handedness):
                handedness = results.multi_handedness[idx].classification[0].label

            norm_landmarks = [
                (lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark
            ]
            px_landmarks = [
                (int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks.landmark
            ]
            detections.append(
                {
                    "label": handedness,
                    "landmarks_norm": norm_landmarks,
                    "landmarks_px": px_landmarks,
                    "raw_landmarks": hand_landmarks,
                }
            )
        return detections

    def draw(self, frame_bgr, hands) -> None:
        if not config.SHOW_LANDMARKS:
            return
        for hand in hands:
            self._drawer.draw_landmarks(
                frame_bgr,
                hand["raw_landmarks"],
                self._mp_hands.HAND_CONNECTIONS,
            )

    def close(self) -> None:
        self._hands.close()
