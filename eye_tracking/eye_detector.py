"""MediaPipe FaceMesh detector for eye tracking."""

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


class EyeDetector:
    NOSE_TIP_POINT = config.EYE_NOSE_LANDMARK_ID
    IRIS_AND_EYE_POINTS = [
        33,
        133,
        159,
        145,
        362,
        263,
        386,
        374,
        468,
        469,
        470,
        471,
        472,
        473,
        474,
        475,
        476,
        477,
    ]

    def __init__(self) -> None:
        mp_solutions = _resolve_mediapipe_solutions()
        self._mp_face_mesh = mp_solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=config.FACE_MAX_FACES,
            refine_landmarks=True,
            min_detection_confidence=config.FACE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_MIN_TRACKING_CONFIDENCE,
        )

    def detect(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        faces = []
        if not results.multi_face_landmarks:
            return faces

        height, width = frame_bgr.shape[:2]
        for face_landmarks in results.multi_face_landmarks:
            normalized = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            nose = normalized[self.NOSE_TIP_POINT]
            faces.append(
                {
                    "landmarks_norm": normalized,
                    "raw_landmarks": face_landmarks,
                    "nose_norm": (nose[0], nose[1]),
                    "nose_px": (int(nose[0] * width), int(nose[1] * height)),
                }
            )
        return faces

    def draw(self, frame_bgr, faces):
        if not config.SHOW_LANDMARKS:
            return
        height, width = frame_bgr.shape[:2]
        for face in faces:
            landmarks = face["landmarks_norm"]
            for idx in self.IRIS_AND_EYE_POINTS:
                x = int(landmarks[idx][0] * width)
                y = int(landmarks[idx][1] * height)
                cv2.circle(frame_bgr, (x, y), 1, (0, 255, 255), -1)

            # Highlight the nose tip used as primary eye-mode cursor anchor.
            nose_px = face.get("nose_px")
            if nose_px is None:
                nx, ny = landmarks[self.NOSE_TIP_POINT][:2]
                nose_px = (int(nx * width), int(ny * height))
            cv2.circle(frame_bgr, nose_px, 5, (0, 140, 255), 2)
            cv2.circle(frame_bgr, nose_px, 2, (0, 140, 255), -1)

    def close(self) -> None:
        self._face_mesh.close()
