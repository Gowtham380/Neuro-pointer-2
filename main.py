"""Neuro Pointer entry point.

Run with:
    python main.py
"""

from __future__ import annotations

import time

import cv2
import pyautogui

import config
from calibration.calibrate import CalibrationManager
from controllers.mouse_controller import MouseController
from controllers.volume_controller import VolumeController
from eye_tracking.eye_detector import EyeDetector
from eye_tracking.gaze_tracking import GazeTracker
from hand_tracking.gesture_recognition import HandGestureRecognizer
from hand_tracking.hand_detector import HandDetector
from ui.overlay import OverlayRenderer
from utils import FPSCounter


class NeuroPointerApp:
    def __init__(self) -> None:
        self.mode = "hand"
        self.calibration = CalibrationManager(config.CALIBRATION_FILE)

        self.hand_detector = HandDetector()
        self.hand_gestures = HandGestureRecognizer(self.calibration)

        self.eye_detector = EyeDetector()
        self.gaze_tracker = GazeTracker(self.calibration)

        self.mouse = MouseController(self.calibration)
        self.volume = VolumeController(enabled=True)
        self.overlay = OverlayRenderer()

        self.fps_counter = FPSCounter()
        self.last_hand_metrics = None
        self.last_eye_metrics = None
        self.last_raw_gaze = None

        self.message = ""
        self.message_until = 0.0
        self.recenter_until = 0
        

    def _open_camera(self):
        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
        return cap

    def _set_message(self, text: str, ttl: float = 1.3) -> None:
        self.message = text
        self.message_until = time.perf_counter() + ttl

    def _active_message(self) -> str:
        if time.perf_counter() > self.message_until:
            return ""
        return self.message

    def _process_hand_mode(self, frame, active_zone=None):
        hands = self.hand_detector.detect(frame)
        self.hand_detector.draw(frame, hands)

        debug_lines = []
        gesture_label = "No Hand"
        cursor_point = None

        if hands:
            hand = hands[0]
            result = self.hand_gestures.recognize(hand)
            self.last_hand_metrics = result["metrics"]
            frame_h, frame_w = frame.shape[:2]
            cursor_point = (
                int(result["cursor_norm"][0] * (frame_w - 1)),
                int(result["cursor_norm"][1] * (frame_h - 1)),
            )

            self.mouse.move_from_hand(result["cursor_norm"], active_zone)

            if result["left_click"]:
                self.mouse.left_click()
            if result["right_click"]:
                self.mouse.right_click()
            if result["scroll_amount"] != 0:
                self.mouse.scroll(result["scroll_amount"])

            self.mouse.update_drag(result["drag"])

            if result["volume_active"] and result["volume_level"] is not None:
                self.volume.set_volume(result["volume_level"])

            gesture_label = result["gesture_label"]
            finger_states = result["finger_states"]
            thresholds = result["dynamic_thresholds"]
            brightness = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
            diagnostics = list(result["diagnostics"])
            if brightness < config.HAND_MIN_BRIGHTNESS:
                diagnostics.append("lighting low")
            elif brightness > config.HAND_MAX_BRIGHTNESS:
                diagnostics.append("lighting harsh")

            debug_lines.extend(
                [
                    "Fingers "
                    f"T{int(finger_states['thumb'])} I{int(finger_states['index'])} "
                    f"M{int(finger_states['middle'])} R{int(finger_states['ring'])} "
                    f"P{int(finger_states['pinky'])}",
                    f"gesture={gesture_label} conf={result['gesture_confidence']:.2f}",
                    f"pinch={result['metrics']['pinch_distance']:.3f} / {thresholds['pinch_click_threshold']:.3f}",
                    f"index-middle={result['metrics']['index_middle_distance']:.3f} / {thresholds['right_click_threshold']:.3f}",
                    f"fist={result['metrics']['fist_ratio']:.3f} / {thresholds['fist_threshold']:.3f}",
                    f"scale={result['metrics']['hand_scale']:.3f} brightness={brightness:.0f}",
                    f"diag={', '.join(diagnostics) if diagnostics else 'ok'}",
                ]
            )
        else:
            self.hand_gestures.reset_tracking()
            self.mouse.update_drag(False)

        return gesture_label, cursor_point, debug_lines

    def _process_eye_mode(self, frame):
        faces = self.eye_detector.detect(frame)
        self.eye_detector.draw(frame, faces)

        debug_lines = []
        gesture_label = "No Face"
        cursor_point = None

        # Eye mode should never hold left drag.
        self.mouse.update_drag(False)

        if faces:
            face = faces[0]
            result = self.gaze_tracker.estimate(face)

            if time.time() < self.recenter_until:
                return "Recentering", None, []

            self.mouse.move_normalized(
                result["cursor_norm"],
                alpha=self.calibration.get("eye_smoothing_alpha"),
            )

            if result["blink_click"]:
                self.mouse.left_click()

            h, w = frame.shape[:2]
            cursor_point = (
                int(result["cursor_norm"][0] * (w - 1)),
                int(result["cursor_norm"][1] * (h - 1)),
            )
            gesture_label = result["gesture_label"]
            debug_lines.extend(
                [
                    f"EAR={result['ear']:.3f}  th={self.calibration.get('blink_ear_threshold'):.3f}",
                    f"gaze_raw=({result['raw_gaze'][0]:.3f}, {result['raw_gaze'][1]:.3f})",
                    f"gaze_center=({self.calibration.get('gaze_center')[0]:.3f}, {self.calibration.get('gaze_center')[1]:.3f})",
                ]
            )

        return gesture_label, cursor_point, debug_lines

    def _handle_key(self, key: int) -> bool:
        step=20
        if key in (ord("q"), ord("Q"), 27):
            return False

        if key in (ord("h"), ord("H")):
            self.mode = "hand"
            self.mouse.stop_drag()
            self._set_message("Hand mode enabled")

        elif key in (ord("e"), ord("E")):
            self.mode = "eye"
            self.mouse.stop_drag()
            self._set_message("Eye mode enabled")

        elif key in (ord("t"), ord("T")):
            was_active = self.calibration.training_active
            self.calibration.toggle_training(self.mode)
            if was_active and not self.calibration.training_active:
                self._set_message("Training stopped")
            elif self.calibration.training_active:
                self._set_message(f"Training started for {self.mode} mode")

        elif key == 32:  # SPACE
            if self.calibration.training_active:
                metrics = (
                    self.last_hand_metrics
                    if self.calibration.training_mode == "hand"
                    else self.last_eye_metrics
                )
                if metrics and self.calibration.capture_sample(metrics):
                    if self.calibration.training_active:
                        self._set_message("Sample captured")
                    else:
                        self._set_message("Training complete and saved", ttl=1.8)
                else:
                    self._set_message("No valid sample available", ttl=1.6)

        elif key == ord("c"):   # calibration
            if self.mode == "eye" and self.last_raw_gaze:
                self.calibration.set(
                    "gaze_center",
                    [float(self.last_raw_gaze[0]), float(self.last_raw_gaze[1])],
                )
                self.calibration.save()
                self._set_message("Eye center calibrated")

        elif key == ord("x"):   # cursor center reset
            screen_w, screen_h = pyautogui.size()
            pyautogui.moveTo(screen_w // 2, screen_h // 2)

            # reset face center
            if self.mode == "eye" and self.last_raw_gaze:
                self.calibration.set(
                    "gaze_center",
                    [float(self.last_raw_gaze[0]), float(self.last_raw_gaze[1])],
                )
                self.calibration.save()

            # pause tracking for 0.5 sec
            self.recenter_until = time.time() + 0.5

            self._set_message("Cursor recentered")

        elif key in (ord("r"), ord("R")):
            self.calibration.reset_defaults()
            self._set_message("Calibration reset to defaults", ttl=1.8)

        elif key == ord('w'):
            self.overlay.move_zone(0, -step)
            self._set_message("Zone Up")

        elif key == ord('s'):
            self.overlay.move_zone(0, step)
            self._set_message("Zone Down")

        elif key == ord('a'):
            self.overlay.move_zone(-step, 0)
            self._set_message("Zone Left")

        elif key == ord('d'):
            self.overlay.move_zone(step, 0)
            self._set_message("Zone Right")

        return True

    def run(self) -> None:
        cap = self._open_camera()
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam. Check camera permissions and index.")

        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)

        try:
            keep_running = True
            while keep_running:
                ok, frame = cap.read()
                if not ok:
                    continue

                if config.MIRROR_FEED:
                    frame = cv2.flip(frame, 1)

                # --- PROCESS INPUT FIRST ---
                if self.mode == "hand":
                    active_zone = self.overlay.get_active_zone(frame.shape, normalized=True)
                    gesture, cursor_point, debug_lines = self._process_hand_mode(frame, active_zone)
                else:
                    gesture, cursor_point, debug_lines = self._process_eye_mode(frame)

                fps = self.fps_counter.update()
                training_text = self.calibration.status_text()
                message = self._active_message()
                volume_level = self.volume.last_level

                # --- DRAW OVERLAY (returns active zone) ---
                self.overlay.draw(
                    frame,
                    self.mode,
                    fps,
                    gesture,
                    cursor_point,
                    training_text,
                    volume_level,
                    debug_lines,
                    message,
                )

                cv2.imshow(config.WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                keep_running = self._handle_key(key)

        finally:
            cap.release()
            self.mouse.close()
            self.hand_detector.close()
            self.eye_detector.close()
            cv2.destroyAllWindows()


def main() -> None:
    app = NeuroPointerApp()
    app.run()


if __name__ == "__main__":
    main()
