"""On-screen overlays for mode, FPS, gesture feedback and active zone."""

from __future__ import annotations
import cv2
import config


class OverlayRenderer:
    def __init__(self) -> None:
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Active zone position offset (in pixels)
        self.offset_x = 0
        self.offset_y = 0

    def move_zone(self, dx, dy):
        self.offset_x += dx
        self.offset_y += dy

        # prevent zone from going too far off-screen
        self.offset_x = max(-400, min(400, self.offset_x))
        self.offset_y = max(-300, min(300, self.offset_y))

    def get_active_zone(self, frame_shape, normalized: bool = False):
        height, width = frame_shape[:2]

        margin_x = int(width * config.ACTIVE_ZONE_MARGIN)
        margin_y = int(height * config.ACTIVE_ZONE_MARGIN)

        x1 = max(0, min(width - 2, margin_x + self.offset_x))
        y1 = max(0, min(height - 2, margin_y + self.offset_y))
        x2 = max(x1 + 1, min(width - 1, width - margin_x + self.offset_x))
        y2 = max(y1 + 1, min(height - 1, height - margin_y + self.offset_y))

        if not normalized:
            return (x1, y1, x2, y2)

        width_norm = max(width - 1, 1)
        height_norm = max(height - 1, 1)
        return (
            x1 / width_norm,
            y1 / height_norm,
            x2 / width_norm,
            y2 / height_norm,
        )

    @staticmethod
    def _draw_panel(frame, top_left, bottom_right, alpha=0.45, color=(10, 10, 10)):
        overlay = frame.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def draw_active_zone(self, frame):
        """
        Draws the interaction box where finger movement controls the cursor.
        Outside this area = dead zone.
        """
        x1, y1, x2, y2 = self.get_active_zone(frame.shape, normalized=False)

        # Red active control zone
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Label
        cv2.putText(
            frame,
            "ACTIVE ZONE",
            (x1 + 5, y1 - 10),
            self.font,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        return (x1, y1, x2, y2)

    def draw(
        self,
        frame,
        mode: str,
        fps: float,
        gesture_label: str,
        cursor_point=None,
        training_text: str = "",
        volume_level=None,
        debug_lines=None,
        message: str = "",
    ):
        height, width = frame.shape[:2]

        # Draw Active Zone
        active_zone = self.draw_active_zone(frame)

        # Info panel
        self._draw_panel(frame, (10, 10), (560, 210), alpha=0.45)

        mode_text = f"Mode: {mode.upper()}"
        cv2.putText(frame, mode_text, (25, 40), self.font, 0.75, (50, 255, 150), 2)

        cv2.putText(frame, f"FPS: {fps:.1f}", (25, 70), self.font, 0.65, (255, 220, 0), 2)

        cv2.putText(
            frame,
            f"Gesture: {gesture_label}",
            (25, 100),
            self.font,
            0.65,
            (255, 255, 255),
            2,
        )

        blink_state = "N/A"
        if mode.lower() == "eye":
            blink_state = "YES" if "blink" in gesture_label.lower() else "NO"

        cv2.putText(frame, f"Blink: {blink_state}", (25, 130), self.font, 0.60, (170, 230, 255), 2)

        if cursor_point is not None:
            cv2.putText(
                frame,
                f"Cursor tgt: ({cursor_point[0]}, {cursor_point[1]})",
                (25, 158),
                self.font,
                0.55,
                (180, 255, 180),
                2,
            )

        if mode.lower() == "eye":
            cv2.putText(
                frame,
                "Nose tip: orange marker",
                (25, 184),
                self.font,
                0.50,
                (0, 170, 255),
                1,
            )

        if training_text:
            self._draw_panel(frame, (10, height - 92), (width - 10, height - 12), alpha=0.50)
            cv2.putText(frame, training_text, (20, height - 36), self.font, 0.60, (80, 255, 255), 2)

        if message:
            cv2.putText(frame, message, (25, 214), self.font, 0.55, (180, 255, 180), 2)

        # Volume bar
        if volume_level is not None:
            bar_x, bar_y = width - 220, 25
            bar_w, bar_h = 180, 20

            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)

            fill_w = int(bar_w * float(volume_level))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 220, 255), -1)

            cv2.putText(
                frame,
                f"Volume {int(volume_level * 100)}%",
                (bar_x, bar_y - 7),
                self.font,
                0.5,
                (0, 220, 255),
                1,
            )

        # Cursor indicator
        if cursor_point is not None:
            cv2.circle(frame, cursor_point, 10, (0, 255, 0), 2)
            cv2.circle(frame, cursor_point, 2, (0, 255, 0), -1)

        # Debug lines
        if config.SHOW_DEBUG and debug_lines:
            y = 250
            for line in [line for line in debug_lines if line][:14]:
                cv2.putText(frame, line, (25, y), self.font, 0.50, (200, 200, 200), 1)
                y += 22

        # Bottom control help
        controls = "WASD:Move Zone  H:Hand  E:Eye  T:Train  P:Test  SPACE:Sample  C:Center Eye  R:Reset  Q:Quit"
        text_size = cv2.getTextSize(controls, self.font, 0.50, 1)[0]
        x = max(12, width - text_size[0] - 12)

        cv2.putText(frame, controls, (x, height - 16), self.font, 0.50, (240, 240, 240), 1)

        return active_zone
