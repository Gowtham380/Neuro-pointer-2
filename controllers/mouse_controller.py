"""Mouse action wrapper around PyAutoGUI."""

from __future__ import annotations

import win32api
import win32con

import config
from utils import ExponentialSmoother, clamp, remap


class MouseController:
    def __init__(self, calibration) -> None:
        self.calibration = calibration
        self.screen_width = win32api.GetSystemMetrics(0)
        self.screen_height = win32api.GetSystemMetrics(1)
        self.smoother = ExponentialSmoother(config.CURSOR_SMOOTHING_ALPHA)
        self.dragging = False
        self.last_screen_pos = None

    def normalized_to_screen(self, point_norm):
        nx = clamp(point_norm[0], 0.0, 1.0)
        ny = clamp(point_norm[1], 0.0, 1.0)
        sx = int(nx * (self.screen_width - 1))
        sy = int(ny * (self.screen_height - 1))
        return sx, sy

    def hand_to_normalized(self, point_norm, hand_region):
        min_x, max_x, min_y, max_y = hand_region
        margin = config.ACTIVE_ZONE_MARGIN

        nx = clamp(remap(point_norm[0], min_x + margin, max_x - margin, 0.0, 1.0), 0.0, 1.0)
        ny = clamp(remap(point_norm[1], min_y + margin, max_y - margin, 0.0, 1.0), 0.0, 1.0)

        return nx, ny

    def move_from_hand(self, point_norm, active_zone=None):

        if active_zone is not None:
            x1, y1, x2, y2 = active_zone

            zone_w = max(x2 - x1, 1e-6)
            zone_h = max(y2 - y1, 1e-6)

            nx = (point_norm[0] - x1) / zone_w
            ny = (point_norm[1] - y1) / zone_h

            nx = clamp(nx, 0.0, 1.0)
            ny = clamp(ny, 0.0, 1.0)

        else:
            hand_region = self.calibration.get("hand_region")
            nx, ny = self.hand_to_normalized(point_norm, hand_region)

        alpha = self.calibration.get("hand_smoothing_alpha")
        self.move_normalized((nx, ny), alpha=alpha)

        return nx, ny

    def move_normalized(self, point_norm, alpha=None):
        sx, sy = self.normalized_to_screen(point_norm)
        smx, smy = self.smoother.update(float(sx), float(sy), alpha=alpha)
        if self.last_screen_pos is None:
            self.last_screen_pos = (smx, smy)
        move_delta = abs(smx - self.last_screen_pos[0]) + abs(smy - self.last_screen_pos[1])
        if move_delta >= config.CURSOR_MIN_MOVE_PIXELS:
            win32api.SetCursorPos((int(smx), int(smy)))
            self.last_screen_pos = (smx, smy)

    def left_click(self):
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)

    def double_click(self, second_click_only: bool = False):
        if second_click_only:
            self.left_click()
            return
        self.left_click()
        self.left_click()

    def right_click(self):
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,0,0)

    def scroll(self, amount: int):
        if amount != 0:
            win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, amount * 120)

    def update_drag(self, should_drag: bool):
        if should_drag and not self.dragging:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
            self.dragging = True
        elif not should_drag and self.dragging:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
            self.dragging = False

    def stop_drag(self):
        self.update_drag(False)

    def close(self):
        self.stop_drag()
