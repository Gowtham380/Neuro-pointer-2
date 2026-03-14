"""System volume control using Pycaw."""

from __future__ import annotations

import time
from ctypes import POINTER, cast

import config
from utils import clamp

try:
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
except Exception:  # pragma: no cover - fallback for missing optional deps
    CLSCTX_ALL = None
    AudioUtilities = None
    IAudioEndpointVolume = None


class VolumeController:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and config.ENABLE_VOLUME_GESTURE
        self.endpoint = None
        self.last_update_time = 0.0
        self.last_level = None
        if not self.enabled:
            return
        if not (AudioUtilities and IAudioEndpointVolume and CLSCTX_ALL):
            self.enabled = False
            return
        try:
            speakers = AudioUtilities.GetSpeakers()
            interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.endpoint = cast(interface, POINTER(IAudioEndpointVolume))
        except Exception:
            self.enabled = False
            self.endpoint = None

    def set_volume(self, level: float) -> None:
        if not self.enabled or self.endpoint is None:
            return
        now = time.perf_counter()
        if (now - self.last_update_time) < config.VOLUME_UPDATE_INTERVAL_SECONDS:
            return
        self.last_update_time = now
        safe_level = clamp(level, 0.0, 1.0)
        self.endpoint.SetMasterVolumeLevelScalar(float(safe_level), None)
        self.last_level = safe_level
