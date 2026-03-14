"""Calibration and gesture training utilities."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import config
from utils import clamp


class CalibrationManager:
    HAND_STAGES = [
        {
            "id": "pinch_click",
            "metric": "pinch_distance",
            "prompt": "Pinch thumb + index and press SPACE",
        },
        {
            "id": "right_click",
            "metric": "index_middle_distance",
            "prompt": "Touch index + middle fingertips and press SPACE",
        },
        {
            "id": "fist",
            "metric": "fist_ratio",
            "prompt": "Make a closed fist and press SPACE",
        },
    ]

    EYE_STAGES = [
        {
            "id": "eye_open",
            "metric": "ear",
            "prompt": "Keep eyes open naturally, then press SPACE",
        },
        {
            "id": "eye_closed",
            "metric": "ear",
            "prompt": "Close your eyes (blink hold), then press SPACE",
        },
    ]

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.data = dict(config.DEFAULT_CALIBRATION)
        self.training_active = False
        self.training_mode = None
        self.stage_idx = 0
        self.samples = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.data.update(loaded)
        except Exception:
            pass

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def reset_defaults(self) -> None:
        self.data = dict(config.DEFAULT_CALIBRATION)
        self.save()

    def get(self, key: str):
        return self.data.get(key, config.DEFAULT_CALIBRATION.get(key))

    def set(self, key: str, value) -> None:
        self.data[key] = value

    def _stages(self):
        if self.training_mode == "hand":
            return self.HAND_STAGES
        if self.training_mode == "eye":
            return self.EYE_STAGES
        return []

    def start_training(self, mode: str) -> None:
        self.training_active = True
        self.training_mode = mode
        self.stage_idx = 0
        self.samples = {stage["id"]: [] for stage in self._stages()}

    def stop_training(self) -> None:
        self.training_active = False
        self.training_mode = None
        self.stage_idx = 0
        self.samples = {}

    def toggle_training(self, mode: str) -> None:
        if self.training_active and self.training_mode == mode:
            self.stop_training()
        else:
            self.start_training(mode)

    def current_stage(self):
        stages = self._stages()
        if not stages or self.stage_idx >= len(stages):
            return None
        return stages[self.stage_idx]

    def status_text(self) -> str:
        if not self.training_active:
            return ""
        stage = self.current_stage()
        if not stage:
            return ""
        sample_count = len(self.samples[stage["id"]])
        return (
            f"Training ({self.training_mode.upper()}): {stage['prompt']} "
            f"[{sample_count}/{config.TRAINING_SAMPLES_PER_STAGE}]"
        )

    def capture_sample(self, metrics: dict) -> bool:
        if not self.training_active:
            return False
        stage = self.current_stage()
        if not stage:
            return False
        metric_name = stage["metric"]
        if metric_name not in metrics:
            return False

        self.samples[stage["id"]].append(float(metrics[metric_name]))
        if len(self.samples[stage["id"]]) >= config.TRAINING_SAMPLES_PER_STAGE:
            self.stage_idx += 1
            if self.stage_idx >= len(self._stages()):
                self._apply_training()
                self.save()
                self.stop_training()
        return True

    def _apply_training(self) -> None:
        if self.training_mode == "hand":
            pinch_avg = mean(self.samples["pinch_click"])
            right_avg = mean(self.samples["right_click"])
            fist_avg = mean(self.samples["fist"])

            self.data["pinch_click_threshold"] = clamp(pinch_avg * 1.30, 0.02, 0.16)
            self.data["right_click_threshold"] = clamp(right_avg * 1.30, 0.02, 0.20)
            self.data["fist_threshold"] = clamp(fist_avg * 1.20, 0.08, 0.35)

        elif self.training_mode == "eye":
            open_avg = mean(self.samples["eye_open"])
            closed_avg = mean(self.samples["eye_closed"])
            blink_threshold = (open_avg + closed_avg) * 0.5
            self.data["blink_ear_threshold"] = clamp(blink_threshold, 0.10, 0.32)
