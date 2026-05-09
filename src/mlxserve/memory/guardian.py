import time
from dataclasses import dataclass
from typing import Literal
import subprocess

import psutil


MemoryZone = Literal["green", "yellow", "red"]


@dataclass
class MemorySnapshot:
    used_gb: float
    swap_used_gb: float
    total_gb: float
    pressure: Literal["normal", "warning", "critical"]


class MemoryGuardian:
    def __init__(self, soft_limit_gb: float, hard_limit_gb: float, idle_unload_minutes: int) -> None:
        self.soft_limit_gb = soft_limit_gb
        self.hard_limit_gb = hard_limit_gb
        self.idle_unload_seconds = idle_unload_minutes * 60
        self._last_zone: MemoryZone = "green"

    @staticmethod
    def _memory_pressure_level() -> Literal["normal", "warning", "critical"]:
        try:
            output = subprocess.run(
                ["memory_pressure", "-Q"],
                capture_output=True,
                text=True,
                timeout=1.0,
                check=False,
            )
            text = (output.stdout + "\n" + output.stderr).lower()
            if "critical" in text:
                return "critical"
            if "warn" in text:
                return "warning"
        except Exception:
            pass
        return "normal"

    def snapshot(self) -> MemorySnapshot:
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        used_gb = vm.used / (1024**3)
        total_gb = vm.total / (1024**3)
        swap_used_gb = swap.used / (1024**3)
        return MemorySnapshot(
            used_gb=used_gb,
            swap_used_gb=swap_used_gb,
            total_gb=total_gb,
            pressure=self._memory_pressure_level(),
        )

    def classify_detail(self, estimated_request_gb: float = 0.0) -> tuple[MemoryZone, str]:
        """Return memory zone and a short deny reason code (empty when not red)."""
        snap = self.snapshot()
        projected = snap.used_gb + estimated_request_gb
        if snap.pressure == "critical":
            self._last_zone = "red"
            return "red", "macos_memory_pressure_critical"

        # Hysteresis to avoid oscillation near the threshold.
        soft_to_green = max(0.0, self.soft_limit_gb - 0.5)
        hard_to_yellow = max(0.0, self.hard_limit_gb - 0.5)

        if projected >= self.hard_limit_gb:
            self._last_zone = "red"
            return "red", "projected_over_hard_limit_gb"
        if projected >= self.soft_limit_gb and snap.swap_used_gb >= 2.0:
            self._last_zone = "red"
            return "red", "soft_limit_with_high_swap_gb"

        if self._last_zone == "red" and projected >= hard_to_yellow:
            return "red", "hysteresis_stay_red_until_below_hard_margin"
        if self._last_zone in ("red", "yellow") and projected >= soft_to_green:
            self._last_zone = "yellow"
            return "yellow", ""

        if projected >= self.soft_limit_gb or snap.pressure == "warning":
            self._last_zone = "yellow"
            return "yellow", ""

        self._last_zone = "green"
        return "green", ""

    def classify(self, estimated_request_gb: float = 0.0) -> MemoryZone:
        zone, _ = self.classify_detail(estimated_request_gb)
        return zone

    def should_unload_idle(self, last_used_ts: float) -> bool:
        return (time.time() - last_used_ts) > self.idle_unload_seconds
