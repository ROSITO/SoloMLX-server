import time
from dataclasses import dataclass
from typing import Literal

import psutil


MemoryZone = Literal["green", "yellow", "red"]


@dataclass
class MemorySnapshot:
    used_gb: float
    swap_used_gb: float
    total_gb: float


class MemoryGuardian:
    def __init__(self, soft_limit_gb: float, hard_limit_gb: float, idle_unload_minutes: int) -> None:
        self.soft_limit_gb = soft_limit_gb
        self.hard_limit_gb = hard_limit_gb
        self.idle_unload_seconds = idle_unload_minutes * 60

    def snapshot(self) -> MemorySnapshot:
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        used_gb = vm.used / (1024**3)
        total_gb = vm.total / (1024**3)
        swap_used_gb = swap.used / (1024**3)
        return MemorySnapshot(used_gb=used_gb, swap_used_gb=swap_used_gb, total_gb=total_gb)

    def classify(self, estimated_request_gb: float = 0.0) -> MemoryZone:
        snap = self.snapshot()
        projected = snap.used_gb + estimated_request_gb
        # Red only when memory is critically high, or swap usage is significant.
        if projected >= self.hard_limit_gb or (projected >= self.soft_limit_gb and snap.swap_used_gb >= 2.0):
            return "red"
        if projected >= self.soft_limit_gb:
            return "yellow"
        return "green"

    def should_unload_idle(self, last_used_ts: float) -> bool:
        return (time.time() - last_used_ts) > self.idle_unload_seconds
