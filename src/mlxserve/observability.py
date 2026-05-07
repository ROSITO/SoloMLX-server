from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import quantiles


@dataclass
class MetricsStore:
    requests_total: int = 0
    chat_requests_total: int = 0
    errors_total: int = 0
    rate_limited_total: int = 0
    generated_tokens_total: int = 0
    latency_sum_seconds: float = 0.0
    latency_count: int = 0
    recent_latencies: list[float] | None = None
    memory_zone: str = "green"
    memory_used_gb: float = 0.0
    swap_used_gb: float = 0.0
    chat_generation_tps: float = 0.0

    def __post_init__(self) -> None:
        if self.recent_latencies is None:
            self.recent_latencies = []

    def observe_latency(self, seconds: float) -> None:
        self.latency_sum_seconds += max(0.0, seconds)
        self.latency_count += 1
        self.recent_latencies.append(max(0.0, seconds))
        if len(self.recent_latencies) > 2048:
            self.recent_latencies = self.recent_latencies[-1024:]

    def latency_p95(self) -> float:
        if not self.recent_latencies:
            return 0.0
        if len(self.recent_latencies) < 20:
            return max(self.recent_latencies)
        return quantiles(self.recent_latencies, n=100)[94]

    def render_prometheus(self) -> str:
        avg = (self.latency_sum_seconds / self.latency_count) if self.latency_count else 0.0
        p95 = self.latency_p95()
        lines = [
            "# HELP mlxserve_requests_total Total HTTP requests",
            "# TYPE mlxserve_requests_total counter",
            f"mlxserve_requests_total {self.requests_total}",
            "# HELP mlxserve_chat_requests_total Total chat completion requests",
            "# TYPE mlxserve_chat_requests_total counter",
            f"mlxserve_chat_requests_total {self.chat_requests_total}",
            "# HELP mlxserve_errors_total Total application errors",
            "# TYPE mlxserve_errors_total counter",
            f"mlxserve_errors_total {self.errors_total}",
            "# HELP mlxserve_rate_limited_total Total rate-limited requests",
            "# TYPE mlxserve_rate_limited_total counter",
            f"mlxserve_rate_limited_total {self.rate_limited_total}",
            "# HELP mlxserve_generated_tokens_total Total generated tokens (estimated)",
            "# TYPE mlxserve_generated_tokens_total counter",
            f"mlxserve_generated_tokens_total {self.generated_tokens_total}",
            "# HELP mlxserve_request_latency_seconds_avg Average request latency in seconds",
            "# TYPE mlxserve_request_latency_seconds_avg gauge",
            f"mlxserve_request_latency_seconds_avg {avg:.6f}",
            "# HELP mlxserve_request_latency_seconds_p95 Recent p95 request latency in seconds",
            "# TYPE mlxserve_request_latency_seconds_p95 gauge",
            f"mlxserve_request_latency_seconds_p95 {p95:.6f}",
            "# HELP mlxserve_memory_used_gb Current used system memory in GB",
            "# TYPE mlxserve_memory_used_gb gauge",
            f"mlxserve_memory_used_gb {self.memory_used_gb:.4f}",
            "# HELP mlxserve_swap_used_gb Current used swap in GB",
            "# TYPE mlxserve_swap_used_gb gauge",
            f"mlxserve_swap_used_gb {self.swap_used_gb:.4f}",
            "# HELP mlxserve_memory_zone Current memory zone (green=0,yellow=1,red=2)",
            "# TYPE mlxserve_memory_zone gauge",
            f"mlxserve_memory_zone {0 if self.memory_zone=='green' else 1 if self.memory_zone=='yellow' else 2}",
            "# HELP mlxserve_chat_generation_tps Last observed chat generation tokens per second",
            "# TYPE mlxserve_chat_generation_tps gauge",
            f"mlxserve_chat_generation_tps {self.chat_generation_tps:.4f}",
            f"mlxserve_uptime_seconds {time.time():.0f}",
        ]
        return "\n".join(lines) + "\n"
