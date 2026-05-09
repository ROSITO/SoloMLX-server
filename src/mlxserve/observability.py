from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import quantiles


def _escape_prometheus_label_value(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\r", " ")


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
    macos_memory_pressure: int = 0
    memory_chat_denied_total: int = 0
    memory_chat_denied_by_reason: dict[str, int] = field(default_factory=dict)
    chat_generation_tps: float = 0.0
    chat_completions_by_zone: dict[str, int] = field(default_factory=dict)
    chat_completions_by_model: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.recent_latencies is None:
            self.recent_latencies = []

    def observe_memory_chat_denied(self, reason: str) -> None:
        self.memory_chat_denied_total += 1
        key = reason if reason else "unknown"
        self.memory_chat_denied_by_reason[key] = self.memory_chat_denied_by_reason.get(key, 0) + 1

    def observe_chat_completion(
        self,
        memory_zone: str,
        model_label: str,
        *,
        label_zone: bool,
        label_model: bool,
    ) -> None:
        if label_zone:
            self.chat_completions_by_zone[memory_zone] = (
                self.chat_completions_by_zone.get(memory_zone, 0) + 1
            )
        if label_model and model_label:
            self.chat_completions_by_model[model_label] = (
                self.chat_completions_by_model.get(model_label, 0) + 1
            )

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

    def render_prometheus(
        self,
        *,
        label_chat_zone: bool = True,
        label_chat_model: bool = True,
    ) -> str:
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
            "# HELP mlxserve_macos_memory_pressure macOS memory_pressure level (0=normal,1=warning,2=critical)",
            "# TYPE mlxserve_macos_memory_pressure gauge",
            f"mlxserve_macos_memory_pressure {self.macos_memory_pressure}",
            "# HELP mlxserve_memory_chat_denied_total Chat completions rejected in red memory zone",
            "# TYPE mlxserve_memory_chat_denied_total counter",
            f"mlxserve_memory_chat_denied_total {self.memory_chat_denied_total}",
        ]
        if self.memory_chat_denied_by_reason:
            lines.extend(
                [
                    "# HELP mlxserve_memory_chat_denied_by_reason_total Chat denied in red zone by reason code",
                    "# TYPE mlxserve_memory_chat_denied_by_reason_total counter",
                ]
            )
            for reason, n in sorted(self.memory_chat_denied_by_reason.items()):
                re = _escape_prometheus_label_value(reason)
                lines.append(f'mlxserve_memory_chat_denied_by_reason_total{{reason="{re}"}} {n}')
        lines.extend(
            [
                "# HELP mlxserve_memory_zone Current memory zone (green=0,yellow=1,red=2)",
                "# TYPE mlxserve_memory_zone gauge",
                f"mlxserve_memory_zone {0 if self.memory_zone=='green' else 1 if self.memory_zone=='yellow' else 2}",
                "# HELP mlxserve_chat_generation_tps Last observed chat generation tokens per second",
                "# TYPE mlxserve_chat_generation_tps gauge",
                f"mlxserve_chat_generation_tps {self.chat_generation_tps:.4f}",
            ]
        )
        if label_chat_zone and self.chat_completions_by_zone:
            lines.extend(
                [
                    "# HELP mlxserve_chat_completions_labeled_total Successful chat completions (by memory zone at request start)",
                    "# TYPE mlxserve_chat_completions_labeled_total counter",
                ]
            )
            for z, n in sorted(self.chat_completions_by_zone.items()):
                ze = _escape_prometheus_label_value(z)
                lines.append(f'mlxserve_chat_completions_labeled_total{{memory_zone="{ze}"}} {n}')
        if label_chat_model and self.chat_completions_by_model:
            lines.extend(
                [
                    "# HELP mlxserve_chat_completions_by_model_total Successful chat completions by model id (high cardinality risk)",
                    "# TYPE mlxserve_chat_completions_by_model_total counter",
                ]
            )
            for m, n in sorted(self.chat_completions_by_model.items()):
                me = _escape_prometheus_label_value(m)
                lines.append(f'mlxserve_chat_completions_by_model_total{{model="{me}"}} {n}')
        lines.append(f"mlxserve_uptime_seconds {time.time():.0f}")
        return "\n".join(lines) + "\n"
