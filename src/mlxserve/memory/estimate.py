"""Heuristic memory pre-admission (no model load): prompt tokens + optional KV-cache bound."""


def heuristic_prompt_tokens(messages: list[dict[str, str]]) -> int:
    raw = "\n".join(f"{m['role']}:{m['content']}" for m in messages)
    return max(1, len(raw) // 4)


def estimated_request_gb(
    prompt_tokens: int,
    max_new_tokens: int,
    tokens_per_gb: float,
    cap_gb: float,
) -> float:
    total = prompt_tokens + max_new_tokens
    est = total / max(tokens_per_gb, 1.0)
    return min(max(est, 0.0), cap_gb)


def estimate_kv_cache_gb(
    seq_len: int,
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_element: float = 2.0,
) -> float:
    """Conservative upper bound for decoder KV cache RAM (K+V per layer, one slot per token).

    Uses a standard transformer layout (no GQA grouping factor). Intended for admission
    heuristics only, not exact accounting per architecture.
    """
    sl = max(1, int(seq_len))
    per_layer = 2.0 * float(sl) * float(num_kv_heads) * float(head_dim) * float(bytes_per_element)
    return float(num_layers) * per_layer / (1024.0**3)


def admission_memory_gb(
    prompt_tokens: int,
    max_new_tokens: int,
    tokens_per_gb: float,
    cap_gb: float,
    *,
    kv_enabled: bool,
    kv_max_gb: float,
    kv_layers: int,
    kv_heads: int,
    kv_head_dim: int,
    kv_bytes_per_element: float,
) -> float:
    """Token-derived estimate plus optional capped KV overhead for pre-admission."""
    base = estimated_request_gb(prompt_tokens, max_new_tokens, tokens_per_gb, cap_gb)
    if not kv_enabled:
        return base
    seq = max(1, prompt_tokens + max_new_tokens)
    kv = estimate_kv_cache_gb(
        seq,
        num_layers=kv_layers,
        num_kv_heads=kv_heads,
        head_dim=kv_head_dim,
        bytes_per_element=kv_bytes_per_element,
    )
    return base + min(max(kv, 0.0), max(kv_max_gb, 0.0))
