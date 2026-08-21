from __future__ import annotations

from arc.models.base import ARCAdapter


def create_adapter(source: str, block_size: int = 4, dtype: str | None = None, device: str | None = None) -> ARCAdapter:
    from arc.models.jetmoe import JetMoeAdapter, build_tiny_jetmoe, load_jetmoe

    if source == "tiny":
        return JetMoeAdapter(build_tiny_jetmoe(), block_size=block_size)
    hf_model = load_jetmoe(source, dtype=dtype, device=device)
    return JetMoeAdapter(hf_model, block_size=block_size)
