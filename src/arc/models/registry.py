from __future__ import annotations

from arc.models.base import ARCAdapter


def create_adapter(
    source: str,
    block_size: int = 4,
    device_map: str | None = "auto",
    architecture: str = "jetmoe",
) -> ARCAdapter:
    from arc.models.jetmoe import JetMoeAdapter, build_tiny_jetmoe, load_jetmoe

    if architecture != "jetmoe":
        raise ValueError(f"unknown architecture: {architecture}")
    if source == "tiny":
        return JetMoeAdapter(build_tiny_jetmoe(), block_size=block_size)
    hf_model = load_jetmoe(source, device_map=device_map)
    return JetMoeAdapter(hf_model, block_size=block_size)
