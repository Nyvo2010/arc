from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Scale = Literal["base", "l", "b", "m"]


@dataclass(frozen=True)
class ModelVariant:
    """The only values allowed to differ between comparable model runs."""

    architecture: str
    path: str
    scale: Scale = "base"
    recurrence: int = 1
    block_size: int = 4

    def __post_init__(self) -> None:
        if self.scale not in ("base", "l", "b", "m"):
            raise ValueError(f"unknown scale: {self.scale}")
        if self.recurrence < 1:
            raise ValueError("recurrence must be at least 1")
        if self.scale == "base" and self.recurrence != 1:
            raise ValueError("base recurrence must be 1")

    def model_args(self) -> dict[str, str | int]:
        return {
            "architecture": self.architecture,
            "path": self.path,
            "scale": self.scale,
            "recurrence": self.recurrence,
            "block_size": self.block_size,
        }


@dataclass(frozen=True)
class BenchmarkProtocol:
    """Frozen settings shared by every variant in one experiment set."""

    tasks: tuple[str, ...] = (
        "mmlu",
        "gpqa_main_zeroshot",
        "gpqa_diamond_zeroshot",
    )
    num_fewshot: int = 0
    limit: int | None = 500
    random_seed: int = 1234
    numpy_random_seed: int = 1234
    torch_random_seed: int = 1234
    fewshot_random_seed: int = 1234
    batch_size: str = "auto:2"
    device: str = "cuda:0"
    dtype: str = "int8"

    def as_dict(self) -> dict:
        return asdict(self)

    def harness_args(self) -> dict:
        return {
            "tasks": list(self.tasks),
            "num_fewshot": self.num_fewshot,
            "limit": self.limit,
            "batch_size": self.batch_size,
            "device": self.device,
            "random_seed": self.random_seed,
            "numpy_random_seed": self.numpy_random_seed,
            "torch_random_seed": self.torch_random_seed,
            "fewshot_random_seed": self.fewshot_random_seed,
        }
