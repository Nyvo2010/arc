from arc.recurrence.halt_head import ModelHaltHead, BlockHaltHead, LayerHaltHead
from arc.recurrence.controller import ThresholdController
from arc.models.base import ARCAdapter


def make_controller(scale: str, adapter: ARCAdapter, max_loops: int = 4, compute_budget: float | None = None):
    """Create a controller with appropriate halt head per scale.

    Halt head is built per Notion spec:
    - model adaptive: halt head every model turn
    - block adaptive: halt head every transformer block turn
    - layer adaptive: halt head every layer turn
    """
    # hidden_dim is adapter-specific; use a default if unknown
    hidden_dim = getattr(adapter, "hidden_dim", 768)

    if scale == "model":
        head = ModelHaltHead(hidden_dim=hidden_dim)
    elif scale == "block":
        head = BlockHaltHead(hidden_dim=hidden_dim)
    elif scale == "layer":
        head = LayerHaltHead(hidden_dim=hidden_dim)
    else:
        head = None

    return ThresholdController(max_loops=max_loops, compute_budget=compute_budget, halt_head=head)
