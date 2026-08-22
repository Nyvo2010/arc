# ARC Models & Adapters

## 7 Variants

All variants share the same `ARCAdapter` interface and inference signature:
`model(input_ids, attention_mask=None, position_ids=None) -> RecurrenceResult`

|Name|Scale|Adaptive|Builder|
|---|---|---|---|
|base|base|False|`build_arc_model(source, scale='base')`|
|model_fixed|model|False|`scale='model', adaptive=False, recurrence=R`|
|block_fixed|block|False|`scale='block', adaptive=False, recurrence=R`|
|layer_fixed|layer|False|`scale='layer', adaptive=False, recurrence=R`|
|model_adaptive|model|True|`scale='model', adaptive=True, max_loops=M`|
|block_adaptive|block|True|`scale='block', adaptive=True, max_loops=M`|
|layer_adaptive|layer|True|`scale='layer', adaptive=True, max_loops=M`|

## Production ready

- Tests: `tests/test_parity.py`, `tests/test_smoke.py`, `tests/test_contract.py`
- Benchmarks: CSV runners + `lm-evaluation-harness` bridge for free community tasks
- Kaggle scripts: `run_benchmarks.sh`, `run_lm_eval.sh`
- Packaging: `pyproject.toml` with dev deps

## Adapter Contract

`ARCAdapter` provides:
* `embed`, `forward_native`, `prepare`
* `forward_layer`, `forward_block`, `forward_model`
* `normalize`, `project_logits`, `final_logits`
* `num_layers`, `num_blocks`
* `lm_head_flops_per_token`, `unit_flops(scale, unit_index, seq_len, batch_size)`

Implement once per base model, reuse for all 7 variants.

## Adaptive HALT Decision

Controller runs after each unit execution at the chosen granularity.

Features computed from logits and hidden state:
* entropy, entropy_delta
* JS divergence between successive distributions
* top-1 stability
* hidden cosine change
* recurrence count, compute used/budget

Default `ThresholdController` halts when distribution and hidden state stabilize or max loops/budget reached.

Halt timing:
* model adaptive: after each complete model traversal
* block adaptive: after each block execution
* layer adaptive: after each layer execution

## Usage

```python
from arc.models.factory import build_arc_model

model, adapter = build_arc_model(
    source="tiny",
    scale="model",
    adaptive=True,
    max_loops=4,
)
result = model(input_ids)
logits = result.logits
state = result.state  # compute_used, executions, unit_loop_counts
```

Benchmarks treat all variants identically; only `scale` and `adaptive` change architecture.
