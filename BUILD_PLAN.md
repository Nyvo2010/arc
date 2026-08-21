# Goal
Build ARC as a modular research codebase for studying **adaptive recurrent computation inside pretrained Mixture-of-Experts Transformers**. The implementation must support layer, block, and model recurrence independently, measure actual computation, and make it possible to compare fixed and adaptive computation at matched compute.
The implementation is designed around free Kaggle and Google Colab tiers wherever practical. These environments are appropriate for inference, architecture development, profiling, controlled experiments, LoRA/QLoRA-scale adaptation, and controller experiments. They are not a realistic environment for pretraining a new 8–20B MoE from scratch.
## Core implementation principle
ARC separates:
- **Depth allocation:** repeated computation through layer, block, or model recurrence.
- **Width / activation:** ordinary MoE routing and expert activation inside each model execution.
MoE routing is not treated as a recurrence mechanism and is not given an expert-diversity objective.
The implementation should remain modular enough that the recurrence engine, controllers, compute accounting, and evaluation code can be reused across multiple MoE architectures.
---
# 1. Required training strategy
ARC **must be fine-tuned** for two independent reasons.
## 1.1 Recurrence-aware architectural adaptation
A normal pretrained model was optimized for its original forward trajectory. Repeatedly applying the same layer, block, or complete model traversal changes the distribution of hidden states and exposes the model to computation patterns that were not present during pretraining.
Therefore, untouched pretrained weights are a **compatibility baseline**, not the final ARC model.
Fine-tuning must explicitly expose the model to the recurrent execution pattern it will use. Training should include multiple recurrence depths where practical rather than optimizing only one fixed loop count.
## 1.2 General model improvement and current information
The adaptation stage should also improve the model more generally using appropriately current, high-quality training data. This prevents the ARC model from simply inheriting an older knowledge state while the architecture is being adapted.
The implementation must therefore distinguish two effects:
1. improvement caused by recurrence-aware architectural training;
2. improvement caused by additional/current training data.
Controlled data ablations are required so these effects are not conflated.
---
# 2. Development order
The scientific order and engineering order are different.
### Scientific order
Each recurrence scale is evaluated independently:
1. layer recurrence;
2. block recurrence;
3. model recurrence;
4. cross-scale comparison;
5. pairwise combinations;
6. full ARC only if justified.
### Engineering order
For the first end-to-end implementation, start with **model recurrence**, because it is the simplest recurrent mechanism to validate without exposing individual internal layer boundaries. Then implement block recurrence and finally layer recurrence.
This does **not** change the research comparison: all three mechanisms still receive independent fixed and adaptive experiments.
---
# 3. Base-model strategy
Use an existing open-weight MoE model rather than pretraining a new model.
### Primary research target
**DeepSeekMoE-16B Base**
Reasons:
- appropriate parameter scale for the project;
- conventional Transformer/MoE structure;
- clear layer boundaries;
- useful expert routing structure;
- large enough to make recurrence research meaningful.
### Prototype target
**JetMoE-8B**
Use this for fast architecture development and debugging when the larger model is too expensive for the available GPU.
### Secondary targets
Other modern sparse MoE architectures can be added after the ARC abstraction works. The adapter must not assume that all MoE architectures expose identical internal structures.
Base-model selection should always check:
- license;
- Hugging Face / Transformers support;
- actual layer and block structure;
- routing implementation;
- memory requirements;
- ability to expose intermediate hidden states;
- compatibility with PEFT and quantized loading.
---
# 4. Software stack
## Core
- Python 3.11+
- PyTorch
- Hugging Face Transformers
- Hugging Face Hub
- Accelerate
- Safetensors
## Adaptation
- PEFT
- bitsandbytes where supported for 4-bit/8-bit loading
- TRL only if the chosen training objective requires SFT or preference-style tooling
## Data and evaluation
- datasets
- evaluate
- lm-evaluation-harness where compatible
- custom ARC evaluation code
## Profiling
- torch.profiler
- CUDA events for latency
- custom FLOP accounting
- optional fvcore/ptflops where compatible
## Experiment tracking
Use simple reproducible artifacts first:
- YAML configuration files;
- JSON/JSONL experiment logs;
- CSV result summaries;
- TensorBoard logs;
- saved model/config metadata.
External tracking services are optional and must not be required for reproduction.
---
# 5. Repository architecture
```plain text
arc/
├── README.md
├── pyproject.toml
├── configs/
│   ├── base.yaml
│   ├── layer_fixed.yaml
│   ├── layer_adaptive.yaml
│   ├── block_fixed.yaml
│   ├── block_adaptive.yaml
│   ├── model_fixed.yaml
│   ├── model_adaptive.yaml
│   ├── training.yaml
│   └── evaluation.yaml
├── src/arc/
│   ├── models/
│   │   ├── base.py
│   │   ├── deepseek_moe.py
│   │   ├── jetmoe.py
│   │   └── registry.py
│   ├── recurrence/
│   │   ├── state.py
│   │   ├── layer.py
│   │   ├── block.py
│   │   ├── model.py
│   │   └── scheduler.py
│   ├── controllers/
│   │   ├── base.py
│   │   ├── rules.py
│   │   └── features.py
│   ├── routing/
│   │   └── instrumentation.py
│   ├── compute/
│   │   ├── flops.py
│   │   ├── latency.py
│   │   └── budget.py
│   ├── training/
│   │   ├── dataset.py
│   │   ├── objectives.py
│   │   ├── lora.py
│   │   └── trainer.py
│   └── evaluation/
│       ├── benchmarks.py
│       ├── metrics.py
│       ├── pareto.py
│       └── logging.py
├── scripts/
│   ├── benchmark.py
│   ├── profile.py
│   ├── train.py
│   ├── run_recurrence_sweep.py
│   └── analyze_results.py
├── notebooks/
│   ├── 00_baseline.ipynb
│   ├── 01_model_recurrence.ipynb
│   ├── 02_block_recurrence.ipynb
│   └── 03_layer_recurrence.ipynb
└── results/
    ├── raw/
    ├── processed/
    └── figures/
```
---
# 6. Model adapter interface
Every supported base model gets a model-specific adapter. The adapter exposes real computation boundaries without forcing different architectures into an artificial common implementation.
Conceptual interface:
```python
class ARCModelAdapter:
    def forward_native(self, input_ids, **kwargs): ...
    def embed(self, input_ids, **kwargs): ...
    def forward_layer(self, layer_idx, hidden_state, state, **kwargs): ...
    def forward_block(self, block_idx, hidden_state, state, **kwargs): ...
    def forward_model(self, hidden_state, state, **kwargs): ...
    def get_logits(self, hidden_state, **kwargs): ...
    def get_router_stats(self): ...
    def estimate_compute(self, operation, **kwargs): ...
```
The adapter must preserve the original model's:
- embeddings;
- positional encoding / positional state;
- normalization;
- residual connections;
- attention implementation;
- MoE routing;
- expert execution;
- output head;
- generation semantics.
Before recurrence is enabled, the ARC adapter must reproduce native model behavior within an explicitly measured numerical tolerance.
The adapter should expose hidden states at the exact boundaries used by recurrence and should never silently replace a model's native computation with an approximation.
---
# 7. Recurrence runtime
The recurrence engine should maintain an explicit runtime state containing at least:
```python
@dataclass
class RecurrenceState:
    model_loop: int
    block_loop_counts: dict[int, int]
    layer_loop_counts: dict[int, int]
    compute_used: float
    compute_budget: float | None
    max_model_loops: int
    max_block_loops: int
    max_layer_loops: int
```
The hidden state must be passed forward from one recurrent execution to the next.
Recurrence must therefore behave conceptually as:
```plain text
h0 → recurrent computation → h1 → recurrent computation → h2 → ...
```
It must never restart a recurrence from the original hidden state.
Repeated executions reuse the same model parameters unless a separate experiment explicitly changes the parameterization.
---
# 8. Layer recurrence implementation
Layer recurrence repeats an individual Transformer layer before normal traversal continues.
$$
h_{l,r+1}=F_l(h_{l,r})
$$
Example:
```plain text
L1 ×1
L2 ×3
L3 ×1
L4 ×2
L5 ×1
```
The runtime must support:
- uniform fixed repetition;
- heterogeneous fixed repetition;
- adaptive repetition.
For adaptive layer recurrence:
```plain text
enter layer i
      ↓
execute layer
      ↓
measure progress
      ↓
HALT → continue to next layer
CONTINUE → execute layer i again
```
The decision boundary is after an actual layer execution. Every recurrent execution must record:
- layer index;
- local recurrence index;
- controller decision;
- estimated compute;
- actual measured timing where available;
- hidden-state change;
- output change when available.
A major implementation concern is that a pretrained layer normally receives representations produced by the preceding layer. Repeating it may therefore move representations outside its training distribution. This is one reason recurrence-aware fine-tuning is required.
---
# 9. Block recurrence implementation
The exact definition of a **block** must be fixed per architecture before experiments begin.
For a conventional Transformer, the block should correspond to the model's complete repeated computational unit, including the architecture's attention, residual/normalization structure and MoE/FFN computation as implemented by the model.
Conceptually:
```plain text
block i
 ├─ attention
 ├─ residual / normalization
 └─ MoE / FFN
```
Block recurrence is:
$$
h_{b,r+1}=B_b(h_{b,r})
$$
Support:
- uniform fixed counts;
- heterogeneous fixed counts;
- adaptive counts.
The adapter must define exactly which parameters and operations belong to a block. This definition cannot change between experiments.
---
# 10. Model recurrence implementation
Model recurrence repeats a complete model traversal.
$$
H_{t+1}=F(H_t)
$$
A model loop is one complete traversal through the model's layers/blocks.
```plain text
Model loop t
    ↓
L1 → L2 → ... → LN
    ↓
progress measurement
    ↓
HALT / CONTINUE
    ↓
Model loop t+1
```
The MoE router runs according to the base architecture during each complete model execution. Routing events must be instrumented from the actual implementation rather than inferred from nominal expert counts.
For model recurrence, the output hidden state from model loop `t` becomes the input hidden state for model loop `t+1`.
---
# 11. MoE routing implementation
MoE routing is treated as conditional width, not recurrence.
The initial implementation must not introduce:
- expert-diversity rewards;
- forced expert rotation;
- expert-change requirements between loops;
- routing regularizers whose purpose is to increase diversity.
The router should naturally select whichever experts are useful for the current hidden state.
Routing statistics should be recorded as observations:
- expert IDs selected;
- routing probabilities if exposed by the model;
- expert utilization;
- load balance;
- number of active experts;
- routing changes between model loops.
If the same experts are repeatedly selected, that is a valid outcome. If expert assignments change, that is also a valid outcome.
---
# 12. Fixed recurrence implementation
Every recurrence scale needs two fixed controls.
## Uniform fixed recurrence
All applicable units receive the same count.
Example:
$$
[2,2,2,2,2]
$$
## Heterogeneous fixed recurrence
Different units receive predetermined counts.
Example:
$$
[1,2,4,1,2]
$$
The heterogeneous fixed condition is essential. Without it, adaptive recurrence could appear superior simply because it discovers a better non-uniform allocation rather than because dynamic decisions are useful.
Initial fixed sweeps should approximately cover:
- layer: ×1, ×2, ×3, ×4;
- block: ×1, ×2, ×3, ×4;
- model: ×1, ×2, ×3, ×4, ×6, ×8 where feasible.
The exact maximum must be selected from measured memory, latency and compute constraints.
---
# 13. Adaptive controller implementation
The initial ARC controller should use **deterministic progress-based rules and thresholds**. A learned HALT head is not a required component of the architecture.
Define a common controller interface:
```python
class RecurrenceController:
    def decide(self, features, state) -> bool:
        """True = continue recurrence, False = halt."""
```
Candidate features:
- output entropy;
- change in entropy;
- output-distribution distance;
- top-token stability;
- hidden-state cosine change;
- recurrence count;
- remaining compute budget.
Example rule:
```plain text
if distribution_change < threshold:
    HALT
else:
    CONTINUE
```
Thresholds must be tuned on validation data and evaluated across multiple values. They must not be selected after looking at the final test set.
The controller should estimate whether **another computation step is worthwhile**, not merely classify the query as easy or hard.
Controller computation must be negligible relative to the model itself.
---
# 14. Hard compute constraints
Every recurrence mechanism must have explicit hard limits:
$$
T_{layer,max},\quad T_{block,max},\quad T_{model,max},\quad B_{compute}
$$
Execution must terminate if:
1. the controller says HALT;
2. the relevant recurrence maximum is reached;
3. the compute budget is exhausted.
Conceptually:
```plain text
if controller says HALT:
    stop recurrence
elif recurrence_count >= maximum:
    stop recurrence
elif compute_budget exhausted:
    stop computation
else:
    continue
```
These limits are enforced by the runtime, not by the controller.
---
# 15. Progress measurement
The controller and analysis code should expose several measurable progress signals.
Output distribution:
$$
p_t=softmax(z_t)
$$
Distribution change:
$$
D_t=D(p_t,p_{t-1})
$$
Entropy:
$$
H(p_t)=-\sum_i p_{t,i}\log p_{t,i}
$$
Hidden-state change:
$$
\Delta h_t=1-\cos(h_t,h_{t-1})
$$
Validation loss improvement where labels are available:
$$
\Delta L_t=L_t-L_{t+1}
$$
These signals are diagnostics, not proofs of improvement. A confident wrong prediction can have low entropy and low distribution change, while a large hidden-state change does not necessarily imply better output.
---
# 16. Compute accounting
ARC must evaluate systems using **actual computation**, not nominal loop counts alone.
Track at least:
- total FLOPs;
- active expert FLOPs;
- average compute per example;
- maximum compute;
- layer execution count;
- block execution count;
- model-loop count;
- sequence length / tokens processed;
- wall-clock latency;
- peak GPU memory.
For MoE layers, compute accounting should distinguish total model parameters from active parameters and should count the actually executed expert paths as accurately as practical.
Where exact FLOP counting is unavailable, document the approximation and use the same method consistently across compared systems.
---
# 17. Equal-compute evaluation
Raw benchmark score is insufficient.
For every adaptive result, compare against fixed and predetermined heterogeneous systems with approximately the same **actual average compute**.
If an adaptive system averages 2.3 model loops, a fixed control should be selected with comparable average compute rather than comparing only against ×2 or ×3 by nominal count.
The central evaluation is the quality-compute frontier:
$$
Q(C)
$$
where `C` is actual compute.
The strongest result is an improvement in the Pareto frontier rather than simply a higher score obtained with more computation.
---
# 18. Training data and adaptation
The training pipeline should support a current, high-quality corpus appropriate to the selected base model and license.
The data pipeline must record:
- dataset version;
- collection date where available;
- filtering steps;
- deduplication method;
- tokenizer/version;
- sequence length;
- train/validation split;
- number of training tokens.
Data should be cleaned and deduplicated before training. Benchmark evaluation data must remain isolated from training data.
The implementation should support two controlled adaptation tracks:
### Track A — recurrence adaptation
Use the same or a controlled data distribution while exposing the model to recurrent execution. This measures whether the architecture becomes more compatible with repeated computation.
### Track B — recurrence + current-data improvement
Use recurrence-aware training plus the selected current data mixture. This measures the practical final-model improvement.
A third baseline should preserve the pretrained model while using only the current-data adaptation where feasible. This isolates the contribution of recurrence-aware architecture training.
---
# 19. Fine-tuning strategy under free compute
Full fine-tuning of a 16B-class model is not the default path on free GPUs.
Start with PEFT:
1. LoRA;
2. QLoRA where supported;
3. targeted modules;
4. broader adaptation only if resources permit.
Initial trainable-module candidates:
- attention projections;
- selected MLP / expert projections;
- normalization parameters if needed;
- router parameters as an explicit ablation, not automatically enabled.
The experiment configuration must explicitly record which parameters are trainable.
Use:
- 4-bit or 8-bit base loading where compatible;
- mixed precision;
- gradient checkpointing;
- small micro-batches;
- gradient accumulation;
- short contexts during development;
- checkpointing between notebook sessions.
Do not assume that a quantized model that fits inference will also fit training.
---
# 20. Recurrence-aware training objective
The base task objective remains the language-model objective:
$$
L_{task}
$$
For multiple recurrent states, optionally supervise intermediate states:
$$
L_{states}=\sum_t\lambda_tL_t
$$
This can encourage intermediate recurrent states to remain useful rather than only the final maximum-depth state.
For adaptive computation, a compute-aware objective can be used:
$$
L=L_{task}+\lambda C
$$
where `C` is actual computation.
Multiple values of `λ` must be evaluated because:
- too large a penalty can force premature halting;
- too small a penalty can make the controller consume excessive computation.
The final evaluation must report the complete quality-compute curve rather than a single selected penalty.
---
# 21. Training progression
## Stage 0 — Native baseline
Establish:
- native model output;
- loss/perplexity where applicable;
- task benchmark performance;
- parameter count;
- memory usage;
- latency;
- compute estimate;
- MoE routing statistics.
Verify that the ARC adapter reproduces native behavior.
## Stage 1 — Frozen recurrence compatibility
Implement fixed layer, block and model recurrence independently.
Test uniform counts first, then heterogeneous predetermined schedules.
Measure whether repeated pretrained computation:
- improves performance;
- produces progressive refinement;
- has diminishing returns;
- eventually degrades;
- changes expert routing.
This stage determines whether recurrence is mechanically valid and whether the untouched checkpoint provides a useful starting point.
## Stage 2 — Recurrence-aware fine-tuning
For each promising recurrence family, adapt the model using the corresponding recurrent execution.
Training must expose multiple recurrence depths where practical.
Compare the adapted model against the frozen recurrence baseline at matched compute.
## Stage 3 — Current-data improvement ablations
Separate:
- frozen pretrained baseline;
- recurrence-aware adaptation only;
- current-data adaptation only where feasible;
- recurrence-aware adaptation + current-data improvement.
This identifies which gains come from architecture and which come from general model improvement.
## Stage 4 — Adaptive recurrence
Implement deterministic rule/threshold controllers for each recurrence scale.
Tune thresholds on validation data.
Compare adaptive systems against uniform fixed and heterogeneous fixed systems at matched actual compute.
## Stage 5 — Cross-scale comparison
Compare layer, block and model recurrence at approximately equal compute.
Identify:
- strongest recurrence scale;
- useful recurrence locations;
- diminishing returns;
- recurrence-compatible regions of the network.
## Stage 6 — Pairwise combinations
Test:
- layer + block;
- layer + model;
- block + model.
Only retain combinations that produce evidence of benefit at matched compute.
## Stage 7 — Full ARC
If justified by the preceding experiments, combine the useful recurrence scales with:
- deterministic adaptive controllers;
- compute budgets;
- hard recurrence limits;
- natural MoE routing;
- recurrence-aware fine-tuning;
- current training data.
The final architecture must be evidence-driven rather than requiring every mechanism.
---
# 22. First concrete experiments
## Experiment 0 — Adapter correctness
Use 50–200 prompts and compare the native model against the ARC adapter.
Verify:
- token-level equality where deterministic;
- logits within numerical tolerance;
- matching generation behavior;
- no unintended routing changes;
- comparable memory and latency.
## Experiment 1 — Recurrence trajectories
For selected layers, blocks and model loops collect:
$$
h^{(0)}\rightarrow h^{(1)}\rightarrow h^{(2)}\rightarrow h^{(3)}
$$
Measure:
- loss;
- output distribution;
- output distribution change;
- hidden-state change;
- task performance;
- latency;
- compute;
- expert routing.
## Experiment 2 — Fixed recurrence sweeps
Run uniform and heterogeneous fixed schedules for layer, block and model recurrence.
## Experiment 3 — Equal-compute comparison
Construct approximate compute-matched comparisons between the three recurrence scales.
## Experiment 4 — Recurrence-aware fine-tuning
Fine-tune the most promising recurrence configurations with PEFT and compare them with their frozen counterparts.
## Experiment 5 — Adaptive rules
Apply deterministic threshold-based controllers to the promising recurrence scales.
Compare against matched heterogeneous fixed schedules.
## Experiment 6 — Pairwise combinations
Test combinations only after independent mechanisms have been characterized.
---
# 23. Kaggle and Google Colab workflow
## Kaggle
Prefer Kaggle for:
- longer experiment runs;
- parameter sweeps;
- repeated fixed-recurrence experiments;
- PEFT training runs;
- controller threshold sweeps;
- profiling and result generation.
The code must checkpoint frequently because free compute availability and session duration are not guaranteed.
## Google Colab Free
Prefer Colab for:
- interactive development;
- debugging model internals;
- architecture changes;
- quick ablations;
- validating individual experiments;
- reproducing selected runs.
## Hardware abstraction
Never hard-code a T4, L4 or P100.
Detect the available hardware at runtime and configure:
- device;
- dtype;
- quantization;
- batch size;
- sequence length;
- gradient accumulation;
- checkpointing.
Every experiment must save the detected hardware and software versions in its result metadata.
---
# 24. Memory and checkpoint strategy
For large MoE models, memory management is a first-class implementation concern.
### Inference
Start with:
- 4-bit/8-bit loading where compatible;
- batch size 1;
- short context lengths;
- CPU offload only when necessary.
### Training
Use:
- QLoRA/LoRA;
- gradient checkpointing;
- mixed precision;
- gradient accumulation;
- small micro-batches;
- frequent checkpoints.
Checkpoints should include:
- adapter weights;
- optimizer state when needed;
- scheduler state;
- training step;
- configuration;
- base-model identifier;
- tokenizer identifier;
- dataset version;
- random seeds.
The experiment must be restartable after a Kaggle or Colab session ends.
---
# 25. Evaluation and logging schema
Every run should produce a machine-readable record containing at least:
```plain text
experiment_id
base_model
model_revision
tokenizer_revision
training_mode
trainable_parameters
adapter_type
recurrence_type
recurrence_schedule
controller_type
controller_thresholds
max_recurrence
compute_budget
sequence_length
batch_size
hardware
seed
training_dataset
training_tokens
validation_loss
benchmark_scores
avg_compute
max_compute
avg_latency
peak_memory
avg_layer_loops
avg_block_loops
avg_model_loops
expert_utilization
routing_statistics
checkpoint_path
```
This allows results from different notebook sessions to be merged without relying on notebook state.
---
# 26. Required ablations
Every claimed improvement must answer:
1. Did recurrence cause the improvement?
2. Did recurrence-aware fine-tuning cause it?
3. Did current/newer training data cause it?
4. Did heterogeneous allocation cause it?
5. Did adaptivity cause it?
6. Did additional average compute cause it?
7. Did MoE routing change cause it?
8. Does the effect remain at matched actual compute?
At minimum, the main experiments should compare:
<table header-row="true">
<tr>
<td>System</td>
<td>Recurrence</td>
<td>Training</td>
<td>Allocation</td>
<td>Adaptive</td>
</tr>
<tr>
<td>Baseline</td>
<td>none</td>
<td>pretrained</td>
<td>none</td>
<td>no</td>
</tr>
<tr>
<td>Fixed uniform</td>
<td>one scale</td>
<td>frozen</td>
<td>uniform</td>
<td>no</td>
</tr>
<tr>
<td>Fixed heterogeneous</td>
<td>one scale</td>
<td>frozen/adapted</td>
<td>predetermined</td>
<td>no</td>
</tr>
<tr>
<td>Adaptive</td>
<td>one scale</td>
<td>adapted</td>
<td>dynamic</td>
<td>threshold/rule</td>
</tr>
<tr>
<td>Combined</td>
<td>multiple scales</td>
<td>adapted</td>
<td>dynamic</td>
<td>threshold/rule</td>
</tr>
</table>
---
# 27. Expected failure modes
The implementation must explicitly detect:
### Repetition provides no benefit
Repeated execution may simply waste compute.
### Repetition helps and then degrades
The recurrent trajectory may have an optimal depth.
### Recurrence only works after adaptation
This would indicate that the pretrained representation trajectory is not naturally recurrence-compatible.
### Adaptive controller collapses to maximum compute
The controller may fail to learn useful stopping behavior.
### Adaptive controller collapses to minimum compute
The compute penalty or thresholds may be too aggressive.
### All examples receive nearly identical computation
The system may be functionally fixed despite having an adaptive mechanism.
### One recurrence scale dominates
This is a valid research result. The final ARC architecture should not retain weaker recurrence mechanisms merely for symmetry.
### Adaptive allocation provides no advantage
If adaptive rules do not beat matched fixed heterogeneous allocation, the dynamic-allocation hypothesis is weakened.
---
# 28. Milestones
## Milestone 1 — Correct recurrence engine
Native model and ARC adapter agree within measured numerical tolerance.
## Milestone 2 — Recurrence effect
At least one recurrence scale produces a reproducible effect on quality, compute or refinement behavior.
## Milestone 3 — Recurrence-aware adaptation
Fine-tuning produces a measurable difference compared with the frozen recurrent model.
## Milestone 4 — Adaptive allocation
A threshold-based adaptive controller produces a better quality-compute tradeoff than matched fixed controls, if the hypothesis is supported.
## Milestone 5 — Cross-scale result
Layer, block and model recurrence have been compared at matched compute and their relative strengths are established.
## Milestone 6 — ARC combination
A combination of recurrence mechanisms produces additional gains that survive equal-compute evaluation, if supported by the evidence.
## Milestone 7 — Final Pareto result
The resulting ARC configuration demonstrates a meaningful quality-compute Pareto improvement over the baseline.
---
# 29. Final implementation target
The mature architecture may look like:
```plain text
QUERY / INPUT
     │
     ▼
INITIAL MODEL COMPUTATION
     │
     ▼
MODEL LOOP t
     │
     ├── native MoE routing
     │
     ├── block traversal
     │      ├── block-level recurrence when enabled
     │      │
     │      └── layer-level recurrence when enabled
     │
     ▼
OUTPUT / PROGRESS MEASUREMENT
     │
     ▼
RULE-BASED CONTROLLER
     │
     ├── HALT ───────► OUTPUT
     │
     └── CONTINUE ───► MODEL LOOP t+1
```
This is an implementation endpoint, not a requirement that every recurrence scale must survive the experiments.
The final ARC system should contain only mechanisms that demonstrate a useful quality-compute tradeoff.
---
# 30. Final design principles
Do not assume:
- every layer should loop equally;
- every block should loop equally;
- every model should execute the same number of loops;
- harder inputs automatically need more computation everywhere;
- recurrent executions need different experts;
- expert diversity is inherently useful;
- maximum computation is inherently better;
- deterministic adaptive rules are inferior to learned controllers;
- frozen pretrained weights are sufficient for the final ARC model.
Instead:
- fine-tune the model for recurrence compatibility;
- also use current training data for general model improvement;
- isolate architectural gains from data gains;
- compare uniform and heterogeneous fixed controls;
- use deterministic threshold-based adaptive controllers;
- enforce hard compute limits outside the controller;
- measure actual compute rather than nominal loop count;
- compare systems at matched compute;
- retain only recurrence mechanisms supported by ablation results.
The implementation goal is therefore:
$$
\boxed{\text{Build a model that can learn where, when, and how much recurrent computation is worth spending.}}
$$