## Research objective
ARC investigates whether a pretrained Mixture-of-Experts Transformer can achieve better **quality-per-compute** by dynamically allocating computation rather than always executing a fixed computational depth.
The project separates computation into two dimensions:
- **Depth / recurrence:** layer, block, and model looping.
- **Width / activation:** ordinary MoE expert routing.
The three recurrence mechanisms are first studied independently, each with fixed and adaptive variants. Only after their individual effects are understood are they combined.
## Core hypothesis
A model may not need the same amount of computation for every input or at every stage. If additional computation is allocated where its marginal value is highest, ARC may reach the quality of a more expensive fixed-computation system at lower average compute.
$$
\boxed{\text{maximize useful computation per unit of compute}}
$$
## Recurrence scales
1. **Layer recurrence** — repeatedly execute an individual Transformer layer.
2. **Block recurrence** — repeatedly execute a defined Transformer block.
3. **Model recurrence** — repeatedly execute the entire model.
MoE routing is not a fourth recurrence type. It controls conditional width inside each model execution.
## Fixed vs adaptive
Each recurrence scale gets:
- uniform fixed recurrence,
- heterogeneous predetermined recurrence,
- adaptive HALT/CONTINUE recurrence.
This separates the value of recurrence from the value of non-uniform allocation and from the value of dynamic decision-making.
## Routing principle
The router should initially remain natural. ARC does **not** optimize for expert diversity, force expert rotation, or require different recurrent executions to use different experts.
For the initial architecture, MoE routing occurs once per complete model execution. Multiple model loops therefore create multiple routing events; layer/block recurrence does not independently trigger a new global routing pass.
## Evaluation principle
The main result is not simply a higher benchmark score. ARC is successful if it improves the **quality/compute frontier** at matched or controlled compute.
Important measurements include:
- quality, loss, perplexity and task performance,
- actual FLOPs and active expert FLOPs,
- average and maximum compute,
- wall-clock latency,
- recurrence counts by layer/block/model,
- computation distribution across examples,
- hidden-state and output-distribution changes,
- MoE routing behavior.
## Research progression
**Baseline → fixed layer → adaptive layer → fixed block → adaptive block → fixed model → adaptive model → cross-scale comparison → combinations → full ARC.**
The final architecture is deliberately an endpoint rather than an assumption: weak recurrence mechanisms should be dropped rather than retained for architectural symmetry.
## Source
This page is a concise project description derived from the internal ARC research plan.