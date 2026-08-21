## 1. Research objective
ARC investigates whether a pretrained Mixture-of-Experts (MoE) Transformer can achieve better **quality-per-compute** by dynamically allocating computation rather than always executing a fixed computational depth.
ARC treats computation as having two separate dimensions:
- **Depth / recurrence:** how many times computation is performed.
- **Width / activation:** how much of the model is active during each computation through MoE expert routing.
The depth side is investigated through **three distinct looping mechanisms**:
1. **Layer looping** — repeatedly execute an individual Transformer layer.
2. **Block looping** — repeatedly execute a Transformer block.
3. **Model looping** — repeatedly execute the entire model.
MoE routing is a separate mechanism. It is **not** considered a fourth type of looping. Instead, MoE controls conditional width/activation inside the model.
The central hypothesis is:
> A model does not necessarily need to spend the same amount of computation on every input, at every depth, or at every stage of processing. If additional computation is allocated where its marginal value is highest, a model may reach the quality of a substantially more expensive fixed-computation system at lower average compute.
The primary objective is therefore:
$$
\boxed{\text{maximize useful computation per unit of compute}}
$$
The project is explicitly designed to determine **which recurrence mechanisms are useful before combining them**. Layer, block, and model recurrence are first studied as separate model families, each with both predetermined and dynamic loop counts. Only after these mechanisms are understood will they be merged.
---
## 2. Core conceptual model
ARC separates the problem into two dimensions.
### Depth: recurrent computation
The model can perform additional computation at different structural scales:
$$
\text{Layer recurrence}
\rightarrow
\text{Block recurrence}
\rightarrow
\text{Model recurrence}
$$
These are not assumed to be equivalent. They may perform different computational functions and may have different quality/compute tradeoffs.
### Width: conditional activation
The underlying Transformer is an MoE model. During a model execution, the router selects the experts that are active.
Therefore:
$$
\text{ARC computation}
=
\text{recurrent depth allocation}
+
\text{MoE conditional width}
$$
The project should not assume that recurrence and expert routing need to be coordinated by an explicit diversity objective. Routing should initially remain as natural as possible.
---
## 3. The three looping mechanisms
### 3.1 Layer looping
Layer recurrence repeatedly executes an individual Transformer layer:
$$
h_{l,r+1}=F_l(h_{l,r})
$$
where:
- $`l`$ is the layer index,
- $`r`$ is the local recurrence index,
- $`F_l`$ is the layer transformation.
Example:
```plain text
L1 ×1
L2 ×3
L3 ×1
L4 ×2
L5 ×1
```
The key research question is whether repeated application of an individual layer can provide useful refinement.
A skeptical possibility must remain open: pretrained layers are normally trained on the distribution of representations produced by the preceding layers, so repeated execution may eventually move the hidden state outside the distribution for which the layer was optimized. Layer recurrence may therefore be useful only in particular layers, for limited numbers of repetitions, or after recurrence-aware adaptation.
### 3.2 Block looping
Block recurrence repeatedly executes a Transformer block:
$$
h_{b,r+1}=B_b(h_{b,r})
$$
where $`b`$ indexes the block.
The exact definition of “block” must be fixed in the implementation before experiments begin. The research should not allow the terminology to change between experiments.
Block recurrence tests a larger unit of repeated computation than layer recurrence. It may provide more coherent refinement because the repeated unit contains a larger transformation.
### 3.3 Model looping
Model recurrence repeatedly executes the complete Transformer:
$$
H_{t+1}=F(H_t)
$$
where $`t`$ is the model-loop index.
For a model with $`N`$ blocks/layers, one model loop corresponds to a complete traversal:
```plain text
L1 → L2 → L3 → ... → LN
```
A second model loop then performs another complete traversal over the resulting state.
Model recurrence tests whether another global computation phase is useful.
---
## 4. MoE routing is separate from looping
MoE is a conditional computation mechanism, not a recurrence mechanism.
The router determines which experts are active during a model execution. ARC should not initially introduce an objective that encourages expert diversity.
The principle is:
> The router should use whichever experts are useful for the current state.
If repeated computation causes different hidden states and therefore different expert assignments, that is an interesting observational result. If the same experts remain optimal, that is equally valid.
There should initially be:
- no expert-diversity reward,
- no forced expert rotation,
- no requirement that different model loops use different experts,
- no requirement that recurrent executions use different experts.
### Important routing constraint
For the initial ARC architecture, the **MoE router runs once per complete model execution**.
Therefore:
- If there is **no model looping**, the router runs once for the inference.
- If there are **multiple model loops**, the router runs once during each model loop.
- Layer and block looping do **not** independently trigger a new global router pass.
Conceptually:
$$
R_t=Router(H_t)
$$
where $`t`$ indexes model loops.
Within one model loop, layer/block recurrence uses the routing decisions associated with that model execution according to the chosen implementation.
This makes the experimental distinction explicit:
$$
\boxed{\text{Looping controls depth; MoE routing controls conditional width/activation.}}
$$
---
## 5. Why study the three recurrence types separately?
The project should not start with a full hierarchical controller.
If layer, block, and model recurrence are combined immediately, a performance improvement would be difficult to attribute.
Instead, the first research stage isolates them:
```plain text
                   BASE MoE MODEL
                         │
      ┌──────────────────┼──────────────────┐
      │                  │                  │
      ▼                  ▼                  ▼
   LAYER              BLOCK              MODEL
 RECURRENCE         RECURRENCE         RECURRENCE
      │                  │                  │
  ┌───┴───┐          ┌───┴───┐          ┌───┴───┐
  ▼       ▼          ▼       ▼          ▼       ▼
Fixed  Adaptive    Fixed  Adaptive    Fixed  Adaptive
```
This produces six fundamental experimental model families:
- Layer Fixed
- Layer Adaptive
- Block Fixed
- Block Adaptive
- Model Fixed
- Model Adaptive
plus the normal one-pass baseline.
The first major question is therefore not “Does full ARC work?” but:
> **Which recurrence scale, if any, provides useful additional computation, and does dynamic allocation improve on predetermined allocation?**
---
## 6. Fixed versus adaptive recurrence
Every recurrence mechanism must have both a predetermined and a dynamic version.
### Fixed recurrence
The loop count is determined before inference.
For example:
$$
T=4
$$
means the model always executes four recurrent iterations for that mechanism.
Fixed recurrence provides a controlled baseline for asking whether recurrence itself helps.
### Adaptive recurrence
The model dynamically decides whether additional computation is useful.
Conceptually:
$$
\text{HALT or CONTINUE}
$$
The controller can use signals such as:
- query representation,
- current hidden state,
- output distribution,
- output distribution change,
- hidden-state change,
- remaining compute budget,
- previous computation decisions.
The adaptive model should not simply learn “hard query = many loops.” It should learn whether another computation step is useful **given the current state**.
---
## 7. Fixed recurrence should include uniform and heterogeneous controls
A fixed loop count does not necessarily mean every layer receives the same number of executions.
Two fixed configurations should eventually be distinguished.
### Uniform fixed recurrence
Every applicable unit receives the same number of repetitions:
$$
[2,2,2,2,2,\ldots]
$$
### Predetermined heterogeneous recurrence
Different units receive different predetermined loop counts:
$$
[1,2,4,1,2,1,3,\ldots]
$$
This distinction is important because an adaptive system could outperform uniform recurrence simply because **non-uniform allocation is better**, not necessarily because dynamic decision-making is better.
The heterogeneous fixed baseline therefore becomes an important control before the final adaptive system is evaluated.
---
## 8. Initial model families
### Version A — Baseline
Normal pretrained MoE Transformer.
```plain text
L1 → L2 → L3 → ... → LN
```
One model execution.
Purpose:
- establish baseline quality,
- establish baseline FLOPs,
- establish baseline latency,
- establish normal MoE routing behavior.
### Version B — Fixed layer recurrence
Repeat individual layers using predetermined loop counts.
Example:
```plain text
L1 ×1
L2 ×2
L3 ×2
L4 ×1
L5 ×3
...
```
Initial experiments should include uniform counts such as:
$$
T_l\in\{1,2,3,4\}
$$
followed by selected heterogeneous fixed patterns.
Purpose:
- determine whether layer recurrence helps,
- identify which layers benefit from repetition,
- measure diminishing returns,
- determine whether useful recurrence is uniform or localized.
### Version C — Adaptive layer recurrence
Each applicable layer can dynamically determine whether another execution is useful.
```plain text
Layer
  ↓
Execute
  ↓
Local controller
  ├── HALT → continue forward
  └── CONTINUE → execute layer again
```
Purpose:
- determine whether local adaptive allocation improves quality/compute,
- identify whether different examples require different recurrent layer patterns.
### Version D — Fixed block recurrence
Repeat blocks using predetermined counts.
Example:
```plain text
B1 ×1
B2 ×2
B3 ×2
B4 ×1
```
Purpose:
- establish the independent contribution of block recurrence,
- compare uniform and heterogeneous block repetition.
### Version E — Adaptive block recurrence
Each block can dynamically decide whether to execute again.
Purpose:
- test dynamic local computation at block granularity,
- determine whether block-level recurrence provides a better tradeoff than layer-level recurrence.
### Version F — Fixed model recurrence
Repeat the entire model a predetermined number of times:
```plain text
Model loop 1
    L1 → L2 → ... → LN
Model loop 2
    L1 → L2 → ... → LN
...
```
Test:
$$
T_m\in\{1,2,3,4,6,8,\ldots\}
$$
subject to compute availability.
The MoE router runs once during each model loop.
Purpose:
- establish whether complete recurrent computation phases help,
- measure the quality/compute curve for model recurrence.
### Version G — Adaptive model recurrence
After each complete model execution, a global controller decides:
$$
\text{HALT or CONTINUE}
$$
The MoE router runs once for each model execution.
Purpose:
- determine whether the model can learn when another complete computation phase is useful,
- determine whether different examples naturally receive different numbers of model loops.
---
## 9. The first critical comparison matrix
<table header-row="true">
<tr>
<td>Variant</td>
<td>Layer loops</td>
<td>Block loops</td>
<td>Model loops</td>
<td>Adaptive?</td>
</tr>
<tr>
<td>Baseline</td>
<td>1</td>
<td>1</td>
<td>1</td>
<td>No</td>
</tr>
<tr>
<td>Layer ×2</td>
<td>2</td>
<td>1</td>
<td>1</td>
<td>No</td>
</tr>
<tr>
<td>Layer ×4</td>
<td>4</td>
<td>1</td>
<td>1</td>
<td>No</td>
</tr>
<tr>
<td>Adaptive Layer</td>
<td>Dynamic</td>
<td>1</td>
<td>1</td>
<td>Layer</td>
</tr>
<tr>
<td>Block ×2</td>
<td>1</td>
<td>2</td>
<td>1</td>
<td>No</td>
</tr>
<tr>
<td>Block ×4</td>
<td>1</td>
<td>4</td>
<td>1</td>
<td>No</td>
</tr>
<tr>
<td>Adaptive Block</td>
<td>1</td>
<td>Dynamic</td>
<td>1</td>
<td>Block</td>
</tr>
<tr>
<td>Model ×2</td>
<td>1</td>
<td>1</td>
<td>2</td>
<td>No</td>
</tr>
<tr>
<td>Model ×4</td>
<td>1</td>
<td>1</td>
<td>4</td>
<td>No</td>
</tr>
<tr>
<td>Adaptive Model</td>
<td>1</td>
<td>1</td>
<td>Dynamic</td>
<td>Model</td>
</tr>
</table>
The exact loop counts should be adjusted to the base model and available compute.
The important property is that every recurrence type has a fixed family and an adaptive family.
---
## 10. Equal-compute evaluation
Raw benchmark performance is insufficient.
A system that uses twice the computation should not automatically be considered better because it obtains a higher score.
The primary evaluation should be:
$$
\boxed{\text{quality versus actual compute}}
$$
Actual compute should be measured rather than inferred from nominal loop counts.
Important metrics include:
- total FLOPs,
- active expert FLOPs,
- average number of recurrent executions,
- average model loops,
- average block loops,
- average layer loops,
- tokens processed,
- wall-clock latency.
The key comparison is whether one system achieves a given quality at less compute than another.
---
## 11. Quality-compute Pareto frontier
Each model family should be evaluated across a range of compute budgets or recurrence counts.
Conceptually:
```plain text
Quality
  │
  │                         ●
  │                  ● ARC/adaptive
  │             ● fixed recurrence
  │        ●
  │   ● baseline
  └────────────────────────────── Compute
```
The strongest result is not necessarily the highest absolute quality.
The strongest result would be a model that moves the Pareto frontier outward:
$$
Q_{ARC}(C) > Q_{baseline}(C)
$$
for a meaningful range of compute budgets $`C`$.
---
## 12. Research questions
### RQ1 — Does recurrence help at all?
Does repeatedly executing computation improve performance compared with a normal one-pass pretrained MoE Transformer?
This should be tested separately for layer, block, and model recurrence.
### RQ2 — Which recurrence scale is most useful?
How do:
$$
\text{layer recurrence}
\quad vs.\quad
\text{block recurrence}
\quad vs.\quad
\text{model recurrence}
$$
compare when evaluated at approximately equal compute?
### RQ3 — Does adaptivity improve over fixed computation?
For each recurrence mechanism:
$$
\text{fixed} \quad vs.\quad \text{adaptive}
$$
Does dynamic allocation provide a better quality/compute tradeoff?
### RQ4 — Is heterogeneous allocation better than uniform allocation?
Does a predetermined non-uniform recurrence pattern outperform uniform repetition at similar compute?
This separates the value of **allocation** from the value of **adaptivity**.
### RQ5 — Are the three recurrence mechanisms complementary?
After isolated experiments, does combining mechanisms produce more than the sum of their independent benefits?
### RQ6 — Where should additional computation be spent?
Do different layers or blocks have different marginal computational value?
For example:
$$
L_1\times1,\quad L_2\times1,\quad L_3\times3,\quad L_4\times1
$$
versus a uniform allocation.
### RQ7 — Are pretrained layers and blocks recurrence-compatible?
Do repeated applications produce useful progressive refinement, or does performance degrade because the recurrent input distribution differs from the distribution seen during pretraining?
This is a fundamental diagnostic question.
### RQ8 — Do different examples require different computation?
Does the adaptive system allocate different amounts of computation to different examples?
If every example receives approximately the same computation, the controller may not be providing meaningful adaptive behavior.
### RQ9 — Does computation correlate with problem difficulty?
Do harder examples systematically receive more computation?
This is a useful sanity check, but not itself proof of useful adaptive computation.
### RQ10 — Are there diminishing returns?
Measure the marginal quality improvement from additional computation:
$$
\frac{\Delta Q}{\Delta C}
$$
The expected pattern is diminishing returns:
```plain text
additional computation 1 → large benefit
additional computation 2 → smaller benefit
additional computation 3 → smaller benefit
...
additional computation n → negligible benefit
```
A successful adaptive controller should tend to stop near the point where additional computation has low marginal value.
### RQ11 — Does model recurrence compensate for local recurrence, and vice versa?
Can global model loops achieve most of the benefits of local layer/block recurrence?
Conversely, can local recurrence provide the benefits of another complete model traversal?
### RQ12 — How does MoE routing behave under recurrence?
Does recurrent computation naturally change expert assignments because the hidden state changes?
This is observational rather than an optimization objective.
### RQ13 — How much adaptation is required?
Compare:
- frozen pretrained model,
- LoRA/adapters,
- router adaptation,
- expert adaptation,
- combinations,
- potentially full fine-tuning if resources permit.
### RQ14 — Does adaptive recurrence improve quality-per-compute rather than merely quality?
The central success criterion is not simply:
$$
Q_{ARC}>Q_{baseline}
$$
but rather:
$$
\boxed{Q_{ARC}(C)\text{ is better at a given compute budget }C}
$$
---
## 13. Recurrence compatibility experiment
Before investing heavily in adaptive controllers, directly test whether repeated pretrained computation produces useful trajectories.
For a layer or block, observe:
$$
 h^{(0)}
\rightarrow
 h^{(1)}
\rightarrow
 h^{(2)}
\rightarrow
 h^{(3)}
$$
Measure whether repeated computation produces:
- decreasing validation loss,
- improving task performance,
- stable hidden-state trajectories,
- useful output refinement,
- diminishing returns,
- eventual degradation.
This experiment distinguishes two very different failure modes:
1. **Recurrence itself is not useful for the pretrained model.**
2. **Recurrence is useful, but the adaptive controller is not learning to exploit it.**
This distinction should be preserved throughout the project.
---
## 14. Recurrence placement experiments
For a fixed additional compute budget, test where the computation is placed.
For example, compare approximately equal-cost configurations such as:
```plain text
Configuration A
L1 ×2
L2 ×2

Configuration B
L5 ×2
L6 ×2

Configuration C
L3 ×4

Configuration D
L2 ×1
L3 ×2
L4 ×1
```
The purpose is to determine whether the marginal value of computation depends on location.
If it does, this provides evidence against the assumption that all depth is computationally interchangeable.
---
## 15. Adaptive controller design
The controller should be small relative to the base model.
The initial controller can receive a compact state representation containing some combination of:
- query representation $`q`$,
- current hidden-state summary,
- output logits or compact output statistics,
- entropy,
- output distribution change,
- hidden-state change,
- previous controller decision,
- recurrence count,
- remaining compute budget.
The controller should output a decision such as:
$$
\text{HALT}\quad /\quad\text{CONTINUE}
$$
or, in a more advanced formulation, a probability of continuing.
The query should act as an initial prior rather than a complete computation schedule:
$$
q=f(x)
$$
followed by state-dependent allocation:
$$
\text{decision}=f(q,h_t,\text{progress},\text{budget})
$$
The goal is not to learn a simplistic rule such as “hard query → maximum loops.”
---
## 16. Progress signals
Potential controller inputs include:
### Output distribution
$$
p_t=softmax(z_t)
$$
### Output distribution change
$$
D_t=D(p_t,p_{t-1})
$$
### Entropy
$$
H(p_t)=-\sum_i p_{t,i}\log p_{t,i}
$$
### Hidden-state change
$$
\Delta h_t=1-\cos(h_t,h_{t-1})
$$
### Training loss improvement
$$
\Delta L_t=L_t-L_{t+1}
$$
These should be treated as features rather than perfect measures of progress.
For example, a model can be confidently wrong while showing very little output change. Likewise, a large hidden-state change does not necessarily imply useful improvement.
The controller should ultimately be trained against the actual task objective and compute cost.
---
## 17. Compute budget
ARC should explicitly model computation as a limited resource.
Let:
$$
B_0=\text{available compute budget}
$$
and each operation consume:
$$
C_t=\text{cost of computation }t
$$
Then:
$$
B_{t+1}=B_t-C_t
$$
The controller should learn something related to:
$$
\text{expected quality improvement per unit compute}
$$
rather than simply maximizing loop count.
The desired behavior is approximately:
```plain text
Easy:
    minimal computation

Normal:
    moderate computation

Difficult:
    substantially more computation

Very difficult:
    approach the allowed maximum
```
But this behavior must be measured rather than assumed.
---
## 18. Hard limits
Every recurrence mechanism must have a hard maximum.
Define:
$$
T_{layer,max},\quad
T_{block,max},\quad
T_{model,max}
$$
and an overall compute budget.
Execution must terminate when any relevant constraint is reached.
Conceptually:
```plain text
if controller says HALT:
    stop this recurrence
elif recurrence_count == maximum:
    stop this recurrence
elif compute_budget exhausted:
    stop computation
else:
    continue
```
This guarantees bounded inference cost.
---
## 19. Training objectives
The base objective remains language modeling:
$$
L_{LM}
$$
Potential recurrent-state supervision can be added where useful:
$$
L=\sum_t\lambda_tL_t
$$
A compute-aware objective can be expressed as:
$$
L=L_{LM}+\lambda C
$$
where $`C`$ is actual computation.
The compute penalty must be treated as a tunable experimental variable. If it is too strong, the controller may learn to halt too early. If it is too weak, the controller may learn to use maximum computation.
Therefore, sweep multiple values of:
$$
\lambda
$$
rather than selecting one value without comparison.
Potentially, distillation can encourage earlier recurrent states to approximate stronger later states:
$$
L_{distill}=D(p_t,p_T)
$$
This may help create progressively useful intermediate states.
---
## 20. Training strategy
The research should initially minimize the amount of trainable material.
### Stage 0 — Baseline reproduction
Establish:
- normal pretrained inference,
- validation loss,
- perplexity,
- task accuracy,
- baseline FLOPs,
- latency,
- MoE expert utilization.
### Stage 1 — Fixed layer recurrence
Test:
$$
T_l=1,2,3,4
$$
and selected heterogeneous patterns.
Goal: determine whether layer recurrence is useful at all and identify where it helps.
### Stage 2 — Adaptive layer recurrence
Introduce a local controller.
Goal: determine whether dynamic layer allocation improves over fixed layer allocation at equal compute.
### Stage 3 — Fixed block recurrence
Test multiple predetermined block-loop counts and heterogeneous patterns.
Goal: establish the independent value of block recurrence.
### Stage 4 — Adaptive block recurrence
Introduce dynamic block decisions.
Goal: test adaptive computation at block granularity.
### Stage 5 — Fixed model recurrence
Test:
$$
T_m=1,2,3,4,6,8,\ldots
$$
within resource limits.
The MoE router runs once per model loop.
Goal: establish the model-recurrence quality/compute curve.
### Stage 6 — Adaptive model recurrence
Introduce global HALT/CONTINUE after each complete model execution.
Goal: determine whether the model can learn when another global computation phase is useful.
### Stage 7 — Compare the three recurrence scales
Compare:
$$
L_A\quad vs.\quad B_A\quad vs.\quad M_A
$$
and their fixed counterparts at matched compute.
Goal: identify the strongest recurrence scale and determine whether different scales have distinct behavior.
### Stage 8 — Pairwise combinations
Test:
- Layer + Block,
- Layer + Model,
- Block + Model.
Each combination should have fixed and adaptive variants where feasible.
Goal: determine whether recurrence mechanisms are complementary.
### Stage 9 — Full three-scale recurrence
Combine:
- layer recurrence,
- block recurrence,
- model recurrence.
Initially test fixed combinations, then adaptive combinations.
Goal: determine whether hierarchical recurrence provides a further quality/compute improvement.
### Stage 10 — Mature ARC
The final architecture can combine:
- query-conditioned initial allocation,
- adaptive layer recurrence,
- adaptive block recurrence,
- adaptive model recurrence,
- compute budgeting,
- hard maximums,
- ordinary MoE routing once per model loop.
The mature system should only retain recurrence mechanisms that the experiments justify.
---
## 21. Combination matrix
After the independent studies, construct the following hierarchy:
<table header-row="true">
<tr>
<td>Combination</td>
<td>Fixed</td>
<td>Adaptive</td>
</tr>
<tr>
<td>Layer</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td>Block</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td>Model</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td>Layer + Block</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td>Layer + Model</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td>Block + Model</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<td>Layer + Block + Model</td>
<td>✓</td>
<td>✓</td>
</tr>
</table>
This makes it possible to distinguish:
- recurrence effects,
- adaptive effects,
- interaction effects,
- hierarchical effects.
---
## 22. Measuring synergy
Suppose layer recurrence provides improvement $`A`$, block recurrence provides improvement $`B`$, and their combination provides $`AB`$.
The research should investigate whether:
$$
AB>A+B
$$
under an appropriate normalized comparison.
More rigorously, compare quality/compute frontiers rather than raw scores.
A combination is particularly interesting if it reaches a quality level that neither component can reach efficiently at the same compute budget.
---
## 23. Oracle and average-compute controls
Adaptive computation requires strong controls.
If an adaptive model averages 2.3 loops per example, compare it against fixed or predetermined systems with approximately the same average compute.
This addresses:
> Is adaptivity itself useful, or is the gain simply caused by giving the model more average computation?
An additional useful control is an oracle allocation where computation is allocated using information unavailable to the inference controller but available for analysis. This can estimate the theoretical headroom of better allocation.
---
## 24. What to measure
Every experiment should log at least the following.
### Performance
- validation loss,
- perplexity,
- task accuracy,
- reasoning benchmark performance where appropriate.
### Compute
- total FLOPs,
- active expert FLOPs,
- average compute per example,
- maximum compute,
- average model loops,
- average block loops,
- average layer loops,
- tokens processed,
- wall-clock latency.
### Allocation
- loop count by layer,
- loop count by block,
- model-loop count,
- computation per example,
- computation distribution across examples,
- computation distribution across depth.
### Convergence / refinement
- output distribution change,
- hidden-state change,
- loss improvement,
- marginal quality improvement,
- halt decisions.
### MoE behavior
- expert assignments,
- expert utilization,
- load balance,
- active experts per routing event,
- routing changes between model loops.
Expert diversity is observational only.
---
## 25. MoE-specific experimental principle
The project should not optimize for expert diversity.
The router should be allowed to discover whatever routing behavior is useful.
For example, these are all valid outcomes:
```plain text
Model loop 1 → Experts A, B
Model loop 2 → Experts A, B
Model loop 3 → Experts A, B
```
or:
```plain text
Model loop 1 → Experts A, B
Model loop 2 → Experts A, C
Model loop 3 → Experts D, C
```
The second pattern may be scientifically interesting because the recurrent hidden state changes, but the first pattern is not a failure.
The research question is whether routing improves computation—not whether routing becomes more diverse.
---
## 26. Model variants and training regimes
The research should eventually vary the amount of adaptation applied to the pretrained MoE.
### Variant 1 — Frozen base model
No base-model parameters are updated.
Purpose: determine whether recurrence can work directly on a pretrained model.
### Variant 2 — LoRA / lightweight adapters
Train a small number of additional parameters.
Purpose: test whether recurrence compatibility can be learned cheaply.
### Variant 3 — Controller-only training
Keep the base model fixed and train the recurrence controller.
Purpose: isolate the value of computation allocation.
### Variant 4 — Router adaptation
Allow MoE routing behavior to adapt while keeping most other parameters fixed.
Purpose: determine whether recurrence benefits from recurrence-aware routing.
### Variant 5 — Expert adaptation
Allow selected expert parameters to adapt.
Purpose: determine whether experts need to become more recurrence-compatible.
### Variant 6 — Full adaptation
If resources permit, fine-tune the complete model.
Purpose: establish an upper bound on recurrence-aware training.
These should be introduced only after the fixed/adaptive recurrence phenomena are established.
---
## 27. Expected failure modes
ARC should explicitly test for failure rather than interpreting every result as evidence for the hypothesis.
### Failure mode 1 — Repetition provides no benefit
Repeated computation may simply waste FLOPs.
### Failure mode 2 — Repetition helps initially but then degrades
A recurrent trajectory may have an optimal depth beyond which representations become unstable or less useful.
### Failure mode 3 — Maximum computation wins
The controller may fail to learn meaningful stopping behavior and simply use its full budget.
### Failure mode 4 — Premature halting
A strong compute penalty may cause the controller to stop before useful refinement occurs.
### Failure mode 5 — All examples receive the same computation
The adaptive controller may collapse to an almost fixed policy.
### Failure mode 6 — Recurrence is useful only after training
This would suggest that pretrained models are not naturally recurrence-compatible but can be adapted for recurrent inference.
### Failure mode 7 — One recurrence scale dominates
This is not a failure of the research. It is an important result.
If block recurrence dominates layer and model recurrence, the final ARC system should not artificially preserve the weaker mechanisms.
### Failure mode 8 — Adaptive allocation provides no advantage
If adaptive models perform similarly to fixed models at matched compute, the hypothesis that dynamic allocation is useful is weakened.
---
## 28. Critical ablation philosophy
Every new component should earn its place.
The project should avoid adding multiple mechanisms simultaneously without a corresponding ablation.
For every claimed improvement, ask:
1. Did recurrence cause it?
2. Did adaptivity cause it?
3. Did non-uniform allocation cause it?
4. Did additional compute cause it?
5. Did MoE routing change cause it?
6. Did training adaptation cause it?
7. Is the effect still present at equal compute?
This is especially important because ARC has many interacting variables.
---
## 29. Main experimental progression
The cleanest overall progression is:
$$
\boxed{
\text{Baseline}
\rightarrow
\text{fixed layer}
\rightarrow
\text{adaptive layer}
\rightarrow
\text{fixed block}
\rightarrow
\text{adaptive block}
\rightarrow
\text{fixed model}
\rightarrow
\text{adaptive model}
}
$$
Then:
$$
\boxed{
\text{compare recurrence scales}
\rightarrow
\text{pairwise combinations}
\rightarrow
\text{full combination}
\rightarrow
\text{adaptive full ARC}
}
$$
This progression is deliberately incremental.
---
## 30. Final target architecture
The mature ARC system may eventually resemble:
```plain text
                    QUERY
                      │
                      ▼
            Initial computation prior
                      │
                      ▼
               MODEL LOOP t
                      │
               MoE routing
               (once per model loop)
                      │
     ┌────────────────┼────────────────┐
     ▼                ▼                ▼
  Layer 1          Layer 2          Layer 3 ...
     │                │                │
local policy     local policy     local policy
     │                │                │
   × n₁             × n₂             × n₃
     │                │                │
     └────────────────┼────────────────┘
                      ▼
               Block-level decisions
                      │
                 block recurrence
                      │
                      ▼
               complete model state
                      │
               progress / output
                      │
                      ▼
               Global controller
                 /           \
              HALT         CONTINUE
               │               │
               ▼               ▼
             OUTPUT       MODEL LOOP t+1
```
However, this architecture is a **research endpoint, not an assumption**.
The experiments may show that only one or two recurrence scales are useful. ARC should follow the evidence.
---
## 31. What success would look like
A strong result would not simply be:
> ARC obtains a higher benchmark score.
A convincing result would demonstrate a better quality/compute frontier.
For example:
```plain text
Fixed 1×:
    low compute
    baseline quality

Fixed 4×:
    high compute
    high quality

Adaptive ARC:
    similar quality to fixed 4×
    substantially lower average compute
```
An even stronger result would show differentiated computation:
```plain text
Easy examples:
    few loops
    little additional computation

Medium examples:
    moderate recurrence
    selective depth expansion

Hard examples:
    more computation
    recurrence concentrated in useful layers/blocks

Very hard examples:
    approach maximum allowed compute
```
The strongest evidence would be that these patterns emerge naturally under a compute constraint rather than being hard-coded.
---
## 32. The central scientific claim
ARC should ultimately test the following proposition:
> **Transformer inference can potentially be made more compute-efficient by dynamically allocating recurrent computation at multiple structural scales, while an MoE architecture independently provides conditional width through expert activation.**
The project therefore separates:
$$
\boxed{
\text{Depth allocation}
=
\text{layer + block + model recurrence}
}
$$
from:
$$
\boxed{
\text{Width / activation allocation}
=
\text{MoE routing}
}
$$
The recurrence mechanisms are first tested independently, with both fixed and adaptive variants. Only after their independent effects are established are they combined.
The project should not assume that maximum computation is better, that all layers deserve equal recurrence, that all examples deserve equal computation, or that different recurrent executions should use different experts.
Instead, the central question is:
$$
\boxed{
\text{Can a model learn where, when, and how much computation is worth spending?}
}
$$
---
## 33. Final research roadmap
### Phase 0 — Baseline
- [ ] Reproduce normal pretrained MoE inference.
- [ ] Establish quality metrics.
- [ ] Establish FLOPs and latency.
- [ ] Record MoE routing statistics.
### Phase 1 — Layer recurrence
- [ ] Implement fixed layer recurrence.
- [ ] Test uniform loop counts.
- [ ] Test heterogeneous predetermined patterns.
- [ ] Measure recurrence compatibility.
- [ ] Identify useful layers.
- [ ] Implement adaptive layer recurrence.
- [ ] Compare adaptive versus fixed at equal compute.
### Phase 2 — Block recurrence
- [ ] Define the exact block abstraction.
- [ ] Implement fixed block recurrence.
- [ ] Test uniform loop counts.
- [ ] Test heterogeneous predetermined patterns.
- [ ] Measure recurrence compatibility.
- [ ] Implement adaptive block recurrence.
- [ ] Compare adaptive versus fixed at equal compute.
### Phase 3 — Model recurrence
- [ ] Implement fixed model recurrence.
- [ ] Test multiple model-loop counts.
- [ ] Verify router execution once per model loop.
- [ ] Measure quality versus compute.
- [ ] Implement adaptive model recurrence.
- [ ] Compare adaptive versus fixed at equal compute.
### Phase 4 — Cross-scale comparison
- [ ] Compare layer, block, and model recurrence at matched compute.
- [ ] Identify which recurrence scales are useful.
- [ ] Measure where computation is most valuable.
- [ ] Measure diminishing returns.
### Phase 5 — Pairwise combinations
- [ ] Layer + Block.
- [ ] Layer + Model.
- [ ] Block + Model.
- [ ] Fixed combinations.
- [ ] Adaptive combinations.
- [ ] Equal-compute comparisons.
### Phase 6 — Full ARC
- [ ] Combine layer, block, and model recurrence depending on measurements.
- [ ] Add hierarchical adaptive controllers.
- [ ] Add explicit compute budget.
- [ ] Add hard recurrence limits.
- [ ] Evaluate quality-compute Pareto frontier.
### Phase 7 — Training investigation
- [ ] Frozen pretrained model.
- [ ] Controller-only training.
- [ ] LoRA/adapters.
- [ ] Router adaptation.
- [ ] Expert adaptation.
- [ ] Full adaptation if resources allow.
### Phase 8 — Final evaluation
- [ ] Benchmark performance.
- [ ] Quality/compute frontier.
- [ ] Average and maximum compute.
- [ ] Latency.
- [ ] Computation allocation by example.
- [ ] Computation allocation by depth.
- [ ] MoE routing behavior.
- [ ] Ablation analysis.
- [ ] Failure-mode analysis.
---
## 34. Final design principle
ARC should not prescribe the optimal computational structure.
It should create a system in which different computational structures can be tested and, eventually, selected dynamically.
Do not assume:
- every layer should loop equally,
- every block should loop equally,
- every model should loop the same number of times,
- every difficult query needs more computation everywhere,
- different recurrent executions must use different experts,
- expert diversity is inherently beneficial,
- maximum computation is inherently better,
- adaptive computation is automatically better than fixed computation.
Instead, test these assumptions experimentally.
The final principle is:
$$
\boxed{
\text{ARC dynamically allocates depth through three recurrence scales and width through MoE activation.}
}
$$
The research program exists to determine **which recurrence scales are useful, when they should be combined, whether dynamic allocation is actually superior to predetermined allocation, and whether the resulting system can improve quality-per-compute.**