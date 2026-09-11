# R003 implementation handoff: changing the mixture at fixed gate function

Protocol: `r003_same_function_v1`, 2026-09-11.

Protocol status: frozen v1 handoff. Execution completed on 2026-09-11; see the
canonical [`R003_formal_attempt02`](../../results/R003_formal_attempt02/summary.md)
result. This document specifies the protocol and does not itself supply measured
results.
The machine-readable contract is
[r003_same_function_v1.json](../../configs/experiments/r003_same_function_v1.json).

This handoff was checked against repository commit
`a58d417e7e34a33907d8ae33cbcd22b14af1846d`. Read [AGENTS.md](../../AGENTS.md),
[STATE.md](../../docs/STATE.md), and [SPEC.md](SPEC.md) first. Keep R002 source
artifacts immutable and put R003 outputs in new directories.

## Question and interpretation

Can changing a categorical gate mixture, while preserving its effective truth
table, change the subsequent functional response to a loss-gradient step at
actual R002 checkpoints?

The algebra already permits this. R003 measures the size, direction, numerical
resolvability, and observable consequences of that effect in our recurrent
learner. Three claims must be distinguished:

1. The replacement preserves the present relaxed computation.
2. It changes the induced SGD response in effective truth-table coordinates.
3. That change reaches the observed output after recurrent execution.

A change in a gate-coordinate matrix need not produce a measurable output
change for the current loss and probes. None of these claims establishes better
long-run optimization, naturally occurring motion along fibers, or a causal
explanation of the extracted circuit-size difference in R002.

The current size observation is motivating evidence: under the same constructive
simplifier, update-500 common-hard visible cores span 2–36 binary operations for
categorical coordinates and 44–236 for direct truth coordinates. Those are not
minimum or complete description lengths. A uniform longer-training continuation
remains a separate, unlaunched experiment. R003 neither resumes Adam training nor
changes its duration, objective, or readout contract.

## Explanation to carry into the report

Assume a reader comfortable with linear algebra, probability, and gradients,
but new to differentiable logic-gate networks. Introduce a Boolean circuit,
its trainable gate mixtures, reuse of the circuit over recurrent ticks, and
separate evaluation of the extracted Boolean circuit.

Lead the representation discussion with this example. In the truth-table input
order `00, 01, 10, 11`,

\[
\begin{aligned}
\tfrac12\underbrace{(0,0,0,1)}_{\mathrm{AND}}
+\tfrac12\underbrace{(0,1,1,1)}_{\mathrm{OR}}
&=(0,\tfrac12,\tfrac12,1),\\
\tfrac12\underbrace{(0,0,1,1)}_{\mathrm{copy}\ a}
+\tfrac12\underbrace{(0,1,0,1)}_{\mathrm{copy}\ b}
&=(0,\tfrac12,\tfrac12,1).
\end{aligned}
\]

The two distributions over gates differ, but their mean truth table and
multilinear function, `(a + b) / 2`, agree. Under a sampled AND/OR gate, the
entries at `01` and `10` always agree; under a sampled copy-a/copy-b gate they
always disagree. The map retains their means and discards their dependence.
This is an interpretation of the local categorical distribution, not an exact
ensemble of recurrent Boolean circuit executions.

Now define `T` as the 16-by-4 table matrix and `p` as a column vector of gate
probabilities:

\[
\Phi:\Delta^{15}\to[0,1]^4,\qquad q=\Phi(p)=T^\top p,
\qquad
\mathcal F_q=\{p\ge0:\mathbf1^\top p=1,\ T^\top p=q\}.
\]

The fiber is exactly this preimage: an 11-dimensional convex polytope for
interior q. Logits add one constant-shift gauge dimension. Boundary dimensions
can shrink; the term does not assert a constant-dimensional smooth fiber bundle
over the closed cube. Appendix A contains the full matrix and implementation
conventions; keep it out of the main report exposition.

## Fixed cohort and source resolution

| Item | v1 choice |
| --- | --- |
| Parent experiment | R002 `formal_attempt01` |
| Conditions | `categorical_reference_decay`, `categorical_no_decay` |
| Seeds | Every seed 0 through 15 in each condition |
| Saved optimizer updates k | 0, 250, 500 |
| Logical checkpoint cases | 32 runs × 3 checkpoints = 96 |
| Mixture coordinates alpha | 0, 0.5, 1 |
| Logical arm cases | 288 |
| Independent one-step sizes | Primary eta = 0.001; numerical audit eta = 0.0001 |
| Logical step cases | 576, before failures or duplicate-computation reuse |
| Runtime ticks t | 20; synchronous, 16 × 16 × 8, zero exterior |
| Loss | Terminal visible-channel summed squared error on two fixed inputs |
| Diagnostic optimizer | Plain Euclidean SGD in categorical logits |
| Decay, cost, clipping, momentum, noise | All absent from the diagnostic step |

The conditions label training history. Both receive the same diagnostic with
zero decay. Direct-truth R002 checkpoints are not included: their current q
differs, so they cannot control a same-function intervention. An analytic
direct-truth-coordinate comparison at the identical q is specified below.

Resolve source manifests at the pinned commit using
`results/R002_formal_attempt01_seed{seed:02d}_{condition}/manifest.json`.
Read each `checkpoints` entry for k = 0, 250, 500; verify the selected NPZ against
its manifest SHA-256 and byte length before loading with `allow_pickle=False`.
Check format version 2, update index, condition, seed, representation, config
hash, parameter shapes, and wiring hash. Preserve source-code and dirty-patch
metadata as recorded; do not relabel the old training as clean when its manifest
records untracked results.

Read the actual wiring arrays from each run's hash-verified
`final_common_circuit.npz`, using the existing `wire_{network}_{layer:02d}_{a|b}`
keys. Reconstruct the parameter tree with the existing validated layout and
check the wiring tree hash against checkpoint metadata and manifest pairing.
Do not regenerate wiring or probes from seed numbers under a changed precision
mode. The existing release `experiment/R002` provides artifact recovery if
local files are absent. Historical absolute paths may be relocated by a
declared artifact root; artifact identity remains the hash, not that path.

Use the existing fixed evaluation NPZ, SHA-256
`d49ecf23fd932fba25b1068f8c90820265a05fd90daadc372089520889090b2f`, containing
`inputs` and `target`. Use inputs 0 and 1 for the diagnostic loss and gradient.
Use inputs 0–3 for state-invariance and output-response measurements; indices 2
and 3 are additional observations excluded from the diagnostic loss. This is a
fixed diagnostic subset, not a new estimate of generalization or success rate.

The input ledger should resolve 96 checkpoint references, 32 wiring-artifact
references, and one shared probe artifact, with hashes and provenance. Record
missing or invalid cases without substituting a different seed or checkpoint.
Identical update-0 models can reuse computation if wiring and all diagnostic
inputs also match; retain both condition labels and record the reuse. Gates,
arms, checkpoints, and decay conditions are not independent training trials.

## Precision and numerical eligibility

Run the primary mechanism diagnostic in FP64 in the existing locked modern JAX
environment, with `JAX_ENABLE_X64=1` set before imports. Promote the stored FP32
logits and the already saved inputs/target to FP64; keep integer wiring and
keys unchanged. Assert actual dtypes of logits, p, q, rollout states, gradients,
and JVPs. Use the existing common multilinear kernel and highest dot precision.
Retain the repository's deterministic settings and record them.

This is a declared FP64 re-evaluation of models trained in FP32, not an FP64
training result or a claim of identical numerical trajectories across dtypes.
The preflight separately measures original FP32 versus promoted FP64 outputs
and rounding on its source checkpoints. Those differences must not be counted
as effects of changing the categorical mixture.

Compute p from the original logits by stable softmax and q from p. A gate is
eligible for replacement only if all four entries satisfy
`1e-8 <= q_r <= 1 - 1e-8`, its logits and probabilities are finite, and its
probabilities are strictly positive. Apply this mask from the original FP64
checkpoint once. Leave ineligible gates' logits unchanged in every arm and
record the reason. They remain trainable in the single SGD step in every arm;
their initial coordinate response is consequently shared across arms.

Report eligible counts by layer and checkpoint, plus the fraction of baseline
`sum_i ||v_i||^2` lying on eligible gates, where v is defined below. A tiny
coverage fraction can make a null effect uninformative. Nonfinite baseline
logits, states, or gradients invalidate the checkpoint comparison; leaving a
gate unreplaced does not repair an invalid source model. Do not clamp q, invent
finite logits for zero probabilities, or silently drop a failed replacement.
If an eligible gate fails the numerical checks, retain a failed-case record;
changing eligibility or precision policy requires a versioned protocol revision.

## Replacement construction

For each eligible gate, construct the factorized distribution

\[
p^{\mathrm{fact}}_g
=\prod_{r=1}^4 q_r^{T_{gr}}(1-q_r)^{1-T_{gr}},
\qquad
p^{\alpha}=(1-\alpha)p+\alpha p^{\mathrm{fact}}.
\]

This gives an explicit straight path in probability space with exactly constant
q in real arithmetic. Compute the product using logs and `log1p(-q)`; stable
normalization of those log weights is part of evaluating the defined product
distribution, not a repair of an arbitrary off-fiber proposal. Record its
normalization residual and verify the resulting q. Use alpha = 0, 0.5, 1 for
every eligible gate simultaneously; wiring, inputs, target, state initialization,
parameter sharing, and all other settings stay fixed.

Use the exact original logits for alpha = 0. For alpha > 0, encode each new
distribution as logits with the same per-gate mean as the original:

\[
z_g^\alpha=\log p_g^\alpha+
\left(\overline z-\overline{\log p^\alpha}\right).
\]

Recompute the probabilities and q from these actual encoded logits. Checks
must use those recomputed arrays, not the desired q substituted into a forward
pass. Store maximum changes in normalization, p, q, and logit mean.

**Materialize each arm as independent parameter leaves before differentiation.**
Use a copy/detach or `stop_gradient` at the intervention boundary. Do not
differentiate through the operation that constructs p-fact from the source
checkpoint. Following the reset, all 16 logits per gate are unconstrained SGD
coordinates; the step is not restricted to the factorized family, and the
distribution is not factorized again after the step. This is essential to
testing representative dependence rather than a different four-parameter model.

## Invariance checks before taking any step

For each checkpoint and arm, compare with the original FP64 arm:

- Normalization and nonnegativity for all gates, strict positivity for eligible
  gates, q, and the declared unchanged ineligible gates.
- Every full-state tensor at ticks 0 through 20 on probe inputs 0–3, using
  identical deterministic chunks of two inputs.
- The summed terminal-channel-0 loss on inputs 0 and 1.
- State JVPs of the full terminal state with respect to the initial-state batch
  on inputs 0 and 1, for two fixed directions.
- The gradient of the diagnostic loss with respect to all shared q coordinates.

Construct the state-JVP directions once with NumPy
`Generator(PCG64(20260911))`: draw integers in {0,1} with shape
`(2, 2, 16, 16, 8)` and dtype `int8`, convert to FP64 signs using `2*x - 1`, and
normalize each leading-index batch tensor to unit Euclidean norm. Save and hash
the direction array. The first dimension indexes two directions; the second
indexes the two input grids. These derivatives use the polynomial extension at
the Boolean initial states, which is defined even for perturbations pointing
outside the Boolean cube.

Use elementwise `abs(actual-reference) <= atol + rtol*abs(reference)`:

| Check | atol | rtol |
| --- | ---: | ---: |
| Probability sum versus one | 5e-14 | 0 |
| Effective truth entries | 1e-12 | 1e-10 |
| All recurrent states | 1e-9 | 1e-7 |
| Scalar diagnostic loss | 1e-8 | 1e-7 |
| State JVPs and q gradients | 1e-9 | 1e-6 |
| Local analytic versus autodiff Jacobians | 1e-12 | 1e-10 |
| Full logit chain-rule check in preflight | 1e-9 | 1e-6 |

Report maximum absolute error, relative L2 error when defined, pass/fail, and
the first offending tick/layer when localized. A failed soft-invariance check
invalidates a claim of a controlled same-function comparison for that case;
preserve its diagnostics and continue the cohort ledger without pooling it
with valid cases. Do not loosen tolerances to obtain a pass.

Record native argmax gate IDs with the existing lowest-index tie convention.
Native extraction may legitimately change. Also record common `q >= 0.5` IDs:
exact same-q replacement preserves them mathematically. Any numerical common
ID mismatch must identify the gate, column, both q values, and distance to 0.5.
Flag a mismatch within `1e-10` of the threshold as rounding-ambiguous; do not
claim exact discrete neutrality for it. A mismatch farther from the threshold
is a failed control. Do not snap q to force agreement. No circuit-size decrease
under common rounding is attributable to an exact fiber move alone.

## Geometry, gradient response, and the independent SGD steps

Index distinct shared gate slots by i; there are 3,040, even though a cell applies
more gate instances. Let Q collect their effective truth entries and let

\[
L(Q)=\sum_{b\in\{0,1\},x,y}
\left[S_{20}(Q;X_b)_{xy,0}-Y_{xy,0}\right]^2,
\qquad h_i=\nabla_{q_i}L.
\]

Use the source loss reduction exactly: sum over the two inputs and 256 visible
pixels, without dividing by batch size or pixel count. Parameter sharing and
all 20 recurrent uses participate in differentiation.

For each arm, evaluate these local matrices in FP64:

\[
J_i=\frac{\partial q_i}{\partial z_i}
=T^\top(\operatorname{diag}p_i-p_i p_i^\top),
\quad (J_i)_{rg}=p_{ig}(T_{gr}-q_{ir}),
\quad M_i=J_iJ_i^\top.
\]

The entrywise expression avoids constructing a 16-by-16 matrix. M is a 4-by-4
positive semidefinite SGD response matrix, not a Fisher matrix or an RG flow
constant. Compute each arm's actual h from its recomputed Q, and check it against
the original arm's h as specified above. Record both:

- `v_common_i = -M_i @ h_original_i`, which isolates the coordinate matrix on a
  shared functional gradient.
- `g_i = J_i.T @ h_arm_i` and `v_actual_i = -M_i @ h_arm_i`, which describe the
  step at the actually represented numerical point.

Use `v_common` for between-arm mechanism comparisons and `v_actual` for the
finite-step prediction check. Report their discrepancy. In preflight, compare
g against independent autodiff through the complete logit-to-loss computation;
do not validate only one algebraic formula against a rearrangement of itself.

Report per-gate and per-case Frobenius norms and all four eigenvalues of M,
matrix differences, predicted q-response norms and angles, and

\[
\frac{dL}{d\eta}\bigg|_{0}
=-\sum_i h_i^\top M_i h_i.
\]

This is a response per unit logit-SGD learning rate. The same nominal eta can
produce different functional step sizes; do not normalize the steps to match
their q displacement. Compute the analytic direct-truth comparator
`M_truth = diag((q * (1-q))**2)` at the original Q, with response `-M_truth @ h`.
Label it a counterfactual coordinate comparison, not another trained arm or
an update of the R002 direct-truth runs.

Measure whether coordinate differences reach observable behavior. Define O(Q)
as the terminal visible outputs on inputs 0–3, and compute

\[
w^\alpha=D_Q O(Q^\alpha)\,v^\alpha_{\mathrm{actual}}
\]

with a JVP. Report norms/angles separately for loss inputs 0–1 and additional
inputs 2–3. This avoids equating a change in an internal coordinate response
with a change in the visible output. Use the common Q and `v_common` for a
second, shared-point output-response comparison; reuse the calculation when
the arrays are identical rather than performing duplicate work.

For each arm and each declared eta, start again from the untouched cloned arm:

\[
z_+^\alpha=z^\alpha-\eta g^\alpha,
\qquad
Q_+^\alpha-Q^\alpha=\eta v^\alpha_{\mathrm{actual}}+O(\eta^2).
\]

Evaluate the new q, visible outputs on inputs 0–3, diagnostic loss, and native
and common gate IDs. Compare actual q/output/loss changes with their first-order
predictions. Measure changes from each arm's own pre-step arrays so encoding
roundoff is not mislabeled as an SGD effect. The 0.0001 step is an independent
numerical audit, not a second training step or an opportunity to choose the
better outcome. Report absolute and relative linearization residuals for both
step sizes; failure to enter a linear regime is an observation, not permission
to tune eta after inspecting the outcome.

Use no Adam moments, decay, regularization, clipping, stochastic update masks,
or new training samples. Read and preserve checkpoint optimizer/RNG metadata,
but never consume or overwrite that state. Ordinary Euclidean loss gradients
are orthogonal to the exact logit-fiber tangent; R003 explicitly changes the
representative rather than claiming that SGD naturally explores that tangent.

## Controls and preflight

Add focused tests for mechanisms that could invalidate the experiment:

1. All 16 truth tables and their multilinear extensions agree with the source
   gate algebra; verify rank 5 of `[ones; T.T]`.
2. The AND/OR and copy-a/copy-b example has identical q. For finite-logit tests,
   mix each distribution with 20% uniform mass over all 16 gates. Both then have
   strictly positive p and q = `(0.1, 0.5, 0.5, 0.9)`. Check equality of the soft
   gate, different local J/M, and agreement with independent autodiff.
3. Identity/round-trip and `z -> z + 1` gauge controls preserve q, M, and the
   predicted functional step. Factorizing an already factorized distribution
   is an identity control. An off-fiber perturbation must be detected.
4. A gate with h = 0 contributes zero v even if its M changes; zero norms produce
   explicit undefined-angle/ratio statuses. Do not insert arbitrary epsilons
   to turn a zero or unresolved denominator into an effect size.
5. Eligibility boundaries, underflow/nonfinite inputs, and common-rounding ties
   follow the declared policy. No clipping or automatic outcome-based retry.
6. Replacement is detached, and a subsequent step has 16 independent logit
   coordinates. The unchanged original checkpoint/optimizer/key hashes survive
   the diagnostic. Round-trip artifact loading retains dtype and shape.

The recurrent preflight uses seed 0, both categorical histories, updates 0, 250,
and 500: six logical checkpoints. Run all three arms and both independent step
sizes. Include the full chain-rule check and the separate FP32-to-FP64 source
comparison here. The synthetic and recurrent numerical controls must pass before
launching the full cohort; a null geometry or output effect is not a failed
implementation check. Failed controls should be diagnosed and recorded under a
new attempt; scientific protocol changes require a new version. Routine code
fixes do not erase the original failed record.

## Analysis and deliverables

Report the original-to-factorized contrast as primary, with the halfway arm
showing how the response changes along a known constant-q path. Summarize each
condition and saved update separately. Useful bounded effect sizes are
`||M_fact-M_original||_F / max(||M_fact||_F, ||M_original||_F)` and the analogous
relative response difference; use null plus a reason when both norms vanish.
Always retain absolute effects, coverage, and numerical residuals alongside
relative values. Distinguish structural rank from numerical rank near saturation.

Aggregate gate measurements within a checkpoint before summarizing over seeds.
Treat k = 0, 250, 500 as repeated observations, and pair both training histories
by seed. Show all 16 seed values rather than presenting thousands of gates as
independent evidence. This diagnostic need not force a significance test onto
a mathematically known possibility. Report what is resolved on these checkpoints
and probes, including any lack of observable effect.

Suggested small committed outputs:

- `results/R003_formal_attempt01/manifest.json`: protocol/config/code/lock hashes,
  actual backend and hardware, source ledger, status, failures, deduplication,
  wall/compile times, peak memory, and artifact identities.
- `validation.json`, `cases.jsonl`, and `summary.md`: all control outcomes,
  counts for the 96/288/576 logical cases, per-case effects, and conclusions.
- Compact SVG figures: invariance residuals versus effect sizes; matrix and
  response changes by k and history; predicted versus observed q/output steps;
  eligible coverage and gate response by layer. State which are diagnostic
  observations rather than results of additional training.

Store derived arrays under `artifacts/R003_<run_id>/` with SHA-256 and byte
lengths. Retain h, eligibility, per-arm q/response vectors, measured post-step
q/output changes, state-JVP directions, and sufficient scalar diagnostics to
reconstruct the reported comparisons from source logits. J and M can be
recomputed from saved p or logits; avoid storing redundant dense Jacobians or
duplicate Adam states. Retain full state comparisons for failed cases and a
declared seed-0 review subset. Produce a compact review archive with all small
results, the six preflight source/checkpoint references and required arrays,
and a manifest; use the existing release-artifact conventions for large files.

The report should answer, in order: did neutrality hold numerically; how much
did the coordinate response change; did that reach the observed outputs; did
the finite step agree with the local prediction? Native argmax changes are a
separate extraction observation. Do not call this improved training, a size
optimization experiment, or a proof of naturally traversed neutral paths.

## Implementation surfaces and execution contract

Reuse `recurrent_circuit_learning/r002.py` for truth-table ordering, checkpoint
format, common q kernel, loss convention, and hardening. Add an isolated
`recurrent_circuit_learning/r003.py` and `scripts/r003.py` plus focused tests;
preserve R000–R002 behavior. Routine module organization is discretionary.

Implement `preflight` and `run` commands accepting the frozen config, an
explicit run ID, and optional artifact-root relocation. The following commands
are the CLI contract to implement; they are not runnable until that code exists:

```bash
JAX_ENABLE_X64=1 uv run --project envs/jax-gpu --locked python scripts/r003.py preflight --config configs/experiments/r003_same_function_v1.json --run-id preflight_attempt01
JAX_ENABLE_X64=1 uv run --project envs/jax-gpu --locked python scripts/r003.py run --config configs/experiments/r003_same_function_v1.json --run-id formal_attempt01
```

Use one process initially. Preserve FP64 and the scientific batch/loss contract
when choosing memory management or compilation strategy. A CPU FP64 fallback
is a recorded runtime adaptation requiring the same controls; do not silently
downgrade the formal mechanism diagnostic to FP32. The FP32 source comparison
can run in a separate process in the same locked environment.

The implementation partner should verify the source ledger, implement and run
the focused preflight, then proceed to the fixed cohort when it passes. Record
status and results truthfully, commit small reviewable outputs, and publish
large artifacts with the established manifest/hash workflow. No additional
training continuation, optimizer intervention, learned wiring, or counter task
is included in this handoff.

## Appendix A: full truth-table mapping

Gate indices g = 0–15 follow the binary value of the four output bits. Columns
are input pairs `00, 01, 10, 11`; for example g = 1 is AND, 3 copies a, 5 copies
b, 6 is XOR, and 7 is OR.

\[
T=\begin{pmatrix}
0&0&0&0\\
0&0&0&1\\
0&0&1&0\\
0&0&1&1\\
0&1&0&0\\
0&1&0&1\\
0&1&1&0\\
0&1&1&1\\
1&0&0&0\\
1&0&0&1\\
1&0&1&0\\
1&0&1&1\\
1&1&0&0\\
1&1&0&1\\
1&1&1&0\\
1&1&1&1
\end{pmatrix},\quad
p=\begin{pmatrix}p_0\\p_1\\\vdots\\p_{15}\end{pmatrix},\quad
q=\begin{pmatrix}q_{00}\\q_{01}\\q_{10}\\q_{11}\end{pmatrix}=T^\top p.
\]

The 16 entries of p are nonnegative and sum to one. The four q entries are
output values and need not sum to one. For fractional inputs the gate is

\[
\widetilde f_q(a,b)=q_{00}(1-a)(1-b)+q_{01}(1-a)b
+q_{10}a(1-b)+q_{11}ab.
\]

The mathematical convention uses column vectors. The implementation stores
each gate's probabilities along the last array axis, so its equivalent operation
is `q = p @ T`, with leading axes indexing shared gate slots. Holding all these
q values fixed preserves the full relaxed recurrent rule in exact arithmetic.

## Appendix B: source anchors

- [Original baseline and dimensionality handoff](../../docs/RECURRENT_BASELINE_HANDOFF.md)
- [R002 implementation](../../recurrent_circuit_learning/r002.py) and
  [runner](../../recurrent_circuit_learning/r002_runner.py)
- [R002 frozen config](../../configs/experiments/r002_paired_gate_coordinates.json)
- [Independent trajectory review](../../reviews/R002_formal_attempt01/REVIEW.md)
- [Independent runtime and structure review](../../reviews/R002_runtime_profile/REVIEW.md)
- [Artifact release](https://github.com/trentonstrong/recurrent-circuit-learning-exps/releases/tag/experiment/R002)

This document specifies new work. Its equations and synthetic examples are
derivations; its R002 observations refer to the recorded prior results. They
must not be presented as completed R003 measurements.
