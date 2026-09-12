# R005 implementation handoff: stochastic neutral exploration

Status: specified for implementation, 2026-09-12. Read [AGENTS.md](../../AGENTS.md),
[STATE.md](../../docs/STATE.md), and [SPEC.md](SPEC.md). This document and the
[v1 configuration](../../configs/experiments/r005_stochastic_neutral_exploration_v1.json)
define new work; no runner, GPU timing, or recurrent intervention result is
claimed. The prior [proposal](../../docs/proposals/neutral_exploration/PROPOSAL.md)
contains the derivation and completed NumPy algebra checks.

## Question and scope

Does a stochastic rule that favors locally predicted response rotation produce
more varied subsequent functional motion than uniform neutral exchanges, at
equal probability displacement? Does that variation reach outputs beyond those
used for the loss? How much GPU update time does the operation add?

R003 already demonstrated representative-dependent response. Merely finding
another nonzero response difference is insufficient for the new comparison.
Distinguish stochastic spread, mean drift, and response magnitude. Greater
spread is an exploration observation; it need not improve optimization.
The local score predicts squared displacement from the current direction,
which includes drift. Whether it also increases centered output dispersion is
an empirical question, not guaranteed by the scoring formula.

Use original R002 checkpoints, not factorized R003 descendants or R004 states.
The labels with/without decay describe their training histories. Every R005
task step uses plain Euclidean logit SGD with zero decay, momentum, clipping,
description cost, and extra functional noise. Saved Adam moments and keys are
not consumed. The formula below does not describe an AdamW intervention.

## Frozen source and cohort

Pin source resolution and reference semantics to repository commit
`35bea6cf9a84d2ca267bd1d83d8430b81f4f7acb`. Use the source/checkpoint loader,
truth ordering, common multilinear kernel, and numerical controls of
[R003](../R003/HANDOFF.md), with an R005 adapter. Preserve R003's validator;
do not relabel this configuration to bypass it. Record hashes of reused or
adapted implementation files and any discrepancy from the pin.

| Item | Fixed v1 choice |
| --- | --- |
| Parents | R002 `formal_attempt01`, categorical_reference_decay and categorical_no_decay |
| Training seeds | All 0–15 |
| Saved updates | 0, 250, 500: 96 logical checkpoint cases |
| Runtime | 20 synchronous ticks; 16 × 16 grid; eight channels; zero exterior |
| Gates | 3,040 shared LUT2 slots; fixed wiring |
| Loss | Summed terminal channel-0 squared error on fixed inputs 0 and 1 |
| Observations | Terminal channel 0 on fixed inputs 0–3; primary response on 2 and 3 |
| Arithmetic | Saved FP32 logits promoted to FP64; highest dot precision |
| Task steps | One step per clone, independently at eta = 0.001 and 0.0001 |

Resolve the 96 checkpoint and 32 wiring references from the pinned R002 run
manifests. Verify byte lengths, SHA-256, checkpoint format/version, parameter
layout, seed, condition, update, config identity, and wiring. Load saved arrays;
do not reconstruct weights or wiring from a seed. Follow R003's artifact-root
relocation convention. The probe artifact is
`artifacts/R001_20260909T231016Z_seed23_jaxgpu_attempt01/fixed_evaluation_set.npz`,
SHA-256 `d49ecf23fd932fba25b1068f8c90820265a05fd90daadc372089520889090b2f`.
Inputs 2 and 3 have been observed in earlier diagnostics; call them additional
observation inputs, not a newly unseen test set.

Freeze eligibility from the source FP64 case: finite logits, strictly positive
finite probabilities, and every effective truth entry in [1e-8, 1-1e-8]. Leave
ineligible logits unchanged in every neutral arm; they remain trainable during
SGD. This conservative inherited mask is an experimental choice, not the
mathematical limit of sparse exchanges. Report gate counts and the fraction of
baseline sum_i ||v_i||² on eligible gates. Nonfinite source computation invalidates
the case; masking does not repair it. Keep all scheduled identities and failures.

## Operator and probability law

For one gate, T has shape 16 × 4, with rows in gate-ID order and columns
00, 01, 10, 11. Let

\[
p=\operatorname{softmax}(z),\quad q=T^\top p,\quad
A=\begin{pmatrix}\mathbf1^\top\\T^\top\end{pmatrix}.
\]

Construct the 24 square-face directions of the four-dimensional truth-table
cube. Enumerate varying-position pairs (a,b), a<b, lexicographically; enumerate
the fixed bits on the remaining increasing positions as 00,01,10,11. Assign
signs (+1,-1,-1,+1) to varying bits 00,01,10,11. Every direction d has four
nonzeros, ||d||₂=2, and Ad=0. Their span has rank 11. For example,
`p_AND += delta; p_OR += delta; p_copy_a -= delta; p_copy_b -= delta` is neutral.

For each direction,

\[
\ell=-\min_{g:d_g=+1}p_g,\qquad
r=\min_{g:d_g=-1}p_g,\qquad
\delta=\rho[\ell+U_\delta(r-\ell)],\quad \rho=0.1.
\]

U_delta is uniform on [0,1). Thus p'=p+delta*d stays strictly positive for
positive p in exact arithmetic. Preserve total mass and q algebraically; do
not clip, renormalize, or project q. Encode the new independent logits as
`log(p') + mean(z) - mean(log(p'))`, then stop differentiation through the
intervention. Zero-delta and ineligible gates retain their original logits exactly.

Let h=grad_q L include every use of the shared gate in the recurrent graph.
Define c_g=T_g-q and

\[
J_{rg}=p_g c_{gr},\quad M=JJ^\top,\quad v=-Mh,\quad
u=v/\|v\|,\quad B_{:g}=-2p_g c_g(c_g^\top h).
\]

Holding q and h fixed on the fiber gives

\[
D_pu[d]=(I-uu^\top)Bd/\|v\|,\qquad
s_k=\rho^2(\ell_k^2+\ell_k r_k+r_k^2)\|D_pu[d_k]\|^2/3.
\]

Select a direction with P(k)=(1-gamma)/24 + gamma*s_k/sum_j(s_j). Use gamma=0
for uniform selection and gamma=0.75 for the weighted rule. The latter keeps
25% uniform exploration. rho and gamma are initial fixed choices, not optimized
hyperparameters; do not sweep them or select winners within R005 v1.

If v is zero, directional sensitivity is undefined; use uniform selection and
record the reason. If all valid scores are zero, also use uniform selection.
Use scaled norms and score algebra to avoid underflow. Report unresolved
underflow and its uniform fallback explicitly; nonfinite invalid arithmetic is
a failed control. Do not insert an epsilon that invents a response direction.
Mask unsafe inputs before evaluating branches so discarded JAX branches cannot
introduce NaNs. The two histories share the same policy.

These interval proposals generally have nonzero mean. Neither arm is asserted
to sample the fiber uniformly or to implement a particular Langevin process.
The unit-response score removes pure per-gate response scaling, which deliberately
omits some potentially useful mobility changes. Full h=0 remains stationary
under plain SGD for every representative. A Boolean-vertex q has no nontrivial
fiber; the feasible absolute movement contracts near such corners.

## Paired draws and amplitude controls

Flatten gate slots in this fixed order: perceive, then update; increasing
layer index; C order within each parameter leaf. Save the path/shape/offset
layout and direction table. Retain the 3,040 shared slots; do not expand them
into separate parameters for spatial cells or runtime ticks.

For repetition j=0–15, derive a case key by successively folding the training
seed, checkpoint update, and j as uint32 into PRNGKey(20260912). Do not fold in
history or arm: paired arms and histories use common random numbers. Split it
into choice and amplitude keys, producing independent FP64 uniform arrays of
shape (3040,). Select direction k by inverse CDF, counting the first 23 CDF
entries <= U_choice. This always returns 0–23. Use the same U_delta for the
selected feasible interval in both arms. Do not use Python hash() or a global
NumPy RNG for the on-device draws. Save key data, random-array hashes, selected
direction IDs, and deltas. Scheduling or restart order must not change draws.

From each source checkpoint, construct:

| Arm | Representatives | Purpose |
| --- | ---: | --- |
| identity | 1 | Original logits and ordinary SGD |
| uniform | 16 | Uniform direction/interval proposals |
| direction_weighted | 16 | Geometry-weighted direction proposals |
| uniform_matched | 16 | Uniform draws with coupled amplitude cap |
| direction_weighted_matched | 16 | Weighted draws with the same cap |

For each gate and paired draw, let a=min(|delta_U|,|delta_G|). In the matched
arms replace each delta by sign(delta)*a, retaining its own direction. Build
both from the untouched source. Shrinking toward zero preserves feasibility,
and ||d||₂=2 makes the per-gate probability displacement exactly equal. A zero
move in either member makes both matched moves zero at that gate. Do not
rescore, redraw, or change gamma after capping. Record realized displacement
after the logit round trip, plus zero/underflow movement counts.

This is a coupled control distribution, not either original proposal law.
Report both raw and matched results. All 65 representatives per checkpoint
receive two independent step sizes: 6,240 representatives and 12,480 steps.
Do not duplicate the identity 16 times as independent observations. Byte-identical
update-0 cases can reuse computation if their complete inputs and keys match;
retain their logical source labels and record the reuse.

## GPU update and numerical audits

The production-shaped kernel does one recurrent forward/backward computation
to get Q0,h0, computes the local scores and draw, makes the affine probability
move, reconstructs actual p',Q', and applies

\[
z_{\mathrm{next}}=z'-\eta J(p')^\top h_0.
\]

J(p') uses the actual post-encoding softmax and marginals. Reusing h0 is exact
under ideal neutrality; reusing the old logit gradient is wrong. Do not
differentiate through the sampler or re-encoding, and keep all sixteen logit
coordinates independently trainable in the following task step. Identity takes
its step from original logits. Each eta starts independently from the same
representative, not from the other eta's endpoint.

Batch local operations within `jax.jit`; the direction library is a constant
array. No per-step SVD, inverse, circuit enumeration, or whole-network Hessian
is required. Parameters, gradients, and PRNG arrays stay on device. Keep source
loading, artifact hashing, report formatting, and checkpoint I/O outside the
kernel. JAX's [JIT](https://docs.jax.dev/en/latest/jit-compilation.html),
[random](https://docs.jax.dev/en/latest/random-numbers.html), and
[asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)
documentation describe the applicable execution model.

The full formal audit intentionally does more work than the timed update.
For every representative, check probability normalization, Q neutrality, all
runtime states at ticks 0–20 on observation inputs 0–3, loss, and a freshly
computed q-gradient h'. Compare the reused-h logit gradient with J(p')^T h'
under the inherited R003 tolerances. Do these checks outside the benchmark;
their cost is not the overhead of an eventual one-backward-pass learner.

Retain the FP64 tolerance policy explicitly copied into the configuration.
Recompute from the logits actually materialized. A nonambiguous common-rounding
flip fails the hard control. Record threshold-ambiguous flips separately and
make no hard-invariance claim for them. Native argmax changes are an extraction
observation, not evidence of changed relaxed behavior. A soft-invariance failure
invalidates that representative's same-function comparison; retain it without
retrying for a more favorable draw or relaxing tolerances.

For each arm save v_a=-M_a*h0 and compute shared-point output response
w_a=D_Q O(Q0)[v_a]. O is flattened terminal channel 0; keep loss-input,
additional-input, and all-four views distinct. Also compute the actual-point
JVP at Q_a. Compare measured (Q_next-Q_a)/eta with v_a, and
(O_next-O_a)/eta with the actual-point output JVP. Subtract each arm's own
pre-step value, so round-trip drift is not counted as an SGD effect. Retain
absolute residuals and relative residuals when resolved. For a smooth local
step, normalized residuals should decrease roughly tenfold for tenfold smaller
eta; do not require an exact ratio at floating-point floors or retune eta.

## Outcomes and uncertainty

The primary contrast is direction_weighted_matched minus uniform_matched at saved update
250, reported separately for both histories. This checkpoint was selected from
the prior R003 response evidence before observing R005 outcomes. Updates 0 and
500 are secondary context. Use output JVPs on additional observation inputs
2 and 3, which do not contribute to h0.

For an arm a, normalize each full output-response vector w_ar to u_ar. Across
R=16 repetitions define

\[
\bar u_a=R^{-1}\sum_r u_{ar},\qquad
V_a=\frac1{R-1}\sum_r\|u_{ar}-\bar u_a\|^2.
\]

V is the trace of the unbiased sample covariance: stochastic directional
spread, invariant to positive scaling of an entire response vector. The
primary per-seed measurement is DeltaV=V_direction_weighted_matched-V_uniform_matched.
It can be negative. A positive result supports more directional spread under
this coupled movement control, not better training.

Separately record ||mean(u_a)-u_identity||², mean squared distance from identity,
and response angles/norms. The exact sample decomposition
mean_r ||u_ar-u_identity||² = (R-1)*V_a/R + ||mean(u_a)-u_identity||²
distinguishes spread from drift. Also report unnormalized response covariance
and means; unit normalization can magnify tiny physical effects.

Report raw-law results, the same q-response metrics, and baseline-response-
weighted per-gate angular displacement. Fix gate weights from ||v_i0||² before
intervening. Include absolute probability and functional displacement, eligible
response coverage, all four eigenvalues of each M, trace-normalized M shape,
and movement-versus-response plots. Normalizing v_a to a common global q-step
norm before a linear output JVP is an analytic speed control; label it and do
not introduce tuned finite-step learning rates.

Record the eleven higher truth-table Walsh moments E_p[product_(r in S)
(2*T_gr-1)] for |S|=2,3,4 and the six connected two-entry correlators
E[s_r*s_t]-E[s_r]E[s_t]. First moments are fixed by q. These concern truth
tables drawn under p, not cellular runtime states. This experiment does not
estimate RG constants or a stationary distribution. Pre/post SGD common/native
gate changes and post-step task loss are descriptive; no circuit-size cause or
complete description-length claim follows from a single step.

Zero or unresolved response norms give undefined direction metrics with reasons,
not epsilon-normalized vectors. For the primary dispersion require all 16
valid draws in both matched arms; do not silently shrink R or replace failures.
An undefined identity direction invalidates the drift-from-identity metric,
but does not invalidate dispersion if all matched-arm directions are resolved.
Retain valid absolute effects when a unit-direction observable is undefined.
Report scheduled, resolved, invalid, undefined, and reused counts separately.

First aggregate draws and gates within each checkpoint. Training seed is the
uncertainty unit; keep both histories, all times, and coupled draws together.
Show all 16 seed values. For the complete cohort, use 100,000 paired bootstrap
index rows from Generator(PCG64(20260912)), int64 shape (100000,16), uniformly
0–15. Report percentile 95% intervals for mean paired DeltaV. Other contrasts
are descriptive; no significance hunt or pooled all-checkpoint win is defined.
If seed pairs are missing/undefined, show available pairs descriptively and
their identities; do not label that a full-cohort 16-seed interval. Repetitions,
gates, probe cells, and repeated checkpoints are not independent training trials.

## Focused preflight

Before formal execution, run synthetic operator checks and seed 0 in both
histories at updates 0,250,500 using repetitions 0,1: six checkpoints, 54
representatives, 108 single steps. A null exploration effect is a valid result,
not a preflight failure.

- Verify exact integer A*d=0, direction norm 2, and rank 11; compare a clear
  NumPy reference with compiled JAX using identical supplied uniform arrays.
  Check scoring derivatives against finite differences and the proposal's
  equal-norm 77.3196-degree fixture. Test feasible bounds and amplitude matching.
- Verify gamma=0 reproduces the uniform arm for identical randomness. A rho=0
  control must preserve original logits exactly via an identity fast path.
  Check zero gradients/scores, underflow, ineligible gates, and rounding ties.
- Compare the reused-h compiled task update against fresh full logit autodiff;
  validate detachment and independent coordinates. Check recurrent neutrality,
  output JVPs, and both finite step sizes under the frozen policies. Keep the
  separate original-FP32/promoted-FP64 comparison distinct from neutrality.
- Check identical identities/draws after serialization, restart, and reordered
  case scheduling. Verify source arrays, Adam states, and keys are unchanged.
  Reject a resumed case with mismatched source, config, layout, or random keys.
- Check the device-transfer boundary of the compiled operator and benchmark.
  Use the runtime's transfer guard or equivalent recorded trace to detect
  accidental host transfers. Warm up compilation before checking timed work.

Preserve failed preflight attempts and fixes. Changing numerical or scientific
choices requires a versioned protocol update, not a silent response to results.
Routine implementation fixes retain the same protocol with recorded provenance.

## Cost benchmark

Benchmark the six preflight checkpoints and three raw arms separately in FP64,
eta=0.001. The coupled amplitude controls are offline analysis arms and are
excluded from the production-cost comparison. Compile the 20-update block
separately, execute one warm-up block, then five timed blocks of 20 updates.
Reset each block to its untouched source;
carry parameters and the intervention key through every update within the block.
Recompute h through the recurrence each time. Use the fixed loss batch 0,1.
These 100-update benchmark paths are not a training-success study.

Use PRNGKey(20260913), folding training seed, checkpoint update, and block ID;
split the evolving key once per update to obtain that update's sampling key.
Pair histories/arms; timed block IDs are 0–4 and the warm-up ID is uint32 max
(4294967295). Implement a jitted
lax.scan with evolving parameters, so loop-invariant reuse cannot accidentally
remove the backward passes from the measurement. Discard warm-up descendants.

Synchronize with block_until_ready at block boundaries; record compile time,
all block times, median time/update, incremental time and ratio versus identity,
peak device memory, and failure/fallback counts. Record memory measurement
method and distinguish allocator reservations from live usage. Keep validation,
hashing, and artifact I/O outside timed blocks; report their wall time separately.
Inspect finite outputs/statuses after each block. Preserve a failed block instead
of omitting it from a throughput headline. No speedup or FP32 extrapolation is
assumed. A CPU FP64 mechanism fallback can be recorded, but cannot count as a
GPU benchmark. Queue GPU work without interrupting R004.

## Deliverables and execution

Implement an isolated R005 operator/runner, a transparent NumPy reference, and
`scripts/r005.py` with preflight, run, analyze, and benchmark subcommands. The
runner uses the fixed config, explicit run IDs, artifact relocation, and a
failure/restart ledger. After passing preflight, run the fixed cohort, analyze
it, and benchmark when the GPU is available. No new long-training arm is part
of v1. Keep shared-code changes small and preserve R000–R004 semantics.

The commands below define the CLI to implement; they are not currently runnable
commands claimed by this handoff:

```bash
JAX_ENABLE_X64=1 uv run --project envs/jax-gpu --locked python scripts/r005.py preflight --config configs/experiments/r005_stochastic_neutral_exploration_v1.json --run-id preflight_attempt01
JAX_ENABLE_X64=1 uv run --project envs/jax-gpu --locked python scripts/r005.py run --config configs/experiments/r005_stochastic_neutral_exploration_v1.json --run-id formal_attempt01
JAX_ENABLE_X64=1 uv run --project envs/jax-gpu --locked python scripts/r005.py analyze --config configs/experiments/r005_stochastic_neutral_exploration_v1.json --run-id formal_attempt01
JAX_ENABLE_X64=1 uv run --project envs/jax-gpu --locked python scripts/r005.py benchmark --config configs/experiments/r005_stochastic_neutral_exploration_v1.json --run-id benchmark_attempt01
```

Retain `jax_threefry_partitionable=false`, the repository's deterministic XLA
settings, highest dot precision, and the validated locked GPU environment.
Record actual hardware, versions, code commit/dirty patch, config and lock
hashes, reference file hashes, array layout, RNG convention, and source identities.
Do not call an unavailable or changed environment verified without checking it.

Put small manifests, validation, cases.jsonl, analysis.json, summary.md, and
figures under results/R005_<run_id>. Keep benchmark records separately under
results/R005_<run_id>_benchmark. Use corresponding artifacts/ paths outside git
for arrays. Store source references, h0, masks, sampled direction IDs/deltas,
and enough post-encoding representatives, q/output response vectors, finite
step deltas, and statuses to recompute the comparisons. Matrices and correlators
may be reconstructed from saved p/logits; do not duplicate dense J arrays or
parent Adam states. Save array hashes, byte lengths, and stable locations.

Estimate artifact volume during preflight and avoid redundant copies. Keep
full failed-control evidence and a fixed review subset: seed 0, both histories,
all three checkpoints, all arms, repetitions 0 and 1, both etas. Include all
small cohort records in the review archive. Use the experiment/R005 release
workflow for large artifacts after execution.

Completion means every planned case is accounted for, passing and failed
controls are visible, the spread/drift/amplitude comparisons are reproducible,
and the benchmark reports actual device cost. A valid null or adverse outcome
completes the experiment. A later training protocol can use these findings to
choose its optimizer, intervention frequency, precision, and fixed budget;
do not silently turn this checkpoint diagnostic into that experiment.
