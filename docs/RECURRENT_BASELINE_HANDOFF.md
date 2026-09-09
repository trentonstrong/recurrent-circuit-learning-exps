# Recurrent DiffLogic: baseline and dimensionality handoff

Status: proposed implementation and experiment instructions, 2026-09-09.
No new learner has been implemented or trained for this handoff. The source
audit and small algebraic checks described below have been completed.

Repository adaptation: see [workstation setup](WORKSTATION_SETUP.md) for the
RTX 5090 runtime plan, [current state](STATE.md) for experiment status, and the
individual experiment specifications for the frozen additional evaluation seed.
The historical source manifest remains unchanged.

## Research objective and division of work

Establish a published, functioning recurrent learner before further changes to
our sequence-generator architecture. Then measure how gate coordinates,
redundancy, wiring, and discretization affect optimization. The eventual target
remains efficient complete descriptions of sequences, with increment and other
generators leading toward more difficult cases such as pi.

The repository will hold the accepted language, experiment specifications,
implementation, decisions, and result manifests. Local Codex can implement and
run each specification on the user's machines. This conversation can develop
hypotheses, review diffs and results, and specify subsequent experiments against
a concrete repository commit. A suggested change becomes an experimental result
only after its code, configuration, and outputs are recorded.

The first baseline should be the **synchronous checkerboard experiment from
DiffLogic CA**. It trains through recurrent rollouts. Game of Life is useful as
a local-rule verification task, but its one-step supervised training does not
establish that our recurrent optimization works. Checkerboard reconstruction is
also easier than emitting a counter stream: success there is a positive control,
not evidence that the same recipe will discover a counter. [1]

## Audited reference

Primary sources:

1. [DiffLogic CA paper, v1](https://arxiv.org/html/2506.04912v1).
2. [Official notebook at the audited commit](https://github.com/google-research/self-organising-systems/blob/3d5547ca48b60ecac459834e2c05c9ff5df87991/notebooks/diffLogic_CA.ipynb).
3. [Light Differentiable Logic Gate Networks, v1](https://arxiv.org/html/2510.03250v1).

Pin repository `google-research/self-organising-systems`, commit
`3d5547ca48b60ecac459834e2c05c9ff5df87991`, path
`notebooks/diffLogic_CA.ipynb`. The retrieved UTF-8 file has SHA-256
`a9b3829db0d9fe0eb46148d516c18358aa648e3538aeaa682003b77cfbe757c8`.
This pins the currently audited public implementation, not a claim that this was
the exact source revision used for the paper. Keep its Apache-2.0 attribution
when vendoring source. Notebook cell IDs and extracted settings are in
[source_manifest.json](../references/difflogic_ca/source_manifest.json).

The implementation uses JAX, with JAX/JAXLIB 0.4.33 specified. Other dependencies
must be resolved and locked in the working environment; the notebook does not
provide a complete version lock. Preserve the deterministic XLA flag recorded
in the manifest for the reference run. Record any hardware or package adaptation.

| Setting | Synchronous reference | Asynchronous notebook recipe |
| --- | --- | --- |
| Grid; channels | 16 × 16; 8 | 14 × 14; 8 |
| Checker square width | 2 | 2 |
| Runtime ticks | 20 | 50 |
| Batch; training updates | 2; 500 | 1; 800 |
| Update-layer widths | 256 repeated 10 times, then 128,64,32,16,8,8 | 256 repeated 14 times, same tail |
| Perception | 16 kernels, widths 9,8,4,2 | Same |
| Cell update probability | 1 | 0.6 |

Both use fixed wiring, zero exterior padding, fresh random Boolean initial
grids, terminal-channel squared-error loss summed over the batch, and softmax
gate logits biased toward A. AdamW uses learning rate .05, betas (.9,.99),
weight decay .01; gradients are clipped **by value at 100**. Gate A starts at
logit 10 and the other logits at zero. Native hard inference uses argmax. [2]

The asynchronous recipe is a separate reference. Its result cannot isolate
the effect of masking because several other settings change. An eventual
mask-only experiment must define its own matched protocol.

## R000: source capture and implementation parity

Implement this first. No large training sweep is needed for R000.

1. Vendor or fetch the pinned notebook, verify its checksum, retain its license,
   and record all deviations from it. Extract a standalone reference runner
   without changing its mathematics. Do not use the existing TAPE learner as
   the implementation of the published baseline.
2. Make the synchronous configuration explicit and independent of notebook
   execution order. In particular, construct the optimizer from that config
   rather than inheriting a previously defined global optimizer.
3. Save wiring arrays, initial logits, initial grids, targets, and RNG states.
   A seed number alone is insufficient for parity across frameworks.
4. Verify all 16 Boolean gate tables and their multilinear extensions on a
   fractional input grid. Verify patch extraction and boundary ordering, kernel
   parameter sharing, every intermediate tensor shape, one state update, a
   complete rollout, loss reduction, and a gradient/optimizer update.
5. If using PyTorch, retain JAX as the reference oracle. Export identical arrays
   and use them in both implementations; do not expect matching RNG seeds to
   generate matching arrays. Test parameter gradients within each representation
   across frameworks, not between the 16- and four-parameter representations.

Use exact equality for Boolean execution and wiring. For numerical checks,
start with float64 unit-scale probes (for example `atol=1e-10, rtol=1e-8`) and
FP32 local probes (`atol=1e-6, rtol=1e-4`). Full deep rollouts may accumulate
larger differences: report the earliest differing layer/tick and measured
errors, and justify any tolerance change. Do not silently loosen a check until
it passes. Test both the reference initialization and nontrivial fixed logits.

Preserve gather order, gate input orientation, tensor flattening, dtype, summed
loss, optimizer epsilon semantics, clipping order, and weight-decay placement.
The notebook is authoritative if a diagram or comment differs from executable
code. Record that discrepancy rather than silently correcting the baseline.

A successful PyTorch port gives us a convenient implementation for later work.
Running the native reference first avoids conflating a porting error with an
optimization failure. GPU bitwise trajectory agreement across different models
of GPU is not an acceptance requirement; numerical settings and differences
must be documented.

## R001: reproduce recurrent learning

Run the pinned synchronous recipe first with its reference seed 23 and full
500-update budget. Save the initial, final, and every-50-update states of the
optimizer, model, and RNG. Preserve any reference evaluation performed during
training. Separately evaluate the exported Boolean circuit after update 500;
do not confuse a pre-update logged loss with the final model's loss.

Also define a fixed evaluation set of 32 initial grids, generated from a
separate RNG stream and reused at every checkpoint. This set must not affect
training, checkpoint selection, or tuning. Measure terminal soft loss, native
hard bit error, fraction of perfectly reconstructed grids, and the trajectory
of both visible and hidden state. Additional larger-grid, longer-runtime,
damage, and asynchronous tests are distinct evaluations with their own contracts.
Use the source's target construction when claiming to reproduce its tests.

The exact reconstruction criterion for a declared finite set is zero hard
error on every grid in that set. Report both the reference-compatible metrics
and this additional criterion; do not imply that the paper promised success on
our additional 32-grid set. One successful seed verifies a working example, not
a success rate. If the reference fails, inspect parity and environment evidence
before interpreting results from altered learners.

This stage retains the published optimizer and task. Our previous cross-entropy,
truth-table noise, learned origin seed, emit interface, and description penalty
are not part of this reference recipe. They remain available as named subsequent
interventions. There is no transfer of trained weights between independent runs.

## R002: matched gate parameterizations on the working recurrent task

After R001, use the same checkerboard architecture and observer. Compare the
16-function softmax with four independent sigmoid truth entries. This is the
first controlled change; learned wiring and counter adaptation come later.

Use 16 paired seeds, 0–15, and four conditions:

| Condition | Gate coordinates | AdamW weight decay |
| --- | --- | ---: |
| `categorical_reference_decay` | 16-way softmax | .01 |
| `truth_reference_decay` | Four sigmoids | .01 |
| `categorical_no_decay` | 16-way softmax | 0 |
| `truth_no_decay` | Four sigmoids | 0 |

The first pair compares the two parameterizations under the reference recipe.
The second removes the coordinate-dependent weight decay. Equal decay applied
to different parameter coordinates is not the same functional regularizer;
report the interaction rather than attributing it all to redundant dimensions.

Hold the original 500 updates, .05 learning rate, clipping convention, loss,
precision, fixed wiring, input batches, and evaluation set constant. Start all
runs from scratch. No added exploration noise or description pressure is part
of this comparison. First profile one run to select safe concurrency on the
available hardware; do not change batch size or unroll length to increase it.

For each paired seed, construct categorical probabilities p at the reference
initialization, form its effective table q, and initialize sigmoid logits with
`logit(q)`. Share the actual wiring and input arrays. Check equality of initial
gate functions, complete soft rollouts, and state Jacobian-vector products.
Do not independently sample the two initial parameterizations. The models have
different parameter gradients by design.

Keep native categorical argmax and entrywise table rounding as separate
evaluations. Use common table rounding, with a declared tie rule, to compare
the learned effective functions. Label which extraction each success rate uses.
Record clipping frequency and effective table displacement per step: the same
nominal learning rate is not the same functional step size. Any later learning-
rate sweep gets equal declared budgets and a separate experiment ID.

Primary outputs are per-seed correctness, hard error, soft loss, soft/hard gap,
time to the first exact evaluation checkpoint, and final-update results. Report
paired outcomes and uncertainty over training seeds; grids from the same seed
are not independent training trials. Report wall time, compilation time, peak
memory, and throughput alongside update counts. This is a comparison of learning
and runtime costs, not yet a complete-description compression result.

## What dimensionality means here

Let T be the 16 × 4 matrix containing the Boolean truth tables in rows, ordered
by gate IDs 0–15; truth-table columns are input pairs 00,01,10,11. Then

\[
z\in\mathbb R^{16}\longmapsto p=\operatorname{softmax}(z)
\in\Delta^{15}\longmapsto q=T^Tp\in[0,1]^4.
\]

For a two-input gate:

| Quantity | Dimension |
| --- | ---: |
| Stored categorical logits | 16 |
| Probability simplex, after the constant-logit-shift gauge | 15 |
| Effective multilinear truth function | 4 |
| Interior mixture directions preserving that function | 11 |
| Stored independent sigmoid logits | 4 |

The 11 follows from `rank([ones; T.T])=5`: a perturbation v satisfying both
`sum(v)=0` and `T.T @ v=0` leaves normalization and q unchanged. There is also
one trivial gauge direction in the 16 logits. These are interior statements;
on simplex faces the allowable fiber dimension can shrink. At an exact Boolean
truth-table vertex the representing distribution is concentrated on that gate.

For k-input LUTs, the analogous counts are `2^(2^k)` categorical logits,
`2^(2^k)-1` simplex dimensions, and `2^k` effective table dimensions. Their
generic interior difference is `2^(2^k)-1-2^k`. This is one reason the direct
table representation becomes attractive at larger fan-in. It does not prove
better optimization on every task. [3]

Those are local gate-function counts, not the effective dimension of an entire
circuit or a finite dataset. Duplicate inputs, unreachable rows, inactive gates,
symmetries, and limited observations can reduce the latter further. For G
independently parameterized LUT2 slots with fixed wiring, `4G` is an upper bound
on the formal gate-function coordinates; it need not be an identifiable model
dimension.

In the audited synchronous reference, there are 224 distinct perception gate
slots and 2,816 update slots, for 3,040 total. Perception parameters are reused
across channels. Consequently there are 48,640 categorical logits, 45,600
simplex coordinates, and at most 12,160 formal gate-function coordinates. A
cell applies 4,608 gate instances per tick because sharing reduces parameter
count without eliminating evaluations. These counts are derived from [2], and
are recorded separately in the manifest.

Our existing four-cell, six-channel, 16-LUT counter template is different:

| Parameter group | Current scalar count per independent run |
| --- | ---: |
| Truth logits | 64 |
| Gate input-selection logits | 944 |
| Next-state selection logits | 228 |
| Initial origin-state logits | 6 |
| Total | 1,242 |

Thus 1,172 parameters describe wiring choices. Changing only gate coordinates
to 16-way softmax raises the total to 1,434, not four times the whole model.
After the 38 wiring-softmax gauges, the current formal coordinate count is
1,204. This is still only an upper bound on functional identifiability.

For a finite fixed-readout trace, let `F_H(theta)` be the vector of m predicted
output values and `J_H = dF_H/dtheta`. Its rank is at most m. Our 80-output
counter trace therefore constrains at most 80 first-order directions at a point,
even if every one is independent. This does not make every vector in the local
Jacobian nullspace an exact neutral path or a safe finite step. Higher-order
effects and additional observations can distinguish those directions.

Finally, runtime state dimension is separate: our tape has 24 Boolean state
bits; the 16 × 16 × 8 reference has 2,048. Unrolling recurrent computation creates
more activations and constraints while reusing the same learned parameters.
Deterministic autonomous finite-state instances remain eventually periodic. Learning a size-general
rule is a different claim from recovering a bounded instance.

## R003: test the redundant directions directly

This is a post-reproduction mechanism study, not required to launch R001.

At declared checkpoints, replace a categorical distribution p with another
strictly positive distribution p' having exactly the same q. One convenient
choice is the factorized distribution

\[
p'_g=\prod_{r=1}^{4}q_r^{T_{gr}}(1-q_r)^{1-T_{gr}}.
\]

It is a representative of the same mean truth table, not an exact distribution
over recurrent executions. Alternatively, choose a vector in the 11-dimensional
nullspace above and a step that preserves positivity. Do not clip or renormalize
arbitrary proposals and assume neutrality survived; verify the constraints.

Freeze all other parameters, inputs, and masks. With cost and weight decay
excluded from the diagnostic, verify equality of soft rollouts, output loss,
and state Jacobian-vector products. Record whether native argmax changed despite
that equality. Then compare gate-coordinate Jacobians and a single explicitly
specified SGD step on cloned states. No training trajectory is overwritten.

For ordinary Euclidean SGD in logits, define

\[
J_{qz}=T^T(\operatorname{diag}(p)-pp^T),\qquad
\Delta q\approx-\eta J_{qz}J_{qz}^T\nabla_q\mathcal L.
\]

Two points with the same q can have different induced update matrices. This
tests whether movement neutral for the soft function changes later functional
learning. Adam also carries moments and coordinatewise scaling; report a
separate, fully specified optimizer-state policy before extending this test to
Adam. Neither the extra dimensions nor their removal is assumed beneficial.

The small algebraic audit for this handoff confirmed exact matched tables between
the reference initial p and its factorized representative, but different
`J_qz @ J_qz.T`. This is an illustration, not a recurrent-training result.

## Trajectory instrumentation

Distinguish optimizer update k, runtime tick t, and observation position j in
every record. Save raw gate logits or truth entries, effective q, hard gates,
wiring, optimizer moments, and RNG state at recoverable checkpoints. Log cheap
loss and gradient summaries every update. Store selected full state trajectories
at fixed checkpoints rather than every activation of every training batch.

Measure gate-family frequencies, rounding margins, input influence, reachable
output cones, soft/hard divergence, and changes in complete descriptions where
a codec exists. Use Boolean spins `sigma = 2s-1` for spatial and temporal
correlators, and distinguish raw correlations from connected correlations
`<sigma_i sigma_j> - <sigma_i><sigma_j>`. The spin transform alone is not mean
centering. Specify which cells, initial states, runtime window, and conditioning
are averaged. A constant circuit's high raw correlation should not be mistaken
for useful organization.

At a small declared checkpoint set, estimate output-Jacobian spectra with JVPs
and VJPs on a fixed probe set. Separate the truth-coordinate Jacobian from the
logit-coordinate Jacobian; spectrum and effective rank depend on coordinate
metric and scaling. State the estimator, numerical rank threshold, precision,
and uncertainty. A few leading singular values do not establish a full rank.
One optional summary is the participation ratio
`(sum(sigma^2))^2 / sum(sigma^4)`, with an explicit convention for a zero Jacobian.

RG flow constants require a defined coarse-graining map, family of system
sizes, and scaling observables. Do not infer them from a single loss curve.
Keep enough raw state and circuit data to formulate that experiment later.

Float32 should first follow the reference. A float64 replay can diagnose
rounding and derivatives without becoming a different primary run. Mixed
precision, compilation optimizations, and alternative gate evaluation kernels
are named numerical/performance interventions, validated against the established
reference before their results are pooled.

## Repository contract and first Codex task

Suggested layout (adapt names to an existing repository):

| Path | Role |
| --- | --- |
| `docs/STATE.md` | Current goal, accepted language, latest result IDs, next question |
| `docs/DECISIONS.md` | Dated decisions and links to evidence |
| `experiments/Rxxx/SPEC.md` | Hypothesis, controls, frozen budget and success criteria |
| `configs/` | Machine-readable, versioned experiment configurations |
| `src/` and `tests/` | Implementation and reference-parity tests |
| `results/<run_id>/` | Small manifest, metrics, summary, artifact locations and hashes |

Keep large checkpoints and arrays in durable artifact storage, with stable
locations and checksums committed to the repo. The repo remains the authority
for which artifacts belong to which run. Each manifest needs code commit and
dirty-patch hash, configuration hash, reference pins, wiring/data hashes, seeds
and RNG conventions, dependency lock, hardware/driver, precision/compiler flags,
start/end status, and the checkpoint selection rule. Failed and incomplete runs
remain in the ledger. A seed is a trial, not a fallback slot to rerun until it
works.

Preserve the existing reports and results as legacy evidence. In particular,
the latest readout pilot found 0/32 correct counter circuits, and direct-copy
initialization bypassed all LUTs throughout its near-copy runs. Its small
128-update budget and architecture are not the published reference recipe.
No prior trained weights become initializers for the new experiments.

Suggested instruction to local Codex:

> Implement R000 and the R001 runner from this specification. Inspect the repo's
> existing instructions first. Pin and audit the official source; establish the
> native recurrent baseline and, if using PyTorch, prove parity from identical
> arrays before interpreting training outcomes. Add explicit configuration,
> Boolean export, checkpoints, and result manifests. Run the focused checks and
> the canonical synchronous seed-23 experiment on the available hardware. Report
> deviations, soft and hard results, runtime/memory, and artifact locations.
> Leave R002 and later interventions as separate experiment specifications;
> do not silently fold them into the reference implementation or tune the
> reference budget after seeing the result.

After reviewing that result, the next handoff can freeze R002's implementation
and launch plan. Returning to the counter will require a labeled task/architecture
adaptation. Simply freezing the current learned wiring does not guarantee that
the resulting topology can represent increment; validate expressivity for that
specific topology, or retain it as an explicitly different control. All sequence
description scores must continue to charge the declared machine, initialization,
observer, runtime budget, and exceptions under a complete codec.
