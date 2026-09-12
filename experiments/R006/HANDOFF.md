# R006 handoff: longer training and the approach to gate hardening

Protocol: r006_training_and_hardening_v1, 2026-09-12.

Status: specified for implementation. Implement, validate, and run the fixed
continuation and frozen-checkpoint probes below. No R006 execution is claimed
by this handoff. Read [AGENTS.md](../../AGENTS.md), [STATE.md](../../docs/STATE.md),
[SPEC.md](SPEC.md), and the
[machine-readable configuration](../../configs/experiments/r006_training_and_hardening_v1.json).

Reference commit: c7680962270b0cd2a1ca22f46a4d7dbc7f0ba72e, which contains the
R004 implementation, small results, and original/failed preflight records.
[SOURCE_PINS.json](SOURCE_PINS.json) records all 64 parent-manifest identities,
update-2,000 checkpoints, common/native exports, wiring identities, release
assets, and the four seed-0 update-1,950 checkpoints used for historical replay.
Listed repository files were checked against their Git blobs at this commit.
Full parent checkpoint/export verification and GPU replay remain preflight work.

## Question and rationale

Does substantially more training of the unchanged learner close the difference
between relaxed-output correctness and correctness after gate hardening? As
training proceeds, does the rounding perturbation shrink, does the recurrent
computation tolerate more of that perturbation, or do these quantities separate?

R004's [independent review](../../reviews/R004_continuation/REVIEW.md) establishes
16/64 exact common-hard circuits at update 2,000, compared with 4/64 at 500.
At least 47/64 final relaxed models have original-probe summed squared error
below 0.25, a sufficient bound for correct thresholded terminal outputs.
Thirty-two of those models fail common gate hardening. This is a measured gap
at finite training budgets, not evidence that longer training cannot close it.

The late records remain mixed: common-hard successes rise from 13/64 at 1,500
to 16/64 at 2,000. Of the 32 final SSE-certified relaxed models with failed hard
circuits, 22 reduce their hard error from 1,500 to 2,000 and 10 increase it.
These retrospective observations motivate the new protocol; they are not
prespecified tests of R004 or a reason to select only promising descendants.

A useful analytic example is the relaxed gate with truth entries (0,0,a,a),
so its first input evolves as x[t+1] = a*x[t]. From x[0]=1 with target x[20]=0,
a=0.8 gives output 0.8^20, approximately 0.01153, and squared loss 0.8^40,
approximately 0.0001329. Common hardening yields the identity gate for a>=0.5,
which preserves the wrong value 1. Longer training could push a below 0.5,
but immediately above that crossing the soft loss is already near 2^-40.
Small relaxed loss can coexist with insufficient training for hard correctness.
This example is a validation fixture, not a claimed mechanism for an R004 run.

## Frozen continuation

| Item | R006 v1 choice |
| --- | --- |
| Parent cohort | All 64 R004 formal_attempt01 update-2,000 checkpoints |
| Seeds and conditions | Seeds 0–15; categorical/truth crossed with decay 0.01/0 |
| New optimizer updates | 2,001 through 10,000 inclusive |
| Budget | 8,000 additional updates per run; 512,000 total |
| Primary model | Actual post-update-10,000 checkpoint |
| Learning rate | Constant 0.05 |
| Optimizer | Original AdamW, betas 0.9 and 0.99, eps 1e-8, eps_root 0 |
| Gradient clipping | Elementwise at 100, before AdamW |
| Loss | Channel-0 terminal squared error, summed across batch and grid |
| Training batch | Two fresh random Boolean initial grids each update |
| Runtime computation | 20 synchronous ticks, 16 by 16 grid, eight channels |
| Target and boundary | Original anchored 2 by 2 checkerboard; constant zero exterior |
| Precision | Original FP32; X64 disabled during training |
| Ordinary evaluation | Every 50 global optimizer updates |
| Resumable checkpoints | Every 250 global updates |
| Structure | Every 500 updates from 2,000 through 10,000 |
| Hardening curves | Updates 2,000, 4,000, 6,000, 8,000, 10,000 |
| Extended hard runtime | Updates 2,000 and 10,000 |

The exact condition names remain categorical_reference_decay,
truth_reference_decay, categorical_no_decay, and truth_no_decay.
Preserve every parent's own parameters, optimizer leaves/counters, model key,
data key, wiring, and representation. Do not rematch Q between representations,
reset moments, change logits' gauges, add noise or entropy/description costs,
anneal anything, change the loss, or feed a hardening-probe model into training.
R003/R005-intervened checkpoints are not eligible parents. Continue successful
and unsuccessful runs alike to the fixed endpoint.

These are longer trajectories of the same 16 paired seed groups. They are not
new independent trials or a cross-task curriculum. More updates also mean more
exposure to the existing AdamW decay. The experiment measures the unchanged
recipes at a later budget, not independently tuned coordinate systems.

The checkpoint interval changes from R004's 50 to 250 solely to limit storage.
Ordinary evaluations remain every 50 and per-update records remain complete.
Intermediate evaluation states can be reconstructed by exact replay from a
preceding checkpoint if needed. No outcome changes a scheduled save or probe.

## Sources, exact resumption, and provenance

Use the source-pins file to resolve each parent at the reference commit, rather
than trusting an unversioned local manifest. Verify lengths and SHA-256 hashes
before loading NPZ with allow_pickle=False. Validate format version 2, global
update 2,000, seed, condition, representation, original R004 config hash, and
complete original R002 lineage. Preserve the dirty-source identities recorded
during R004; the later clean source commit does not rewrite their provenance.

Load actual wiring from the verified update-2,000 Boolean exports, and require
common/native wiring equality and agreement with the parent wiring hash.
Validate raw gate IDs as integer arrays in [0,15] before uint8 conversion.
Validate wire dtype, shape, range, and architecture before any conversion.
The pinned R004 analysis loader casts gates before checking their range;
R006's loader must check the raw values first. Do not inherit that weakness.

Use R002's existing mathematical step, sampler, effective-truth-table kernel,
optimizer factory, and checkpoint machinery. The R004 wrapper freezes R002
update-500 parents and a 2,000 endpoint in several places; it is not a generic
continuation API. Add an R006 parent loader, metadata validator, orchestration,
and analysis adapter rather than changing R004's historical validators or
globally substituting its constants.

Creating optimizer/state shape templates is permitted, but all live values must
come from the verified parent. Save descendants under R006 metadata with the
R006 config hash and explicit parent-manifest/checkpoint/wiring identities.
Do not relabel, overwrite, re-encode, or resave the original R004 parent.

Each new logical update retains this order:

    data_key, inputs = sample_training_batch(data_key, config)
    state, pre_update_loss, gradients, updates = train_step(
        state, inputs, target, wires
    )
    assert int(state.update_index) == global_update

The internal model-key split remains unchanged even where its secondary key
is unused. The reported training loss belongs to the pre-update model;
checkpoint evaluations belong to the post-update model. Each training example
starts a fresh 20-tick CA rollout, as before.

Keep the GPU uv environment and lock:
SHA-256 7a8e627cd9851d7426502bc3734bcedcfe48277575aec363877d4e7b717b2125.
The recorded versions are Python 3.11.6, JAX/JAXLIB 0.10.2, NumPy 2.4.6, and
Optax 0.2.8. Require JAX_ENABLE_X64=0 before imports, legacy
jax_threefry_partitionable=false, highest dot precision for categorical p@T,
and XLA_FLAGS=' --xla_gpu_deterministic_ops=true'. No dependency upgrade or
mixed precision belongs in this continuation. Record actual OS, GPU, driver,
uv, runtime versions, compiler flags, lock, and source hashes.

Verify identical actual new training batches across each seed's four conditions.
Hash each typed array using the R002 dtype/shape/C-order convention. The new
indexed stream digest is SHA-256 of the ASCII lines

    2001:<batch_hash>
    ...
    10000:<batch_hash>

with one trailing LF per line. Keep the R002 and R004 stream digests separately;
concatenated digests are not a digest of the combined raw history.

Write complete checkpoints atomically. On interruption, resume the latest
valid matching checkpoint, restore every state/key/counter, retain superseded
tail records, and produce one authoritative record per global update. A resumed
attempt is the same trial. Preserve failures and unresolved outcomes; never
replace them with another seed or a favorable checkpoint.

Evaluation, compilation, export, and diagnostics must not consume the live
keys or alter the optimizer/state. Warmups use disposable clones. Run the
hardening and extended-runtime analyses offline against saved checkpoints.

## Ordinary evaluation and structure

Retain per-update loss, gradient maximum/L2, clipped-element count, parameter-
update norm, Q-step norms, native/common gate changes, actual batch hash,
global update, continuation update k-2000, and synchronized update time.

At every 50-update position, retain the existing original-32 soft loss,
common/native hard errors, perfect-grid counts, all-32 exactness, and extraction
diagnostics. Add the actual thresholded relaxed-output error count at y>=0.5,
all-32 relaxed exactness, and minimum signed output margin. Report the sufficient
SSE<0.25 certificate separately; its failure is not a soft classification failure.

Keep the original R004 update-2,000 records and label the R006 replay separately.
There are 161 evaluation positions per run, or 10,304 labeled positions including
the baseline. There are 32 new resumable checkpoints per run, or 2,048 total.
At each saved checkpoint retain the first original probe's soft/native/common
states at every runtime tick 0–20.

Use the same FactoredDAG implementation and recurrent visible-core closure as
R004. Measure common and native structure at the 17 scheduled positions:
2,176 labeled modes, of which 2,048 require new exports. Retain visible-core
binary nodes/channels, all-channel binary nodes, AND/XOR counts, and relevant
local inputs. Plot size with correctness. Equal node counts do not establish
equal functions; these counts are constructive bounds, not minimum or complete
description lengths.

## Frozen hardening path

At each of the five diagnostic updates, compute and materialize the checkpoint's
effective truth tables Q in FP32 using its existing representation. Q has four
entries per shared two-input gate: categorical p@T or direct sigmoid entries.
Use the established truth-column order 00,01,10,11 and gate-ID bit weights
8,4,2,1. Retain source hashes for Q and the original parameter checkpoint.

Compute H = 1[Q>=0.5] once and freeze it for the entire curve. Define

\[
Q_\alpha = Q + \alpha(H-Q), \qquad 0\leq\alpha\leq1.
\]

Use explicit endpoint branches: alpha 0 returns the original Q arrays exactly;
alpha 1 returns H exactly. Do not obtain endpoint arrays by subtracting and
adding rounded floating-point values. Intermediate alpha values use the
declared affine expression in FP32. Record any collapsed adjacent Q-alpha arrays
due to finite precision, retaining every alpha label.

The fixed alpha set is

\[
\{j/16:j=0,\ldots,16\}
\;\cup\;\{2^{-j},\,1-2^{-j}:j=5,\ldots,12\}.
\]

It has 33 distinct values. The config stores their sorted integer numerators
over 4,096, so the sampling grid is exactly representable in FP32. Do not adapt
the grid after seeing results, bisect an assumed threshold, or stop at the first
failure. Report failures and later recoveries in their original order.

Execute Q-alpha directly through the common multilinear gate kernel. Do not
convert it back to categorical probabilities or sigmoid logits, change gate
temperature, sample gates, train on it, or threshold intermediate states.
Sharing remains identical across every cell, layer application, and runtime tick.
Each alpha starts from the same initial grids, never the previous alpha's state.

Every table stays in the same common-rounding region along the path, including
the upward tie convention at Q=0.5. Check H(Q-alpha)=H(Q). Thus the discrete
endpoint rule is fixed throughout the curve even while the relaxed computation
changes. This path changes Q and is not the same-function neutral path in R005.
Native categorical argmax extraction remains in ordinary metrics; no native-
argmax interpolation arm is part of v1.

Evaluate all 64 runs, all five snapshots, and all 33 alphas: 320 checkpoint
cases and 10,560 labeled alpha rollouts, each containing 66 initializations.
That is 696,960 logical 20-tick initialization trajectories. Verified numerical
reuse can reduce work, but never the logical accounting.

### Fixed initializations and boundaries

Use the original 32-grid probe, the reused additional 32-grid runtime probe,
and the all-zero/all-one eight-channel grids, in that order. Preserve the target,
zero exterior, grid size, observer, and 20-tick readout for every alpha.

The original probe NPZ hash is
d49ecf23fd932fba25b1068f8c90820265a05fd90daadc372089520889090b2f.
Its canonical path is recorded in the config.

The additional NPZ hash is
4a64f801d604478f7d8ffa40db8255b8d7aaabbdb560173fe1946e00659d1478.
Its uint8 inputs have shape (32,16,16,8) and C-order raw hash
0bd05a2d3f281ff9f5e26c118677ba91ff9fc9c3bf777eed4ede6bca60387d65.
If needed, recover that exact array with NumPy Generator(PCG64(20260910)),
integers(0,2,size=(32,16,16,8),dtype=uint8), verify the raw hash, and record the
new file identity separately. This reused array is not an unseen holdout.

Use batches of 32,32,1,1 for the four groups and one alpha at a time. This
retains the original batch-32 probe shape and bounds GPU memory. JIT the
recurrent computation; no gradient is required for these probes. Measure
performance before claiming that a different batching strategy is equivalent.

### Required observations

For each alpha and input group, and separately for all 66 combined, retain:

- Terminal channel-0 summed SSE, actual bit errors after y>=0.5, perfect-grid
  count, all-input exactness, maximum absolute target error, and minimum signed
  margin (2*b-1)*(y-0.5).
- Terminal RMS and maximum output displacement from the same checkpoint's
  alpha-0 rollout, on identical inputs.
- At every tick 0–20, RMS and maximum state displacement from alpha 0, separately
  for the visible channel and all eight channels.

Ties are classified by y>=0.5. A signed margin of zero is recorded as a tie,
not automatically treated as correct or incorrect without the target bit.
Tick 0 displacement must be exactly zero. At alpha 1 all state values must be
Boolean and the visible outputs must match independent Boolean execution.

Store terminal visible FP32 arrays with shape (33,66,16,16) for all cases, so
an independent reviewer can recompute thresholds, margins, and SSE. Accumulate
new curve summaries in NumPy FP64 over those stored FP32 values. Retain the
source-comparable FP32 SSE at alpha 0 as well; do not replace the historical
training/evaluation reduction with a different one. Retain per-tick summary
arrays and the first original input's full states at alpha 0,0.5,0.9375,1.
Full state dumps for all inputs and all alphas are unnecessary.

For Q-to-H displacement, report L2, RMS, maximum absolute displacement, and
absolute-entry quantiles 0,0.05,0.25,0.5,0.75,0.95,1. Include mean binary
truth-entry entropy and minimum rounding margin. Compute these for all
effective entries and each network layer; use the number of truth entries,
not the number of logits, in RMS normalizations.

Also record the actual Q-alpha minus Q norms and typed-array hashes at each
alpha. Alpha is a fraction of each checkpoint's own rounding displacement;
equal alpha at two checkpoints need not mean equal perturbation magnitude.
Global distances/entropies include gates that may not influence the observed
computation. They do not measure effective circuit size or prove robustness.
Do not call the existing one-cell dependency mask a full recurrent causal cone.

### Curve summaries and their limits

Let E(a) be a group's terminal thresholded bit-error count, N its number of
visible bits, and a[i] the ordered grid. Report the trapezoidal grid estimate

\[
A_E=\sum_i (a_{i+1}-a_i)
       \frac{E(a_i)+E(a_{i+1})}{2N}.
\]

This is an alpha-weighted estimate of integrated bit-error rate, not an exact
integral or a continuous robustness certificate. Retain the full curve and
the normalization N: 8,192 for each 32-input group, 256 for each constant start,
and 16,896 for all 66.

Report the first sampled failure, every adjacent pass/fail transition, and
whether a sampled recovery follows a sampled failure. The last alpha in the
initial all-success sampled prefix is defined only if alpha 0 passes. If alpha
0 fails, report that flag, first failure at 0, and prefix endpoint null.
If every sampled alpha passes, first failure is null and the sampled prefix
ends at 1. Do not silently replace a missing prefix with zero.

A last sampled pass followed by a sampled failure is not a certified bracket
for the first continuous failure: an unsampled failure and recovery could have
occurred earlier. Never report the largest successful alpha as an uninterrupted
tolerance, and never infer monotonicity or a basin radius from this one direction.
Keep curves for alpha-0 failures and alpha-1 successes too.

Increasing tolerance along these sampled paths is evidence about these paths.
It can accompany smaller rounding displacements, changed sensitivities, or both.
The curves and time-resolved state differences can guide subsequent mechanism
work; they do not establish an unavoidable analog solution or identify a unique
causal gate.

## Numerical audit and extended Boolean runtime

The primary training and hardening curves remain FP32. In a separate X64-enabled
process, audit seed 0 in all four conditions at updates 2,000 and 10,000, at
alpha numerators 0,2048,3840,4080,4095,4096 over 4,096. Use all 66 starts:
eight checkpoint cases, 48 alpha rollouts, 3,168 initialization trajectories.

Promote the already materialized FP32 Q and H to FP64. Do not recompute the
softmax or sigmoid in FP64, which would change the source point. Retain maximum
and RMS output residuals, thresholded-bit disagreements, and SSE differences
against the corresponding FP32 curves. Flag max absolute output residual above
1e-4 and any threshold disagreement. Both precisions must match the Boolean
oracle exactly at alpha 1.

Interior numerical flags are reported evidence about numerical sensitivity;
they are not permission to replace the primary result, shift a threshold,
alter a curve, or claim a mathematical discontinuity. If numerical differences
affect an interpretation, label it unresolved pending a separately documented
follow-up. Failures of source identity or Boolean endpoint parity block the
diagnostic until the implementation is corrected.

At updates 2,000 and 10,000, retain the existing hard runtime/orbit and universal
propagation analysis through tick 256 for both common/native extraction and
all 66 starts. This accounts for 256 labeled modes and 16,896 trajectories;
the 2,000 baseline can be reused only with verified compatible identities.
The new endpoint contributes 8,448 trajectories. Keep the R004 phase,
initialization-counterexample, per-orbit onset, universal onset, persistent-
failure, and inconclusive distinctions. These additional readouts do not
change primary tick-20 success. Universal Boolean certificates apply to the
hard endpoints, not the interior relaxed models.

## Focused preflight and execution order

Before the full continuation:

1. Verify all source pins, probe identities, metadata, raw gate/wire validity,
   architecture, environment/lock, and all 64 parent checkpoints and exports.
   Replay the update-2,000 original-probe baseline. Common-hard success counts
   must be 3,2,8,3 out of 16 in the config condition order. Boolean outputs and
   exports require exact agreement; compare source soft outputs/loss with the
   existing atol=1e-6, rtol=1e-4 policy and report the residuals.
2. For seed 0 in all four conditions, restore the original update-1,950
   checkpoint and reproduce updates 1,951–2,000. Require exact equality of
   parameters, optimizer state, model/data keys, and counters with the parent.
   Compare arrays, not new NPZ serialization hashes.
3. From the actual update-2,000 sources, compare four steps of the reference
   mathematical update and R006 orchestration. Then compare ten uninterrupted
   steps against five steps, save/load, and five more. Insert ordinary
   evaluation and a complete hardening curve on one disposable path. Require
   exact state/key equality and identical batch hashes throughout. Exercise
   recovery from a superseded uncheckpointed tail.
4. Check the single-gate attenuation example analytically across the complete
   alpha grid, a fully Boolean Q giving a constant curve, upward ties at 0.5,
   and H(Q-alpha)=H(Q). Exercise reducers with a failing-then-recovering curve,
   a baseline failure, and an all-pass curve. Reject invalid raw gate IDs
   before casting; reject nonfinite/out-of-range Q or alpha rather than
   silently clipping them.
5. On seed 0 in all four conditions at update 2,000, run the full curve.
   Alpha 0 must recover the unmodified relaxed execution within the recorded
   FP32 policy; alpha 1 must match independent unsimplified Boolean execution
   exactly on all 66 starts. Record threshold discrepancies at alpha 0 if
   floating-point execution variants place an output on different sides of
   0.5. Check trajectory shapes, initial states, zero boundary, per-tick
   summaries, and fixed source Q/H hashes. Run the four baseline FP64 audits.
6. Validate structure/runtime adapters against the R004 update-2,000 records.
   Preserve simplifier and proof semantics. Source-versus-FactoredDAG Boolean
   execution and stored baseline runtime observations must match.
7. Measure compilation, synchronized curve execution, memory, checkpoint
   sizes, diagnostic-array sizes, and per-update log sizes. Use disposable
   copies for timing and keep all scientific cases/alphas/inputs fixed.

These checks address continuation identity and the new numerical operator.
Do not rerun unrelated R000 experiments as a substitute. Record original failed
preflights, fixes, the final source-matching preflight, and actual residuals.
If the environment cannot reproduce the continuation, record a protocol/runtime
decision before launching a changed experiment; do not quietly weaken equality.

Once preflight passes, launch each formal run from its untouched R004 parent.
Preflight descendants cannot seed the formal cohort. The baseline curves can
be computed before training and reused with identical source/analysis identities.
Later curves are offline evaluation of scheduled saved checkpoints.

Use one GPU training process and schedule R005's GPU work separately when
measuring throughput. R004's synchronized update time suggests approximately
11.5 hours of additional training-update execution for 512,000 updates on the
recorded workstation, before R006 evaluation, compilation, I/O, and diagnostics.
This is an extrapolation, not a measured R006 wall-time promise. Project the
new overhead from preflight and report actual components separately.

## Analysis contract

The primary endpoint is common-hard original-32 correctness at optimizer
update 10,000, compared with the same run at 2,000. Report each seed and all
four gained/lost/retained/failed endpoint categories. Test the paired success
change within each of the four conditions using the two-sided exact
McNemar/binomial calculation, with Holm correction across those four tests.
The primary family concerns longer training; coordinate comparisons are
secondary descriptive results. Do not select a winner from intermediate
budgets or add uncorrected tests after looking at curves.

Use 100,000 paired bootstrap resamples from NumPy Generator(PCG64(20260912)),
with an int64 index array of shape (100000,16), uniform on 0–15. Each resampled
seed carries all four conditions and all snapshots. Reuse indices across
reported contrasts. Retain raw values and percentile 95% intervals for paired
success-fraction and hard-error changes; show secondary pointwise intervals
for changes in Q-distance, integrated curve error, constructive size, and the
truth-minus-categorical size gap. These secondary intervals are exploratory,
not simultaneous confidence bands or additional confirmatory tests.

Preserve complete saved R002/R004/R006 histories, counting boundary evaluations
once. Report first observed soft and hard successes, subsequent losses and
recoveries, and final status. First success is cadence-limited and is not
necessarily persistent. Do not substitute an intermediate best result for
the fixed endpoint.

Cross actual thresholded relaxed correctness with common-hard correctness
at every evaluation. Freeze the update-2,000 relaxed-correct/hard-failed
subgroup by its baseline status, before inspecting descendants. Separately
identify the conservative 32-member SSE<0.25/hard-failed subgroup discussed
in the R004 review. Both are conditional descriptive analyses and retain
their original condition/seed identities. The full cohort remains primary.
Do not define the comparison subgroup by failure at 10,000.

Required figures and tables:

- Original-probe soft loss, actual soft/hard bit errors, exact-success counts,
  and gained/lost histories through the fixed endpoint.
- Full hardening curves by checkpoint, with original/additional/constant input
  groups distinguishable; retain nonmonotonic paths and undefined prefixes.
- Per-seed rounding distances alongside curve summaries, with actual Q-alpha
  displacement and per-tick state divergence available for inspection.
- Constructive size versus hard error and the paired size-gap trajectory.
- Endpoint broader-initialization/runtime/cycle evidence and measured cost.

Repeated alphas, checkpoints, probe grids, channels, and gates are not new
independent trials. Report missing/nonfinite/failed runs explicitly, with
scheduled and resolved denominators and unresolved-success bounds. Full-cohort
paired inference requires the planned seed groups.

Improvement supports usefulness of more compute for these recipes. Continued
soft improvement without hard improvement establishes a continuing finite-budget
gap, not proof of an impossible hard solution. A shrinking Q-distance does not
guarantee hard correctness, and a large global Q-distance can include irrelevant
gates. Stable-looking endpoints do not prove stationarity under constant-rate
AdamW and fresh batches. Neither this study nor its curve measurements establish
minimum circuit descriptions, infinite-sequence learning, or grid-size
generalization.

## Implementation deliverables and publication

Add an R006 runner, frozen hardening probe, and analysis adapter, with routine
module organization left to the implementation partner. Reuse the mathematical
kernels and source validation conventions. Provide preflight, sweep, probe,
analyze, and verified resume entry points. A completed scientific run returns
exit code zero; scientific failure to solve the target is not a process error.

The following define the CLI to implement; they are not commands claimed to
run in this handoff's repository snapshot:

    JAX_ENABLE_X64=0 uv run --project envs/jax-gpu --locked python scripts/r006.py preflight --config configs/experiments/r006_training_and_hardening_v1.json --run-id preflight_attempt01

    JAX_ENABLE_X64=0 uv run --project envs/jax-gpu --locked python scripts/r006.py sweep --config configs/experiments/r006_training_and_hardening_v1.json --concurrency 1 --sweep-id formal_attempt01

    JAX_ENABLE_X64=0 uv run --project envs/jax-gpu --locked python scripts/r006.py probe --config configs/experiments/r006_training_and_hardening_v1.json --sweep-id formal_attempt01 --analysis-id analysis_attempt01

    JAX_ENABLE_X64=0 uv run --project envs/jax-gpu --locked python scripts/r006.py analyze --config configs/experiments/r006_training_and_hardening_v1.json --sweep-id formal_attempt01 --analysis-id analysis_attempt01

Preflight/probe orchestration launches the declared FP64 audit in a separate
process before importing its JAX code; it must not enable X64 in training.
A probe invocation may process currently available scheduled checkpoints and
truthfully report pending cases; completion requires the whole fixed grid.
An analysis may resume from verified case results without recomputing them.
Any caches must include checkpoint/Q/H, alpha, input, boundary, observer,
precision, and analysis-code identities as applicable.

Use results/R006_{sweep_id}_seed{seed:02d}_{condition} for per-run small records,
results/R006_{sweep_id} for the cohort, and
results/R006_{sweep_id}_{analysis_id} for analyses. Store full checkpoints,
all 512,000 per-update JSONL records, dense curve arrays, and large trajectory
payloads under corresponding artifacts paths outside git. Commit small
manifests, evaluation tables, scalar curve/distance summaries, per-seed results,
source-matching preflight evidence, failure/resume records, and release identities.
The storage policy moves bulk logs outside git; it does not sample or discard
their update records or batch hashes.

Every result records source identities, code commit/dirty patch, resolved config
and hash, environment/lock/hardware, precision, counters/keys, start/end status,
compile/training/evaluation/probe/analysis timings, and artifact lengths/hashes.
The release workflow must preserve all original and failed preflight records;
the earlier compact R004 archive omitted its original pre-launch record.

Prepare deterministic experiment/R006 release bundles, targeting at most
1.5 billion bytes per archive shard. Preserve logical inventories, checksums,
and stable asset URLs. Deduplicate identical array payloads only while retaining
all logical source labels and compatibility metadata.

The fixed compact review subset is seed 0, all four conditions, all five curve
checkpoints. Include its original/full source checkpoints, wiring, Q/H, probe
arrays, all terminal curve outputs, selected first-original full-state
trajectories, FP64 audit outputs, config, lock, implementation, and provenance.
Include all small cohort results, scalar curve/distance tables, and preflight
attempts. Reference bulk per-update logs and other source arrays by stable
release identities. Extra illustrative circuits may be added under an explicit
selection rule, without replacing the fixed subset.

Complete the implementation, focused validation, formal continuation, declared
diagnostics, analysis, and source/artifact publication on the workstation.
Keep statuses truthful at every stage. R005's protocol and original R002
checkpoint selection remain unchanged.
