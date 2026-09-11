# R004 handoff: extend every R002 training trajectory to 2,000 updates

Protocol: `r004_training_continuation_v1`, 2026-09-11.

Status: specified for implementation. No continuation runner, preflight, or
longer-training result is claimed by this handoff. Implement, validate, and run
the fixed continuation below, then complete its declared analysis. The
[configuration](../../configs/experiments/r004_training_continuation_v1.json)
is the machine-readable contract.

Reference repository commit: `8d58f06eb1161ce887ffdefe5993ca59cefd8392`.
Read [AGENTS.md](../../AGENTS.md), [STATE.md](../../docs/STATE.md), and
[SPEC.md](SPEC.md). R004 is separate from the
[R003 mixture-intervention diagnostic](../R003/HANDOFF.md). It continues the
original R002 trajectories, with their own parameters, optimizer states, and
random streams. No R003-modified checkpoint enters this experiment.

## Scientific question

Does the separation in circuit size and behavior observed at update 500 persist
when all original optimization recipes receive more updates?

All 64 R002 trials had lower fixed-probe soft loss at update 500 than at 400.
Under the same constructive simplifier, their common-hard visible cores at 500
span 2–36 binary operations for categorical coordinates and 44–236 for direct
truth coordinates. These observations do not establish stationary optimization
or a lasting preference for short descriptions. R004 measures trajectories of
correctness, size, and runtime behavior through a fixed later endpoint.

The only training intervention is a larger update budget. In particular, keep
the 20-tick training rollout: more optimizer updates and more execution ticks
are different interventions. The recipe includes the existing per-step AdamW
decay where applicable, so those runs also accumulate more decay exposure.
The existing no-decay conditions remain the corresponding controls.

R004 is a named continuation of the same 16 seed groups, not 64 new independent
trials or cross-task curriculum learning. It tests the existing recipes at
larger budgets, not whether either representation would win after independent
learning-rate tuning. A failure at 2,000 updates does not prove convergence or
impossibility, and a smaller incorrect circuit is not compression of the target.

## Fixed design

| Item | v1 choice |
| --- | --- |
| Parent | R002 `formal_attempt01`, original update-500 checkpoints |
| Seeds | Every seed 0–15, paired across all four conditions |
| Conditions | categorical/truth × decay 0.01/0, unchanged |
| Starting global optimizer update k | 500 |
| New updates | 501 through 2,000 inclusive |
| Budget | 1,500 additional updates per run; 96,000 across 64 runs |
| Primary endpoint | The actual post-update-2,000 model |
| Learning rate | Constant 0.05 throughout |
| Optimizer | Original AdamW; betas 0.9, 0.99; eps 1e-8; eps_root 0 |
| Clipping | Elementwise gradient clipping at 100, before AdamW |
| Loss | Terminal channel-0 squared error, summed over the batch and grid |
| Training batch | Two fresh random Boolean initial grids per update |
| Recurrent rollout | 20 synchronous ticks, 16 × 16 grid, eight channels |
| Boundary and target | Zero exterior; original anchored 2 × 2 checkerboard |
| Gate coordinates, wiring, sharing | Preserve each parent run's representation and arrays |
| Precision and random convention | Original FP32, X64 disabled, legacy Threefry setting |
| Checkpoint and primary-probe cadence | Every 50 updates, including a baseline at 500 |
| Structural snapshots | 500, 750, 1,000, 1,250, 1,500, 1,750, 2,000 |
| Full runtime profiles | 500, 1,000, 1,500, 2,000 |

Keep all four exact condition names:
`categorical_reference_decay`, `truth_reference_decay`,
`categorical_no_decay`, `truth_no_decay`. Do not restart successful or failed
parents, reset their moments, match their current q values, change parameter
gauges, add exploration noise, alter the loss, anneal the learning rate, or stop
after a favorable checkpoint. Continue every valid run to the fixed endpoint,
including those that solve the task earlier or later lose a correct circuit.

## Parent artifacts and environment

Resolve each parent manifest at the reference commit using
`results/R002_formal_attempt01_seed{seed:02d}_{condition}/manifest.json`.
The `checkpoints` entry for update 500 supplies its checkpoint identity. Verify
its SHA-256 and byte length before loading with `allow_pickle=False`, and check
format version 2, update index, seed, condition, representation, original config
hash, parameter shapes/dtypes, and wiring hash. Preserve the original training
code and dirty-patch provenance exactly as recorded.

Read the actual fixed wiring from the parent's verified Boolean export, using
the existing `wire_{network}_{layer:02d}_{a|b}` keys. Verify it against both
checkpoint and manifest wiring hashes. Check the loaded model's native and
common gate IDs against the original final exports. The categorical and truth
models at update 500 need not have the same q; their shared initialization and
training inputs were established by R002, not recreated at the continuation
boundary.

The parent release is
[`experiment/R002`](https://github.com/trentonstrong/recurrent-circuit-learning-exps/releases/tag/experiment/R002).
Support relocating historical absolute artifact paths to a declared local root,
while preserving identity by hash. Resolve every scheduled parent before a full
launch; never substitute a different seed, an earlier favorable checkpoint, or
a reconstructed model when its update-500 source is missing.

Use the existing `envs/jax-gpu` uv project and lock. The parent records:

- Python 3.11.6; JAX/JAXLIB 0.10.2; NumPy 2.4.6; Optax 0.2.8.
- Lock SHA-256 `7a8e627cd9851d7426502bc3734bcedcfe48277575aec363877d4e7b717b2125`.
- `jax_enable_x64=false`, `jax_threefry_partitionable=false`, highest dot
  precision for categorical `p @ T`, and
  `XLA_FLAGS=" --xla_gpu_deterministic_ops=true"`.
- AdamW defaults `mu_dtype=None` and `nesterov=False`, in addition to the
  epsilon settings above.

Verify these against all parent manifests and the installed runtime. Record
actual GPU, driver, kernel, uv version, environment versions, lock, compiler
flags, and dtypes. Use a fresh process with `JAX_ENABLE_X64=0` before imports;
R003's FP64 process settings must not carry into R004. Enabling X64 and casting
the sampled arrays back to FP32 is not the original RNG contract. No mixed
precision, new optimizer defaults, or dependency upgrade belongs in v1.

## Exact continuation semantics

Restore every field needed to continue the computation:

| Saved item | Required treatment |
| --- | --- |
| Gate parameters | Load each run's own FP32 leaves without conversion or reinitialization |
| Optimizer state | Restore all moments, Adam bias-correction count, and any other state leaves |
| `model_key` | Restore the learner key and preserve the source's split convention |
| `data_key` | Restore the independent training-input stream at the end of update 500 |
| `update_index` | Remains global: 500 at load, 501 after the first new update |
| Wiring | Load the saved, hash-verified arrays; never resample |

Creating an optimizer instance or a shape template is permitted; its freshly
initialized state must never replace the loaded optimizer state. The stateful
object must be restored, not merely the parameter dictionary.

The logical step from k-1 to k is the existing R002 operation:

```python
data_key, inputs = sample_training_batch(data_key, config)
state, pre_update_loss, gradients, parameter_updates = train_step(
    state, inputs, target, wires
)
assert int(state.update_index) == k
```

The model-key split inside `r002.make_train_step` remains part of this contract
even though its secondary loss key is currently unused. Preserve sampler order,
target construction, loss reduction, clipping order, AdamW update semantics,
and parameter sharing. Record `pre_update_loss` as an observation on the model
before update k; checkpoint evaluations describe the post-update-k model.

Compilation, evaluation, export, and profiling must not advance either live
random key or modify optimizer state. Warmups may use disposable copies. Verify
state/key identities around evaluation in preflight.

Use a new R004 runner/config validator. The existing `r002.run_condition` always
initializes a fresh trial, and its validator freezes the 500-update recipe;
merely changing its JSON budget is insufficient. Reuse its mathematical kernels
and checkpoint machinery without weakening R002's historical contract.

When loading a parent, validate its metadata as R002 with its original config
hash. Save descendants under R004 metadata with the R004 config hash, global
update, parent run/checkpoint hashes, and wiring identity. Do not relabel the
source checkpoint or save into an R002 artifact directory. Reusing NPZ format
version 2 is fine if its serialization semantics remain unchanged.

## Pairing, interruption, and the continuation ledger

For a given seed, all four parents must have identical saved `data_key` values
and the recorded pairing identities for wiring and the original training-input
stream. Compare the actual new training batches across conditions using a hash
for every global update 501–2,000. The four condition histories remain distinct;
only their input streams are paired.

For each actual batch, compute SHA-256 over the same typed-array encoding used
by R002: `str(dtype).encode()`, then `json.dumps(shape).encode()`, then contiguous
C-order array bytes. Store that batch hash with its global update in the metric
record. Define `continuation_batch_hashes_sha256` as SHA-256 of the ASCII sequence
`501:<hash>\n502:<hash>\n...2000:<hash>\n`. Verify that every condition for a seed
has the same complete ordered sequence.

Keep the parent's original `training_stream_sha256` separately. A SHA-256 digest
cannot serve as a restored incremental hasher state, and concatenating parent
and continuation digests does not reproduce the digest of the full raw stream.
The new, explicitly named indexed-batch digest avoids that ambiguity.

Write checkpoints atomically after the device has completed the corresponding
update. Resume an interrupted R004 run from its latest valid complete checkpoint,
restoring both keys, all optimizer leaves, and the global index. Hash-check the
saved config and parent identity before resuming. Preserve attempt history and
mark metrics beyond the durable checkpoint as superseded if they must be
recomputed; the final scientific ledger has one authoritative record per update.
Do not silently append duplicate updates or choose the checkpoint with best loss.

Rebuild the indexed-batch digest from committed metric records on resume. If
records are missing, replay the sampler alone from the saved parent data key to
recover the exact relevant hashes and verify the resulting key; do not advance
the live training state while doing so. An emergency checkpoint at a non-cadence
update may support recovery, but does not add a selected scientific endpoint.

A resumed attempt is the same parent trial, not another seed draw. Record failed,
missing, and nonfinite trajectories explicitly. Unresolved outcomes are not zero
errors, and hardware/code failures are not additional independently sampled
training failures. Continue accounting for the full planned cohort and report
any unresolved denominators rather than replacing a run.

## Measurements during and after training

Retain R002's cheap per-update fields: pre-update loss, gradient maximum and L2
norm, clipped-element count, parameter-update norm, q-step maximum and L2 norm,
native/common gate-ID change counts, and synchronized update time. Add global
update, continuation update `k-500`, and actual batch hash. Parameter or gate-ID
motion by itself does not establish a neutral path.

At k = 500 and every 50 updates thereafter, evaluate the unchanged original
32-grid probe at runtime tick 20. Record soft loss; native and common hard bit
errors; perfectly reconstructed grid count; all-32 exactness; soft/hard gap;
gate-family frequencies and rounding margins using the existing diagnostics.
Retain the original R002 record at k = 500 and separately identify its R004
replay check. Do not average the two as repeated measurements.

There are 31 evaluation positions per run including k = 500, and 30 newly
trained checkpoints. Thus a complete continuation adds 1,920 checkpoints and
accounts for 1,984 labeled evaluation positions. Keep the parent baseline by
reference instead of duplicating its large source checkpoint. Preserve the
original pre-500 history when plotting the combined trajectory.

Save full model/optimizer/RNG checkpoints every 50 updates and a compact
first-original-probe soft/native/common state trajectory for ticks 0–20, as in
R002. More expansive state dumps at every optimizer update are unnecessary.
Export native and common Boolean circuits at the structural snapshots, with
checkpoint, wiring, and extraction identities. Original k = 500 exports can be
referenced directly.

Run structural and longer-runtime analysis offline against saved artifacts so
it cannot affect the training stream or contaminate measured GPU update time.
Use the following fixed schedules:

| Analysis | Global updates | Scope |
| --- | --- | --- |
| Structural extraction | 500, 750, 1000, 1250, 1500, 1750, 2000 | All 64 runs, both hardening modes: 896 labeled snapshots |
| Runtime/orbit profile | 500, 1000, 1500, 2000 | All 64 runs, both modes, all 66 starts: 33,792 labeled trajectories |

The 500-profile portion already exists. Verify and reuse compatible results or
recompute them under the same semantics; retain the reference in either case.
The three new runtime checkpoints add 25,344 labeled trajectories. Reuse
byte-identical rules across checkpoints/modes only with compatible boundary,
initializations, target, observer, horizon, and analysis implementation. Preserve
all source labels and do not count reused computations as new evidence.

## Fixed probes and structural/runtime definitions

The original probe NPZ is
`artifacts/R001_20260909T231016Z_seed23_jaxgpu_attempt01/fixed_evaluation_set.npz`,
SHA-256 `d49ecf23fd932fba25b1068f8c90820265a05fd90daadc372089520889090b2f`.

For runtime profiles, reuse the additional 32-grid array from the completed
R002 profile, plus the all-zero and all-one starts. The additional NPZ is
`artifacts/R002_formal_attempt01_runtime_profile_attempt01/fresh_probe.npz`,
SHA-256 `4a64f801d604478f7d8ffa40db8255b8d7aaabbdb560173fe1946e00659d1478`.
Its `inputs` array has shape `(32,16,16,8)`, dtype `uint8`, and raw C-order hash
`0bd05a2d3f281ff9f5e26c118677ba91ff9fc9c3bf777eed4ede6bca60387d65`.
If recovery is required, the exact array is generated by
`Generator(PCG64(20260910)).integers(0,2,size=(32,16,16,8),dtype=uint8)`;
verify its raw hash and separately record any newly serialized NPZ identity.
Do not call this reused array a newly unseen test set.

For circuit size, retain the pinned `FactoredDAG` simplifier in
`reviews/R002_circuits/refine_circuits.py` and the current recurrent visible-core
closure definition. Record `visible_core_binary_nodes`,
`visible_recurrent_channels`, `binary_nodes_all_channels`, AND/XOR counts, and
relevant local inputs. Treat complemented edges, constants, and routing exactly
as in the original analysis. Use common hardening as the primary size comparison;
retain native results separately. Do not introduce a stronger simplifier only
for later checkpoints. These counts are constructive bounds in the declared
representation, not minimum or complete description lengths.

For runtime, preserve states at every integer tick 0–256, the original report
ticks `20,21,22,25,32,64,128,256`, exact full-state cycle detection, and the
sound universal-state propagation capped at 256. It is sufficient to retain
every-tick error/Hamming arrays, cycle evidence, and selected packed witnesses;
storing every full state for all trajectories is unnecessary.

Use the existing profiler's execution and proof machinery through a validated
R004 adapter. Its current source loader and validator assume R002 update 500;
do not bypass their checks or relabel new artifacts as R002. Include training
update in source records and cache provenance. Preserve the integer-range check
and canonical uint8 normalization used for Boolean rule hashes.

Apply the two reporting refinements identified in the existing review: record
observed phase dependence, an explicit initialization counterexample, and an
inconclusive universal test as separate flags; distinguish per-start orbit-
certified onset from a universal all-initial-state onset. Absence of a universal
certificate is not evidence of dependence. The primary training success remains
all-32 correctness at tick 20, even when a later runtime readout succeeds.

## Focused preflight before the 96,000-update continuation

Use seed 0 in all four conditions for recurrent resume tests. Verify each
required source artifact and preserve all originals.

1. **Historical replay:** restore each update-450 checkpoint and run the
   unchanged source operations for updates 451–500. In the recorded environment,
   require exact array equality at 500 for parameters, optimizer leaves, model
   and data keys, and counters against the saved parent. Compare arrays, not
   newly serialized NPZ file hashes. A mismatch requires diagnosis before the
   cohort; do not silently relax this into an approximate continuation claim.
2. **Budget-field parity:** from each actual saved update-500 state, compare
   four new steps using the old step factory/config and the R004 orchestration
   with the 2,000-update budget. Inputs, state, optimizer leaves, and keys must
   match exactly at each step. The old budget is not used inside the constant-
   rate step mathematics, but the wrapper must preserve that fact.
3. **Interruption parity:** compare ten uninterrupted updates from 500 with
   five updates, save/load, then five more. Insert checkpoint evaluation/export
   on one path and verify exact endpoint state and batch-hash sequences. Test
   resume bookkeeping with uncheckpointed tail records and reject mismatched
   config/parent metadata. These preflight descendants do not seed formal runs.
4. **Cohort boundary checks:** load all 64 update-500 parents; verify pairing,
   exports, and source metadata. Replay the original probe metrics before any
   formal update. Require exact Boolean outputs and compare soft outputs/loss
   under the existing FP32 policy (`atol=1e-6`, `rtol=1e-4`), reporting measured
   residuals. Parameters and restored state remain exact copies, regardless of
   the soft-evaluation tolerance.
5. **Analysis adapter:** establish the update-500 baseline with the unchanged
   structure/runtime definitions. It must recover original common-hard tick-20
   counts 1/16, 1/16, 2/16, 0/16 in the stated condition order and the existing
   per-seed structural counts. Check source versus simplified Boolean execution,
   every-tick phase bookkeeping, and the reviewed seed-4 convergence and seed-14
   phase examples. Reporting refinements must not change the underlying replay.

Add focused tests for these real failure modes, not a second broad reproduction
of all R000 work. Record failed preflight attempts and fixes. Do not accept a
reset-optimizer test as checkpoint-resume validation. A changed hardware or
compiler environment that cannot reproduce the intended continuation needs an
explicitly documented protocol/runtime decision before a formal launch.

Once the checks pass, start every formal descendant from its untouched R002
update-500 checkpoint. Resume existing validated R004 attempts only when their
identities agree. No further outcome-dependent checkpoint selection is allowed.

## Analysis questions and uncertainty

The primary endpoint is k = 2,000 under common hardening on the original 32
probes at tick 20. Report all 16 seed outcomes per condition, common/native
error counts, and paired changes from 500. The four endpoint categories are
gained (inexact to exact), lost (exact to inexact), exact at both endpoints, and
inexact at both endpoints. Neither same-status category implies an unchanged
status at the intervening checkpoints.

Separately report whether a solution was ever observed, was later lost, or
was never observed across the complete saved R002/R004 history. A first
successful saved checkpoint is a cadence-limited observation; retain the full
success/failure timeline and do not substitute the best intermediate result
for the final endpoint.

Let `C_{s,r,d}(k)` be the common-hard visible-core operation count for seed s,
representation r, and decay history d. Track the paired size gap

\[
D_{s,d}(k)=C_{s,\mathrm{truth},d}(k)-C_{s,\mathrm{categorical},d}(k)
\]

and its change `D(2000)-D(500)` across seeds. Show size and hard error together,
including connected per-seed paths through the declared structural snapshots.
This distinguishes simplification accompanying better behavior from collapse
to a small failing rule. Comparisons restricted to successful circuits are
descriptive and selected by outcome; they do not establish a causal size bias.

At the runtime checkpoints, track fixed-readout correctness, time to persistent
visible correctness when certified, mixed-phase cycles, persistent failure,
and unresolved cases separately. A frozen circuit's old failure certificate
does not constrain a later trained circuit with different parameters. Do not
use a powers-of-two readout alone to infer settling.

Resample training-seed groups, carrying all four conditions and all time points
together. Use 100,000 paired bootstrap resamples generated with
`Generator(PCG64(20260911))`, drawing an int64 index array of shape `(100000,16)`
uniformly from 0 through 15. Reuse these indices across contrasts and report
percentile 95% intervals for mean paired hard-error and size changes. Include
raw seed values; sparse successes and n = 16 limit precision.

For endpoint categorical-versus-truth success comparisons within each decay
history, reuse the two-sided exact paired McNemar/binomial calculation and
report Holm-adjusted p-values across those two comparisons. Other timepoints,
decay contrasts, and conditional structural summaries are descriptive; do not
search them for an uncorrected winner. Preserve the original R002 statistics
and label the later endpoint separately. Do not present repeated checkpoints,
probe grids, or two hardening modes as independent trials.

If a cohort remains incomplete, show scheduled and resolved counts and the
missing seed identities; report bounds on unresolved success rates. Full-cohort
paired inference requires its planned seed groups. Runtime, correctness, and
constructive size are separate observables; no Kolmogorov-complexity or
sequence/grid-size generalization claim follows from this continuation alone.

## Deliverables and execution contract

Add a separate `recurrent_circuit_learning/r004_runner.py` and `scripts/r004.py`
(routine module organization is discretionary), plus a validated analysis
adapter. Reuse R002's mathematical update code and immutable source conventions.
Add `preflight`, per-run `run`, cohort `sweep`, and `analyze` entry points.
`--resume` must operate only on matching existing R004 identities; completed
runs can be skipped with verified provenance. Successful completed runs must
return exit status zero, even if their result status is named `completed`
rather than a test's `passed`.

The following commands define the CLI to implement; they are not currently
runnable commands claimed by this document:

```bash
JAX_ENABLE_X64=0 uv run --project envs/jax-gpu --locked python scripts/r004.py preflight --config configs/experiments/r004_training_continuation_v1.json --run-id preflight_attempt01
JAX_ENABLE_X64=0 uv run --project envs/jax-gpu --locked python scripts/r004.py sweep --config configs/experiments/r004_training_continuation_v1.json --concurrency 1 --sweep-id formal_attempt01
JAX_ENABLE_X64=0 uv run --project envs/jax-gpu --locked python scripts/r004.py analyze --config configs/experiments/r004_training_continuation_v1.json --sweep-id formal_attempt01 --analysis-id analysis_attempt01
```

Use one GPU training process. Keep the R003 and R004 processes and outputs
separate; schedule GPU work so competing experiments do not obscure throughput
measurements. Measure compile time, synchronized training-update time, total
wall time, and peak memory separately. Compare both update counts and measured
costs; the two representations do not necessarily cost the same per update.

Store per-run small outputs under
`results/R004_{sweep_id}_seed{seed:02d}_{condition}/`, aggregate training results
under `results/R004_{sweep_id}/`, and structural/runtime analysis under
`results/R004_{sweep_id}_{analysis_id}/`. Each manifest records source/checkpoint
lineage, code/dirty patch, config, environment/lock, actual hardware, counters,
keys or their hashes, batch-stream identities, status, and artifact checksums.
Retain per-update metrics, checkpoint evaluations, per-seed structural/runtime
tables, source/control validation reports, and an explicit failure ledger.

Keep full checkpoints and array payloads in corresponding `artifacts/` paths
outside git. At 1,920 new checkpoints, artifact volume is material: measure it
during preflight and report a projection. Preserve the scientific cadence;
reduce redundant copies and state dumps rather than dropping unfavorable or
intermediate checkpoints. Use the existing experiment-scoped release/hash
workflow for `experiment/R004` after completing the run.

Supply a compact review archive with all small cohort results, configuration,
validation and provenance, plus source/derived arrays for seed 0 in all four
conditions at k = 500, 1,000, 1,500, 2,000. Reference the original k = 500 source
without duplicating its entire archive. Additional illustrative circuits may be
included with an explicit selection rule and do not replace this fixed subset.

Figures should show paired error/soft-loss trajectories, success gained/lost,
the size-gap trajectory, joint size-versus-error paths, runtime/cycle evidence
at the four analysis checkpoints, and training cost by condition. Explain the
result in terms of more updates to the same learner. Longer training and R003's
representative intervention answer complementary questions; do not pool them.

Completion means all 64 scheduled descendants are accounted for, every valid
completed run reaches global update 2,000, the fixed analyses are recorded, and
the source/continuation artifacts can be independently reviewed. The handoff
itself is not evidence that any of those runs have occurred.

## Source anchors

- [R002 specification](../R002/SPEC.md) and
  [frozen configuration](../../configs/experiments/r002_paired_gate_coordinates.json)
- [R002 mathematical kernels and checkpoints](../../recurrent_circuit_learning/r002.py)
  and [runner](../../recurrent_circuit_learning/r002_runner.py)
- [Original sampler and optimizer configuration](../../recurrent_circuit_learning/difflogic_ca.py)
- [Independent R002 trajectory review](../../reviews/R002_formal_attempt01/REVIEW.md)
- [Runtime profile handoff](../R002/RUNTIME_PROFILE_HANDOFF.md),
  [profiler](../../scripts/r002_runtime_profile.py), and
  [independent runtime/structure review](../../reviews/R002_runtime_profile/REVIEW.md)
- [R003 handoff](../R003/HANDOFF.md)
