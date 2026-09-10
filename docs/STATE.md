# Research state

Updated: 2026-09-10. The formal 64-run R002 paired sweep is complete. Next:
review its aggregate comparison, then separately validate the successful-circuit
simplification and publish the 317 MB artifact bundle to durable storage.

## Objective

Learn short, complete descriptions of finite sequences whose generators are known
so we can compare against concrete baselines. Study description length and the
trajectory of circuit structure during search. Pi remains a motivating eventual
target; smaller generators establish what the learning machinery can do.

The score must describe a complete decoder: machine, wiring, initialization,
observer, runtime/length contract, and any exceptions. Floating-point parameter
count and entropy penalties are useful proxies, not Kolmogorov complexity or a
complete description by themselves. A compact generator plus a finite exception
stream is allowed if the codec accounts for both.

Primary experiments start from scratch. A curriculum with carried-over weights
would answer a different transfer question. A finite trace cannot by itself
establish an infinite generator; any fixed finite-state autonomous circuit is
eventually periodic. Length and storage-size generalization need separate tests.

## Active path

| Experiment | Purpose | Status |
| --- | --- | --- |
| [R000](../experiments/R000/SPEC.md) | Pinned source, standalone JAX, numerical parity | Direct notebook and cross-runtime checks passed |
| [R001](../experiments/R001/SPEC.md) | Synchronous checkerboard, canonical seed 23 | Completed; fixed 32-grid probe exact at update 500 |
| [R002](../experiments/R002/SPEC.md) | Paired gate-coordinate and weight-decay comparison | Formal 64-run sweep completed; no supported coordinate or decay advantage |
| [R003](../experiments/R003/SPEC.md) | Same-function mixture interventions | Proposed mechanism study |

Use a known recurrent learner as a positive control before redesigning the
sequence learner. The initial reference has fixed wiring, shared recurrent
parameters, synchronous updates, and an external fixed observation schedule.
Its terminal squared-error objective is retained for reproduction. The earlier
cross-entropy, learned wiring, emit interface, noise, and description costs are
available as later named interventions.

Two-input gates have 16 categorical logits but only four effective relaxed truth
entries. Generic interior mixtures have 11 directions preserving the truth
entries, plus the constant-logit gauge. Removing this redundancy changes the
optimizer's geometry; it does not establish that learning must improve. See the
[full derivation and controls](RECURRENT_BASELINE_HANDOFF.md).

## Existing evidence and its limits

[R000](../results/R000_20260909_jax_parity/summary.md) verified the pinned
notebook checksum, exact wiring and Boolean execution, FP64 gate probes, FP32
20-tick rollouts, gradients, an AdamW update, and exact checkpoint resumption.
The modern runtime must retain `jax_threefry_partitionable=false` to reproduce
JAX 0.4.33's seeded arrays. This is implementation/runtime evidence, not a new
training result. The later direct-notebook result closes the independent source
execution and X64-disabled sampling gaps.

[R001 seed 23](../results/R001_20260909T231016Z_seed23_jaxgpu_attempt01/summary.md)
completed the frozen 500-update recipe. Its added fixed 32-grid probe first had
zero native-hard bit errors at checkpoint 250 and remained exact through the
primary update-500 checkpoint. The final soft summed squared error on that set
was 0.1815665. This establishes one working recurrent-learning example, not a
success-rate estimate or a sequence-compression result.

The [checkpoint review](../reviews/R001_checkpoints/REVIEW.md) independently
replays the released hard circuits and all saved hard probe trajectories. Exact
simplification leaves three binary operations in the full eight-channel local
rule at checkpoint 250 and four at checkpoint 500, plus routing and state.
The visible recurrent cores use two and three state channels respectively.
Some recorded gate changes preserve the complete hard update rule; others
preserve visible behavior or only the terminal result. These measurements do
not establish continuous neutral optimizer paths or complete description lengths.

The [rule walkthrough](../reviews/R001_checkpoints/RULE250.md) adds a finite
convergence certificate for checkpoint 250. Sound unknown-value propagation
establishes the target at tick 16 for every initial assignment on its 16 by 16
zero-exterior grid. Its two-channel visible subsystem reaches a fixed point by
tick 17. This is a property of the frozen hard circuit, not another training run
or a size-generalization result.

The source and dimensionality audits are recorded in the
[manifest](../references/difflogic_ca/source_manifest.json). These are completed
audits, not new recurrent-training results.

The earlier counter readout pilot, summarized in the handoff, found 0/32 correct
circuits in its short 128-update runs. Near-copy initialization kept the output
path bypassing the LUTs. That learner and budget differ from DiffLogic CA.
Legacy code and raw artifacts have not been imported into this repository; no
claim of reproducing those runs here is made.

## Current validation boundary

The [direct notebook parity result](../results/R000_20260910_direct_notebook_parity_attempt01/summary.md)
executes the vendored definition cells as an independent historical-CPU oracle
with X64 disabled. Notebook-to-extraction and notebook-array-to-modern-GPU
comparisons pass the existing tolerance policy; Boolean execution, wiring,
initialization, keys, and three successive sampled batches are exact.

The [R002 validation result](../results/R002_20260910_validation/summary.md)
records the common multilinear kernel bridge, FP64 derivative checks, export
checks, and all four two-update seed-23 preflights. Checkpoint resumption is exact
within every condition and the pairing hashes agree. A failed first kernel check
is retained: default GPU matrix-multiply precision was insufficient for `p @ T`,
so the selected kernel declares highest dot precision. That validation supported
the subsequently completed formal launch.

## Formal R002 result

The [`formal_attempt01`](../results/R002_formal_attempt01/summary.md) sweep
completed all 64 frozen trials: seeds 0--15 paired across four gate-coordinate
and decay conditions. Exact update-500 success on the fixed 32-grid probe was
1/16 for categorical/reference-decay, 1/16 for truth/reference-decay, 2/16 for
categorical/no-decay, and 0/16 for truth/no-decay. Native and common hardening
agree on every exact-success classification.

All paired success tests have two-sided exact p-values at least 0.5, and all
paired-bootstrap intervals for mean common-hard error differences cross zero.
This cohort therefore does not support a gate-coordinate or weight-decay
advantage. The sparse successes and wide intervals do not establish equivalence.
R002-specific Boolean simplification of the successful circuits and durable
publication of the 317 MB artifact bundle remain separate follow-up work.

The [review of commit e94789f](../reviews/R001_e94789f/REVIEW.md) identified the
now-closed direct-notebook validation gap. Shared access to the existing R001
checkpoint bundle is provided by the experiment-scoped
[`experiment/R001` GitHub release](https://github.com/trentonstrong/recurrent-circuit-learning-exps/releases/tag/experiment/R001)
for direct circuit and trajectory analysis.

The workstation runs native Arch Linux kernel 7.1.8-arch1-3 with NVIDIA driver
610.57.04. The RTX 5090 uses an explicit modern GPU runtime adaptation, validated
against the historical CPU reference. The independent uv projects retain exact
locks and Python interpreter pins. See [workstation setup](WORKSTATION_SETUP.md).

Trajectory data should distinguish optimizer update, runtime tick, and output
position. Log effective truth tables, discretization, gradients, optimizer state,
and selected runtime states. Use both raw and connected correlators with defined
averaging. RG measurements require a declared coarse-graining map and size family;
no RG constants are being estimated in R001.
