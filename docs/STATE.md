# Research state

Updated: 2026-09-09. R000 implementation and office-5090 parity are complete.
The canonical R001 seed-23 run completed and established the recurrent positive
control. Review that evidence before starting R002.

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
| [R000](../experiments/R000/SPEC.md) | Pinned source, standalone JAX, numerical parity | Passed on CPU oracle and office 5090 |
| [R001](../experiments/R001/SPEC.md) | Synchronous checkerboard, canonical seed 23 | Completed; fixed 32-grid probe exact at update 500 |
| [R002](../experiments/R002/SPEC.md) | Paired gate-coordinate and weight-decay comparison | Proposed; follows R001 review |
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
training result.

[R001 seed 23](../results/R001_20260909T231016Z_seed23_jaxgpu_attempt01/summary.md)
completed the frozen 500-update recipe. Its added fixed 32-grid probe first had
zero native-hard bit errors at checkpoint 250 and remained exact through the
primary update-500 checkpoint. The final soft summed squared error on that set
was 0.1815665. This establishes one working recurrent-learning example, not a
success-rate estimate or a sequence-compression result.

The source and dimensionality audits are recorded in the
[manifest](../references/difflogic_ca/source_manifest.json). These are completed
audits, not new recurrent-training results.

The earlier counter readout pilot, summarized in the handoff, found 0/32 correct
circuits in its short 128-update runs. Near-copy initialization kept the output
path bypassing the LUTs. That learner and budget differ from DiffLogic CA.
Legacy code and raw artifacts have not been imported into this repository; no
claim of reproducing those runs here is made.

## Next question

With the pinned synchronous recipe working under a recorded environment, the
next question is whether the matched gate-coordinate and weight-decay conditions
in R002 change optimization behavior. Do not launch that comparison until the
R000/R001 evidence and protocol are reviewed.

The [review of commit e94789f](../reviews/R001_e94789f/REVIEW.md) supports the
R001 positive control and records a remaining validation gap: the committed R000
fixtures compare the extracted implementation across runtimes. Add direct
vendored-notebook execution parity, including the X64-disabled sampling mode
used by R001, before launching R002. Shared access to the existing checkpoint
bundle is now provided by the experiment-scoped
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
