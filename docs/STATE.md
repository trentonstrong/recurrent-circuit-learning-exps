# Research state

Updated: 2026-09-11. The formal 64-run R002 paired sweep, independent trajectory
review, artifact publication, selected six-run hard-circuit review, and bounded
full-cohort runtime profile are complete. The profile distinguishes late
convergence, phase-dependent success, and persistent Boolean failure without
changing the frozen tick-20 outcomes. R003's same-function diagnostic and R004's
uniform training-budget continuation now have separate fixed handoffs and
configurations. Their implementation and validation are the next steps; no
execution of either new experiment is claimed.

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
| [R002](../experiments/R002/SPEC.md) | Paired gate-coordinate and weight-decay comparison | Sweep and bounded runtime profile completed; no supported coordinate or decay advantage |
| [R003](../experiments/R003/SPEC.md) | Same-function mixture interventions | Handoff and v1 protocol specified; implementation pending |
| [R004](../experiments/R004/SPEC.md) | Continue all R002 runs from 500 to 2,000 updates | Handoff and v1 protocol specified; implementation pending |

The [R003 implementation handoff](../experiments/R003/HANDOFF.md) and
[configuration](../configs/experiments/r003_same_function_v1.json) fix a 96-case
checkpoint diagnostic: all 32 categorical R002 runs at updates 0, 250, and 500.
Three mixture representatives preserve each eligible gate's effective table;
FP64 invariance checks precede geometry measurements and independent plain-SGD
steps at 0.001 and 0.0001. Saved Adam/RNG state is preserved. The handoff tests
representative-dependent response, not a claim of improved training or a causal
explanation of circuit size.

The [R004 implementation handoff](../experiments/R004/HANDOFF.md) and
[configuration](../configs/experiments/r004_training_continuation_v1.json) continue
all 64 R002 runs from their own update-500 checkpoints to global update 2,000,
preserving parameters, Adam moments/counters, random keys, wiring, FP32, and the
original constant-rate, 20-tick training recipe. Checkpoints/evaluation remain
every 50 updates; structure is measured every 250 updates and the full runtime
profile at 500, 1,000, 1,500, and 2,000. The preflight checks historical replay
and exact resumption. These are longer trajectories of the same seed groups,
not new independent trials, and they do not use R003-modified checkpoints.

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
The 317 MB checkpoint, trajectory, and final-circuit collection is published in
the experiment-scoped
[`experiment/R002` GitHub release](https://github.com/trentonstrong/recurrent-circuit-learning-exps/releases/tag/experiment/R002).
All release assets were downloaded and verified after publication. The
[selected circuit review](../reviews/R002_circuits/REVIEW.md) additionally verifies
144 uploaded payloads and independently replays all 66 saved checkpoints from
six selected runs in both hardening modes. The missing shared probe was
reconstructed byte-for-byte against its recorded SHA-256.

The [independent R002 review](../reviews/R002_formal_attempt01/REVIEW.md) verifies
all 128 committed result files and recomputes the paired statistics. Direct truth
coordinates produce more common gate-change events in every paired seed under
both decay settings, without an established exact-solution advantage. All 64
trials have lower fixed-probe soft loss at update 500 than at 400. These are
trajectory observations, not evidence of neutral paths or stationary failures.
The review distinguishes a possible uniform budget extension from R003's direct
same-function mechanism test; neither follow-up has been launched.

The selected circuit review finds that both successful seed-4 categorical runs
share a two-operation, two-channel visible core: a rotated version of R001's
checkpoint-250 generator. Sound set-valued propagation certifies the target
from tick 16 for every initial state on the fixed 16 by 16 zero-exterior grid.
The seed-13 categorical/reference-decay near miss is correct on all original
probes by tick 22 and certified for all initializations by tick 25; it remains
a failure under the frozen tick-20 criterion. Seed 15 categorical/no-decay is
certified from tick 20. The successful seed-14 direct-truth circuit instead has
an exact two-cycle on the probes, with 128 visible errors per grid in the odd
phase; an explicit initial-state witness fails the prescribed even readout.
Seed 8 direct-truth/reference-decay has a visible cell that is provably wrong
from tick 14 onward for every initialization despite its improving soft loss.

In categorical/reference-decay seed 4, 90 gate-ID differences summed between
saved checkpoints 300 through 500 preserve the entire hard local function.
This establishes equality between saved discrete snapshots, not a continuous
neutral optimizer path. Reported simplified gate counts are constructive
representation-dependent bounds, not minimum or complete description lengths.
These selected outcomes do not establish condition-wide complexity differences
or generalization across grid sizes.

The completed
[`R002_runtime_profile_v1`](../results/R002_formal_attempt01_runtime_profile_attempt01/summary.md)
diagnostic holds all update-500 circuits fixed and measures ticks 0 through 256
for common and native hardening. It accounts for 128 run/hardening modes and
8,448 trajectories from the original probes, one shared fresh probe set, and
the zero/one states. Of 120 modes that failed at tick 20, 22 are demonstrably
late generators under a declared later readout or universal certificate. Of
the eight original-success modes, the two seed-14 truth/reference modes remain
phase dependent; all eight also succeed at tick 20 on the sampled fresh and
constant initializations. Universal propagation certifies persistent failure
for 72 modes. Four modes remain unresolved by both the bounded replay and the
universal abstraction, while another two have unresolved sampled orbits but a
persistent-failure certificate. Common and native counts agree for these
headlines, so the corresponding primary common-hard counts are 11, 1, 36, and
2 circuits respectively.

The analysis verified all 256 unique source artifacts before execution, found
96 unique Boolean rules among the 128 labeled modes, and retained all source
identities while reusing 32 byte-identical profiles. Exact full-state cycles,
ten wrong-phase witnesses, every-tick error/Hamming arrays, universal traces,
figures, and a compact review bundle are recorded under the analysis manifest.
These are exploratory diagnostics on the same 16 paired training seeds, not
new training trials, basin-volume estimates, minimum circuit descriptions, or
evidence of grid-size or sequence generalization.

The [independent runtime-profile review](../reviews/R002_runtime_profile/REVIEW.md)
checks all nine committed result-file hashes and all 8,448 trajectory records,
and replays the six available source runs in both hardening modes through tick
256 (792 logical trajectories). It distinguishes 12 universally convergent
common-hard circuits, one additional circuit with correct fixed points on all
66 starts, five with mixed correct/wrong cycle phases, 36 universal persistent
failures, eight with target-free cycles on all 66 starts, and two unresolved
by both forms of analysis. At tick 64, 15 circuits pass the original probes but
14 pass all 66 starts: truth/reference-decay seed 2 has full period eight,
visible period four, and a different phase for the zero initialization.

The same review finds a complete separation in the current simplified visible
core operation counts: categorical circuits span 2–36, direct-truth circuits
44–236, with direct truth larger in every paired seed under both decay settings.
These are representation-dependent constructive counts, not proven minima.
This gives a concrete motivation for R003's proposed geometry diagnostic; it
does not yet identify the cause of the size difference. Source replay in the
independent review covers six runs, not the other 58 or the external tick-metric
and witness payloads.

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
