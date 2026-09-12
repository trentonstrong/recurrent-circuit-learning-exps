# Result ledger

R000 implementation parity is recorded in
[`R000_20260909_jax_parity`](R000_20260909_jax_parity/summary.md). It is runtime
and implementation evidence, not a recurrent-training result. The canonical
R001 positive control is recorded in
[`R001_20260909T231016Z_seed23_jaxgpu_attempt01`](R001_20260909T231016Z_seed23_jaxgpu_attempt01/summary.md).
Its large artifacts are published under the experiment-scoped
[`experiment/R001` GitHub release](https://github.com/trentonstrong/recurrent-circuit-learning-exps/releases/tag/experiment/R001).

The direct vendored-notebook oracle and X64-disabled sampling closure are in
[`R000_20260910_direct_notebook_parity_attempt01`](R000_20260910_direct_notebook_parity_attempt01/summary.md).
The common-kernel validation and four-condition seed-23 development preflight are
in [`R002_20260910_validation`](R002_20260910_validation/summary.md). The formal
seeds 0--15 subsequently completed in
[`R002_formal_attempt01`](R002_formal_attempt01/summary.md): all 64 trials
finished, and the paired analysis supports neither a coordinate-system nor a
weight-decay advantage in this cohort. Its large checkpoint, trajectory, and
final-circuit payloads are published under the experiment-scoped
[`experiment/R002` GitHub release](https://github.com/trentonstrong/recurrent-circuit-learning-exps/releases/tag/experiment/R002).

The fixed R003 post-training mechanism diagnostic is recorded in
[`R003_formal_attempt02`](R003_formal_attempt02/summary.md). All 96 checkpoint
cases passed their same-function controls; changing the categorical
representative changed the induced q response and reached the visible output in
every primary contrast. `preflight_attempt01` retains an implementation failure,
`preflight_attempt02` and `formal_attempt01` are passing development records, and
`preflight_attempt03` plus `formal_attempt02` identify the canonical implementation
whose derived bundle retains p arrays for independent geometry reconstruction.

The R004 uniform continuation is recorded in
[`R004_formal_attempt01`](R004_formal_attempt01/summary.md). All 64 verified R002
descendants reached update 2,000, and the fixed structural/runtime analysis is
recorded in
[`analysis_attempt03`](R004_formal_attempt01_analysis_attempt03/summary.md).
The deterministic assets are published under the experiment-scoped
[`experiment/R004` GitHub release](https://github.com/trentonstrong/recurrent-circuit-learning-exps/releases/tag/experiment/R004)
and were downloaded and SHA-256 verified after publication.

Create one directory per run, for example `R001_20260909_seed23_jaxgpu_attempt01`.
Use separate identities for retries and interventions and explain their relation.
Do not silently overwrite a failure with a successful rerun.

Keep small files such as `manifest.json`, `summary.md`, and `metrics.jsonl` in git.
Place large arrays, checkpoints, full state trajectories, and videos in durable
artifact storage. Record each artifact's stable location, size, and SHA-256.
An expiring download URL or an unqualified temporary path is not durable storage.

A run manifest must record:

- Experiment/specification version; complete resolved configuration and its hash.
- Code commit; dirty status and archived patch/hash when applicable; source pins.
- Environment lock identity; Python/framework/optimizer/CUDA versions; actual OS,
  GPU, driver, CPU, precision, compiler, and allocator settings.
- Seed and PRNG convention; wiring, initialization, training/probe data hashes;
  saved RNG state needed for resume.
- Start/end timestamps, current update, status (`running`, `completed`, `failed`,
  or `interrupted`), failure reason where applicable, and parent/retry relation.
- Primary checkpoint selection rule, metric definitions/reductions, and separate
  training, source-compatible evaluation, and added probe results.
- Checkpoint/artifact identities, compile time, synchronized training runtime,
  peak-memory measurement method, and any numerical/protocol deviations.

Never place credentials or secrets in manifests. Keep unknown values explicitly
unknown. Update `docs/STATE.md` with result links and conclusions supported by the
recorded evidence. Report paired seed outcomes before pooling evaluation grids.
