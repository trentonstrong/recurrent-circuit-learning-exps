# Result ledger

No new experiment has run in this repository yet. The source audit is under
`references/`; it is not a training result.

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
