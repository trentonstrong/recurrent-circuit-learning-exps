# Working in this repository

Read `docs/STATE.md` and the relevant `experiments/Rxxx/SPEC.md` before changing
experimental code. The repository begins as a handoff; do not assume a runner
or dependency lock already exists.

- Implement the active experiment against the pinned source. Record adaptations
  and discrepancies explicitly. Preserve attribution for copied upstream code.
- Keep the scientific recipe separate from hardware/runtime configuration.
  Inspect the local OS, driver, and devices before selecting environment packages.
  Lock dependencies after validating them; do not label an untested lock verified.
- Do not silently alter a frozen budget, optimizer, loss, wiring, initialization,
  readout, precision, or evaluation rule in response to an outcome. Assign a
  separate experiment or documented protocol revision to an intervention.
- Independent trials start from scratch. Reusing trained weights is a distinct
  transfer experiment. A failed seed remains a failed trial in the ledger.
- Preserve executable reference semantics, including RNG splitting, boundary
  order, clipping by value, loss reduction, and optimizer defaults. Across
  frameworks, compare identical arrays rather than seed numbers alone.
- Use focused, meaningful parity and export checks. Establish correctness before
  optimizing kernels or increasing concurrency. Do not replace the baseline
  with a faster mathematically different relaxation.
- Keep the state file, experiment status, and result manifests truthful. Distinguish
  proposed work, implemented code, completed runs, and supported conclusions.
- Commit small manifests, configurations, metrics, and reports. Keep large
  checkpoints, arrays, videos, caches, and environments outside git; record stable
  artifact locations, sizes, and SHA-256 hashes. Never commit credentials or
  environment files containing secrets.
- Record the code commit, dirty patch when applicable, config hash, source pins,
  environment lock, actual hardware, RNG conventions, and artifact identities
  needed to resume and interpret each run.

For R000/R001, implementing and running the specified focused checks and canonical
run is the task. Routine implementation choices should be resolved and recorded
locally; unexplained deviations should not be disguised as reproduction results.
