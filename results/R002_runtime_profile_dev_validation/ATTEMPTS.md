# R002 runtime-profile development validation attempts

These checks were performed while implementing the frozen
`R002_runtime_profile_v1` protocol. They are development evidence, not the
formal analysis result.

1. The first source-validation attempt rejected the update-500 checkpoints
   because it expected a `run_id` field that is not part of the checkpoint
   metadata schema. The validator was corrected to check every metadata field
   the checkpoint actually records: condition, representation, seed, config
   hash, and wiring hash.
2. A later invocation used the nonexistent nested artifact root
   `artifacts/R002_formal_attempt01_release_assets` and failed before profiling
   with `FileNotFoundError` for the first checkpoint. This was an invocation
   error; rerunning against `artifacts/` succeeded.
3. The completed development pass verified all 128 run/hardening modes, 256
   unique source artifacts, tick-20 metrics, saved trajectories, unsimplified
   versus simplified execution, cycle bookkeeping, the selected six-circuit
   regressions, universal certificates, and the published seed-14 phase
   witness. It then profiled all 96 unique Boolean rules in 42 seconds and
   reproduced all 128 logical modes. A matching retry restored all 96 records
   from the integrity-checked resume cache and completed in 6 seconds. The
   transient development outputs were discarded before creating the clean
   formal analysis record.
