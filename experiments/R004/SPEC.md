# R004 — Uniform continuation of the R002 training cohort

Status: specified for implementation, 2026-09-11. No R004 runner, preflight, or
training result is claimed. The authoritative v1 protocol is the
[implementation handoff](HANDOFF.md) and
[configuration](../../configs/experiments/r004_training_continuation_v1.json).

Continue all 64 original R002 `formal_attempt01` trajectories from their own
update-500 checkpoints to global update 2,000. Preserve every parameter,
optimizer moment/counter, model and data key, wiring array, and representation.
Keep all four original conditions, seeds 0–15, FP32, learning rate 0.05,
AdamW/decay, elementwise clipping at 100, summed terminal squared error, batch
size two, and the 20-tick synchronous training rollout.

This adds 1,500 updates per run, or 96,000 across the cohort. It is a continuation
of the same seed groups, not a new independent cohort or a transfer curriculum.
Do not restart, retune, anneal, stop at first success, or use an R003-modified
checkpoint. R003 remains a separate same-function mechanism diagnostic.

The primary endpoint is common-hard correctness on the original 32 probes at
runtime tick 20 after optimizer update 2,000. Preserve evaluations/checkpoints
every 50 updates, measure structure every 250 from 500 through 2,000, and run
the declared 66-start, 256-tick runtime profile at 500, 1,000, 1,500, and 2,000.
Record common and native extraction separately. Compare correctness and
constructive circuit size together, with paired changes from the original 500
endpoint and uncertainty over training seeds.

The preflight must demonstrate exact 450-to-500 historical replay, parity of
the new continuation orchestration, uninterrupted versus save/resume agreement,
and update-500 source/evaluation/analysis checks. Preserve all original R002
results and artifacts. Use separate continuation manifests, output paths, and
failure/resume ledgers; missing outcomes are not silently replaced.
