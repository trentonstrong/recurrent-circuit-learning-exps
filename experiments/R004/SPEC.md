# R004 — Uniform continuation of the R002 training cohort

Status: completed, released, and independently reviewed 2026-09-12. All 64
continuations and the fixed analysis are
recorded in [`formal_attempt01`](../../results/R004_formal_attempt01/summary.md)
and [`analysis_attempt03`](../../results/R004_formal_attempt01_analysis_attempt03/summary.md).
The [review](../../reviews/R004_continuation/REVIEW.md) recomputes the cohort
statistics and directly replays the seed-0 subset. The experiment-scoped
[release](https://github.com/trentonstrong/recurrent-circuit-learning-exps/releases/tag/experiment/R004)
preserves the original dirty-source provenance. The authoritative frozen v1
protocol remains the
[implementation handoff](HANDOFF.md) and
[configuration](../../configs/experiments/r004_training_continuation_v1.json).

Common-hard original-probe successes rose from 4/64 to 16/64. All 64 soft losses
improved, but visible-core size increased in 62/64 runs. At least 47/64 final
relaxed outputs meet a sufficient thresholded-output correctness bound; 32
still yield failed hard circuits. These are improved fitting and hard-success
results, not evidence of compression or a proven coordinate advantage.

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

The final cohort contains all 96,000 added optimizer updates and 1,920 new
checkpoints with an empty failure ledger. Common-hard exactness at update 2,000
is 3/16, 2/16, 8/16, and 3/16 in the condition order above, versus 1/16, 1/16,
2/16, and 0/16 at update 500. The corrected paired categorical-versus-truth
endpoint tests do not reject equality. Constructive truth-minus-categorical
size gaps grew under both decay histories, but these representation-dependent
bounds do not establish minimum description length or causality. The complete
runtime report retains phase, orbit, universal-certificate, failure, and
unresolved outcomes separately.

Full checkpoints, continuation analysis, runtime arrays, the fixed compact
review subset, and their identity manifest are published in the experiment-
scoped [`experiment/R004` release](https://github.com/trentonstrong/recurrent-circuit-learning-exps/releases/tag/experiment/R004).
Every published asset was downloaded and SHA-256 verified after publication.
