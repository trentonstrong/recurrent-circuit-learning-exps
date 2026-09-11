# R003 independent record and implementation review

Reviewed 2026-09-11 at source commit
[fe14ca65cf62cdc3e283cf8891a745a46967bd42](https://github.com/trentonstrong/recurrent-circuit-learning-exps/commit/fe14ca65cf62cdc3e283cf8891a745a46967bd42).
Canonical experiment: [R003_formal_attempt02](../../results/R003_formal_attempt02/summary.md).
Supplement: [review_metrics.json](review_metrics.json);
reproduction: [recompute.py](recompute.py).

## Finding

The committed implementation and records support the stated conclusion:
changing the categorical representative while holding the effective truth
tables fixed changes the local Euclidean-SGD response, including the visible
outputs. The response changes include substantial changes of direction at
update 250. They cannot in general be reduced to a scalar learning-rate change.

This is evidence about the local geometry of the actual saved models, beyond
the algebraic possibility established by the synthetic example. It does not
establish better training, spontaneous exploration of exact fibers, or the
cause of R002's circuit-size separation. R002 used AdamW; R003 deliberately
measures a different, plain-SGD diagnostic.

## What was independently checked

The review fetched the pinned source, all 96 committed case records, the
canonical analysis and manifest, and the six final-preflight records. It
verified the Git blob identity of the case file and the recorded SHA-256
identities of four implementation files, the frozen config, and analysis.json.

The supplemental script checks the complete seed/condition/update cross-product,
all 288 reported pre-step invariance outcomes, all 576 step labels, source and
derived-artifact ledger consistency, all 18 reported preflight chain-rule
checks, the 935 native extraction changes, and zero common extraction changes.
It independently recomputes the case-derived global and grouped summary
statistics and the single-arm linearization-residual summaries.

Source inspection confirms the replacement is detached before differentiation,
the three arms use their own effective tables and gradients, the shared-gradient
comparison isolates M, and both learning rates start independently from each
untouched arm. Saved optimizer moments and random keys are not used by the
diagnostic. The original failed preflight records a target batch-shape mismatch;
it was retained, with subsequent attempts recording the corrected execution.

**Review boundary:** this is a source and committed-record audit. The external
checkpoint and derived-array payloads were not available in this review. Their
hashes are recorded, but their bytes and the GPU computations were not
independently replayed. The published cross-arm finite-output-delta statistics
require those arrays and were not independently recomputed. The reported
27-test run was not independently rerun.

To reproduce the supplemental record audit from the repository root:

    python reviews/R003_same_function/recompute.py

It requires only Python's standard library and the committed files. The input
hashes pin this review to the examined result; a changed canonical input should
receive an explicit new review.

## Why the direction measurements matter

Write the effective truth entries of all shared gate slots as Q, and the
unregularized loss gradient as h = grad_Q L. Per gate,

\[
p=\operatorname{softmax}(z),\qquad q=T^\top p,\qquad
J_{rg}=p_g(T_{gr}-q_r),\qquad M=JJ^\top.
\]

A small logit-SGD step induces

\[
v=\frac{dQ}{d\eta}\bigg|_{\eta=0}=-M h,\qquad
w=D_Q O(Q)\,v,
\]

where O observes terminal channel zero on the four fixed probe grids. In the
shared-point comparison, Q, h, and the derivative of O are held fixed. Only
the categorical representative and hence M change.

A change in response norm alone could amount to a different effective step
size on these probes. A nonzero response angle demonstrates a change that no
single positive scalar learning rate can reproduce to first order.

The supplemental angles use the recorded common-gradient q responses and
shared-Q output responses. They are reconstructed from the norms by

\[
\cos\theta=
\frac{\|u\|^2+\|v\|^2-\|u-v\|^2}{2\|u\|\|v\|}.
\]

Zero norms would be reported as undefined; none occur in these primary
comparisons. Full-array dot products remain a useful payload-level confirmation,
especially for angles very near zero.

All entries below are medians over 16 seeds. Output measurements concatenate
all four observed grids. The norm ratio is factorized/original.

| Training history | Update | q-response angle | Output-response angle | Output norm ratio | Relative output-response difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference decay | 0 | 17.54° | 0.23° | 1.762 | 0.432 |
| Reference decay | 250 | 22.32° | 24.27° | 1.012 | 0.617 |
| Reference decay | 500 | 7.29° | 1.63° | 0.996 | 0.0587 |
| No decay | 0 | 17.54° | 0.23° | 1.762 | 0.432 |
| No decay | 250 | 56.18° | 68.34° | 3.862 | 0.966 |
| No decay | 500 | 6.36° | 1.76° | 1.003 | 0.0398 |

The initialization rows duplicate the same 16 starting models across the two
decay histories; they are not independent replications. There are 16 seed
groups, not 96 independent trials or 288 independent experiments.

At initialization, the output effect is predominantly a magnitude change even
though the internal q-response rotates. At update 250, substantial rotations
reach the output. The median output angles on loss inputs alone are 24.28°
and 68.34°; on the two additional inputs they are 24.25° and 68.34°. Thus this
observation is not specific to the inputs defining the diagnostic gradient,
although two additional inputs are a very limited probe.

At update 500, the typical output effect is much smaller. Relative output
differences and output angles both decrease from update 250 in 14/16
reference-decay histories and all 16 no-decay histories. This is a statement
about this intervention, loss, and observer at three saved times. It does not
show that fibers disappear, that training is stationary, or that all functionally
relevant directions have become insensitive.

There are outliers: the largest reference-decay output angle at update 500 is
about 71°. Medians should not be presented as universal behavior.

## Immediate descent is a separate measurement

The factorized/original ratio of the predicted local loss decrease,
(-dL/deta)_factorized / (-dL/deta)_original, has medians:

| Update | Reference decay | No decay |
| --- | ---: | ---: |
| 0 | 1.764 | 1.764 |
| 250 | 1.152 | 2.378 |
| 500 | 1.002 | 1.003 |

At update 250, the factorized representative has a larger predicted immediate
decrease in 13/16 reference-decay and 14/16 no-decay cases. This uses the same
small SGD step size and the two selected loss inputs. It is not a training
speedup, an AdamW prediction, or evidence of a better final solution. Norm and
direction must both be considered in any follow-up claiming a search benefit.

## Numerical scope and coverage

The pre-step same-function controls are strong: maximum effective-table error
is 7.77e-16, maximum recurrent-state error is 6.33e-12, and all common-rounded
gate IDs remain unchanged. These are comparisons between FP64 diagnostic arms.

The reported factorized-arm output linearization residual is normalized by
the larger of the measured and predicted step norms. Reducing eta from 0.001
to 0.0001 reduces this residual by a median factor of 9.999 across paired
checkpoints; individual reduction factors span 9.652–10.057. That is the
expected scaling for a second-order remainder divided by a first-order step.
An unnormalized remainder would instead be expected to decrease by about 100.

The worst factorized-arm relative output residual is 0.150 at eta=0.001 and
0.0149 at eta=0.0001. The smaller-step audit supports the local interpretation;
it would be inaccurate to describe every primary finite step as having
negligible linearization error.

The median 0.331 headline is the L2 difference between output response
derivatives on the **two loss inputs**, per unit eta. It is not an output
change of 0.331 caused by the finite step. The published median difference
between actual finite output steps at eta=0.001 is 0.000465 over **all four
observed grids**. These numbers differ in both scaling and observation scope.

The separate FP32-to-promoted-FP64 output comparison records 30/96 cases outside
its diagnostic threshold (atol=1e-6, rtol=1e-5). The largest maximum absolute
output difference is 0.00108356, at reference-decay seed 5, update 250.
This does not invalidate the within-FP64 neutrality result: all arms use the
same declared precision. It does mean that the passing neutrality headline
must not be read as passing FP32/FP64 parity or direct validation of the
original FP32 optimizer's response. These precision differences are retained
in the supplemental metrics.

Gate-count coverage and response-weighted coverage differ markedly in a few
cases. No-decay seed 13 at update 500 replaces about 98% of gates but only
0.00319% of the baseline q-response squared norm. Its small observed effect
therefore tests a limited part of the currently active response. However, the
small median endpoint effects are not solely explained by these extreme cases:
among update-500 cases with at least 99% response-weighted coverage, median
relative output differences remain 0.0423 (11 reference-decay cases) and
0.0365 (10 no-decay cases). This threshold analysis is exploratory and does not
replace the full-cohort result.

## Extraction, remaining reporting details, and next step

The 935 native argmax changes are gate-slot differences summed across the 96
original-to-factorized checkpoint contrasts, not 935 unique reusable gadgets.
They consist of 137 and 532 changes at reference-decay updates 250 and 500,
and 87 and 179 without decay; initialization contributes zero. Since common
rounding is unchanged, the same-function reset supplies no common-hard circuit
size improvement before training.

Some requested instrumentation is incomplete in the committed report. This
review adds the missing original-to-factorized response angles from existing
norms. The runner computes all four eigenvalues per gate but retains only
extrema and numerical-rank counts in JSON; the saved p arrays would permit
reconstruction of each M and its spectrum. The separate FP32/FP64 common-rounding
comparison is not recorded by the runner. The synthetic gauge control checks q,
but does not explicitly check the requested M and response invariance.
These gaps do not overturn the central response measurement and should remain
visible when describing protocol coverage.

Keep R004's continuation unchanged: it asks whether the existing paired runs
develop differently with more training. A subsequent, separately specified
intervention can test whether same-function representative changes improve
search. Update 250 is a motivated candidate for that experiment because the
present probe shows large functional direction changes there; this is a
post-result design choice, not a predeclared training result. Any such experiment
needs a stated optimizer-state policy and controls for step magnitude before
attributing a benefit to neutral exploration.
