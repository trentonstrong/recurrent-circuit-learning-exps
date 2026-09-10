# R002 formal paired sweep — attempt 01

Status: completed. All 64 frozen trials finished successfully. This 16-seed
paired cohort does not support an advantage for either gate coordinate system or
either AdamW decay setting.

## Protocol and integrity

The run used code commit `d220b65da822707cd7db3dc90df69d9250aa0f77`,
configuration SHA-256
`9632f2135e8262903c40cdaeba211e4f5bebac22c2b8f45c0e2100b993111bb9`,
and fixed-probe SHA-256
`d49ecf23fd932fba25b1068f8c90820265a05fd90daadc372089520889090b2f`.
It ran with JAX X64 disabled on the recorded RTX 5090 environment, concurrency
one, and the common multilinear truth-table kernel.

All 64 manifests are `completed` at the primary update 500. Each condition has
16 trials and each seed has all four conditions. The four conditions within
every seed have identical wiring, categorical initialization, truth
initialization, and ordered training-stream hashes. All initial effective-table
comparisons passed; the maximum absolute discrepancy was
`5.960464477539063e-08`.

The closeout independently parsed all 64 metrics streams and found exactly the
ordered updates 1 through 500 in each. It also recomputed size and SHA-256 for
all 1,536 checkpoint, trajectory, and final-circuit artifact records with no
failure. No update clipped a gradient element; the largest recorded gradient
element magnitude was `48.4465`, below the clipping threshold of 100.

The first run recorded a clean worktree. The later 63 manifests record a dirty
worktree because completed result directories from earlier trials were then
untracked; all record the empty tracked-patch SHA-256, so the executed tracked
code remained the named commit.

## Final outcomes

Exact success means zero common-hard bit errors on all 32 fixed probe grids at
update 500. The intervals are two-sided 95% Wilson score intervals over the 16
training seeds. Probe grids are evaluations, not additional independent trials.

| Condition | Exact seeds | Success rate (95% interval) | Median common errors (IQR) | Median soft SSE | Median common soft/hard gap |
| --- | --- | ---: | ---: | ---: | ---: |
| categorical, decay 0.01 | 4 | 1/16, 6.25% (1.11–28.33%) | 1,373 (772–2,654.5) | 617.654 | 849.618 |
| truth, decay 0.01 | 14 | 1/16, 6.25% (1.11–28.33%) | 2,083.5 (975.5–2,630.25) | 517.237 | 1,442.060 |
| categorical, no decay | 4, 15 | 2/16, 12.5% (3.50–36.02%) | 1,938 (567.5–3,061) | 865.853 | 1,071.976 |
| truth, no decay | none | 0/16, 0% (0–19.36%) | 2,069.5 (1,350–2,219) | 635.234 | 1,210.828 |

Seed 4 first became exact at saved checkpoint 250 under both categorical
conditions. Categorical/no-decay seed 15 and truth/reference-decay seed 14 first
became exact at the primary checkpoint 500. Native and common extraction agree
on exact-success classification for every trial.

The paired common-hard error counts at update 500 are:

| Seed | Categorical, decay | Truth, decay | Categorical, no decay | Truth, no decay |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1,211 | 1,399 | 1,956 | 840 |
| 1 | 2,677 | 2,377 | 3,045 | 2,739 |
| 2 | 2,757 | 3,105 | 2,682 | 2,402 |
| 3 | 2,647 | 3,750 | 1,627 | 2,042 |
| 4 | **0** | 211 | **0** | 2,158 |
| 5 | 592 | 994 | 3,912 | 4,036 |
| 6 | 1,535 | 2,002 | 955 | 2,136 |
| 7 | 434 | 1,595 | 689 | 1,360 |
| 8 | 1,948 | 2,472 | 1,920 | 2,097 |
| 9 | 1,111 | 3,475 | 203 | 749 |
| 10 | 1,935 | 692 | 103 | 1,473 |
| 11 | 3,469 | 2,299 | 3,136 | 2,132 |
| 12 | 1,179 | 2,165 | 3,109 | 4,323 |
| 13 | 44 | 3,330 | 2,487 | 1,984 |
| 14 | 832 | **0** | 4,115 | 448 |
| 15 | 3,244 | 920 | **0** | 1,320 |

## Paired comparisons

Success differences use the two-sided exact McNemar test over discordant seeds.
Common-error differences use a two-sided percentile interval from 100,000
paired-seed bootstrap resamples of the mean, with Python `random.Random` seed
`20260910`. Positive common-error differences mean the first condition is worse
because lower error is better.

| Contrast | Success-rate difference | Discordant wins | Exact p | Mean common-error difference (bootstrap 95% interval) |
| --- | ---: | --- | ---: | ---: |
| truth − categorical, decay 0.01 | 0.0% | truth 1, categorical 1 | 1.0 | +323.2 (−318.2, +979.9) |
| truth − categorical, no decay | −12.5% | truth 0, categorical 2 | 0.5 | +143.8 (−549.9, +737.7) |
| no decay − decay, categorical | +6.25% | no decay 1, decay 0 | 1.0 | +270.2 (−564.5, +1,122.7) |
| no decay − decay, truth | −6.25% | no decay 0, decay 1 | 1.0 | +90.8 (−591.4, +793.2) |

Every interval for the mean paired hard-error difference crosses zero. With
only zero, one, or two successes per condition, the exact-success intervals are
wide. The observed categorical/no-decay success count is therefore descriptive,
not evidence for a coordinate or decay advantage.

## Native versus common hardening

The two extraction rules produce different final nonzero error counts in six
categorical trials, as expected from their definitions: seed 0 no-decay
`1956/1981` common/native; reference-decay seeds 5 `592/1052`, 6 `1535/2399`,
10 `1935/1967`, 11 `3469/3472`, and 15 `3244/2638`. Truth-coordinate native and
common hardening are the same rule. No native/common difference changes an exact
success classification.

## Runtime, records, and boundaries

The cohort ran from `2026-09-10T02:40:16.344084+00:00` through
`2026-09-10T03:50:03.417661+00:00`, 4,187.1 seconds elapsed. Summed across runs,
training compilation took 569.1 seconds, evaluation compilation 140.5 seconds,
synchronized optimizer updates 2,623.2 seconds, and training-loop wall time
3,470.0 seconds. Median synchronized update totals are approximately 41 seconds
per 500-update trial. The recorded 1,681,023,744-byte device-memory peak is a
single-process allocator high-water mark, not an independent condition-level
comparison.

The 64 manifest/metrics directories occupy 19,425,729 bytes. The checkpoint and
trajectory artifacts occupy 317,110,063 bytes and remain outside git under
`artifacts/R002_formal_attempt01_*`; every identity is recorded in its run
manifest. Those local paths are not durable publication. Artifact publication
and R002-specific validated Boolean simplification remain explicit follow-up
work. No post-hoc analysis changed trial status or the frozen training recipe.

Machine-readable aggregate statistics and analysis methods are in
[`analysis.json`](analysis.json). The per-run manifests and metrics are in the
64 sibling `R002_formal_attempt01_seedXX_CONDITION` directories.
