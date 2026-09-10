# R002 runtime-profile handoff

Prepared on 2026-09-10 against
`46ab6cf94896edb5b2c2e768d5cab43459741dd8`.
Protocol ID: `R002_runtime_profile_v1`.
Status: protocol ready for implementation; the full-cohort profile has not run.

Implement, validate, and run a bounded posthoc runtime profile of all 64 final
circuits in `R002_formal_attempt01`. The purpose is to determine which circuits
already contain useful generators, how long they take to produce the target,
and how their behavior depends on initial state and observation phase. This
is the next analysis agreed in the research conversation. After the focused
checks pass, complete the profile without a separate training-launch approval.

Read current `AGENTS.md`, `docs/STATE.md`, this directory's `SPEC.md`, and
`reviews/R002_circuits/REVIEW.md`. Fetch current main before editing and preserve
newer work. Keep the existing uv environments and validated locks. Some older
handoff/setup prose predates the installed runners; inspect the actual files
and current state rather than recreating environments from those old status lines.

The original R002 experiment remains a 500-optimizer-update, 20-runtime-tick
comparison. This diagnostic executes its frozen Boolean programs for longer.
Preserve every original run manifest, metric stream, success classification,
and checkpoint. Put the new results under a separate analysis ID. R003,
additional training, checkpoint selection, other grid sizes, and new sequence
tasks are separate work.

**The bounded protocol.** Encode these scientific choices in a new configuration
at `configs/analysis/r002_runtime_profile_v1.json`. The path and runner below are
implementation targets, not existing commands claimed to have been tested.

| Field | Required value |
| --- | --- |
| Source cohort | `R002_formal_attempt01`, all 16 seeds times all four conditions |
| Frozen model | Optimizer update 500; final exported Boolean circuits |
| Hardening | `common` primary; `native` separately reported |
| State and boundary | 16 by 16 cells, eight Boolean channels, constant-zero exterior |
| Transition | Original synchronous update, exact exported wiring and gate orientation |
| Observer | Channel 0; original anchored 2 by 2 checkerboard |
| Execution horizon | States at every integer tick 0 through 256 inclusive |
| Primary initializations | The original fixed 32-grid probe, unchanged |
| Additional initializations | One shared set of 32 fresh random grids, plus all-zero and all-one grids |
| Universal-state diagnostic | Existing sound set-valued analysis, capped at 256 ticks |
| Report times | 20, 21, 22, 25, 32, 64, 128, 256; retain every-tick measurements |
| Initial analysis ID | `R002_formal_attempt01_runtime_profile_attempt01` |

Every circuit therefore has 66 ordinary evaluation initializations, grouped
as `original_probe` (32), `fresh_probe` (32), `all_zero` (1), and `all_one` (1).
Keep their results separate. Both constant initializations set all eight
interior channels; the exterior remains zero in every case. State at tick 0
is the supplied initial state, before any circuit update.

Generate the fresh probes exactly once using
`numpy.random.Generator(numpy.random.PCG64(20260910)).integers(0, 2,
size=(32,16,16,8), dtype=numpy.uint8)` in the pinned execution environment.
Serialize the resulting array and record its shape, dtype, generator/version,
raw-array hash, and artifact hash before the full profile. Use those actual
bytes for every training seed, condition, and hardening mode. Do not resample
per run or choose grids based on outcomes. These probes were not in the
original evaluation file; no claim is made that they are proven disjoint from
every grid ever sampled during training. The fixed random seed is a protocol
choice, not additional learned information.

The 256-tick cap is an analysis budget, not an assumed convergence bound. It is
longer than the behaviors observed in the selected review while remaining
bounded. Keep unresolved trajectories unresolved at that cap. Any later
extension gets its own recorded protocol revision and preserves these results.

**Inputs and provenance.** Enumerate cohort members from the committed formal
cohort/results, checking that there are exactly 64 completed update-500 runs,
four declared conditions for every seed 0 through 15, and no duplicate or
missing members. The training code was
`d220b65da822707cd7db3dc90df69d9250aa0f77`; record that identity separately from
the implementation commit of this analysis.

Use the existing local payloads or the published
[`experiment/R002` release](https://github.com/trentonstrong/recurrent-circuit-learning-exps/releases/tag/experiment/R002).
Resolve paths by verified run ID and repository-relative artifact name; original
manifests contain workstation-absolute paths. Verify each used file against its
recorded byte count and SHA-256. Check gate ID ranges, array shapes, and wiring
indices before execution. Verify the complete wiring hash and the association
between each export and its run. Retain both native and common labels even if
their arrays are identical.

Required source artifacts are the two final circuit exports, update-500
checkpoint, and saved update-500 hard trajectory for every run, plus the shared
probe. Check each exported circuit against extraction of that checkpoint using
the existing conventions: categorical native argmax; common rounding of the
four effective truth entries at `q >= 0.5`; weights `[8,4,2,1]` for entries
`00,01,10,11`. Direct-truth native and common use that same rounding rule. Use
exported IDs for execution once the association is verified.

The original shared probe's complete-file SHA-256 is
`d49ecf23fd932fba25b1068f8c90820265a05fd90daadc372089520889090b2f`.
If the file is unavailable, `reviews/R002_circuits/reconstruct_probe.py` already
reconstructs and verifies its exact bytes. Its NPZ serialization hash check was
verified in the recorded review environment; if another environment serializes
the container differently, recover the original file or reproduce that
environment. Do not bypass the check and substitute an unverified probe.

The six-run upload alone is insufficient for this task. Missing or mismatched
artifacts should produce an explicit incomplete/blocked analysis record naming
the affected runs. Do not silently reduce the cohort or replace a checkpoint.

**Measurements per trajectory.** Let `X[t]` be the full Boolean state,
`V[t] = X[t][...,0]`, and `Y` the original target. Define
`e[t] = count_nonzero(V[t] != Y)` and retain it for every tick through 256.
Also retain Hamming changes in the visible state and in the full state between
successive ticks. Equality of error counts is not equality of states.

| Quantity | Exact meaning |
| --- | --- |
| `errors_at_20` | Original readout error, checked against the recorded evaluation for original probes |
| `first_hit_tick` | First `t` with `e[t] == 0`; null if no hit by the cap |
| `observed_correct_suffix_start` | Earliest `t` for which every error from `t` through 256 is zero; null if `e[256] != 0` |
| `observed_correct_suffix_length` | Number of observed states in that suffix; zero if absent |
| `cycle_entry_tick`, `cycle_period` | Exact full-state recurrence information, or null when unresolved |
| `cycle_target_fraction` | Fraction of distinct cycle phases with zero visible error, when a cycle is established |
| `cycle_min/mean/max_errors` | Error statistics over exactly one full cycle |
| `certified_visible_settling_tick` | Earliest onset established by a sound certificate; null if unavailable |
| `evidence` | Separately record finite replay, exact orbit recurrence, and universal-state certificate status |

A suffix ending at the cap is a finite observation. Even a very long suffix
does not itself prove persistence forever. Retain its length so a single
correct last frame cannot be mistaken for settling. A null first hit is
right-censored at tick 256, not an infinite waiting time. A certified onset is
an upper bound on settling time unless minimality has independently been shown.

For each initialization group retain the aggregate error curve `E[t] = sum e[t]`,
the number of perfect grids per tick, and the first simultaneous hit of all
grids. Also report whether every grid has an observed correct suffix and the
latest suffix start when it does. Do not conflate each grid hitting the target
at some different time with all grids being correct at one shared readout.

**Exact cycles and phase.** Detect recurrence using the complete 2,048-bit
state for each trajectory, including hidden channels. Retain the earliest index
of each previously seen state. At the first exact repeat `X[b] == X[a]`, record
`mu = a` and `lambda = b-a`. The deterministic autonomous transition then
certifies the future orbit. Packed state bytes can be dictionary keys; if a
hash is used as an index, verify actual byte equality before declaring a match.
Do not use a floating tolerance, visible-state equality, or repeated error
counts as the cycle criterion.

Once a cycle is verified, later observations through tick 256 may be filled by
exact modular indexing instead of further circuit execution. The horizon and
reported tick convention must remain identical. A full-state fixed point has
period one. Hidden-state cycles can nevertheless have a constant, correct
visible output, so report visible persistence separately from full-state period.
Compute the minimal period of the visible image sequence within a known cycle
by checking its divisors; do not infer it from the period of the error counts.

Report cycle behavior as `all_phases_correct`, `some_phases_correct`, or
`no_phases_correct`, according to whether `cycle_target_fraction` is 1, strictly
between 0 and 1, or 0. Without a full-state repeat or another certificate,
report `unresolved_by_cap` and the finite observations. A target-free eventual
cycle may still have passed through the target during its transient.

Record correct phase indices relative to `X[mu]`, plus phase at the original
tick-20 readout if it is already on the cycle. For a mixed cycle, phase is part
of the generator's observation contract. Produce one reproducible wrong-phase
witness per affected run/hardening mode: choose the lowest initialization ID
with a mixed cycle, then its first incorrect phase `j`; initialize with cycle
state `(j-20) mod lambda`, and independently execute 20 ticks to verify the
wrong output. Save the packed initial state, codec, source orbit indices, and
error count. These derived witnesses are diagnostic constructions, not extra
independent random probes and not part of the 66-grid denominators.

For initialization dependence, report outcomes for each of the four groups and
the number of distinct full cycles reached within each circuit. Canonicalize
cycles up to cyclic rotation using full state bytes; store phase offsets so
two starts in different phases of the same orbit are not counted as different
attractors. Report unresolved counts alongside known cycles. Equal visible
patterns do not identify hidden-state attractors. These finite samples do not
measure exact basin volumes or establish arbitrary-initial-state correctness.

**Universal-state certificates.** Apply the existing set-valued Boolean method
to each final circuit, starting every interior bit with `{0,1}` and fixing every
outside-grid read to `{0}`. Use exact set images for individual gates and a
deterministic simplification method. Record known-correct, known-wrong, and
unknown visible-bit counts by tick until the abstraction is invariant or the
256-tick cap is reached.

An invariant abstraction with every visible bit equal to `Y` certifies all
future visible outputs for every initial full-grid assignment. If an earlier
onset is reported, verify correctness at every intervening abstract tick.
An invariant abstraction with a known-wrong visible bit certifies persistent
failure of that bit for every initialization; verify any earlier claimed onset
through the invariant state as well. If neither certificate follows, report
inconclusive. Unknown bits do not prove instability or failure. Distinguish this
universal result from an orbit certificate that covers one explicit initial
state. The fixed grid and boundary are part of every certificate's scope.

**Circuit structure and implementation.** Reuse the independent NumPy Boolean
executor and exact simplification machinery in
`reviews/R001_checkpoints/analyze_checkpoints.py` and `reviews/R002_circuits/`.
The existing six-run scripts contain fixed run lists and paths; make an explicit
cohort-wide runner rather than changing those historical review outputs.
Record the simplifier identity, binary operation count for all channels, count
for the recurrent visible core, and that core's state-channel closure. Use the
same accounting for every condition. These are structural proxies and
constructive implementations, not minimum circuits or complete description
lengths. Do not make exhaustive local-rule equivalence search a prerequisite
for executing a valid circuit.

Start with NumPy Boolean/integer execution on CPU using the existing locked
environment. Profile one simple and one larger final circuit before selecting
worker count. Independent process workers are appropriate; the review's
temporary simplifier-class replacement is not safe for concurrent threads.
Reuse work when gate arrays and wiring are byte-identical, while retaining all
run/hardening records. Equal tick-20 scores alone do not justify deduplication.
If measured CPU cost warrants a JAX GPU executor, keep the existing GPU lock and
require bit-exact agreement with the independent executor before using it.
Record backend, device, compilation, execution time, memory, workers, and
thread settings. No dependency upgrade or training-kernel change is required.

Keep full states transient in memory for cycle detection. All tick-level error
and Hamming summaries are small enough to retain. Save selected packed-state
witnesses and figures; reconstruct other trajectories from the hashed circuit
and initial state. A fresh dump of every full trajectory is unnecessary for
review. Preserve analysis resumption by completed run/hardening record, guarded
by source, configuration, and initialization hashes. Never invoke training
checkpoint resumption or mutate optimizer/RNG state.

**Focused validation and acceptance checks.** Treat implementation errors as
analysis failures, leaving original trials intact. Establish these checks
before aggregating the full profile:

1. Decode all update-500 exports and reproduce all original-probe tick-20 error
   and perfect-grid counts in both hardening modes. Independently replay the
   supplied first-probe 21-state hard trajectory through tick 20 for every run.
   Check the unsimplified Boolean executor against any simplified/accelerated
   path on those identical inputs. These comparisons must be exact.
2. Verify cycle bookkeeping on small deterministic transitions with a transient
   into a fixed point, a two-cycle, and hidden-state motion behind a constant
   visible bit. Verify a no-repeat-within-cap case and a late single correct
   frame that must remain only an observed suffix. Check tick 0/20 conventions
   and cycle-based reconstruction against direct execution.
3. Check abstract gate set images on their finite domains. Reproduce the six
   reviewed final circuits' every-tick original-probe errors through tick 64
   and their recorded certificates. Preserve the known cases below.

| Required regression | Expected result |
| --- | --- |
| Seed 4 categorical, both decay settings | Zero original-probe errors from tick 20 onward; all-initial-state visible target certified from tick 16; two-operation, two-channel visible core |
| Seed 13 categorical/reference decay | Aggregate errors 44, 2, 0 at ticks 20, 21, 22; universal target certificate from tick 25 |
| Seed 15 categorical/no decay | Universal visible target certificate from tick 20 |
| Seed 14 truth/reference decay | Zero errors at even ticks from 20, 4,096 at odd ticks on the original probes; full-state period two; packed phase witness gives 128 errors at tick 20 |
| Seed 8 truth/reference decay | 2,472 aggregate errors at tick 20; cell `(y=0,x=12)` certified wrong from tick 14 onward |

A semantically equivalent simplifier may strengthen an abstract certificate.
Document that improvement and retain the old regression's guaranteed bound;
do not require a weaker onset to match exactly. Reuse the published seed-14
witness independently of the new cycle-witness constructor.

Implement a single entry point, proposed as
`scripts/r002_runtime_profile.py`, with automatic validation before the full run
and a `--validate-only` option. Support configuration, analysis ID, artifact
root, backend, and worker count. For example, after implementation and local
validation, the intended full command is:

```bash
JAX_ENABLE_X64=0 uv run --project envs/jax-gpu --locked python scripts/r002_runtime_profile.py \
  --config configs/analysis/r002_runtime_profile_v1.json \
  --analysis-id R002_formal_attempt01_runtime_profile_attempt01 \
  --artifact-root artifacts --backend numpy --workers 1
```

This uses the installed GPU project's dependencies while selecting CPU NumPy
execution; it does not launch training. The implementation partner should
return the actual validated command and chosen worker count. Fail clearly on
an existing incompatible analysis directory; explicit matching resumption may
skip only records whose integrity checks pass.

**Deliverables and interpretation.** Write a new result directory
`results/R002_formal_attempt01_runtime_profile_attempt01/` containing:

- A manifest identifying protocol/configuration, source and analysis commits,
  dirty patch if any, environment/lock hashes, all input artifact hashes, probe
  identities, cap, backend, completeness, failures, and timings.
- Validation results, one compact per-trajectory record for every run/hardening/
  initial-state ID, per-run summaries, and a machine-readable cohort aggregate.
  Save every-tick error and Hamming data in a compressed artifact if keeping
  them inline would make the JSON unwieldy. Record missing evidence explicitly.
- `summary.md`, figures, and packed counterexamples sufficient to audit the
  principal conclusions. Retain large arrays outside git and use the existing
  release/asset-manifest convention if publication is needed. Provide a compact
  review bundle so another checkpoint upload is not necessary.

The main comparison table should give, for each condition and hardening, the
unchanged original tick-20 success count, the number correct on all original
probes at each declared later readout, the number whose original probes are
each certified eventually always correct, the number with mixed correct/wrong
cycle phases, and the unresolved count. A run can have different behaviors
across its initializations: define whether a count means any or all probes and
allow overlapping flags where appropriate. Supply a per-seed table so the
paired design remains inspectable. Report fresh-probe and constant-state
results separately, including any original-probe successes that fail them.

Show error versus runtime tick for the cohort, certified/observed timing with
unresolved counts, and structural size against runtime behavior. Plot labels
must distinguish optimizer update 500 from runtime ticks. If timing statistics
condition on successful/certified trajectories, state their denominator and
the excluded unresolved count; do not replace nulls by 257 and report an
ordinary mean. Exact initialization-independent certificates deserve a
separate count from success on sampled states.

These are exploratory diagnostics on the same 16 training seeds. Probe grids,
clock ticks, and native/common exports are not independent training trials.
Describe paired per-seed changes; any uncertainty calculation must preserve
the training-seed clusters. Choosing the best readout separately for each
initialization is a diagnostic oracle and must not be reported as a single
usable observation schedule. Keep the schedule or phase information needed by
each proposed decoder explicit. Do not infer that a representation is better
from a few attractive circuit examples.

Return a concise account of how many original failures are demonstrably late
generators, how many original successes depend on phase or initialization,
which failures have certified wrong behavior, and what remains unresolved.
Distinguish these observations from hypotheses about why gradient search found
them. Circuit size, execution time, initialization, and observer cost are
different parts of a complete description; this profile does not compute
Kolmogorov complexity or establish sequence/size generalization.

Commit the implementation, focused checks, and small results; update
`docs/STATE.md` and link the new analysis from R002 without rewriting the
historical result. The completion condition is all 64 runs accounted for in
both hardening modes, the defined measurements and provenance saved, and the
findings ready for review. Finishing this diagnostic does not authorize a
training continuation or launch R003.
