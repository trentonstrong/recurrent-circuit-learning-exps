# R003 formal_diagnostic: R003_formal_attempt01

Status: **passed**.

This is a post-training FP64 mechanism diagnostic on immutable R002 categorical
checkpoints. It is not resumed training and does not establish improved
optimization, causal circuit-size reduction, or naturally traversed neutral paths.

## Coverage and controls

- Checkpoint cases: 96
- Valid same-function arm comparisons: 288/288
- Original-to-factorized native argmax gate changes: 935
- Factorized-arm matrix-difference L2 range: 1.2751917135781e-05 to 1.0572143615124372
- Factorized-arm loss-input output-response L2 range: 0.0022573621272590754 to 91.24716109954056

Detailed invariance residuals, geometry, responses, and both independent finite
steps are recorded in `validation.json` and `cases.jsonl`. Derived arrays are
stored outside git under the corresponding artifact directory with hashes.
