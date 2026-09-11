# R003 preflight: R003_preflight_attempt03

Status: **passed**.

This is a post-training FP64 mechanism diagnostic on immutable R002 categorical
checkpoints. It is not resumed training and does not establish improved
optimization, causal circuit-size reduction, or naturally traversed neutral paths.

## Coverage and controls

- Checkpoint cases: 6
- Valid same-function arm comparisons: 18/18
- Original-to-factorized native argmax gate changes: 43
- Factorized-arm matrix-difference L2 range: 1.2751917135781e-05 to 0.826221838540033
- Factorized-arm loss-input output-response L2 range: 0.31322452717844085 to 2.5777427791435477

Detailed invariance residuals, geometry, responses, and both independent finite
steps are recorded in `validation.json` and `cases.jsonl`. Derived arrays are
stored outside git under the corresponding artifact directory with hashes.
