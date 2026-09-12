# R004 formal_attempt01 analysis

All endpoint results are continuations of the original R002 seed groups, not new trials.

| Condition | Common exact at 500 | Common exact at 2,000 | Native exact at 2,000 |
| --- | ---: | ---: | ---: |
| categorical_reference_decay | 1/16 | 3/16 | 3/16 |
| truth_reference_decay | 1/16 | 2/16 | 2/16 |
| categorical_no_decay | 2/16 | 8/16 | 8/16 |
| truth_no_decay | 0/16 | 3/16 | 3/16 |

## Endpoint transitions

- `categorical_reference_decay`: gained 2, lost 0, exact at both 1, inexact at both 13.
- `truth_reference_decay`: gained 1, lost 0, exact at both 1, inexact at both 14.
- `categorical_no_decay`: gained 6, lost 0, exact at both 2, inexact at both 8.
- `truth_no_decay`: gained 3, lost 0, exact at both 0, inexact at both 13.

## Paired inference

- `reference_decay`: exact paired McNemar p=1, Holm-adjusted p=1; mean truth-minus-categorical hard-error change difference -231.125 (95% paired bootstrap [-1070.8828125, 659.0015624999996]); mean size-gap change 175.688 (95% paired bootstrap [108.1875, 242.37656249999964]).
- `no_decay`: exact paired McNemar p=0.125, Holm-adjusted p=0.25; mean truth-minus-categorical hard-error change difference 447.938 (95% paired bootstrap [-224.87656249999998, 1128.125]); mean size-gap change 191.562 (95% paired bootstrap [134.75, 250.9375]).

## Runtime analysis

The fixed profiles account for 512 labeled modes and 33792 labeled trajectories. Observed phase dependence occurs in 86 modes; 260 universal tests are inconclusive.

Circuit counts are constructive FactoredDAG bounds, not minimum descriptions. Runtime, correctness, and circuit size are separate observables. The repeated checkpoints, probe grids, and two hardening modes are not independent trials. This fixed-grid continuation does not establish Kolmogorov complexity, grid-size generalization, or an infinite generator.
