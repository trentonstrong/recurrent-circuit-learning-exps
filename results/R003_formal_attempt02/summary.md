# R003 formal diagnostic: R003_formal_attempt02

Status: **passed**. All 96 checkpoint cases, 288 same-function arms, and 576
independent one-step diagnostics completed under the frozen v1 protocol.

## Result

The intervention was numerically neutral before the step: the largest q error
was 7.77e-16, the largest recurrent-state error was
6.33e-12, and common hardening changed zero gate IDs. Despite
that fixed relaxed function, the original-to-factorized coordinate matrix changed
in every checkpoint (relative difference 0.333–0.793; median 0.544).

The induced q response also changed in every checkpoint (relative difference
0.00397–1.09; median 0.33). That change reached the visible outputs on the two loss inputs in every case: response-difference L2 ranged from
5.58e-05 to 8.16, with median 0.331.

At eta 0.001, the actual original-to-factorized output-step difference ranged
from 7.9e-08 to 0.0115. The factorized-arm output linearization residual had median relative size 0.000353 and maximum 0.15. At eta 0.0001 those values fell to 3.53e-05 and 0.0149, respectively, supporting the local first-order interpretation.

Factorization changed 935 native argmax gate
IDs across the 96 primary contrasts while changing zero common-rounded IDs. This
is a separate extraction observation. Eligible gates covered 91.283%–100.000% of slots; the smallest eligible share of baseline response squared norm was 3.19e-05, so that low-coverage case should be interpreted cautiously.

| training history | update | median relative M change | median relative q-response change | median visible-response difference L2 |
| --- | ---: | ---: | ---: | ---: |
| reference decay | 0 | 0.333 | 0.331 | 0.482 |
| reference decay | 250 | 0.659 | 0.486 | 0.195 |
| reference decay | 500 | 0.566 | 0.147 | 0.422 |
| no decay | 0 | 0.333 | 0.331 | 0.482 |
| no decay | 250 | 0.69 | 0.852 | 0.441 |
| no decay | 500 | 0.59 | 0.111 | 0.0553 |

These measurements establish representative-dependent local SGD response at the
saved R002 checkpoints and show that it reaches the observed recurrent outputs.
They do not establish improved long-run training, naturally occurring motion
along fibers, or a causal explanation of R002 circuit-size differences.

`analysis.json` retains all 16 seed values for every condition/update group.
`cases.jsonl` contains per-case controls, geometry, responses, and finite steps;
the hashed derived arrays retain p, q, gradients, responses, and measured deltas.

Deterministic derived and review release assets are prepared in
`artifacts/releases/R003/`, with identities recorded in
`release-assets.json`. Upload to the intended `experiment/R003` GitHub release
is pending because the current workstation does not have the `gh` client.
