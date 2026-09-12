# R004 preflight attempt 01

Status: failed during cohort-boundary soft-loss replay.

The seed-0 historical 450-to-500 replays, four-step budget-field comparisons,
and interrupted-versus-uninterrupted continuations all passed exactly in all
four conditions. The first cohort loop evaluated parents eagerly, however,
instead of using the compiled evaluation path used by R002. Two seed-4
categorical aggregate soft losses narrowly exceeded the frozen scalar tolerance,
so this attempt stopped before completing the adapter checks.

The failure is retained as an implementation/preflight failure, not a training
outcome. The next attempt uses the same compiled evaluation form as the source
runner, records every replay residual, and still enforces the original tolerance.
