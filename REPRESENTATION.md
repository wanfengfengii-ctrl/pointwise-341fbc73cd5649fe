# Format-two representation invariants

This note describes how the target storage and the SDK2 analysis represent a
revision. It is deliberately short.

## Storage

* The active directory for format two is `target_model_revisions` plus
  `target_model_heads`.  `model_revisions` / `model_heads` are copied nowhere
  and never read again; they remain as the format-one audit copy.
* Each target row stores the canonical legacy-shaped definition (ordered metrics,
  ordered states with signed reward vectors, ordered events with string state
  identities) together with an explicit computation representation:
  ordered `states`, ordered `metrics`, `residence` rows and event records
  `{id, source, target, rate, payment}`.  Both parts are derived from the same
  validated definition in one transaction, so a revision's identity directory
  and its numbers cannot drift apart.
* Event order, event identity, source, target and payment order are preserved.
  A self event is stored with `source == target`; it is not collapsed.  Parallel
  events stay distinct rows with their own ids, rates and payments.  A zero-rate
  event keeps its identity and its zero contribution.  Coalescing duplicate
  `(source, target, rate)` events is explicitly not done because it would change
  the event directory and lose identities.
* `put_model` (legacy shape) and `put_native_model` (native shape) both validate,
  translate to this single canonical shape, and append to the same immutable
  revision sequence and head.  Native input is translated exactly once, at the
  write boundary; no audit row is consulted for either write path.

## Tasks and results

* `tasks.revision` now has a foreign key to `target_model_revisions`, so a task
  binding names a revision that lives in the active directory.  Request bindings,
  fixed revisions, cohort snapshots, task ids, terminal states, previously
  published result payloads and receipts are copied verbatim.  `legacy_tasks_audit`
  and `legacy_results_audit` hold the format-one rows for audit.
* A historical task therefore still resolves its original revision, and a
  succeeded task still returns its original published report without recompute.
  The model head can advance without rebinding it.

## Propagation

`analysis_archive.sdk2_analysis` builds one column-vector linear system and calls
`vendor.sdk2.propagate(matrix, vector, time)` once.  For `n` states and `k`
metrics the dimension is `n * (1 + k + k(k+1)/2) <= 6 * 10 = 60`.  State
probabilities, means and raw second moments are all read from that one result
vector; the probability query is a projection of the same augmented state.

* Unknowns: `y[i, alpha] = E[1{X=i} Z^alpha]` for every multi-index of degree
  at most two.  Degree-one entries give means; degree-two entries give raw
  second moments, whose centered combination is the full covariance.
* Drift rows encode `r_i . grad`, i.e. residence reward accrued at state `i`
  with the rate sign kept (negative rewards are legal).
* Event rows encode `lambda * (Z + g)^alpha` expanded by the multinomial
  identity; a single jump therefore pays the whole vector `g`, so metrics of the
  same occurrence are correlated exactly as the business requires.  Self events
  produce the same operator with `source == target`; their jump does not move
  state probability but still adds payment moments.
* Direction: rows are the target monomial, columns the source monomial, matching
  `exp(t*A) @ y(0)` and SDK2's documented column convention.  Inputs are plain
  finite float lists; SDK2 does not mutate them.
* Time zero needs no special case: the generator scaling gives the identity.

## Transaction boundaries

`analysis_archive.migration.upgrade` is a single `BEGIN IMMEDIATE` transaction
that creates the target tables, copies models, fires `fault("after_models")`,
renames the old task/result tables, rebuilds the target-bound tasks/results,
updates the format marker and fires `fault("before_commit")`.  Any exception
rolls all of it back, leaving format one and the old objects usable.  A format-
two archive returns immediately without touching the database or calling the
callback.  Readers and writers dispatch on the current format marker, so a
previously obtained `archive.models` / `archive.analysis` object observes the
switch on its next call.

The analysis path for format two never calls SDK1; for format one it keeps the
original SDK1 behaviour so the pre-upgrade regression remains valid.
