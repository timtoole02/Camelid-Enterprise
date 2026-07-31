# Architecture decision records

A record here answers one question that was genuinely open, states what was
decided, and — this is the part that makes it worth keeping — states what the
decision does **not** buy. A record is not a design document and not a status
page: it is the thing a reader consults when they are about to undo a decision
without knowing why it was made.

House rules, so records stay comparable:

- **A claim about the tree carries its evidence.** File and symbol, or the name
  of the test that fails if the claim stops being true. "Verified against the
  current tree" without either is prose.
- **Limits are a section, not a footnote.** Every record ends with what it does
  not close. A reader who assumes a decision covers more than it does is the
  failure mode these documents exist to prevent.
- **Published values are quoted, not paraphrased.** A digest, a header name or a
  route is written out in full, because the record is where someone will come to
  check one.
- **Superseding, not editing.** A decision that changes gets a new record that
  names the one it replaces. Facts that were wrong get corrected in place.

## Index

| # | Record | Status |
|---|---|---|
| 0001 | *Reserved* — service separation. The decision is currently carried by [`docs/architecture/service-separation.md`](../architecture/service-separation.md), which labels itself a planning document rather than a record: it maps a boundary the tree has only partly built. When that separation is decided rather than proposed, it lands here as 0001. | reserved |
| 0002 | [What a replica publishes about itself](0002-replica-identity-surface.md) — the configuration digest's scope, the admission policy's own digest, model identity, resolved worker width, and the served-route allow list. | accepted |
