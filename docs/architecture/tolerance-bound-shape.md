# What a tolerance-admitted posture has to publish

**Status:** evidence and a recommended shape. Not a decision — the numbers belong
to whoever measures them on the backend that needs them. What this records is the
*form* the bound has to take, and the measurements from another campaign that
make each term non-optional.

**Answers:** the item ADR 0004 leaves open — *"It does not give the GPU posture an
acceptance criterion. … What bar a non-bit-identical posture must clear is open."*

---

## Where this evidence comes from, and what it is worth

None of it was measured here. It is from the sibling engine's gemma3 long-prompt
prefill campaign (`GEMMA3_METAL_CONDUCTOR.md` §16–18, Camelid PR #585), which
pinned a numeric envelope *before* measuring, published the failures when the pin
did not survive contact, and amended it. That is the only run of this experiment
this project has access to.

Read it accordingly. It is a different model, a different kernel, a different
platform, and a windowed-attention defect rather than a projection one. **It does
not transfer as values.** What does transfer is which *observables* had power
against a real defect and which did not, because that is a property of the
observable rather than of the defect — and each of the three findings below cost
that campaign a failed gate to learn.

## The three terms, and why none is optional

### 1. The scalar bound must be relative, and per tensor

Their first pin used one **absolute** number across tensors three orders of
magnitude apart — caches reaching `1.4e2`, a final hidden reaching `3.3e4`. It
failed twice: once too loose to constrain the large tensor at all, and once
*tighter than the arithmetic's own round-off floor* on the small one, which is
unreachable by any correct implementation.

Their own words on the second failure: the pinned `5.0e-2` was `3.6e-4` relative
at that tensor's scale, "below the half round-off floor the same paragraph had
just derived."

So: bound each compared tensor on its own magnitude, and state the scale the
fraction is relative to. A single absolute number covering a whole forward pass
is not a bound, it is a coin flip about which tensor it happens to fit.

### 2. A per-position outlier factor, which is the term with the power

Their sharpest case: a defect moved **287,743 elements** while the *median*
per-position divergence stayed **exactly `0.0`**, because only one or two query
positions of 513 clipped. A scalar bound sees that only if it is set absurdly
tight — tight enough to fire on ordinary round-off.

The outlier term bounds any single position as a multiple of the per-position
median, so it fires even against a zero median. Measured separation on the
amended pin: **9.1× on the outlier term against 4.3× on the scalar**. The
campaign's own summary is that "the outlier half is what carries this gate."

A conformance report that publishes only a maximum relative error has published
the weaker half of its own bound.

### 3. Top-1 agreement is the outer gate, and must be labelled as such

The same campaign planted a one-position window defect and measured which
observable caught it across nine items at 513–2400 tokens:

| observable | caught `window_minus_one` | caught `window_plus_one` |
|---|---|---|
| token identity (argmax) | **0 / 9** | **0 / 9** |
| KV / numeric equivalence | 9 / 9 | 9 / 9 |

Zero — including four items built specifically to expose it, with the answer word
planted at exactly q-510/511/512/513. Their conclusion: token identity "has
measured zero power against the campaign's headline defect. It stays as the outer
gate; it is not the gate."

Publish it anyway: it is the one term a client can check without a reference
implementation, and a *drop* in it is still real evidence. But a posture gated on
top-1 alone is gated on nothing, and the field should say so where it is declared
so a high number is not read as a strong claim.

## Two further constraints from the same source

**Compare a final hidden or the logits, not only intermediate state.** §16e found
a mutant confined to the last layer that moved **zero** KV bits — its entire
signature was in the final hidden. A harness that captures only caches passes it.

**A tolerance gate must prove the accelerated path actually ran.** Theirs refuses
vacuity twice over: the result must not be bit-identical to the reference lane,
*and* must not be bit-identical to the path being replaced — "which is what a
silent admission failure would produce." A bound is trivially satisfied by a
fallback to the reference, so conformance without a liveness proof is not
conformance.

## What this means for the CPU posture, today

Nothing changes. `cpu` is bit-identical to the portable reference and is gated on
that directly — `engine-macos` and `engine-windows` each assert their kernel
against `engine_core::tensor::q8_0_dot_rows` to the bit, and
`crates/engine-windows/tests/forward_avx2.rs` compares whole logits vectors
`to_bits` with a one-ULP negative control.

Where bit-identity is achievable it should be required, and this document does not
soften that. The campaign's own stopping rule says the same thing: a bit-identity
failure "is a bug to fix, not a bar to lower." This is for the path where it is
genuinely not achievable — a GPU reducing in an order the CPU cannot reproduce —
and the point of writing it down before that path exists is that the natural first
guess at a bound is the shape that already failed.
