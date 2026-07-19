# Offline Template-Bank Design and Milestone 6A Smoke

## Status

Milestone 6A implemented and tested `foundry-template-bank-v1`. The automatic pipeline accepted
118 of 120 attempts, both exact verifiers agreed on all 120, and deterministic replay matched.
However, a complete Codex inspection found 13 clearly invalid or unnatural questions produced by
systematic frame-label and morphology defects. The final status is therefore **TECHNICAL GATE
FAILED**. The local human-review packet remains available for independent user review, but review
cannot turn the failed implementation into an approved full-generation system without a separately
approved blocker-resolution design.

## Architecture and responsibility boundary

The live-model path is closed. The retained procedural layer owns curriculum, semantic IR,
quantities, units, constraints, target, exact answer, reasoning trace, provenance, and two
independent mathematical verifiers. A template owns only an English sentence plan. It never holds a
benchmark question, benchmark-derived paraphrase, fixed benchmark value, computed answer, free-form
slot, or hidden assumption.

`TemplateSpec` records the version, reasoning category, semantic frame, compatible target kinds,
required event types, typed placeholder roles, allowed units/object families, supported difficulty,
clause order, question form, sentence plans, optional-context policy, output-contract compatibility,
original provenance, review state, normalized hash, and per-plan render-signature hash.
`SentencePlanSpec` records clause order, opening/event/question forms, temporal framing, and
grammatical construction. Any missing, extra, untyped, or incompatible field fails closed.

## Initial bank capacity

| Category | Semantic frames | Plans per frame | Render signatures |
|---|---:|---:|---:|
| Multi-step bookkeeping | 18 | 4 | 72 |
| Rate, ratio, percentage, or average | 20 | 4 | 80 |
| Discrete constraints | 20 | 4 | 80 |
| **Total** | **58** | **4** | **232** |

The plans vary clause ordering, active/passive or record-centered construction, temporal framing,
subject placement, question wording, scenario framing, and operation description. Names and numbers
do not define plan identity. All plans are original, source controlled, and explicitly marked
`human_review_pending`.

## Typed language safety and acceptance pipeline

The bank reuses `ProblemIR`, entity and unit types, explicit singular/plural morphology, semantic
node coverage, target kinds, and output-contract evidence. Every candidate passes, in order:

1. semantic-IR and template compatibility;
2. deterministic slot filling;
3. semantic-node coverage;
4. object/unit and morphology/agreement checks;
5. target-answer consistency;
6. primary and independent exact verification plus agreement;
7. output-contract and deterministic language-quality validation;
8. exact, number-neutral, render-signature, latent-program, and five-token screening;
9. frozen generated-to-development MiniLM screening;
10. separate generated-to-generated diversity evidence and final decision.

The eleven prior typed renderer regression classes still reject. No LLM is used to write, label,
verify, repair, or judge a candidate.

## Diversity and contamination policies

Generated-to-development protection is unchanged: pinned
`sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU float32,
fixed pooling/normalization, five-token Jaccard `0.35`, semantic review `0.75`, and semantic rejection
`0.82`. Benchmark questions are loaded read-only only inside that scanner; labels are never loaded
by the generator and sealed-final content is never opened.

The separate internal policy hard-rejects exact text, identical filled questions, number-neutral
copies, latent-program copies, render-signature reuse, and cross-template structural copies. It
records high lexical or semantic neighbors for review, but same-skill topical similarity alone is
not an automatic rejection. Policy SHA-256 is
`1ee8993754599e86a4ee37f89064588c9fbef42b3bd12996abd241aff00a9604`; its six-class original
fixture-set SHA-256 is `5cb72b962ab9a33370426a26b6e50b34e217309f35ded0762b744f9d47a23931`.
Benchmark and internal thresholds were frozen before the counted smoke and were not tuned afterward.

## Bounded smoke result

The new master seed was `foundry-template-bank-smoke-master-20260718-v1`. Exactly 120 attempts were
made with no replacements: targeted 33/14/13 and generic 20/20/20 across
bookkeeping/rate/discrete. Each group contained exactly 12 output-contract attempts. All 120 render
signatures were unique.

Automatic acceptance was 118/120 (98.33%): targeted 60/60 and generic 58/60; bookkeeping 53/53,
rate 33/34, and discrete 32/33; easy/medium/hard 39/39/40 of 40. All 24 output-contract attempts
passed. One duplicate latent program and one number-neutral copy were correctly rejected. There were
zero verifier failures, disagreements, false labels, deterministic language-rule failures, target
mismatches, benchmark lexical/semantic rejections, unresolved contamination cases, exact text
duplicates, or reused render signatures. Forty-five internal similarity neighbors were recorded.

Counted/replay deterministic SHA-256 is
`bf87e7af166f5dca107c9777337216e6da7a656b4eec3efb372dc98d1bfa5487`. Counted and replay runtime
were 1.579 and 1.585 seconds; peak process RAM was 914,022,400 bytes; raw ignored evidence is
440,657 bytes. No GPU was used.

## Codex inspection and failed gate

Codex inspected all 120 surfaces after the automatic run; this is **not** a human audit. Thirteen
were clearly invalid or unnatural. Content-free categories were duplicated frame/grouping nouns,
invalid ordinal inflection (`1th`/`2th`), malformed compound/grouping nouns, and awkward literal
frame-label realization. These defects recur across categories, so the “no systematic template
defect” requirement fails even though the deterministic rules recorded zero defects. No templates
were patched and no second smoke was run after observing the results.

The ignored user packet is:

`C:\Users\Admin\Projects\Foundry\results\raw\template_bank_smoke\human_review.md`

Human review status is **pending user review**. Review each of the 120 entries for naturalness,
complete facts, clear target, and whether the automatic accept/reject reason is sensible. Record the
candidate IDs you approve or reject and a short defect label for each rejection. Do not paste the
packet into tracked documentation. Because the technical gate already failed, approval of individual
sentences would be evidence for a future design decision, not authorization for full generation.

## Decision required

Do not generate 4,000 + 4,000 examples or train. The user must decide whether to approve a bounded,
architecture-level blocker resolution for reviewed phrase composition and ordinal/frame-name
morphology, with a new bank version and a new single smoke, or stop synthetic-data realization.

## Milestone 6B composition compiler

Milestone 6B freezes the compiler-backed bank as `foundry-template-bank-v2` while preserving the 58
frames, 232 sentence plans, mathematical generators, dual verifiers,
MiniLM artifact, and every contamination threshold. It changes only deterministic surface
composition. The final status is **TECHNICALLY READY — HUMAN REVIEW PENDING**.

### Thirteen prior defect regressions

| # | Sanitized surface | Primary class | Architectural prevention |
|---:|---|---|---|
| 1 | `dispatch record record` | adjacent duplicate head | one typed noun head plus approved surface lexeme |
| 2 | `receiving record record` | adjacent duplicate head | one typed noun head plus approved surface lexeme |
| 3 | `paired collections collections` | duplicated grouping head | grouping/head role uniqueness |
| 4 | `selected share inventory` | internal frame leakage | ID/surface-lexeme separation |
| 5 | `1th group` | invalid ordinal morphology | centralized ordinal renderer |
| 6 | `two resource capacity inventory` | unsupported compound | approved surface lexeme and noun composition |
| 7 | `materials register register` | adjacent duplicate head | one typed noun head plus approved surface lexeme |
| 8 | `equipment register register` | adjacent duplicate head | one typed noun head plus approved surface lexeme |
| 9 | `parallel channels process` | internal frame leakage | ID/surface-lexeme separation |
| 10 | `matched batches collections` | malformed grouping compound | grouping/head role uniqueness |
| 11 | `2th group` | invalid ordinal morphology | centralized ordinal renderer |
| 12 | `paired supply limit inventory` | unsupported compound | approved surface lexeme and noun composition |
| 13 | `dual recipe plan plan` | adjacent duplicate head | one typed noun head plus approved surface lexeme |

All 13 are sanitized fixtures; no complete generated or benchmark question is tracked.

### Lexical boundary, noun phrases, and ordinals

`SurfaceLexemeSpec` provides an approved phrase and explicit head noun for each internal semantic
frame. Internal frame, role, enum, snake-case, and debug identifiers remain metadata and cannot be
normalized into prose. `NounPhraseSpec` separates head, attributive modifier, grouping noun,
container noun, count behavior, and quantity; duplicate lexical roles or adjacent equal heads reject.
The renderer uses explicit irregular morphology inherited from typed `LexemeSpec`.

Numeric ordinals handle `1st`, `2nd`, `3rd`, `4th`, the `11th`/`12th`/`13th` exception, and larger
forms such as `21st`, `22nd`, `23rd`, and `101st`. Word ordinals are allowed only from a bounded
explicit mapping. Zero, negative, or unsupported word forms reject.

### Surface provenance

Every emitted token is classified as fixed grammar, approved lexeme, entity slot, quantity slot,
unit slot, morphology output, punctuation, or approved optional context. The report also proves that
every required semantic node is rendered exactly once and that no invented node appears. It hashes
the complete token/source and node-count evidence. Provenance validation runs before mathematical
verification and contamination screening.

### Full-bank expansion

Every one of 232 plans was rendered over ten deterministic fixtures spanning easy/medium/hard and
output-contract on/off states: 2,320 attempts, 2,320 valid renders, 232 distinct signatures, zero
final noun, identifier, ordinal, morphology, target, or coverage failures. Expansion SHA-256 is
`78802a61a421ed060aeeab9841c8dd139b97b0ddf971a9b5dd85f5e4766d8e99`.

The expansion measured 15 exact and 1,192 number-neutral repeats because it deliberately applies
multiple fixture values to the same plans. Those expansion artifacts are not candidate data. The
counted pipeline still rejects reused signatures, latent programs, exact text, and number-neutral
copies. Codex inspected a deterministic stratified sample of 90 (30 per family) and recorded zero
findings; that is not human review.

### Fresh 120-attempt smoke

Seed `foundry-template-bank-smoke-master-20260719-v2` produced exactly 120 attempts with no
replacement. Automatic acceptance was 116/120: targeted 58/60, generic 58/60;
bookkeeping/rates/discrete 53/31/32; easy/medium/hard 38/39/39. All 24 output-contract attempts
passed. Three latent-program copies and one number-neutral copy were rejected. False labels,
verifier disagreements, deterministic language defects, target mismatches, benchmark rejections,
unresolved contamination, exact accepted duplicates, and reused signatures were zero.

Counted/replay SHA-256 is
`f5caa7e811cbf257c752a15059e25cc20b2f978fb60e8ad0890c64186095a254`. Counted/replay runtime was
1.670/1.612 seconds; peak process RSS was 912,838,656 bytes; ignored evidence was 539,386 bytes; GPU
use was zero.

### Genuine user review instructions

Open this ignored local file in a browser:

`C:\Users\Admin\Projects\Foundry\results\raw\template_bank_smoke_v2\human_review.html`

For each of 120 cards:

1. Read only for natural wording, complete information, clear references, correct units, and an
   unambiguous target.
2. Click **Approve**, **Reject**, or **Unsure**.
3. For Reject/Unsure, choose a defect label and optionally add a short note.
4. Confirm the progress counter reaches `120 of 120 reviewed`.
5. Click **Export review JSON** and retain the downloaded
   `foundry-template-bank-smoke-v2-review.json` locally.
6. Tell Codex whether the bank is approved and provide the exported JSON for deterministic summary
   if desired.

The page stores progress in browser `localStorage`; the HTML, Markdown fallback, and exported JSON
are ignored and must not be committed. Human review status remains **PENDING USER REVIEW**. Full
generation and all training remain unapproved.
