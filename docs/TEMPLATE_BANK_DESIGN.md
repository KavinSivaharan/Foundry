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
