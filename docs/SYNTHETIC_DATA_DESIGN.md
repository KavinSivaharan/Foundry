# Foundry Phase 1 Synthetic-Data Design

Status: frozen design only; no synthetic dataset or training run exists

Design version: `foundry-synthesis-design-v1`
Date: 2026-07-18

## Purpose and boundary

This design converts aggregate weaknesses from Foundry's trusted 814-example development
baseline into independent procedural training examples. It deliberately separates diagnosis
from generation:

- Development questions and model completions were inspected locally to classify failures.
- Their wording, numbers, structures, answers, and paraphrases are forbidden as generator input.
- A future generator may receive only the content-free taxonomy, aggregate counts, difficulty
  targets, random seeds, and the contracts in this document and `configs/synthesis/`.
- The executable latent program, not an LLM response, is the source of every synthetic label.
- The sealed-final partition remains inaccessible until a training recipe and checkpoint are
  frozen under a later approval.

Milestone 3 implements schemas and validation contracts only. It does not implement a full
generator, generate model questions, create a synthetic dataset, or train a model.

## Evidence base and complete failure taxonomy

All 293 development failures were reviewed: 231 extractable-but-wrong outputs and 62
unextractable outputs. Detailed content-bearing classifications remain ignored under
`results/raw/failure_taxonomy/current/`. The committed record contains only aggregate counts,
content-free definitions, hashes, and stable identifier prefixes.

| Primary category | Count | Content-free definition |
| --- | ---: | --- |
| Output format or answer extraction | 69 | Terminal answer is missing, conflicting, truncated, malformed, or extracted from a value other than the completion's clear intent. |
| Multi-step bookkeeping or omission | 68 | A required update, term, branch, or final aggregation is omitted, duplicated, or applied in the wrong order. |
| Target or language interpretation | 53 | The computation answers a related quantity instead of the quantity requested. |
| Rate, ratio, percentage, or average | 28 | A rate, ratio, percentage, fraction, weighted average, or denominator is modeled incorrectly. |
| Constraint, distribution, or discrete reasoning | 27 | A bounded, integral, allocation, capacity, remainder, or distribution constraint is violated or rounded incorrectly. |
| Time, unit, or sequence reasoning | 24 | A conversion, elapsed-time boundary, recurrence, or event ordering is mishandled. |
| Arithmetic execution | 22 | The plan is suitable but an exact elementary operation is wrong. |
| Benchmark ambiguity or annotation risk | 2 | The prompt/reference appears ambiguous, inconsistent, or risky to label automatically. |
| **Total** | **293** | **224 reasoning failures and 69 output-format/extraction failures.** |

Confidence was high for 274 classifications, medium for 17, and low for two. The two low
confidence cases are the benchmark-risk records and are excluded from synthesis. The other 291
appear targetable, independently representable, and deterministically verifiable in principle.
Secondary categories overlap by design. Output-related secondary counts were 42 ambiguous
terminal answers, eight conflicting answers, seven missing terminal answers, three truncated
generations, two malformed terminal answers, and seven confirmed wrong-output false
extractions. The seven remained mathematically wrong and do not change the audited 521-correct
numerator.

This is an exhaustive classification of the observed development failures, not an assertion
that categories are objectively unique. A primary category records the dominant cause; secondary
tags preserve reasonable alternative interpretations.

## Pilot-category ranking

Scores are design judgments from 1 (weak) to 5 (strong). For ambiguity and contamination, 5
means lower risk. `Measure` means the ease of measuring a category-specific change.

| Reasoning category | Prevalence | Verifiable | Independent generation | Expected effect | Low ambiguity | Diversity | Low contamination risk | RTX feasibility | Measure | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Multi-step bookkeeping/omission | 5 | 5 | 5 | 5 | 4 | 5 | 4 | 5 | 5 | Select |
| Target/language interpretation | 5 | 3 | 3 | 5 | 2 | 5 | 2 | 5 | 3 | Defer |
| Rate/ratio/percentage/average | 3 | 5 | 5 | 4 | 4 | 5 | 4 | 5 | 5 | Select |
| Constraint/distribution/discrete | 3 | 5 | 5 | 4 | 4 | 4 | 4 | 5 | 5 | Select |
| Time/unit/sequence | 3 | 4 | 4 | 3 | 3 | 4 | 4 | 5 | 4 | Defer |
| Arithmetic execution | 2 | 5 | 5 | 3 | 5 | 4 | 5 | 5 | 5 | Generic control already covers it |
| Benchmark ambiguity/risk | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 5 | 1 | Exclude |

The three selected categories combine observed prevalence with reliable independent generation
and verification. Target/language interpretation is more prevalent than two selected categories,
but subtle wording changes can alter the target, create ambiguity, or encourage benchmark-like
rendering. It is deferred until the simpler pipeline is proven.

## Selected curriculum tracks

### 1. Multi-step bookkeeping or omission

- **Meaning:** the solver loses, duplicates, or misorders an update in an otherwise manageable
  chain of exact operations.
- **Abstract example:** a fictional inventory state receives and removes several independently
  sampled quantities before an explicitly named remainder is requested. This is a generator
  family description, not a benchmark example.
- **Inputs:** arithmetic-DAG shape, operation count, branch count, signed/unsigned values,
  divisibility constraints, distractor-state count, units, template ID, and seed.
- **Difficulty:** number of steps, dependency depth, branching, cancellation, irrelevant state,
  magnitude, and whether an intermediate result is reused.
- **Primary verification:** topological `Fraction` execution of the latent DAG.
- **Independent verification:** a separately implemented state ledger with conservation and
  inverse checks.
- **Invalid risks:** dead branches, undefined symbols, cyclic dependencies, negative physical
  quantities without an explicit interpretation, a requested target not represented by the
  answer symbol, or a rendered statement that changes operation order.
- **Rejection:** any verifier disagreement, unused required branch, impossible state, ambiguous
  target, duplicate structure, or contamination alert.
- **Expected benefit:** better retention and aggregation across multi-step solutions, the largest
  observed reasoning-failure population.

### 2. Rate, ratio, percentage, or average

- **Meaning:** the solver uses the wrong denominator, base, direction, weighting, or proportional
  relationship.
- **Abstract example:** an invented process combines exact rates over controlled intervals and
  asks for one precisely defined aggregate. Values and narrative are generated independently.
- **Inputs:** relation type, rational operands, percent base, number of groups, weights, exact
  termination requirement, units, template ID, and seed.
- **Difficulty:** nested ratios, weighted versus unweighted mean, successive percentage changes,
  inverse-rate questions, mixed units, and denominator distractors.
- **Primary verification:** exact `Fraction` equation evaluation.
- **Independent verification:** cross-multiplication, inverse substitution, and dimensional
  consistency through a separate path.
- **Invalid risks:** implicit percent base, zero denominator, units that do not cancel, a result
  requiring unspecified rounding, or text that reverses the ratio.
- **Rejection:** any missing base, disagreement, dimensional mismatch, unsafe rounding, duplicate,
  or contamination alert.
- **Expected benefit:** directly targets a recurring, diverse error family with high-quality exact
  labels.

### 3. Constraint, distribution, or discrete reasoning

- **Meaning:** the solver violates integrality, bounds, capacity, allocation, remainder, or a
  floor/ceiling condition.
- **Abstract example:** an original packing system allocates independently sampled component
  types into bounded assemblies while satisfying explicitly stated integer constraints.
- **Inputs:** finite variable domain, constraint graph, capacity, divisibility/remainder rule,
  objective, uniqueness requirement, template ID, and seed.
- **Difficulty:** variable count, domain width, number and interaction of constraints, boundary
  solutions, remainder cases, and floor/ceiling choice.
- **Primary verification:** constructive exact solution of the sampled constraint system.
- **Independent verification:** bounded brute-force enumeration over a frozen finite domain.
- **Invalid risks:** multiple solutions, no solution, implicit rounding, excessive enumeration,
  hidden assumptions, or a natural-language constraint not represented in the program.
- **Rejection:** non-unique or absent solution, out-of-domain value, enumeration timeout, verifier
  disagreement, duplicate, or contamination alert.
- **Expected benefit:** teaches the model to respect discrete constraints rather than applying a
  plausible but invalid continuous calculation.

### Separate output-contract track

`terminal-final-answer-contract-v1` teaches a shared response behavior rather than a reasoning
category. Every training response may include a deterministic reasoning trace, but its final line
must be exactly:

```text
Final answer: <canonical-number>
```

The strict line parser and an independent canonical string round-trip must agree. The track uses
the same number of accepted examples and token budget in targeted and generic datasets so format
training cannot confound the curriculum comparison.

## Typed synthetic-example schema

`src/foundry/synthesis/schema.py` freezes the following fields:

| Field | Contract |
| --- | --- |
| Synthetic example ID | Stable versioned ID independent of benchmark IDs. |
| Generator version and random seed | Reproduce program sampling and rendering. |
| Target category and secondary tags | Content-free curriculum labels. |
| Difficulty | Frozen `easy`, `medium`, or `hard` band plus program-level controls. |
| Latent executable program | Typed parameters, operations, constraints, and answer symbol; source of truth. |
| Rendered question | Controlled-template natural language derived from the latent program. |
| Deterministic solution trace | Replayable exact steps derived from the program. |
| Canonical final answer | Reduced exact rational value; never floating point. |
| Required answer format | Immutable `Final answer: <canonical-number>`. |
| Primary verification evidence | Verifier/version/method, exact result, and evidence hash. |
| Independent verification evidence | Distinct verifier and method family, exact result, and evidence hash. |
| Validity and rejection reason | Accepted candidates have no rejection reason; rejected ones require one. |
| Normalized-text and latent-program hashes | Deduplication and tamper evidence. |
| Provenance | Independent procedural source plus configuration/taxonomy hashes; benchmark input is forbidden. |
| Contamination and deduplication status | Not run, passed, rejected, or manual review. |

Integers and `Fraction` values are canonical. Decimal text may be used only when it maps exactly to
a terminating rational. Binary floating point is never a label source.

## Generator architecture

| Choice | Label source | Advantage | Main risk | First-pilot decision |
| --- | --- | --- | --- | --- |
| A. Procedural program + controlled templates | Exact latent-program execution | Reproducible, local, cheap, auditable, lowest label risk | Less surface diversity | **Selected** |
| B. Procedural program + local-model paraphrasing | Original exact program | More wording diversity | Paraphrase can change semantics; new model/dependency | Deferred |
| C. Frontier-model generation + executable verification | Independently reconstructed execution | Potentially broad language | Paid/cloud, provenance, reproducibility, memorization | Rejected for pilot |

Architecture A follows this fixed flow:

1. Sample an approved abstract program family from a deterministic seed.
2. Sample values and constraints while enforcing the family domain.
3. Execute the program exactly.
4. Render the program with a controlled, versioned natural-language template.
5. Derive a deterministic exact solution trace.
6. Run two logically independent verification methods.
7. Run validity, ambiguity, deduplication, and contamination gates.
8. Save only candidates that pass every gate.

No LLM is needed to make trustworthy labels, and no LLM judge may settle disagreements.

## Independent verification

| Track | Primary verifier | Independent verifier | Timeout/disagreement behavior | Forced rejection |
| --- | --- | --- | --- | --- |
| Multi-step bookkeeping | Topological exact DAG execution | State-ledger replay plus conservation/inverse checks | Reject on either timeout or any disagreement | Unknown/cyclic symbol, dead required branch, division by zero, invalid state |
| Rate/ratio/average | Exact rational equation | Cross-multiplication/inverse substitution plus units | Reject on either timeout or any disagreement | Zero denominator, implicit base, dimension mismatch, unspecified rounding |
| Constraint/discrete | Constructive exact solver | Bounded brute-force enumeration | Reject if enumeration exceeds frozen state/time bound or disagrees | No/multiple solutions, non-integral value, ambiguous floor/ceiling |
| Output contract | Strict final-line parser | Exact canonical-answer round-trip | Reject on any disagreement | Missing/multiple answer lines, trailing text, value mismatch |

Every evidence record includes verifier identity, version, method family, exact computed answer, and
SHA-256 evidence digest. Two calls to the same function or method family are not independent. A
clear disagreement, timeout, unsafe normalization, or unrepresented rendered constraint rejects
the candidate; the system never guesses or asks a model to break a tie.

## Deduplication and contamination controls

### Milestone 4 semantic-artifact selection

Exactly three open-weight, Transformers-compatible candidates were compared before any model
download. Reported download sizes cover only the safetensors/tokenizer/config files needed by the
fixed Transformers path, not alternate ONNX, OpenVINO, TensorFlow, or duplicate PyTorch weights.

| Candidate | Immutable revision | License | Required size | Dimension | Frozen pooling/normalization | Decision |
| --- | --- | --- | ---: | ---: | --- | --- |
| `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | Apache-2.0 | 91,577,897 bytes | 384 | Attention-mask mean, L2 | **Selected** |
| `intfloat/e5-small-v2` | `ffb93f3bd4047442299a41ebb6fa998a38507c52` | MIT | 134,410,262 bytes | 384 | Attention-mask mean, L2, symmetric `query:` prefix | Not selected |
| `BAAI/bge-small-en-v1.5` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | MIT | 134,410,442 bytes | 384 | CLS token, L2 | Not selected |

All three fit the 500 MB, CPU, open-license, no-paid-API, and standard-Transformers constraints.
MiniLM was selected because it is the smallest, is intended directly for symmetric sentence and
short-paragraph similarity, requires no task prefix, and has an official plain-Transformers recipe
for attention-mask-aware mean pooling plus L2 normalization. Foundry uses the existing PyTorch,
Transformers, Hugging Face Hub, and safetensors installations; `sentence-transformers` is not
required and no dependency lock changes are needed.

The exact operational pin is
`configs/synthesis/semantic_all_minilm_l6_v2.yaml`. It requires CPU float32 inference, batch size
32, maximum length 256, `trust_remote_code=False`, safetensors, local-only loading after the one
approved download, and cosine similarity through normalized dot products. The frozen thresholds
remain pass below 0.75, manual review from 0.75 to below 0.82, and rejection at or above 0.82.
Original fixtures must pass this policy before procedural generation may begin.

Checks are staged from cheapest to most semantic. Benchmark content may be loaded locally only by
this comparison stage. It is never generator input, never included in an LLM prompt, and never
committed.

1. Reject an exact normalized-text match.
2. Replace signed, comma-grouped, decimal, and fractional values with `<num>` and reject a
   normalized numeric-template match.
3. Reject an equivalent latent-program structure, even if words and values differ.
4. Reject token five-gram Jaccard similarity at or above `0.35`.
5. With a future pinned local encoder, send semantic similarity from `0.75` to manual review and
   reject at or above `0.82`.
6. Manually audit the closest accepted matches and all semantic-band candidates before training.

If the semantic backend has not run, a pair is marked for manual review and cannot auto-pass.
Milestone 3 deliberately does not download an embedding model or run a large semantic scan. The
thresholds are frozen, while the exact local encoder artifact, revision, and embedding protocol
must be pinned under Milestone 4 approval before any pilot candidate can be accepted.

Original unit fixtures prove that case/punctuation copies, number swaps, structural copies, and
high semantic similarity are rejected, and that absence of a semantic result escalates rather
than silently accepting. A future integration test must add locally held obvious paraphrases and
structure-preserving rewrites before pilot generation.

Only content-free counts, policy hashes, rejection reasons, and comparison hashes may be tracked.

## Targeted versus generic control

Two matched datasets isolate curriculum selection:

| Property | Targeted | Generic control |
| --- | --- | --- |
| Accepted examples | 4,000 | 4,000 |
| Synthetic train/validation | 3,600 / 400 | 3,600 / 400 |
| Output-contract examples | 800, included above | 800, included above |
| Expected tokens | about 1.4 million | about 1.4 million |
| Generator framework | Same | Same |
| Quality, verification, contamination gates | Same | Same |
| Difficulty and length distribution | Matched | Matched |
| Curriculum | Failure-weighted three-category sampling | Broad approximately uniform arithmetic sampling |

The targeted reasoning allocation is 1,760 multi-step, 720 rate/ratio, and 720
constraint/discrete examples, plus 800 output-contract examples. The generic control uses broad
or approximately uniform family selection but the same total size, output track, difficulty,
length, validation ratio, and acceptance rules.

A later training comparison must use the same base checkpoint, QLoRA architecture, optimizer-step
count, training-token budget, evaluation contract, and controlled seeds for:

1. untouched base model;
2. base model fine-tuned on generic synthetic data;
3. base model fine-tuned on Foundry-targeted synthetic data.

No training is part of this milestone.

## Pilot size, compute estimates, and stages

These are planning estimates, not measurements or guarantees:

| Item | Estimate |
| --- | --- |
| Generator smoke | 120 candidates total across all four tracks |
| Accepted targeted pilot | 4,000 examples; about 1.4M tokens |
| Accepted generic pilot | 4,000 examples; about 1.4M tokens |
| Generation time | 5–15 minutes per dataset on CPU |
| Verification and screening | 10–30 minutes per dataset before manual audit |
| Combined accepted/metadata disk | 40–100 MB, excluding any embedding-model cache |
| Future QLoRA run | 2–4 hours per run on the RTX 3080 |
| Future QLoRA peak VRAM | approximately 7.5–9.5 GiB |

The QLoRA range is consistent with the prior 3.123 GiB inference reservation but is not a training
measurement. Sequence length, batch size, optimizer state, gradient checkpointing, and Windows
allocator behavior could still cause an out-of-memory error on the 10 GiB card.

The staged plan is:

1. Implement three generator families and run a 120-candidate smoke after separate approval.
2. Audit generator validity, rejection behavior, diversity, and contamination safeguards.
3. Generate matched 4,000-example pilots only after the smoke gates pass and approval is given.
4. Run a one-seed targeted-versus-generic training signal check.
5. Run two-seed confirmation only if the signal check passes.
6. Increase dataset size only after reproducible evidence justifies it.

## Frozen go/no-go criteria

Generation gates:

- 100% of accepted examples have two distinct agreeing verifiers.
- Zero unresolved contamination candidates are accepted.
- Zero exact, numeric-template, or latent-program duplicates remain.
- At least 30 accepted and 30 rejected candidates per track receive human audit before training.
- Any verifier disagreement, timeout, ambiguity, or missing semantic screen rejects or escalates;
  it never auto-accepts.

One-seed signal gates:

- Targeted training exceeds the 64.0049% base development score by at least 1 percentage point.
- Targeted training exceeds the matched generic control by at least 0.5 percentage points.

Two-seed claim gates:

- Mean targeted gain over base is at least 2 percentage points.
- Mean targeted gain over generic control is at least 1.5 percentage points.
- At least two targeted categories improve by at least 5 percentage points.
- Untargeted development accuracy declines by no more than 2 percentage points.
- Extractability remains at least 91.38%, one point below the frozen 92.38% base rate.
- Gains reproduce across the approved seeds before a strong claim.
- Sealed-final data remains untouched until the recipe and checkpoint are frozen.

These are targets for future go/no-go decisions, not promised outcomes. A failure ends or redesigns
the experiment rather than redefining success after results are known.

## Frozen artifacts and unresolved risks

- Complete detailed taxonomy hash:
  `964d0c18b60d4f0262f0ec711b2f13b396ca4fa1c921f0dc2e91205d393cb692`
- Content-free taxonomy-contract hash:
  `021837a1f1a3bb5a189b1f39c808bb907e415e28d8fa722a8a03c3114717cf28`
- Synthesis architecture/curriculum/gate contract hash:
  `910bf21dba7cef833fd9f7bd83842034e9e7261cf93979d7cdddc0479094d347`
- Phase 1 synthesis configuration hash:
  `7c087ac45c9027ab872cfecbc0dbf6123b60ec088b35e6d6ddc4dfd9094a99d5`

Remaining risks include controlled-template language being too narrow, category classifiers being
interpretive, a semantic encoder producing false positives/negatives, synthetic-validation data
being easier than benchmark prose, QLoRA memory pressure, and multiple-comparison noise in
category metrics. Target/language interpretation remains a known weakness but is intentionally
excluded from the first pilot because reliable rendering is not yet proven.

## Exact proposed Milestone 4

Milestone 4 should be separately scoped to:

1. pin one local semantic-similarity encoder and protocol without changing the benchmark
   evaluator;
2. implement only the three approved procedural generator families and shared output track;
3. run at most a 120-candidate generator smoke;
4. exercise both verifiers, all rejection paths, deduplication, and contamination controls;
5. manually audit the required smoke sample and report acceptance/rejection statistics;
6. stop before generating the full 4,000 + 4,000 pilot or running any training.

The user must explicitly approve that milestone and the local semantic artifact before any
synthetic example generation begins.

## Milestone 4 measured smoke result

The approved bounded implementation selected
`sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
under Apache-2.0. Its eight required files occupy 91,577,897 bytes, use 384-dimensional CPU float32
attention-mask mean pooling plus L2 normalization, and require no dependency beyond the existing
PyTorch/Transformers stack. Original fixtures separated exact/number-swapped/semantic paraphrases
from related and unrelated questions under the unchanged 0.75/0.82 policy. Deterministic fixture
output SHA-256 is `4998fa509da71f7e1f681059d8fd68ea91deae0e3e6a3b38a912c6341cd73ba0`.

Exactly 120 fixed candidates were processed: 60 targeted and 60 generic, with 12 output-contract
attempts in each group. The pipeline accepted 24 and rejected 96. Accepted counts were four
bookkeeping, 16 rate/ratio, and four discrete. Rejections were 25 numeric-template copies, 50
five-token overlaps, seven semantic automatic rejections, and 14 manually confirmed
generated-to-generated near-template rejections. There were zero verifier failures, verifier
disagreements, ambiguous targets, generator exceptions, unresolved contamination cases, or false
labels. Final decision SHA-256 is
`661410933e90680d34a06c1836c7aca6fecfd5bba507c2dfaf3d8ecd5340c8b9`; aggregate SHA-256 is
`eb85cf9efe130d34164bca20badb9b3dce8f050493abf0e014614332b68f8771`, and dry replay matched both.

Manual review covered all 120 attempts. Five accepted examples were invalid: four bookkeeping
renderings failed to define a common unit before combining heterogeneous object counts, and one
discrete capacity rendering had grammar and tied-constraint difficulty defects. Other systematic
weaknesses were lowercase continuation pronouns and insufficient hand-authored template diversity.
The readiness gate therefore failed at 20% acceptance, with fewer than 15 accepted bookkeeping and
discrete examples and nonzero invalid acceptances. Thresholds were not changed and rejected attempts
were not replaced.

Full pilot generation is blocked. A future separately approved blocker-resolution smoke should
retain the same semantic artifact, thresholds, curriculum, output-track proportion, verifier
contracts, and 120-attempt limit while changing only the generator version, fresh master seed,
entity/unit-consistent rendering, grammar, genuinely discriminating discrete constraints, and
hand-authored template diversity. Local-model paraphrasing remains an architectural change and is
not implied by this result.

## Milestone 4.1 measured blocker-resolution result

The approved repair introduced explicit object families, countability, units, compatible
locations, typed quantities and transfers, singular/plural forms, supported verbs, renderer
quality metadata, and pre/post-render checks. Bookkeeping now exposes two mathematical families,
eight rendering variants per family, and 24 scenario domains. Rate/ratio retains five mathematical
families with six rendering variants and 20 domains. Discrete reasoning exposes four families,
six rendering variants, 20 domains, untied capacity constraints, and measured search-space bands
of 9–35 (easy), 36–80 (medium), and 81–200 (hard).

The fresh master seed was
`foundry-phase1-procedural-smoke-master-v2-rendering-diversity`. Exactly 120 attempts were
processed with the original 60/60 curriculum and 12 output-contract attempts per group. The
pipeline accepted 86 and rejected 34. Targeted yield was 52/60 and generic yield was 34/60;
family yield was 30/53 bookkeeping, 29/34 rate/ratio, and 27/33 discrete. Exact normalized-text,
number-neutral-template, and latent-structure rejections were all zero. Six candidates failed the
unchanged five-token threshold, 19 failed the unchanged 0.82 semantic threshold, and nine
same-scenario/same-family cases in the 0.75–0.82 review band were conservatively rejected. No
candidate was replaced.

Replay matched decision SHA-256
`84bd6c622b30034a5932a4098c166b8710e39bbf4756e74b1c7c51cf54ce84a3` and aggregate SHA-256
`0e2e20a3516beacb651dfafea96be9b3e95760fbede8804ae6bea76eb6657ed6`. All exact labels and
dual-verifier decisions were correct, all 24 output contracts were valid, and no unresolved
contamination or benchmark resemblance remained.

Manual audit rejected production readiness. Eleven accepted renderings had residual grammar,
duplicate-group, target-consistency, or unit-expression defects. Consequently the unchanged gate
failed both the minimum 90/120 yield and zero-invalid-acceptance requirements. The current
generator version must not produce the full pilot. A new milestone is not implied: the user must
choose whether to stop this procedural lineage or explicitly approve another narrowly scoped
renderer-quality decision. Threshold changes, model paraphrasing, full generation, and training
remain outside the frozen design.

## Milestone 4.2 typed-realizer stress result and procedural stop

The live generator path now emits `ProblemIR` rather than final prose. Typed entities, lexemes,
quantities, units, rates, groups, constraints, and answer targets feed a centralized realization
compiler. The compiler emits sentence plans, explicit morphology evidence, a stable
`RenderSignature`, and exact semantic-node-to-clause coverage. Irregular forms are explicit; unknown
morphology rejects deterministically. Weighted means, rates, capacities, counts, and grouping
targets have distinct question contracts.

All eleven Milestone 4.1 defect classes have sanitized regression coverage. The bounded in-memory
stress used 300 deterministic attempts per mathematical family and persisted no question corpus.
All 900 passed typed checks with zero exact or latent-structure duplicates and 900 distinct render
signatures. Nevertheless, 99 number-neutral templates repeated, and MiniLM nearest-neighbor scores
placed 899/900 successful renders at or above the unchanged 0.82 automatic-rejection threshold
(median 0.936379). Manual audit of 60 renders found 13 unnatural imperative-question surfaces and
zero false mathematical labels.

The renderer stress gate therefore failed. The fresh 120-candidate contamination smoke was not
authorized by the gate and did not run. The pure procedural-renderer lineage is closed; thresholds
must not be changed and a Milestone 4.3 must not be created.

### Separately approvable architectural pivot

Retain the typed latent programs, exact labels, deterministic traces, independent verifiers,
target/unit contracts, and contamination pipeline. Replace only the final surface realization with
a pinned local model operating under a constrained interface. The model must never generate or
choose the label. Each surface must round-trip into the typed semantics with exact recovery of all
entities, quantities, units, relations, targets, and required nodes; disagreement or missing
evidence rejects. A bounded design milestone must pin the model/revision/license, quantify local
compute and dependency impact, freeze the constrained prompt/schema and round-trip verifier, and
define a new smoke before any realization generation.

## Milestone 5A constrained local-model realization pivot

The design milestone selected
`Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` as primary and the existing pinned
Qwen2.5-1.5B-Instruct model as fallback. Both are Apache-2.0 and use standard Transformers with
`trust_remote_code=False`; thinking is disabled. Qwen3 requires a separately approved realization
lock with Transformers at least 4.51 but is estimated to fit at 4.5–5.5 GiB FP16/BF16 VRAM.

The model receives a content/value-blind `RealizationRequest` with ordered events, typed opaque
placeholders, target kind, exact question intent, allowed discourse orders, forbidden
transformations, and bounded style controls. It returns strict JSON containing a slot-preserving
question template, exact placeholder inventory, clause-to-node map, target/intent echoes, and no
answer. Missing, extra, altered, duplicated, or misplaced slots/nodes; raw numeric literals;
target, unit, interval, or question-intent changes; answer text; and malformed structure reject
before the compiler inserts values.

Round-trip admission then uses the existing typed morphology/reference/coverage checks, exact latent
execution, independent verifier, terminal-output contract, deterministic quality rules, and complete
answer-blind human audit. No LLM can accept an example, choose a label, judge correctness, or break a
verifier tie.

Generated-to-development contamination retains the pinned MiniLM model and 0.75/0.82 thresholds.
Generated-to-generated diversity retains exact, number-neutral, latent-structure, and five-token
Jaccard 0.35 controls but receives a separately calibrated semantic policy based only on original
fixtures. That policy must be frozen before model-generated smoke output is examined.

The proposed, separately approved Milestone 5B uses 120 new IRs under the existing 60/60 curriculum
and 20% output-contract track. Three deterministic beams per IR create at most 360 candidate
sequences, with stable ordering, no sampling, retry, or replacement. At least 90 clean IRs and 15 per
family are required, with zero false labels, accepted semantic drift, invalid acceptances, verifier
disagreements, or unresolved contamination. Full dataset generation and training remain blocked.

The complete model comparison, schemas, audit protocol, policy analysis, compute estimates, risks,
and next-decision boundary are in `docs/LOCAL_MODEL_REALIZATION_DESIGN.md`.

## Milestone 5B realization-smoke consequence

The procedural source of truth and label path remain valid: all 120 new IRs were constructed
deterministically, both exact verifier families agreed, and no false label occurred. The
value/benchmark firewall also held. The local Qwen3 wording layer, however, admitted zero clean IRs
from 360 fixed beams.

This is a realization-contract failure, not permission to bypass verification. Parsed model output
systematically omitted semantic events or declared invalid clause maps, while many unparsed JSON
objects exhausted the frozen 256-token budget. The complete hidden-label audit confirmed that every
automatic rejection was appropriate. Consequently:

- no generated question from Milestone 5B is approved training data;
- no targeted or generic 4,000-example dataset may be generated;
- the matched-control and QLoRA plans remain designs only;
- the frozen development benchmark and sealed-final partition remain outside generation;
- contamination thresholds, labels, and verifier mathematics remain unchanged.

A future compact-protocol design may retain the same procedural IR, exact arithmetic, verifier,
curriculum, output-contract, and contamination contracts. It must make semantic coverage
deterministically recoverable without asking the model to echo a long inventory and separate clause
map, and it must freeze original fixtures before any new inference. No such milestone is currently
approved.

## Milestone 5C compact-protocol consequence

The compact tagged protocol removed every redundant model echo. Qwen returned only one `<E…>` block
per fact and one `<Q>` block, with immutable value placeholders and deterministic semantic-anchor
tokens. The compiler derived coverage from tags and inserted all approved predicate phrases after
validation. The exact generators, answers, dual verifiers, output-contract track, development
firewall, and internal-diversity policy were unchanged.

This improved structural compliance but not usable realization. All 90 beams parsed; 87 preserved
the full token assignment; no verifier disagreed. Yet automatic and audited clean acceptance were
0/30 because the model echoed the listed tokens rather than composing them into grammatical
predicate-argument clauses. Every beam was unnatural and semantically drifted. Exact replay passed.

Therefore no compact output is accepted training data, the 120-IR Qwen3 follow-up is blocked, and
the 4,000 + 4,000 pilot remains unapproved. The final Qwen3 prompt-patching stop rule is active. The
single recommended next experiment is a separately approved stronger local realization model using
the same frozen compact protocol and gates; changing the protocol again is not recommended.

## Milestone 5D stronger-model conclusion

The final approved model substitution used the exact M5C control set and compact protocol with
Qwen3-4B-Instruct-2507. Artifact, memory, and exact-replay checks passed, but clean acceptance remained
0/30 and every returned beam failed blinded naturalness/semantic-preservation review. Therefore live
LLM surface realization is removed from the proposed production path.

The recommended replacement preserves the independently verified core: sample exact latent programs,
compute labels exactly, and retain both mathematical verifiers. Natural language would come from a
versioned offline bank of independently authored templates, each manually accepted before use.
Deterministic compatibility rules would select templates and fill typed slots; the existing language
quality, exact/template/structural duplicate, development-contamination, and internal-diversity screens
would remain mandatory. No template may contain or be derived from benchmark content. This architecture
requires a separately approved design and bounded smoke before any 4,000 + 4,000 generation.

## Milestone 6A offline-bank evidence

`foundry-template-bank-v1` supplies 18 bookkeeping, 20 rate/ratio, and 20 discrete semantic frames,
four sentence plans each (232 signatures). The bank is source controlled, original, fail closed on
untyped or incompatible IR, and marked human-review pending. Generated-to-development MiniLM and
lexical thresholds remain unchanged; internal diversity uses separate identity/structure controls
without treating same-skill topical similarity alone as duplication.

The bounded smoke accepted 118/120 automatically with exact replay, zero false labels, and zero
verifier disagreements. One latent duplicate and one number-neutral duplicate were rejected. Codex
inspection then found 13 invalid or unnatural surfaces, so the implementation failed the unchanged
systematic-language-defect gate. The packet remains ignored and human review remains pending. Full
generation and training are not ready.

## Milestone 6B composition and review boundary

The offline-bank path now uses an explicit surface compiler. Internal IDs never become prose;
approved surface lexemes, typed noun phrases, centralized ordinals, and token provenance are checked
before the frozen contamination pipeline. Ten deterministic fixtures per plan exercised all 232
plans (2,320 total), with zero final rule failures and all 13 prior defects blocked.

The new fixed-seed smoke retained the targeted 33/14/13 and generic 20/20/20 curricula and the 20%
output-contract track. It accepted 116/120 and rejected three latent copies plus one number-neutral
copy. Every mathematical check agreed; no label, target, language-rule, benchmark-screen, or replay
failure occurred. This establishes **technical readiness for review only**. The 116 candidates are
not an approved dataset, and the bank remains `human_review_pending` until the user exports and
approves the local review JSON.

## Milestone 6C-R review-derived revision and diversity blocker

The genuine v2 review is now a frozen content-free input contract: 120 matching IDs, 60 approvals,
60 rejections, no unsure decisions, and SHA-256 `564a8ca...791`. Approved historical uses and
quarantined uses are recorded separately. Twelve affected sentence-plan families were reauthored;
generator mathematics, labels, verifiers, target contracts, benchmark firewall, and contamination
thresholds were unchanged.

The v3 static matrix validates 2,320/2,320 renders and 232 distinct signatures with no deterministic
failure. The fresh precommitted smoke nonetheless accepted 104/120 because 15 rendered questions
collided under number-neutral duplicate detection and one latent program duplicated. Exact replay and
all correctness/safety checks passed. The required 110 gate therefore failed, no review packet was
created, and neither an 8,000-example generation nor training is permitted. A future proposal must
address runtime selection/diversity without weakening the frozen duplicate or contamination policies.

## Milestone 6D cross-dataset capacity boundary

The proposed pilot contains 4,000 targeted and 4,000 generic-control acceptances. Applying the frozen
125% candidate budget independently to each group/category quota produces 10,003 required attempts:

| Family | Accepted quota | Fixed attempts | Active plan identities | Domain-aware identities | Number-neutral identities |
|---|---:|---:|---:|---:|---:|
| Bookkeeping | 3,534 | 4,418 | 72 | 1,728 | 768 |
| Rate/ratio/percentage/average | 2,266 | 2,834 | 80 | 400 | 88 |
| Discrete constraints | 2,200 | 2,751 | 80 | 1,600 | 320 |
| **Total** | **8,000** | **10,003** | **232** | **3,728** | **1,176** |

The same question-identity pool serves output-contract enabled and disabled examples because the
contract changes the completion, not the question. Every family/difficulty/output stratum is below
its required quota. The capacity gate therefore fails before scheduling. Milestone 6D creates no
allocator, latent schedule, 120-attempt schedule, fresh smoke, raw candidate evidence, or packet.

The minimum current number-neutral shortfalls are 3,650 bookkeeping, 2,746 rate/ratio, and 2,431
discrete surfaces. Adding IDs alone is not sufficient: future capacity work must add independently
authored and reviewed plan structures and scenario/lexical domains that survive the unchanged
number-neutral and contamination controls. Capacity must be re-audited before selection logic is
implemented.

## Milestone 6E bounded-reuse contract and remaining program-space limit

Generated-to-generated identity is now four explicit layers:

1. The normalized, fully rendered question is globally unique across datasets and splits.
2. The complete executable latent program plus instantiated parameters is globally unique.
3. Content-free structural skills may repeat under balanced family/frame/target/difficulty/output
   quotas because repeated reasoning structure is the curriculum.
4. Reviewed sentence plans, scenarios, lexical families, render signatures, and number-neutral
   signatures may repeat only under frozen quota-derived caps.

Generated-to-development matching is unaffected: exact, n-gram, structural, and MiniLM screening
remain fixed at revision `1110a243...b4d41`, 0.75 manual review, and 0.82 automatic rejection.

The cap rule is `ceil(1.25 * stratum quantity / active identities)`. The principal plan and
number-neutral caps are:

| Dataset / family | Required attempts | Plan attempt/accepted cap | Number-neutral attempt/accepted cap | Frame attempt/accepted cap |
|---|---:|---:|---:|---:|
| Targeted bookkeeping | 2,750 | 48 / 39 | 5 / 4 | 191 / 153 |
| Targeted rate/ratio | 1,167 | 19 / 15 | 17 / 14 | 73 / 59 |
| Targeted discrete | 1,084 | 17 / 14 | 5 / 4 | 68 / 55 |
| Generic bookkeeping | 1,668 | 29 / 24 | 3 / 3 | 116 / 93 |
| Generic rate/ratio | 1,667 | 27 / 21 | 24 / 19 | 105 / 84 |
| Generic discrete | 1,667 | 27 / 21 | 7 / 6 | 105 / 84 |

Plan-plus-domain, target-type, difficulty, and output-contract caps are derived by the same formula
and recorded in the content-free capacity audit. No cap is learned from generated outputs.

| Family | Accepted quota | Required attempts | Bounded surface capacity | Verified latent capacity under balance | Ratio | Result |
|---|---:|---:|---:|---:|---:|---|
| Bookkeeping | 3,534 | 4,418 | 5,524 | 5,524 | 125.03% | pass |
| Rate/ratio/percentage/average | 2,266 | 2,834 | 3,544 | 1,632 | 57.59% | fail |
| Discrete constraints | 2,200 | 2,751 | 3,441 | 2,073 | 75.35% | fail |

The full 10,003-attempt schedule is not feasible. Exact finite shortages occur in rate-total (96),
ratio-scale (336), percentage (104), combined-rate (384), equal-distribution (253), and dual-capacity
(90) program domains. Abundant modes are still bounded by semantic-frame balance. Per the capacity
gate, no allocator, candidate schedule, smoke, replay, or review packet follows.

## Signal-first 1,000 + 1,000 capacity result

The reduced pilot freezes 1,000 accepted examples per dataset, 900/100 train/validation splits, 200
output-contract examples per dataset, and 2,504 fixed attempts. Accepted family quotas are
550/233/217 targeted and 334/333/333 generic for bookkeeping/rates/discrete. Attempt quotas are
688/292/272 and 418/417/417 respectively. Difficulty is divided as evenly as integer counts allow.

The compatibility-aware audit applies the unchanged bounded-reuse caps to actual mode/target/frame
edges and shares finite mode supply across datasets:

| Family | Required attempts | Compatible capacity | Shortfall | Result |
|---|---:|---:|---:|---|
| Bookkeeping | 1,106 | 1,384 | 0 | pass |
| Rate/ratio/percentage/average | 709 | 695 | 14 | fail |
| Discrete constraints | 689 | 598 | 91 | fail |

Each targeted family and generic bookkeeping/rates pass independently. Generic discrete fails at
399/417, so all its attempt-level difficulty, output, and split reservations are short. Cross-dataset
disjointness then makes both rate and discrete family totals fail. The audit is exact over the frozen
compatibility graph; no allocator, schedule, smoke, replay, or packet was produced.
