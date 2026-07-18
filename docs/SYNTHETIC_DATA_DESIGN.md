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
