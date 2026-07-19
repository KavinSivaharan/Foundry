# Foundry constrained local-model realization design

## Status and scope

Milestone 5A is a design-only architectural pivot. The pure procedural English renderer is closed
under D-017: its typed compiler preserved mathematics but failed the naturalness and scale-diversity
stress gate. This design retains the reusable parts and inserts a local language model only between
typed semantic planning and deterministic compilation.

No model weight was downloaded or loaded in this milestone. No inference, benchmark access,
synthetic-example generation, dataset creation, dependency change, or training occurred. A future
implementation smoke requires separate approval.

## Responsibility boundary

The procedural layer remains the source of truth. It samples the exact latent program, values,
units, entities, ordered events, target, canonical answer, deterministic trace, and dual-verifier
evidence. The local model has exactly one responsibility:

> Express an already-complete semantic problem specification in natural, varied English.

The model must not invent a mathematical structure, choose or see numeric values, calculate or see
the label, change units, choose the answer target, modify the latent program, see a GSM1K question or
answer, judge correctness, or resolve a verifier disagreement. It receives semantic roles and opaque
placeholders, not their eventual surfaces. Deterministic code fills values only after the proposed
template passes every required check.

## Candidate comparison

Facts in the next table come from the official model cards and pinned Hugging Face repository
metadata. VRAM and speed ranges are explicitly planning estimates for short structured generation on
the project's RTX 3080; they have not been measured for this task.

| Candidate | Immutable revision and license | Documented size and runtime requirements | Non-thinking and structure behavior | Local planning estimate | Decision |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; Apache-2.0 | 1.54B parameters; 3,087,467,144 weight bytes and 3,098,973,447 total repository bytes; Transformers >=4.37; standard Qwen2 support | It is not a hybrid reasoning model, so no thinking switch is needed. It follows instructions well but has no documented strict-JSON guarantee. `trust_remote_code=False`. | 3.5–4.5 GiB FP16/BF16 VRAM; 2–3 GiB optional 4-bit; roughly 60–100 output tokens/s. This exact model already ran locally at about 74 output tokens/s in a longer benchmark workload. | **Fallback.** Lowest dependency and reproducibility risk; already cached and proven on this desktop. |
| `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; Apache-2.0 | Card reports 1.7B parameters and 1.4B non-embedding; repository tensor metadata totals about 2.03B; 4,063,515,592 weight bytes and 4,079,450,110 repository bytes; Transformers >=4.51 | Official chat-template control supports `enable_thinking=False`. No strict-JSON guarantee is documented, so schema/slot checks remain mandatory. Standard Qwen3 support permits `trust_remote_code=False`. | 4.5–5.5 GiB FP16/BF16 VRAM; 2.5–3.5 GiB optional 4-bit; roughly 50–85 output tokens/s, or 20–45 effective tokens/s for three-beam search. | **Primary.** Best balance of instruction following, natural wording, hard non-thinking control, 10 GiB fit, reproducibility, and speed. |
| `HuggingFaceTB/SmolLM3-3B` | `a07cc9a04f16550a088caea529712d1d335b0ac1`; Apache-2.0 | 3.075B repository tensor parameters; 6,150,235,008 weight bytes and 6,167,865,576 repository bytes; card requires Transformers >=4.53 | Official usage supports non-thinking via the chat template. It has no documented strict-JSON guarantee. Standard SmolLM3 support permits `trust_remote_code=False`. | 6.5–8.0 GiB FP16/BF16 VRAM; 3.5–5.0 GiB optional 4-bit; roughly 30–55 output tokens/s. | Not selected. It offers credible instruction following but consumes the most VRAM, requires the largest dependency move, and is likely slowest without a demonstrated placeholder-fidelity advantage. |

Official references: [Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
[Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B), and
[SmolLM3-3B model card](https://huggingface.co/HuggingFaceTB/SmolLM3-3B).

The pinned Qwen3 chat template SHA-256 is
`a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`. The fallback
Qwen2.5 chat template SHA-256 is
`cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`.

### Dependency impact

The current environment has Transformers 4.46.3 and tokenizers 0.20.3. Qwen3 requires a separately
approved, dedicated realization lock with Transformers at least 4.51 and its compatible tokenizers
dependency. The implementation milestone should pin an exact tested patch release (proposed:
Transformers 4.51.3), regenerate only a new `requirements-realization.lock.txt`, and retain PyTorch
2.5.1 CUDA 12.1 unless dependency verification proves that impossible. Qwen2.5 needs no change.
SmolLM3 would instead require Transformers at least 4.53. No dependency changed in Milestone 5A.

Windows/CUDA execution is measured for Qwen2.5. Qwen3 and SmolLM3 use ordinary
PyTorch/Transformers CUDA paths and should run on Windows, but their Windows behavior is an estimate
until a future approved smoke. Their cards do not provide a project-specific Windows guarantee.

## Slot-preserving interface

`model_contracts.py` freezes a value-blind request containing:

- category and semantic frame;
- ordered typed semantic events and stable node IDs;
- typed immutable entity, quantity, unit, rate-interval, location, constraint, and target slots;
- explicit target type and required question intent;
- allowed discourse orders;
- explicit forbidden transformations;
- bounded style, voice, context, and difficulty controls.

Placeholders use an exact uppercase grammar such as `<ENTITY_1>`, `<QUANTITY_1>`, `<UNIT_1>`,
`<RATE_INTERVAL_1>`, `<LOCATION_1>`, and `<TARGET_ENTITY>`. The model never receives the surfaces
that replace them. Every event declares which placeholder tokens its clause must contain.

The frozen system prompt SHA-256 is
`9d3e808d5c887d974919728d5afb51df9daa5760d467d3c08648799aeddcc393`. It requires exactly one
JSON object, byte-exact placeholders, only supplied events, and no values, answers, calculations,
explanations, or extra fields.

The strict response schema contains exactly:

```json
{
  "question_template": "<ENTITY_1> ... <QUANTITY_1> <UNIT_1>. ... <TARGET_ENTITY>?",
  "placeholder_inventory": ["<ENTITY_1>", "<QUANTITY_1>", "<UNIT_1>", "<TARGET_ENTITY>"],
  "clause_to_semantic_nodes": [
    {"clause_index": 0, "semantic_node_ids": ["EVENT_1"]},
    {"clause_index": 1, "semantic_node_ids": ["TARGET"]}
  ],
  "requested_target_type": "count",
  "requested_question_intent": "ask for the target count",
  "style_id": "plain-active"
}
```

There is deliberately no answer field. Unknown, missing, or duplicate JSON fields are rejected.

### Deterministic rejection rules

A model realization is rejected before values are filled if it has any of the following:

- missing, extra, altered, duplicated, or overused placeholders;
- a placeholder inventory that differs from the request;
- a raw numeric literal or answer-bearing text;
- missing, invented, duplicated, or incorrectly mapped semantic nodes;
- a clause that does not contain the slots required by its declared node;
- an unapproved discourse order;
- changed target type, question intent, or style identifier;
- missing unit or rate-denominator slot;
- conflicting questions, duplicated clauses, malformed punctuation, or malformed JSON;
- any later deterministic morphology, reference, coverage, math, answer-contract, or contamination
  failure.

The deterministic compiler receives an exact replacement dictionary only after these checks. It
requires equality with the placeholder set, fills tokens in stable order, rejects unresolved slots,
and records template and replacement hashes.

## Round-trip semantic validation

Acceptance requires all layers, in this order:

1. Strict JSON and typed schema validation with no unknown fields.
2. Exact placeholder-set and occurrence equality.
3. Semantic-node coverage: every required node exactly once unless repetition was preauthorized.
4. Target-type and question-intent equality with the original `TargetSpec`.
5. Unit, numerator, and explicit denominator-slot preservation.
6. Entity-reference and placeholder-role integrity.
7. Clause-to-node mapping and allowed discourse-order validation.
8. Filled-question numeric, morphology, reference, and unit consistency.
9. Existing exact execution of the untouched latent program.
10. Existing independently implemented mathematical verifier and agreement check.
11. Existing terminal-final-answer contract validation where enabled.
12. Deterministic natural-language quality checks.
13. Existing benchmark-contamination pipeline and the separately frozen internal-diversity policy.

Any required deterministic failure rejects the realization. An optional local-model reverse parse
may later be studied only as an extra rejection signal; it can never accept a candidate or break a
tie. An LLM is never a mathematical verifier or judge.

## Naturalness audit

Automatic checks continue to reject malformed grammar metadata, punctuation, noun-number
disagreement, unsupported morphology, duplicated clauses, missing or ambiguous referents,
target mismatches, missing rate intervals, illegal noun elision, and semantic coverage defects.
These checks do not claim to prove natural English.

The future smoke audits every generated candidate, not just the candidate ultimately selected for
an IR. The first audit pass sees the template, typed IR, node map, and filled question but hides the
canonical answer and all verifier outcomes. The reviewer records independently:

- natural, unnatural, or uncertain;
- semantics preserved, drifted, or uncertain;
- a short content-free defect code;
- whether deterministic validation and the final pipeline decision were appropriate.

Uncertain cases are rejected. Only after the semantic-preservation decision is frozen may the
reviewer inspect the procedural answer and verifier evidence. Benchmark labels do not exist in this
workflow and benchmark questions are never shown as realization examples. Detailed wording-bearing
audit records remain ignored; only aggregate categories and hashes may be committed.

## Semantic-similarity policy

`all-MiniLM-L6-v2` creates normalized sentence embeddings whose cosine similarity reflects broad
sentence meaning and topical/lexical proximity. It does not prove program equivalence. Arithmetic
questions from one skill family legitimately share concepts such as totals, groups, rates, and
questions, so distinct latent programs can occupy a tight embedding neighborhood. The 899/900
Milestone 4.2 result demonstrates that applying a general semantic threshold to generated peers can
mistake same-domain membership for duplication.

The opposite error is also serious: weakening the generated-to-benchmark threshold after observing
high rejection rates could admit paraphrases or structural copies of development questions.
Benchmark contamination and internal curriculum diversity therefore serve different scientific
roles.

**Selected future policy: option 2.** Preserve the current MiniLM artifact, pooling, normalization,
0.75 review threshold, and 0.82 rejection threshold for generated-to-development contamination.
For generated-to-generated diversity, retain exact normalized text, number-neutral templates,
latent-program structure, and five-token Jaccard 0.35, then add a separately calibrated semantic
policy. Do not reuse 0.75/0.82 automatically and do not replace the artifact in this lineage.

Before any realization smoke output is examined, a future approved implementation milestone must
freeze the internal policy using an original hand-authored fixture set containing obvious copies,
number swaps, structural copies, close paraphrases, the same skill with a different latent program,
related-but-distinct questions, and unrelated questions. Fixtures must be independent of GSM1K and
the generated smoke. The calibration report must publish content-free pair counts, error rates,
chosen behavior, and a policy hash. If no policy separates true duplicates from legitimate
same-family examples, the implementation smoke must stop before generation rather than lower a
threshold post hoc.

## Deterministic generation strategy

Three strategies were considered:

| Strategy | Benefit | Risk | Decision |
|---|---|---|---|
| One greedy realization per IR | Simplest replay and lowest compute | One surface choice per IR is unlikely to solve diversity | Not selected |
| Fixed deterministic beam search | Small, stable candidate set; no sampling; beam rank supplies stable order | More compute and correlated beams | **Selected** |
| Seeded sampling | Highest surface variety | CUDA/kernel and library drift can change samples; harder audit and replay | Not selected |

The future policy uses the pinned model and chat template, thinking disabled, fixed seed, PyTorch
deterministic settings where supported, `do_sample=False`, three beams, exactly three returned
sequences per IR, 256 maximum new tokens, 90-second timeout, and stable IR/beam ordering. The first
candidate by beam rank that passes every deterministic and human gate is selected. If none passes,
the IR is rejected. There is no retry, regeneration, or budget replacement.

The design configuration file SHA-256 is
`d6e6ca82681b702e07c71a9732a8c81159ea7a9bca78c73193228f72ca4ec3a5`; a unit test pins the
complete file bytes in addition to typed field and gate checks.

Replay must reproduce exact response hashes on the same environment. If a documented CUDA kernel
cannot be bit-exact, the milestone must predeclare deterministic-equivalence fields—exact slots,
nodes, target, filled question, validation decisions, and selected beam—and report byte-level drift;
that exception requires approval before the run.

## Future bounded implementation smoke

The separately approvable smoke uses 120 new procedural semantic IRs and the existing curriculum:

- targeted 60: 33 bookkeeping, 14 rate/ratio/percentage/average, 13 discrete;
- generic 60: 20 per family;
- output contract: 12 IRs in each group;
- one fixed three-beam request per IR, so 120 model calls and at most 360 candidate sequences;
- no benchmark-derived generator input, no replacement, and no retry-until-success.

The ordered pipeline is: typed IR validation, constrained model request, strict JSON/slot/node/target
validation, deterministic filling, existing morphology/reference/coverage checks, dual mathematical
verification, output-contract checks, benchmark contamination, separately frozen internal diversity,
and complete manual audit.

Measured metrics are IR/model requests, candidate sequences, JSON parse rate, placeholder
preservation, semantic-node coverage, target preservation, naturalness acceptance, mathematical
validity, contamination and internal-diversity rejections, clean IR acceptance, failure categories,
runtime, generated tokens/s, peak allocated/reserved VRAM, false labels, semantic drift, and invalid
acceptances.

Readiness requires exactly 120 IRs accounted for, at least 90 clean accepted IRs, at least 15 per
family, zero false labels, zero accepted semantic drift, zero invalid accepted questions, zero
verifier disagreements, zero unresolved contamination, no systematic wording defect, deterministic
replay or a preapproved equivalence contract, and evidence that fixed-budget yield can plausibly
scale to 8,000 accepted examples. The gate cannot be lowered after results are seen.

## Compute and storage estimates

These are planning estimates, not measured results:

- **Download:** Qwen3 repository metadata totals about 4.08 GB decimal (3.80 GiB), including 4.06 GB
  of weights. Budget 4.5–5 GB of cache space.
- **Inference:** FP16 is recommended because it fits and is already the project's proven CUDA dtype.
  Expected peak is 4.5–5.5 GiB VRAM for short prompts and three-beam decoding. BF16 should fit but is
  not needed; 4-bit would add quantization dependencies and reproducibility risk without solving a
  memory constraint.
- **System RAM:** budget 12–16 GB for model staging, Python, validation, and the semantic encoder;
  the desktop's 32 GB is adequate.
- **Per realization:** approximately 250–500 input tokens and 100–180 output tokens for strict JSON.
  Three-beam effective output speed is estimated at 20–45 tokens/s.
- **120-IR smoke:** about 15–45 minutes of GPU generation plus deterministic checks; complete manual
  audit of up to 360 candidates is likely 2–4 hours.
- **8,000 accepted examples:** yield is unknown. At 75% clean IR yield, roughly 10,700 IRs and at
  most 32,100 beam candidates would be required, approximately 3–6 million output tokens and 20–80
  GPU hours. This is a capacity estimate, not authorization or a promise of yield.
- **Disk:** budget 6–10 GB total for the 4.1 GB repository snapshot, cache overhead, MiniLM, ignored
  raw metadata/templates, hashes, and temporary files. Accepted text alone should be well below 1 GB.

Ordinary Transformers is sufficient for the 120-IR smoke and preserves the simplest auditable
runtime. vLLM does not materially help such a small serial workload and lacks the same straightforward
native-Windows path; it should not be introduced. Windows remains adequate for the smoke. WSL2/Linux
may be considered only for a separately approved, high-throughput 8,000-example run after correctness
and yield are demonstrated.

## Risks and exact next decision

The major risks are placeholder noncompliance, semantic drift hidden behind fluent wording, model or
chat-template version drift, beam homogeneity, human-audit burden, and an internal semantic policy
that still confuses topical similarity with duplication. Deterministic rejection makes these risks
reduce yield rather than corrupt labels, but a low yield could still make the architecture
impractical.

The next decision is whether to approve a bounded Milestone 5B that may create a dedicated pinned
realization dependency lock, download only Qwen3-1.7B at the recorded revision, calibrate and freeze
the internal-diversity semantic policy on original fixtures before generation, implement the local
runtime, and run the 120-IR/360-candidate maximum smoke under the gates above. Until that approval,
no weight download, model generation, synthetic dataset, full pilot, QLoRA, SFT, GRPO, benchmark
inference, or sealed-final access is authorized.

## Milestone 5B measured implementation result

Milestone 5B implemented the design without changing its fixed budget. The dedicated environment is
CPython 3.12.10, PyTorch 2.5.1+cu121, Transformers 4.51.3, tokenizers 0.21.4, and PyYAML 6.0.2;
`pip check` passes. The exact Qwen3 snapshot revision is
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, Apache-2.0, `trust_remote_code=False`, and
4,079,450,110 bytes. A fresh `local_files_only=True` FP16 reload succeeded with thinking disabled.

Before Qwen generation, 24 original fixture pairs compared exactly three internal-diversity
policies. `evidence-gated-balanced-v1` was selected with policy SHA-256
`26c030e8497c4727e286ff3e89d4720cee1c2681a224b8a93b8c515ef521cc90`: 22/24 exact fixture
outcomes, zero duplicate escapes, and zero distinct automatic rejections. Two deliberately
ambiguous pairs passed instead of entering review and were documented rather than used to tune a
threshold. The generated-to-development MiniLM revision and 0.75/0.82 policy were unchanged.

The one counted run used master seed `foundry-m5b-ir-master-20260718-v1`, 60 targeted and 60 generic
IRs, 12 output-contract IRs per group, and exactly three beams per IR. No replacement or retry was
available. Aggregate results were:

| Measure | Result |
|---|---:|
| IRs / returned beams | 120 / 360 |
| Exact JSON parses | 181/360 (50.28%) |
| Full placeholder preservation among parsed JSON | 41/181 (22.65%) |
| Parsed beams passing semantic-node coverage | 0/181 |
| Parsed beams preserving target/intent | 171/181 (94.48%) |
| Beams consuming all 256 generated tokens | 219/360 |
| Unparsed beams at 256 tokens | 160/179 |
| Automatically selected IRs | 0/120 |
| Dual-verifier disagreements / false labels | 0 / 0 |
| Backend failures / per-IR timeouts | 0 / 0 |

The dominant response pattern was a target-only question that omitted every input event. The other
semantics-preserving pattern repeated imperative event descriptions as one run-on or all-caps
instruction rather than producing a natural word problem. Verbose placeholder inventories and
clause maps also exhausted the 256-token budget before many JSON objects closed. Because no beam
reached the later semantic screens, there is no benchmark-contamination or internal-diversity pass
rate to reinterpret.

The first audit pass hid answers and verifier evidence. All 360 beams were reviewed through 37 exact
template groups: 63 were natural but semantically drifted, 59 preserved semantics but were
unnatural, 301 drifted in total, and 297 were unnatural. Every automatic rejection was correct;
invalid acceptances and incorrect rejections were zero. Exact replay reproduced every beam and
decision with SHA-256 `a2e6fb565da817ec5e2e6e3c87ba8a54643b2b5ec294dd8f5d24204083d06dcf`.

Measured generation time was 802.093 seconds (812.465 seconds end to end), or 6.684 seconds per IR.
The run generated 86,199 output tokens at an aggregate 107.47 returned tokens/s. Peak allocated and
reserved VRAM were 4,289,053,184 and 4,716,494,848 bytes; peak process working set was
7,531,028,480 bytes. The complete ignored raw/audit/replay directory occupies 967,542 bytes.

At this measured rate, 8,000 IR attempts—not accepted examples—would take about 15.0 hours and
roughly 65 MB of similarly shaped raw evidence, plus the fixed 4.08 GB model cache and 4.87 GB
environment. With zero accepted IRs, no finite or credible projection exists for 8,000 accepted
examples under this protocol.

## Resulting stop and next design boundary

The frozen readiness gate failed at 0 clean IRs versus 90 required, so full pilot generation is not
ready. The result does not justify the fallback model, more beams, a longer limit, repairs, retries,
or lower gates after observing the outputs.

The narrowest scientifically valid follow-up is design-only: replace imperative event descriptions
and verbose echoed inventories/maps with a compact declarative response whose slot and semantic
coverage can still be derived deterministically. Original hand-authored fixtures must prove the new
contract before any separately approved model inference. The alternative is to stop the local
surface-realization route. Neither choice authorizes full generation or training.

## Milestone 5C compact tagged micro-smoke

Milestone 5C implemented `foundry-compact-tagged-v1`. The model returns only consecutive `<E1>`…
`<En>` fact blocks and one terminal `<Q>` block. Each block receives an exact set of immutable value
and semantic-anchor tokens. Deterministic code derives node coverage from tags, enforces token-to-
event assignment, fills approved predicates and values only after validation, and retains every
mathematical, target, verifier, output-contract, contamination, and diversity decision.

Frozen hashes:

- system prompt: `d5ea32af1d6df2c6bc06f7a315cab68084c26c0c719d5d08d1cb7c4628630222`;
- user protocol: `6ed50a9c49434ec2e61d4d6065a9854aca414523b7b14eb5e33c60eee3285454`;
- combined protocol: `6aec762647b03708268da0ae85c5c584821bdc2c82064a6534251c695a48fcf8`;
- normalized config: `78d1f3e37dad7e97c7e857a0f60efa5459af55b246a34dc653efceb01b47438b`;
- generation configuration: `fe9614dea4be2d5c7ec6983e54deb86ebd19787d61e53a62bc6bd5fb63315968`.

The counted run used 30 new IRs: targeted 8/4/3 and generic 5/5/5 across bookkeeping/rates/
discrete, with three output-contract IRs per group. Three deterministic beams per IR produced 90
outputs. Tag parse was 90/90; placeholder, semantic-anchor, and target preservation were each 87/90;
automatic selections were 0/30. All 90 hidden-label audit decisions were unnatural, drifted, and
correctly rejected. False labels, verifier disagreements, invalid acceptances, incorrect rejections,
timeouts, and backend errors were zero. No beam reached semantic screening because earlier language
or fill checks failed.

Generation took 83.474 seconds and the complete counted run 92.869 seconds, with 11,530 input and
11,472 output tokens. Peak allocated/reserved VRAM was 3,728,026,112/3,825,205,248 bytes; peak
process RAM was 7,532,572,672 bytes. Exact replay reproduced run SHA-256
`b9b1a7bc8214c2656b6cd45cb089252f63fbe572c52f910e1148a34cd6a4358a`.

The micro-gate failed at 0 clean IRs versus 22 required and zero accepted examples in every family.
No 120-IR Qwen3 compact smoke is justified. Under the final stop rule, the single recommendation is
to test a stronger local realization model with this protocol frozen; no further Qwen3 prompt patch,
full generation, or training is authorized.

## Milestone 5D: final stronger-model substitution

The frozen compact protocol was tested without modification using official Apache-2.0
`Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`. The complete snapshot is
8,060,917,568 bytes and loaded offline as 4,022,468,096 FP16 parameters on the RTX 3080 without CPU
offload. The three-beam probe peaked at 8,489,271,296 reserved bytes and left 946,094,080 bytes free,
so compatibility passed.

Every M5C control field matched: 30 plans, semantic-IR hashes, latent hashes, and compact-request
hashes; control manifest `794ade78741918113cfab2f223c58007c9a5a3289196cdd36c6aa3d16a8cad66`.
The compact system/user/combined hashes remained unchanged. The counted 90-beam run parsed 71 outputs,
preserved placeholders/anchors/targets in 47/50/47, and selected zero IRs. Blinded audit found 90
unnatural and 90 drifted outputs. Replay exactly reproduced
`7043fb5f94cbd95fe76391fa167ba766acf5080f77b0fede7197c00b8b9a9f01`.

The final local-model stop rule is active. No other local realization model or compact-prompt revision
is recommended. The proposed successor is an offline, manually vetted template bank with deterministic
typed slot filling; procedural IR, labels, verifiers, and contamination controls remain unchanged.

## Live-model route closure confirmed by Milestone 6A

Milestone 6A did not load or run any realization model. It implemented the D-021 successor as a
source-controlled bank with typed deterministic filling. This preserves the final live-model stop
rule: Qwen3-1.7B, Qwen3-4B, the compact protocol, and further prompt/model substitutions remain closed.

The bank's automatic yield was high, but Codex inspection found systematic frame-name and morphology
defects. This is a template-composition blocker, not a reason to reopen live inference. Any next
proposal must remain offline and must be separately approved; full generation is blocked.
