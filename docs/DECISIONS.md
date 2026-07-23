# Foundry Decision Log

This log separates proposals from approved decisions. A proposal does not authorize implementation.

## D-030 — Use the runtime normalizer as the sole scheduling identity and stop on exact capacity

- **Date:** 2026-07-19
- **Status:** implemented in Milestone 7C; full schedule gate failed
- **Decision:** Both scheduling and runtime call `canonical_number_neutral_identity`; scheduler metadata may remain diagnostic but cannot establish uniqueness. A schedule/runtime hash mismatch fails closed.
- **Evidence:** All five Milestone 7B collisions differed only in metadata erased by runtime normalization. Exhaustive evaluation of the fixed weighted-average candidate pool found eight runtime identities. Frozen caps permit at most 40 targeted and 48 generic weighted-average attempts, below the required 70 and 100.
- **Consequence:** The prior 2,504-slot metadata-based schedule remains preserved negative evidence but cannot authorize generation. No fresh smoke or assisted packet is issued. Any next proposal must expand genuinely distinct reviewed surface forms or explicitly revise the frozen reuse policy; it may not weaken runtime normalization post hoc.

## D-029 — Use feasible submode balancing; retain runtime identity as authoritative

- **Date:** 2026-07-19
- **Status:** implemented in Milestone 7B; review gate failed
- **Decision:** Allocate frozen rate and discrete submodes by deterministic water-filling with constrained difficulty assignment instead of infeasible equal-style caps. Preserve the runtime number-neutral normalizer as the source of truth.
- **Evidence:** The 2,504-slot dry schedule was feasible and exactly replayable. Its 120-question smoke accepted 115, with zero mathematical, language, target, or contamination failures, but found five number-neutral collisions that scheduler-only metadata had predicted as distinct.
- **Consequence:** Do not issue a human-review packet until scheduling computes the exact runtime identity and a newly approved smoke has zero collisions.

## D-001 — Recommend arithmetic reasoning for Phase 1

- **Date:** 2026-07-16
- **Status:** accepted for Phase 1 on 2026-07-16
- **Decision:** Use grade-school arithmetic reasoning with a pinned public GSM1K revision and `Qwen/Qwen2.5-1.5B-Instruct` as the Phase 1 experiment.
- **Alternatives considered:**
  - Python function synthesis with EvalPlus HumanEval+ and `Qwen/Qwen2.5-Coder-1.5B-Instruct`.
  - Structured function calling with a pinned BFCL V4 single-turn subset and `Qwen/Qwen2.5-1.5B-Instruct`.
- **Rationale:** GSM1K has objective integer labels, was created as a contamination-aware counterpart to GSM8K, and supports synthetic examples generated from executable arithmetic programs. A 1.54B model should permit a complete local QLoRA loop on an RTX 3080. This combination tests Foundry's core data/evaluation loop with fewer confounding infrastructure problems than sandboxed code or function-call equivalence.
- **Expected consequences:** Phase 1 will be narrower but easier to audit. It should require no paid API or cloud GPU. If successful, it provides reusable evaluation, synthesis, verification, training, and comparison interfaces for a more agentic Phase 2 task.
- **Reconsider if:** the model has too little measurable headroom; the public dataset cannot be pinned or safely split; a smoke test cannot fit the target GPU; exact-answer parsing is unstable; or the user prioritizes product-facing tool use over minimizing research risk.

## D-002 — Prefer the general Qwen checkpoint over a math-specialized checkpoint

- **Date:** 2026-07-16
- **Status:** accepted for Phase 1 on 2026-07-16
- **Decision:** Start from `Qwen/Qwen2.5-1.5B-Instruct`, not a Qwen2.5-Math checkpoint.
- **Alternatives considered:** Qwen2.5-Math-1.5B-Instruct, Qwen3-class small models, and models near 3B parameters.
- **Rationale:** The general 1.5B checkpoint has an Apache-2.0 license, documented 32,768-token context, likely local feasibility, and more likely headroom. A math-specialized baseline could obscure whether Foundry's loop or prior domain post-training caused the result. Newer reasoning-mode models also introduce output-mode and evaluation complexity that is not needed for the first loop.
- **Expected consequences:** The base score may be lower, but a targeted SFT gain should be easier to detect and attribute. The work will demonstrate adaptation rather than incremental polishing of an already-specialized model.
- **Reconsider if:** the baseline is effectively random, cannot follow the answer format, or requires more synthetic data than the local training budget permits.

## D-003 — Put a benchmark firewall between evaluation and synthesis

- **Date:** 2026-07-16
- **Status:** accepted for Phase 1 on 2026-07-16
- **Decision:** Use benchmark labels only inside the scorer. The generator receives only predefined category targets and aggregate failure statistics, never benchmark questions or answers. A separate overlap checker may compare generated text with benchmark prompt fingerprints solely to reject contamination.
- **Alternatives considered:** feed failed benchmark examples to an LLM to create close variants; train on the public benchmark's training split; or allow the generator to see benchmark rationales.
- **Rationale:** Close variants can leak benchmark structure and inflate apparent improvement. An explicit firewall makes the causal claim stronger: improvement must come from learning a category of reasoning rather than memorizing benchmark instances.
- **Expected consequences:** Synthesis will be less directly tailored and may require a better hand-authored taxonomy. The pipeline will need separate permissions/interfaces for scoring, category aggregation, generation, and overlap rejection.
- **Reconsider if:** no measurable transfer occurs despite verified data, but any relaxation must be approved and must preserve a locked final holdout.

## D-004 — Require a reproducible SFT result before GRPO

- **Date:** 2026-07-16
- **Status:** accepted project constraint; implementation not approved
- **Decision:** Do not implement GRPO in the initial SFT loop. Hold a separate admission review only after SFT demonstrates a measurable, reproducible gain and a remaining failure has an objective reward.
- **Alternatives considered:** build SFT and GRPO together; skip SFT; use a learned LLM judge as the initial reward.
- **Rationale:** GRPO adds multiple completions per prompt, reward design, new failure modes, and materially more compute. Exact-answer math can support a strong reward later, but adding it before an SFT baseline would prevent us from knowing which mechanism produced any gain.
- **Expected consequences:** Phase 1 reaches a trustworthy result sooner and at lower cost. GRPO must justify itself against an SFT plateau.
- **Reconsider if:** SFT is correctly implemented and reproducibly plateaus; the proposed reward is independently verifiable; known reward exploits have tests; the extra compute is estimated and approved; and a quantitative success threshold is fixed in advance.

## D-005 — Pin immutable Hub revisions and create a 904/301 split

- **Date:** 2026-07-16
- **Status:** accepted and implemented in Milestone 1
- **Decision:** Pin `Qwen/Qwen2.5-1.5B-Instruct` at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` and `ScaleAI/gsm1k` at revision `bc09569d09a614b9b530edc7f076fb214ac10493`. Deterministically partition the dataset's 1,205 pinned rows into 904 development identifiers and 301 sealed-final identifiers using seed `foundry-gsm1k-v1`.
- **Alternatives considered:** floating `main` revisions; a random split generated at runtime; an 80/20 split; using the entire benchmark as development data; or storing benchmark questions in the manifests.
- **Rationale:** Immutable revisions prevent upstream changes from silently altering an experiment. A roughly 75/25 split leaves enough development examples to categorize failures and 301 final examples for a meaningful paired comparison. Hash-ranking every pinned row makes the partition reproducible without storing benchmark content.
- **Expected consequences:** The manifests are stable only for the exact dataset revision and configuration. They contain hashed IDs and row indices, not questions or labels. The final subset remains inaccessible through normal evaluation paths unless an explicit sealed-final override is supplied.
- **Reconsider if:** the pinned dataset is corrected upstream; the baseline reveals that 301 final examples cannot support the predeclared uncertainty analysis; or a future official benchmark split provides a stronger separation. Any change would create new manifests and a new experiment lineage rather than rewriting these files.

## D-006 — Use Python 3.12 virtual environments and pip-compile locks

- **Date:** 2026-07-16
- **Status:** accepted and implemented in Milestone 1
- **Decision:** Require Python `>=3.12,<3.13`. Because `uv` was not installed on the detected machine, use an isolated `.venv`, exact direct dependency pins in `pyproject.toml`, and pip-compiled transitive locks in `requirements-dev.lock.txt` and `requirements-smoke.lock.txt`.
- **Alternatives considered:** install `uv` globally; use the macOS system Python 3.9.6; use Conda as the project-level environment manager; or commit only unconstrained dependency ranges.
- **Rationale:** The user authorized the simplest reproducible fallback when `uv` was unavailable. Python 3.12.11 was already installed locally, while changing global package managers or adopting Conda would broaden the milestone. Exact locks make the tested development environment reviewable.
- **Expected consequences:** Developers create a normal virtual environment and install from a lock. The CUDA PyTorch wheel must still be installed from PyTorch's official CUDA index on the RTX machine before applying the smoke lock; that hardware-specific installation was not validated on Apple Silicon.
- **Reconsider if:** `uv` becomes an agreed project dependency, target-platform locking becomes difficult with pip-tools, or training requires a CUDA-specific environment format that pip cannot reproduce reliably.

## D-007 — Enforce one explicit final-answer line

- **Date:** 2026-07-16
- **Status:** accepted and implemented in Milestone 1
- **Decision:** Score only the last non-empty line when it is the single line beginning `Final answer:` and contains a signed integer, correctly grouped comma integer, integral decimal such as `42.0`, or a boxed form such as `\boxed{42}`. Reject non-integral decimals, malformed commas, units after the number, multiple final-answer lines, and trailing text.
- **Alternatives considered:** extract the last number anywhere in the response; accept a broad collection of answer phrases; ask an LLM judge to interpret the response; or compare full rationales.
- **Rationale:** A permissive last-number parser can be gamed by printing many numbers and can turn formatting changes into fake accuracy. The explicit contract is deterministic, unit-testable, and stated in the prompt.
- **Expected consequences:** Some semantically correct but noncompliant responses will count as invalid. That invalid-output rate is a useful model behavior metric rather than something to hide.
- **Reconsider if:** a benchmark requires non-integer answers, the prompt format changes before any baseline is run, or manual audit shows a systematic false-rejection pattern. Parser changes after a baseline would require rerunning every compared model.

## D-008 — Do not freeze a prompt or admit Milestone 2 after format calibration

- **Date:** 2026-07-17
- **Status:** accepted Milestone 1.5 stop decision; no prompt selected
- **Decision:** Keep the strict `Final answer: <integer>` parser unchanged, freeze none of the three tested prompts, and do not begin the main-development baseline. The 30 calibration identifiers remain permanently excluded from the future 874-ID main-development baseline.
- **Alternatives considered:** retain the current prompt; select minimal `format_v1`; select explicit-contract `format_v2`; loosen the parser to accept boxed/prose/unit answers; raise the 512-token limit; or continue testing more prompts beyond the approved 90-generation budget.
- **Evidence:** On the same 30 deterministic calibration identifiers, current prompt validity was 16.67%, `format_v1` validity was 10.00%, and `format_v2` validity was 43.33%. No run had a generation failure. `format_v2` reduced average output length from 305.97 to 230.60 tokens and eliminated token-limit hits, but still left 17/30 invalid. None met the predeclared 90% validity gate.
- **Rationale:** Selecting the best relative result would violate the absolute admission rule and would carry a majority-invalid output contract into the expensive baseline. Loosening the parser would change scoring rather than prove model compliance. A fourth prompt would exceed the approved maximum of 90 generations. The evidence supports stopping, not redefining success after seeing results.
- **Expected consequences:** Milestone 2 remains blocked and no evaluation prompt is frozen for it. The repository retains all three prompt configs and aggregate summaries as calibration evidence, while raw predictions remain ignored. Any further prompt, constrained-decoding, parser, or generation-control experiment requires a new explicit scope and must continue to exclude the 30 calibration IDs from the reported main baseline.
- **Reconsider if:** a separately approved format-control experiment reaches at least 90% valid outputs on the fixed 30-ID calibration set, has zero generation failures, avoids unreasonable output growth, preserves the strict final-answer contract, and is selected for compliance rather than math accuracy.

## D-009 — Separate compliance from accuracy, but reject the evaluator after fresh validation

- **Date:** 2026-07-17
- **Status:** accepted Milestone 1.6 stop decision; evaluator not admitted
- **Decision:** Preserve D-007's strict parser as the exact-format compliance metric and retain the deterministic `foundry-terminal-integer-v1` extractor as calibration evidence, but do not freeze the current prompt/extractor pair for Milestone 2. Treat the new 844-ID manifest as a candidate baseline pool only.
- **Alternatives considered:** continue using strict-format validity as benchmark accuracy; accept the canonical extractor solely from its 90-output calibration result; modify it after observing fresh validation; raise the generation limit and rerun; accept the two conservative false rejections post hoc; or begin Milestone 2 despite the failed gate.
- **Evidence:** Re-scoring the original 90 outputs produced extractable rates of 96.67% current, 93.33% `format_v1`, and 90.00% `format_v2`; all 63 newly accepted outputs were manually audited with zero false extractions. On 30 fresh current-prompt IDs, extractability fell to 76.67%, exact compliance was 10.00%, benchmark accuracy was 46.67%, generation failures were zero, and audited false extractions were zero. Rejections were three token-limit completions, two non-integral decimals, and two clear but unsupported terminal phrasings. Even accepting the latter two would reach only 83.33%.
- **Rationale:** Separating compliance from mathematical extraction is conceptually correct, but the fresh result shows the calibrated grammar and generation behavior do not yet satisfy the predeclared reliability threshold. Editing after validation would tune to the admission set, and a second run would not be an independent confirmation.
- **Expected consequences:** Milestone 2 remains blocked. The repository preserves the versioned extractor, dual metrics, 30/30/844 identifier split, aggregate summaries, and content-free manual audits. Any next proposal needs a new untouched admission set and must predeclare grammar/generation changes before inspecting it.
- **Reconsider if:** a separately approved blocker-resolution design preserves the strict compliance metric, uses deterministic auditable extraction, reaches at least 90% extractability on a new untouched set, has zero confirmed false extractions and generation failures, and avoids unreasonable output growth.

## D-010 — Stop evaluator calibration after the final fresh gate fails

- **Date:** 2026-07-17
- **Status:** accepted Milestone 1.7 stop decision; evaluator and 814-ID baseline not admitted
- **Decision:** Preserve D-007's exact-format parser unchanged and retain `foundry-terminal-number-v2`, the current prompt, the 768-token deterministic configuration, and the 814-ID candidate baseline as reproducible evidence only. Do not iterate on the parser or prompt again, do not freeze this evaluator for Milestone 2, and do not begin Milestone 2 without a separate user choice.
- **Alternatives considered:** admit the evaluator because the old-set re-score reached 90%; add post-hoc rules for the four new clear rejections; raise the token limit beyond 768; create another calibration set; proceed to Milestone 2 while predeclaring unextractable outputs as incorrect and reporting coverage separately; or reconsider the base model/benchmark.
- **Evidence:** The final extractor exactly normalizes explicit integers, decimals, and fractions and safely recognizes tested terminal wrappers; all newly accepted calibration outputs and all 25 accepted final-validation outputs were manually audited with zero false extractions. The unchanged Milestone 1.6 output re-score reached 27/30 extractable (90.00%), but the untouched final 30-ID set reached only 25/30 (83.33%), below the gate. Exact compliance was 5/30 (16.67%), accuracy was 13/30 (43.33%), backend generation failures were zero, one response reached 768 tokens, and four complete clear answers used unsupported prose wrappers.
- **Rationale:** Zero false extraction demonstrates conservatism, not sufficient coverage. Adding rules after inspecting the last untouched set would convert validation errors into calibration targets and violate the final planned gate. The evidence no longer justifies another evaluator-calibration milestone under this research direction.
- **Expected consequences:** Milestone 2 remains blocked by the failed admission rule. The deterministic 30 prompt-calibration / 30 extraction-validation / 30 final-validation / 814 candidate-baseline partition and all hashes remain available for audit, but none authorizes a baseline run. No training, synthesis, sealed-final evaluation, SFT, QLoRA, or GRPO follows from this decision.
- **Next user decision:** Choose exactly one: (1) authorize Milestone 2 with every unextractable output scored incorrect and extraction coverage/exact compliance reported separately, using the recorded evaluator and 814-ID set without further calibration; or (2) reconsider the Phase 1 base model or benchmark.

## D-011 — Admit the frozen evaluator through a one-time coverage-gate exception

- **Date:** 2026-07-17
- **Status:** accepted for Milestone 2 by explicit user approval
- **Decision:** Proceed once with the frozen Milestone 1.7 evaluation stack on the untouched 814-ID development baseline. Every unextractable response counts as mathematically incorrect in end-to-end accuracy; extractability and literal exact-format compliance remain separately reported. Do not change the prompt, strict parser, canonical extractor, 768-token generation configuration, model/dataset revisions, or development manifest.
- **Frozen contract:** `Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; `ScaleAI/gsm1k@bc09569d09a614b9b530edc7f076fb214ac10493`; prompt SHA-256 `738ea5a3b94e7c75ac0bd50a229bbf04f3fc5d773e14658bc6728bc7a4b18350`; extractor `foundry-terminal-number-v2` SHA-256 `e099d1c247968fed982cb849022ec3137b1694c15f23a65663a127b8158c06df`; config SHA-256 `5f315d5de645f9563b8d1e61bc8e02c3513c453238ad9e1d6f9473489b5a622b`; 814-ID manifest SHA-256 `5e810d3ab644bef1d43c598a14a6164ba6464b27fde50e92a2f241816ce87897`; greedy decoding and 768 maximum new tokens.
- **Exception rationale:** The final validation found zero false extractions; remaining errors were conservative rejections; end-to-end benchmark accuracy can validly score all unextractable responses wrong; and further evaluator tuning after multiple observed calibration sets would risk evaluator overfitting.
- **Alternatives considered:** keep Milestone 2 blocked; tune another parser/prompt iteration; treat unextractable responses as missing rather than wrong; or reconsider the base model/benchmark.
- **Expected consequences:** The resulting score is a frozen base-model development baseline, not a sealed-final benchmark result. Coverage limits remain visible through extractability and exact-format compliance. Failure analysis may use only development results and may not begin synthesis or training.
- **Reconsider if:** any frozen hash differs, the 814 IDs are not wholly development-only and disjoint from calibration/validation IDs, CUDA cannot execute the run, or repository safety checks reveal benchmark-content leakage. In those cases stop rather than substitute artifacts or settings.

## D-012 — Preserve the frozen baseline and disclose sampled false extractions

- **Date:** 2026-07-18
- **Status:** accepted Milestone 2 completion record; no synthesis or evaluator change authorized
- **Decision:** Preserve the single frozen 814-example result exactly as run and label it the frozen-evaluator development baseline. Do not change the extractor, rescore, retry, or rerun after the failure audit found two false acceptances in a deterministic sample of 100 extractable-but-wrong records. Report the two observed cases as an evaluator-precision limitation and do not claim that the measured 92.38% extractor rate equals audited model-intent coverage.
- **Evidence:** The run produced 521/814 correct (64.00% end-to-end), 752/814 extractable (92.38%), 69.28% accuracy among extracted answers, 130/814 exact-format compliant (15.97%), 62 unextractable, three truncations, and zero backend failures. Both sampled false extractions remained mathematically incorrect, so they do not change the 521-correct numerator; however, the audit did not inspect the 521 records scored correct and therefore cannot rule out accidental label matches elsewhere.
- **Alternatives considered:** silently treat the sample as zero-false-extraction evidence; change the frozen extractor after seeing baseline outputs; rescore or rerun the baseline; begin synthesis despite the new uncertainty; or discard the entire reproducible run.
- **Rationale:** Altering the evaluator after inspecting the baseline would violate the frozen contract and make comparisons vulnerable to post-hoc tuning. Discarding the run would lose valid hardware and aggregate evidence. The scientifically honest outcome is to retain the exact measurement, separate it from claims about model intent, and require a new explicit decision before the result is used as the scoring foundation for trained-model comparisons.
- **Expected consequences:** The aggregate baseline and provisional failure taxonomy remain useful for engineering orientation and hypothesis generation. They are not sufficient by themselves to establish full extractor precision or authorize targeted data generation. Raw responses remain ignored; only aggregate results, deterministic identifier references, and sanitized interpretations are committed.
- **Next user decision:** Choose whether to authorize (1) a bounded audit of records scored correct to estimate accidental extraction risk before candidate comparisons, or (2) targeted synthetic-data design using the provisional taxonomy while explicitly accepting the frozen evaluator's documented precision limitation.

## D-013 — Trust the frozen development baseline after a complete label-blind correct-response audit

- **Date:** 2026-07-18
- **Status:** accepted Milestone 2.1 completion record; no synthesis, generation, or training authorized
- **Decision:** Classify the frozen 521/814 development result as `BASELINE TRUSTED` for development guidance. Preserve the original frozen evaluator score beside the audited result; do not modify the prompt, strict parser, canonical extractor, generation configuration, model/dataset revisions, or manifest. The audited lower bound, upper bound, and adjusted exact accuracy are all 521/814 (64.0049%).
- **Evidence:** All 521 correct-scored completions were reviewed without benchmark answers or question text using exact extraction rules, source spans, terminal context, completion metadata, token counts, and generic suspicion flags. After all decisions were complete, 521 classifications were frozen with SHA-256 `669a866e984c35908bdb9e5443cb989733fd762d11bf62456387a25a5c12e14c` before score metadata was joined. The audit found 521 intended answers, zero false-positive correct answers, zero ambiguous cases, and a 0% confirmed false-positive rate. Correct answers comprised 90 strict-parser and 431 canonical-only extractions.
- **Known-pattern review:** The prior percentage-plus-currency terminal collision occurs zero times among correct-scored responses. Four completions contain closely related negative-intent language, but each explicitly expresses a positive magnitude such as an amount lost; none loses an intended negative sign or creates a coincidental benchmark match. The two false extractions previously observed among wrong answers remain disclosed and still do not affect the numerator.
- **Alternatives considered:** treat the 64.00% score as permanently untrusted despite a complete audit; adjust the score without evidence; reopen the frozen extractor; rerun the model; or proceed directly to synthesis without recording the audit boundary.
- **Rationale:** Label-blind intent classification eliminates benchmark agreement as evidence, and freezing before the score join prevents post-hoc reclassification. Zero false-positive correct answers and zero ambiguity satisfy the predeclared trust rule. The audit is exhaustive for correct-scored development outputs but does not independently solve questions, validate sealed-final behavior, or make the provisional failure taxonomy exhaustive.
- **Expected consequences:** The 231 extractable-but-wrong outputs remain a useful future failure-analysis pool with the two known extractor-intent exceptions disclosed; the 62 unextractable outputs remain interpretable as evaluator rejections counted wrong. The existing provisional taxonomy is reliable enough to guide a separately approved targeted synthetic-data design. No synthetic examples, model training, SFT, QLoRA, GRPO, or sealed-final evaluation follows automatically.
- **Next user decision:** Decide whether to authorize Milestone 3: a bounded, development-only targeted synthetic-data design that freezes generation and verification rules before any examples are created.

## D-014 — Use independently verified procedural data for the first targeted pilot

- **Date:** 2026-07-18
- **Status:** accepted Milestone 3 design record; generation and training remain unapproved
- **Decision:** Select multi-step bookkeeping/omission, rate/ratio/percentage/average, and constraint/distribution/discrete reasoning as the first three targeted curriculum categories. Use a separate shared `terminal-final-answer-contract-v1` output track. Build the first pilot from independently sampled executable arithmetic programs rendered through controlled templates. Derive labels through exact arithmetic, require a second logically independent verifier, and reject every disagreement. Compare a 4,000-example targeted curriculum with a matched 4,000-example broad generic control whose only meaningful difference is curriculum selection.
- **Evidence:** Exhaustive development-only review classified all 293 failures: 69 output format/extraction, 68 bookkeeping/omission, 53 target/language interpretation, 28 rate/ratio/percentage/average, 27 constraint/discrete, 24 time/unit/sequence, 22 arithmetic execution, and two benchmark-risk cases. Confidence was high/medium/low for 274/17/2 records. The selected categories balance prevalence with exact independent generation, dual verification, diversity, low ambiguity, and measurable category impact. Target/language interpretation was deferred despite its prevalence because controlled rendering and contamination risks are higher.
- **Alternatives considered:** (A) fully procedural programs plus controlled templates; (B) procedural programs plus local-model paraphrasing and semantic verification; and (C) frontier-model generation followed by executable verification. B is deferred because paraphrasing can change semantics and needs a new local model path. C is rejected for the first pilot because paid/cloud use, memorization risk, and weaker reproducibility add no label-quality advantage. Generic arithmetic alone is retained only as the matched control.
- **Verification and contamination contract:** Every accepted example needs two distinct verifier/method families agreeing on one exact rational result. Exact text, number-normalized template, latent structure, token five-gram, and semantic-similarity gates run before acceptance; unresolved semantic screening escalates rather than auto-passing. Benchmark content may be used only for local comparison and may never become generator input or a tracked artifact.
- **Pilot boundary:** Milestone 3 adds typed schemas, contracts, original fixtures, aggregate failure metadata, configuration, and documentation only. It does not implement the full generator, download a semantic encoder, generate the 120-candidate smoke or 4,000-example datasets, or run training.
- **Expected consequences:** A later experiment can distinguish failure-targeted selection from generic arithmetic under matched data and QLoRA budgets. Controlled templates may limit language diversity, and the 7.5–9.5 GiB future QLoRA estimate remains unmeasured on this RTX 3080. The exact local semantic encoder must be pinned before any candidate can pass contamination screening.
- **Next user decision:** Decide whether to approve Milestone 4 narrowly: pin one local semantic-similarity artifact, implement only the three approved procedural families plus output track, and run at most the 120-candidate generator smoke. The full 4,000 + 4,000 pilot and all training remain separate decisions.

## D-015 — Block full pilot generation after the procedural smoke fails quality and yield gates

- **Date:** 2026-07-18
- **Status:** accepted Milestone 4 stop decision; full generation and training remain unapproved
- **Decision:** Preserve the frozen semantic thresholds and generator-smoke evidence, but do not generate the matched 4,000 + 4,000 pilot datasets. Treat all 14 semantic review-band cases as conservative generated-to-generated duplicate rejections. Require a separately approved, fresh bounded smoke after repairing only the observed rendering correctness and controlled-template diversity defects.
- **Evidence:** Exactly 120 fixed attempts yielded 24 accepted and 96 rejected. The accepted distribution was four bookkeeping, 16 rate/ratio, and four discrete examples. Rejections were 25 number-neutral template copies, 50 five-token overlaps, seven semantic automatic rejections, and 14 manual semantic rejections. Distinct verifiers agreed on all candidates; there were zero false labels, verifier failures, disagreements, generator exceptions, unresolved contamination cases, or benchmark leaks. Manual audit found five invalid accepted examples: four bookkeeping renderings combined heterogeneous object counts without establishing a shared inventory unit, and one discrete capacity rendering contained grammar and tied-constraint difficulty defects.
- **Alternatives considered:** lower the duplicate or semantic thresholds after observing yield; count review-band cases as accepted; generate replacements; proceed because labels were correct; introduce a local paraphraser; or start full pilot generation/training despite the failed gates.
- **Rationale:** Correct arithmetic labels are necessary but not sufficient training-data quality. Lowering frozen thresholds or replacing rejected attempts would redefine the test after seeing results. The 20% acceptance rate, fewer than 15 accepted examples in two families, and five invalid acceptances independently fail predeclared gates. The simplest next design remains procedural: correct entity/unit rendering, grammar, and constraint difficulty, then add hand-authored template diversity without changing contamination policy.
- **Expected consequences:** The rate/ratio family and dual-verifier architecture are promising, while bookkeeping and discrete rendering require revision. The selected MiniLM artifact and 0.75/0.82 semantic thresholds remain unchanged. No full dataset, QLoRA, SFT, GRPO, or sealed-final work begins.
- **Next user decision:** Decide whether to approve one bounded blocker-resolution smoke. It should fix the four documented generator weaknesses, use a new generator version/master seed, process exactly 120 fresh attempts under the same distribution and gates, and stop again before full dataset generation or training.

## D-016 — Keep full pilot generation blocked after the rendering/diversity repair still fails

- **Date:** 2026-07-18
- **Status:** accepted Milestone 4.1 stop decision; full generation and training remain unapproved
- **Decision:** Preserve the typed generator implementation and fresh smoke as negative engineering evidence, but do not generate the 4,000 + 4,000 pilot datasets. Do not lower the five-token or semantic thresholds, reinterpret review-band cases, replace rejected candidates, or run another seed automatically.
- **Evidence:** Exactly 120 fresh attempts yielded 86 accepted and 34 rejected. Accepted counts were 30/53 bookkeeping, 29/34 rate/ratio, and 27/33 discrete. All 120 number-neutral renderings and structural hashes were distinct; rejections were 19 semantic automatic rejections, nine manually confirmed generated-to-generated near matches, and six five-token overlaps. Verifiers agreed on every candidate, and there were zero false labels, verifier failures, generator exceptions, unresolved contamination cases, or incorrect rejections. Exact deterministic replay matched decision SHA-256 `84bd6c622b30034a5932a4098c166b8710e39bbf4756e74b1c7c51cf54ce84a3` and aggregate SHA-256 `0e2e20a3516beacb651dfafea96be9b3e95760fbede8804ae6bea76eb6657ed6`.
- **Manual-audit finding:** Eleven accepted renderings were still invalid: residual attributive-plural/grouping grammar, a repeated weighted-average group, a weighted-average conclusion inconsistent with the computed mean, an omitted rate denominator, an elided discrete object noun, an awkward plural capacity phrase, and an irregular container plural. The automatic quality layer caught the five sanitized historical fixtures but not these fresh defects.
- **Rationale:** Yield improved from 20% to 71.67% and every family exceeded 15 accepted examples, but the unchanged gate requires at least 90/120 accepted and zero invalid acceptances/systematic defects. Committing the tested negative result without post-hoc repairs preserves the scientific boundary between implementation and validation.
- **Alternatives considered:** count the nine review-band cases as passes; lower lexical/semantic thresholds; treat awkward renderings as valid because their arithmetic is correct; patch the 11 cases and rerun automatically; generate replacements; proceed directly to full generation; or introduce an unapproved paraphrasing model.
- **Expected consequences:** The type system, renderer inventory, dual verification, contamination pipeline, and fresh aggregate remain reusable evidence, but the current generator version is not production-ready. No complete synthetic dataset or training artifact is created.
- **Next user decision:** Decide whether to stop this procedural lineage or explicitly scope another response to the audited renderer-quality defects. Any new smoke, paraphraser, full dataset, QLoRA, SFT, GRPO, evaluator change, or sealed-final work requires separate approval.

## D-017 — Stop the pure procedural-renderer lineage after its typed compiler fails the stress gate

- **Date:** 2026-07-18
- **Status:** accepted Milestone 4.2 stop decision; full generation and training remain unapproved
- **Decision:** Preserve the typed semantic IR, exact procedural programs, independent verifiers, morphology lexicon, target contracts, coverage reports, and frozen contamination controls, but do not run the fresh 120-candidate smoke or pursue another procedural-renderer patch milestone. The next architecture to discuss is constrained local-model surface realization followed by exact round-trip semantic validation and the existing contamination pipeline.
- **Rationale:** The compiler eliminated all eleven prior defect classes and passed typed validation on 900/900 renders, but the predeclared stress gate also required natural language and plausible scale diversity. Manual audit found 13 unnatural surfaces in 60, 99 number-neutral collisions remained, and 899/900 nearest generated neighbors met the frozen 0.82 semantic-rejection threshold. Distinct hashes and correct labels do not compensate for a systematic surface defect or insufficient semantic diversity.
- **Alternatives considered:** Another set of procedural question-form or template patches is rejected by the explicit final-lineage stop rule. Lowering contamination thresholds, accepting the defects, or generating the full pilot would invalidate the frozen gate. Discarding procedural programs is unnecessary because labels and dual verification remain correct. A frontier/paid generator is outside scope and not required for the proposed local pivot.
- **Expected consequences:** No fresh 120-candidate result, production dataset, adapter, benchmark inference, or training artifact exists. The typed compiler remains useful as semantic evidence and a deterministic fallback, while any local-model realization path requires a separately approved design, model pin, dependency/resource assessment, round-trip validator, and bounded smoke.
- **Next user decision:** Decide whether to approve a design-only architectural-pivot milestone for constrained local-model realization. No model download, realization generation, full dataset, or training may begin from this decision alone.

## D-018 — Freeze a value-blind Qwen3 realization pivot and separate semantic policy roles

- **Date:** 2026-07-18
- **Status:** accepted Milestone 5A design; implementation and inference remain unapproved
- **Decision:** Retain the procedural latent programs, typed IR, exact labels, deterministic traces,
  dual verifiers, and existing generated-to-development contamination policy. Select
  `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` as the primary surface model and
  `Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306` as fallback. The model
  receives only typed roles and opaque placeholders and returns strict JSON with a template,
  placeholder inventory, clause-to-node map, target/intent echoes, and no answer. Deterministic
  validation and filling precede the unchanged exact verifiers and human audit.
- **Semantic-policy decision:** Choose option 2. Keep
  `all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41` and 0.75/0.82 unchanged for
  generated-to-development contamination. Retain exact/template/latent/five-token controls for
  generated peers, but calibrate their semantic-diversity policy separately on original,
  benchmark-independent fixtures and freeze it before viewing new model realizations.
- **Generation decision:** Prefer fixed three-beam deterministic search over one greedy surface or
  seeded sampling. A future smoke has exactly 120 IRs, at most 360 returned sequences, stable beam
  order, no retries or replacements, 256 maximum new tokens, and complete manual audit. Readiness
  still requires at least 90 clean IRs, at least 15 per family, and zero false labels, accepted
  semantic drift, invalid acceptances, verifier disagreements, or unresolved contamination.
- **Alternatives considered:** Qwen2.5 primary minimizes dependency risk but offers less evidence of
  a dedicated non-thinking control; SmolLM3-3B is credible but larger and needs a greater dependency
  move; greedy decoding is simpler but yields only one surface; seeded sampling is more diverse but
  less reproducible; using the same MiniLM thresholds for both roles confuses topical similarity
  with duplication; lowering benchmark thresholds risks contamination.
- **Rationale:** Qwen3 is Apache-2.0, fits 10 GiB in FP16, has an official hard non-thinking switch,
  uses standard Transformers without remote code, and is small enough for a bounded local smoke.
  The slot contract prevents it from controlling values or labels. Independent semantic policy
  calibration responds to the 899/900 internal-collision result without weakening the benchmark
  firewall.
- **Expected consequences:** Milestone 5A creates schemas, validators, a design configuration, tests,
  and documentation only. No model weight, inference result, generated example, dependency change,
  dataset, or training artifact exists. Ordinary Transformers on Windows is sufficient for the
  planned smoke; vLLM and quantization are not recommended.
- **Next user decision:** Decide whether to approve Milestone 5B with a dedicated pinned dependency
  lock, one Qwen3 download, pre-generation internal-policy calibration, and the bounded 120-IR smoke.
  Full 4,000 + 4,000 generation and all training remain separately blocked.

## D-019 — Reject the verbose Qwen3 realization protocol after its bounded smoke

- **Date:** 2026-07-18
- **Status:** accepted Milestone 5B stop decision; full generation and training remain unapproved
- **Decision:** Preserve the pinned Qwen3 environment, exact procedural IRs and labels, dual
  verifiers, value/benchmark firewall, calibrated internal-diversity policy, raw evidence, and exact
  replay result. Do not generate the 4,000 + 4,000 pilot, run the fallback model, increase the beam
  budget, change thresholds, or begin training. Reject `foundry-slot-preserving-json-v1` as a
  production realization protocol because it yielded zero clean IRs under its predeclared gate.
- **Evidence:** Exactly 120 fresh IRs produced exactly 360 deterministic beams. JSON parsed for
  181/360; 160 of the 179 unparsed beams reached the frozen 256-token limit. All 181 parsed beams
  omitted required nodes or declared invalid clause/discourse mappings. Manual review of all 360
  beams through 37 exact-template groups found 301 semantic drifts, 297 unnatural surfaces, 63
  natural-but-drifted outputs, and 59 semantics-preserving-but-unnatural outputs. Every automatic
  rejection was correct. False labels, invalid acceptances, incorrect rejections, verifier
  disagreements, backend failures, and unresolved contamination were zero. Exact replay reproduced
  SHA-256 `a2e6fb565da817ec5e2e6e3c87ba8a54643b2b5ec294dd8f5d24204083d06dcf`.
- **Alternatives considered:** loosen slot/node checks; accept target-only questions; raise the
  256-token limit after observing truncation; simplify or repair outputs post hoc; add more beams;
  retry failed IRs; switch automatically to Qwen2.5; lower contamination gates; proceed because
  mathematical labels remained correct; or abandon the reusable procedural/verifier foundation.
- **Rationale:** Deterministic rejection successfully protected label integrity, but zero usable IRs
  makes the frozen protocol infeasible. Post-hoc repairs or a larger token/beam budget would change
  the tested policy after seeing outputs. The dominant failure is architectural: the model was asked
  to echo a long placeholder inventory and clause map while translating imperative event
  descriptions, causing both output truncation and instruction echo. This is narrower than a failure
  of the procedural mathematics or local model runtime.
- **Expected consequences:** The Qwen3 snapshot and dedicated environment remain reproducible local
  evidence, not authorization for further inference. No accepted synthetic question or dataset was
  produced. A future design may retain the model and replace only the verbose protocol with a compact
  declarative, deterministically recoverable representation, but it must be frozen on original
  fixtures before another separately approved bounded model smoke.
- **Next user decision:** Choose whether to stop the local realization route or approve a design-only
  compact-protocol milestone. No fallback-model run, new Qwen generation, full dataset, QLoRA, SFT,
  GRPO, benchmark inference, or sealed-final access follows automatically.

## D-020 — End Qwen3-1.7B prompt engineering after the compact micro-smoke

- **Date:** 2026-07-18
- **Status:** accepted Milestone 5C stop decision; no 120-IR follow-up or full generation approved
- **Decision:** Freeze `foundry-compact-tagged-v1`, its hashes, exact 30-IR evidence, all-beam audit,
  and replay result. Reject a 120-IR Qwen3 compact smoke, another compact-prompt patch, model-output
  repair, threshold reduction, or full dataset generation. Preserve the procedural IR, exact labels,
  dual verifiers, pinned Qwen3 environment, value/benchmark firewall, and contamination policies.
- **Evidence:** Exactly 30 fresh IRs produced 90 deterministic beams. Tag parsing improved from the
  verbose protocol's 50.28% JSON rate to 90/90; 87/90 beams preserved placeholders, anchors, and
  target tokens. Nevertheless, Qwen placed relation anchors after argument lists and omitted the
  grammatical structure needed to bind quantities, entities, origins, destinations, units, and
  targets. Automatic and manual clean acceptance were both 0/30. All 90 surfaces were unnatural and
  semantically drifted. False labels, verifier disagreements, invalid acceptances, incorrect
  rejections, timeouts, and backend failures were zero. Replay exactly reproduced SHA-256
  `b9b1a7bc8214c2656b6cd45cb089252f63fbe572c52f910e1148a34cd6a4358a`.
- **Alternatives considered:** continue patching Qwen3; run 120 IRs despite zero micro yield; accept
  token echo because every anchor is present; repair or reorder generated tokens deterministically;
  switch immediately to the existing fallback; test a stronger local realization model with the
  same compact protocol; or build an offline model-generated, manually vetted template bank.
- **Rationale:** The compact experiment isolates the remaining failure from JSON verbosity: Qwen3
  can copy tags but cannot reliably compose the supplied opaque predicates into natural English.
  More prompt patches would violate the final stop rule and overfit to observed failures. Of the two
  permitted pivots, a stronger-model test is the narrower scientific comparison because it keeps
  the protocol, IRs, labels, validators, and gates fixed while changing only model capacity.
- **Expected consequences:** No Milestone 5C output is training data. No 120-IR Qwen3 run, complete
  synthetic dataset, training, fallback inference, or sealed evaluation follows. Any stronger model
  requires a separately approved design recording exact revision, license, dependency impact, RTX
  3080 feasibility, fixed budget, and unchanged compact-protocol hashes before download or inference.
- **Next user decision:** Decide whether to approve a bounded stronger-local-model design and smoke
  using the frozen compact tagged protocol. Otherwise stop the local surface-realization route.

## D-021 — End live local-model realization and pivot to a manually vetted template bank

- **Date:** 2026-07-18
- **Status:** accepted by the Milestone 5D stop rule
- **Decision:** Reject Qwen3-4B-Instruct-2507 as a production surface realizer after the controlled
  30-IR comparison yielded 0 clean IRs. Do not test another local model or revise the compact prompt.
  Retain exact procedural semantic IR, labels, and dual verifiers, but recommend an offline,
  manually vetted natural-language template bank with deterministic slot filling plus the existing
  contamination and diversity screens.
- **Evidence:** The stronger model used the exact 30 M5C IRs and unchanged protocol. It returned 90
  beams, with 71 tag parses, 47 placeholder-preserving outputs, 50 anchor-preserving outputs, 47
  target-preserving outputs, and no automatically selected beam. All 90 audited outputs were
  unnatural and semantically drifted. Exact replay passed; false labels, invalid acceptances,
  verifier disagreements, and unresolved contamination cases were zero.
- **Rationale:** Changing only model capacity did not resolve the systematic realization defect.
  Continuing model or prompt experiments would violate the final stop rule and increase adaptive
  overfitting risk. A reviewed finite language asset separates linguistic quality from label
  generation while retaining the trustworthy mathematical and safety stack.
- **Expected consequences:** No M5D output becomes training data. A 120-IR Qwen3-4B run, another
  model substitution, full pilot generation, or training remains blocked.
- **Next user decision:** Decide whether to approve a design-and-bounded-smoke milestone for the
  manually vetted offline template-bank architecture.

## D-022 — Implement the offline bank but reject Milestone 6A technical readiness

- **Date:** 2026-07-19
- **Status:** implemented; technical gate failed; human review pending
- **Decision:** Permanently retain the closure of live-model realization. Preserve the 58-frame,
  232-plan offline bank, typed compatibility contracts, exact mathematics, dual verifiers, and
  unchanged contamination policies as experimental evidence, but do not approve full generation.
- **Evidence:** The single 120-attempt smoke automatically accepted 118, rejected one latent-program
  copy and one number-neutral copy, produced zero false labels or verifier disagreements, and replayed
  exactly at `bf87e7af...5487`. Codex inspection of all 120 surfaces—explicitly not a human audit—found
  13 invalid or unnatural questions from repeated frame nouns, invalid ordinals, malformed compound
  nouns, and raw frame-label realization. The systematic-template-defect gate therefore failed.
- **Rationale:** High automatic yield is insufficient when deterministic metadata does not recognize
  visibly bad English. Patching the 13 observed strings after the smoke would overfit this bounded
  result and would violate the one-smoke evidence boundary.
- **Expected consequences:** The ignored packet remains available for user review. No 4,000 + 4,000
  generation, training, benchmark inference, live realizer, or sealed-final access follows.
- **Next user decision:** Review the local packet, then decide whether to approve one architecture-
  level bank-composition blocker resolution or stop the realization program.

## D-023 — Admit the composition compiler to genuine user review

- **Date:** 2026-07-19
- **Status:** technically ready; human review pending
- **Decision:** Freeze the Milestone 6B lexical boundary, ordinal renderer, one-head noun-phrase
  composition, token provenance, full-bank static expansion, and fresh 120-attempt result. Treat the
  bank as technically ready for user review, but not as human-vetted and not as authorized for full
  generation.
- **Evidence:** Every one of 232 plans passed ten deterministic fixtures (2,320/2,320), all 13 prior
  defects are blocked, and the stratified 90-render Codex sample had no finding. The fresh smoke
  accepted 116/120; four duplicate candidates were safely rejected. Dual verifiers agreed on all
  120, false labels and deterministic language defects were zero, contamination remained clear,
  and replay exactly reproduced `f5caa7e8...a254`.
- **Rationale:** Internal metadata can no longer become prose; noun, ordinal, target, and node
  composition now have typed, auditable sources. These checks establish technical consistency, not
  human naturalness. A real reviewer must decide whether each surface reads naturally enough for
  future training data.
- **Alternatives considered:** treat Codex inspection as human review; generate the pilot from the
  automatic result; lower duplicate or contamination gates; run another smoke; reopen live-model
  realization; or stop the synthesis path.
- **Expected consequences:** The ignored HTML/Markdown packets and any exported review JSON remain
  local. No full dataset, benchmark evaluation, model inference, QLoRA, SFT, GRPO, or sealed-final
  access follows automatically.
- **Next user decision:** Open `results/raw/template_bank_smoke_v2/human_review.html`, mark all 120
  items Approve/Reject/Unsure, export the JSON, and explicitly approve or reject the bank. Any Reject
  or Unsure item requires review before a full-generation proposal.

## D-024 — Preserve the language repair but fail the v3 packet gate on runtime diversity

- **Date:** 2026-07-19
- **Status:** implemented; technical gate failed; no second-review packet created
- **Decision:** Accept the genuine 60-Approve/60-Reject review as immutable evidence, retain the
  review-derived quarantine manifest and v3 worksheet-language replacements, but do not admit the
  v3 smoke to user review because only 104/120 candidates passed the fixed 110 gate.
- **Evidence:** Review SHA-256 is `564a8ca...791` with 120 matching unique IDs. Static expansion is
  2,320/2,320 with zero rule failures, zero exact or number-neutral duplicate sentence plans, and a
  clean 90-render Codex sample. The fresh smoke rejected 15 number-neutral rendered-template copies
  and one latent-program copy; all 120 dual verifications agreed, false labels and deterministic
  language defects were zero, contamination remained clear, and replay exactly matched
  `44cd5265...1e0f`.
- **Rationale:** Human-facing language quality and runtime candidate diversity are separate gates.
  The review repair succeeded statically, but issuing a packet below the precommitted automatic
  threshold would silently weaken the experiment. A different seed, replacements, or post-result
  threshold changes are not permitted.
- **Expected consequences:** Failed-run HTML, Markdown, Codex-audit, and assisted-review packets do
  not exist. Ignored attempts/replay remain local; only content-free evidence is published. No full
  generation, training, benchmark evaluation, or sealed-final access follows.
- **Next user decision:** Decide whether to authorize a narrowly bounded template-selection and
  internal-diversity blocker resolution followed by one fresh precommitted 120-attempt smoke.

## D-025 — Stop runtime allocation because the frozen bank lacks pilot-scale unique capacity

- **Date:** 2026-07-19
- **Status:** capacity gate failed; allocator and smoke not implemented
- **Decision:** Preserve all Milestone 6C-R language repairs and unchanged duplicate/contamination
  rules, but do not implement collision-free allocation, a latent schedule, a future-candidate
  schedule, another 120-attempt smoke, or a review packet. The required preflight proves the finite
  bank cannot supply the fixed 10,003-attempt pilot budget without repeated active signatures or
  number-neutral surfaces.
- **Evidence:** All 16 prior rejections have deterministic partners: 15 reused the same sentence-plan
  identity and normalized surface, and one repeated a latent discrete program. Future quotas require
  4,418 bookkeeping, 2,834 rate/ratio, and 2,751 discrete attempts. Active plan-level capacities are
  72, 80, and 80; domain-aware capacities are 1,728, 400, and 1,600; number-neutral capacities are
  768, 88, and 320. Every category, difficulty, and output-contract stratum fails. Audit SHA-256 is
  `8b921822bf10da964cf357cf3851084a2e0bd15ffc5dc549a85e04f84c9ccd7b`.
- **Rationale:** A collision-free allocator cannot create identities that do not exist. Reordering
  seeds could improve one 120-item sample but would conceal the scale limit and cannot satisfy the
  cross-dataset uniqueness contract. Weakening duplicate rules would change the experiment and is
  not authorized.
- **Alternatives considered:** implement an allocator for the small smoke despite the failed full
  capacity gate; reroll seeds; reuse signatures across targeted/generic datasets; count numeric
  substitutions as diversity; or lower duplicate thresholds. Each conflicts with the fixed gate or
  the scientific comparison.
- **Expected consequences:** The tracked result is a content-free capacity audit only. No fresh raw
  candidates, schedule, packet, dataset, model inference, training, or sealed-final access occurs.
- **Next user decision:** Decide whether to authorize independent bank-capacity expansion. At minimum,
  the existing number-neutral pools need 3,650 additional bookkeeping, 2,746 rate/ratio, and 2,431
  discrete identities, accompanied by enough independently reviewed plans/domains to satisfy every
  group, difficulty, and output-contract stratum before allocation is reconsidered.

## D-026 — Permit bounded reviewed-template reuse; stop on latent-program capacity

- **Date:** 2026-07-19
- **Status:** policy corrected; revised capacity gate failed; no allocator or smoke
- **Decision:** Separate exact-question, latent-program, structural-problem, and surface-template
  identities. Keep exact questions and complete latent programs globally unique across datasets and
  splits. Permit structural skills and reviewed sentence plans/number-neutral signatures to repeat
  only within quota-derived caps. Preserve development-contamination screening exactly. Stop before
  allocation because the unchanged rate and discrete generators cannot supply the complete balanced
  125% attempt pools.
- **Evidence:** `bounded-balanced-template-reuse-v1` matches 14/14 original fixtures (policy SHA-256
  `66443bc8...25f0`); the legacy one-use and permissive alternatives match 12/14 and 11/14. Surface
  identity capacity now passes. Constructive dual-verified capacity is 5,524/4,418 bookkeeping,
  1,632/2,834 rate/ratio, and 2,073/2,751 discrete. The finite rate-total, ratio, percentage, combined-
  rate, equal-distribution, and dual-capacity modes are the limiting compatibility constraints.
- **Rationale:** Reusing controlled language is scientifically desirable because targeted and generic
  groups should differ in curriculum rather than style. But template reuse cannot justify duplicate
  mathematical examples or conceal insufficient program diversity. The remaining blocker is a
  generator parameter/program-space limit, not a need for thousands of new sentence plans.
- **Alternatives considered:** retain one-use number-neutral rejection; allow uncapped reuse; adjust
  caps after results; overfill weighted-average or two-type modes beyond frame-balance caps; allocate
  a small smoke despite failed full capacity; or count unconstructed numeric substitutions as
  capacity. All either fail fixtures or violate the predeclared gate.
- **Expected consequences:** No candidate allocator, full schedule, 120-question smoke, replay, v3
  review packet, dataset, benchmark inference, or training is produced. The selected policy and
  negative capacity evidence are published as content-free contracts.
- **Next user decision:** Decide whether to authorize a narrowly scoped expansion of independent rate
  and discrete latent-program ranges/modes, retaining exact arithmetic and both verifiers, then rerun
  the capacity gate before allocator implementation.

## D-027 — Stop signal-first allocation on target-type/frame compatibility

- **Date:** 2026-07-19
- **Status:** reduced-pilot quotas frozen; corrected capacity gate failed; no allocator or smoke
- **Decision:** Freeze the requested 1,000-targeted plus 1,000-generic experiment and its 2,504
  attempts, but stop before allocation. Enforce the existing per-target-type and semantic-frame caps
  jointly with finite modes and targeted/generic disjointness rather than treating each identity
  layer as an independent aggregate.
- **Evidence:** Bookkeeping supports 1,384/1,106 attempts. Rate target types and exact modes jointly
  support 695/709 (shortfall 14). Discrete target types, per-mode frame balance, and the 90-example
  dual-capacity domain jointly support 598/689 (shortfall 91); generic discrete alone supports
  399/417. Internal capacity-audit SHA-256 is `522b5b4e...7aaf`.
- **Rationale:** The language bank and total latent inventory are sufficient for this smaller pilot,
  but a target type is compatible with only particular mathematical modes. Multiplying identity
  count by a uniform cap overstates capacity when two modes share one target or a mode has fewer
  compatible frames. A schedule must not rely on nonexistent compatibility edges.
- **Alternatives considered:** ignore target-type caps; count aggregate target identities without
  compatibility; change family quotas; expand generators; or implement an allocator and discover
  failure later. Each is outside the frozen policy/quota scope or violates the pre-allocation gate.
- **Expected consequences:** No allocator, full schedule, smoke, replay, packet, dataset, inference,
  training, or sealed-final access occurs. The preliminary pass remains visible in the append-only
  DEVLOG but is superseded by exact compatibility evidence.
- **Next user decision:** Decide whether to authorize a separate policy-design milestone that derives
  target-type and semantic-frame caps from the predeclared curriculum compatibility graph while
  preserving plan/number-neutral caps, exact/latent uniqueness, and every benchmark-contamination
  control; otherwise redefine or abandon the current 1,000 + 1,000 quotas.

## Decision: Submode-local surface caps are selected but insufficient at a frozen difficulty stratum

- **Date:** 2026-07-19
- **Status:** policy selected; full capacity gate failed; smoke and packet prohibited
- **Decision:** Select `submode-local-balanced-surface-reuse-v1` from three predeclared fixture-tested
  policies. Derive dataset caps as `ceil(1.25 * submode quota / active runtime identities)` and the
  global cap as the sum of targeted and generic caps. Preserve every exact, latent, plan, scenario,
  normalizer, verifier, and benchmark-contamination control.
- **Evidence:** The selected policy matches 10/10 original fixtures. Aggregate quotas pass for all
  eleven modes. Weighted-average's easy/medium group has only four compatible runtime identities:
  targeted requires 47 under cap 11 (capacity 44), generic requires 66 under cap 16 (capacity 64),
  and combined requires 113 under cap 27 (capacity 108). Corrected audit SHA-256 is
  `7c0f2913...10f5`.
- **Rationale:** Cap granularity must follow the finite submode being taught, but a submode total is
  still not enough when surface identities have narrower difficulty compatibility. The schedule gate
  must evaluate those edges before construction.
- **Alternatives rejected:** Keep family-level caps (underallocates scarce modes); remove surface
  caps (uncontrolled concentration); change the normalizer, difficulty allocation, or wording after
  observing the failure (outside approval).
- **Consequence:** No schedule, smoke, replay, review packet, dataset, or training run occurs.
- **Next user decision:** Authorize one narrow policy/allocation decision that resolves the five-slot
  weighted-average easy/medium compatibility shortfall, or stop the signal-first pilot.

## 2026-07-19 - Correct difficulty minimally; stop on the next exact schedule blocker

- **Decision:** Select `minimal-compatible-difficulty-reallocation-v1`. Move weighted-average
  targeted easy `2` and medium `1` to hard, and generic easy `1` and medium `1` to hard. Compensate
  targeted through ratio-scale hard-to-easy `1`, ratio-scale hard-to-medium `1`, and combined-rate
  hard-to-easy `1`; compensate generic through ratio-scale hard-to-easy `1` and combined-rate
  hard-to-medium `1`.
- **Why:** This is the deterministic minimum that resolves the measured five-slot shortage while
  preserving every row and column margin and every frozen scientific control.
- **Measured consequence:** Weighted-average compatibility passes exactly. Complete scheduling still
  fails in generic complete-packages: exact joint matching cannot place 121 attempts under caps of
  seven per plan, one per plan-plus-scenario, 25 per frame, and two per runtime identity.
- **Alternatives rejected:** More heuristic ordering, cap relaxation, new templates, or generator
  changes. The milestone stop rule applies.
- **Next user decision:** Reduce the signal-pilot attempt pool or stop. Adding weighted-average plans
  would not address the newly measured discrete complete-packages constraint.

## 2026-07-19 - Reject all three reduced fixed-attempt pools after exact preflight

- **Decision:** Evaluate exactly `1.15`, `1.125`, and `1.10` in descending order and select the first
  complete exact schedule. None passed, so select none and invoke the mandatory stop rule.
- **Evidence:** The pools contain 2,302, 2,253, and 2,203 attempts. Final exact preflight fails for
  each at `generic_control/rate_ratio_percentage_or_average/percentage`. Selection configuration
  SHA-256 is `c5840a94...707e`; evidence SHA-256 is `df31ac17...7ad7`.
- **Rationale:** A lower candidate buffer cannot be called feasible from independent capacity
  products. It must admit a complete schedule using actual rendered runtime identities and every
  frozen cap. All three approved candidates fail that standard.
- **Alternatives rejected:** Invent another multiplier, alter a cap, change wording, expand a mode,
  or select from smoke yield. Each is prohibited or would make the decision result-dependent.
- **Consequence:** No selected config, complete schedule, smoke, replay, review packet, dataset, or
  training run is produced.
- **Next user decision:** Decide whether to reduce the accepted 1,000 + 1,000 signal-pilot objective.

## 2026-07-19: end the current synthetic-data architecture after reduced-size exhaustion

- **Status:** selected; verified stopped result
- **Context:** The accepted size was not part of the hypothesis, so Milestone 7G predeclared the
  descending matched sizes 900, 800, 700, 600, and 500 with one fixed 1.10 multiplier.
- **Decision:** Use `largest-feasible-matched-signal-pilot-v1`, stable largest-remainder family
  quotas, and the actual runtime-rendered exact scheduler. Select only the first feasible size.
- **Measured result:** Every size failed exact surface assignment: 900 at generic ratio-scale; 800
  at generic rate-total; 700 at generic two-type allocation; 600 at generic rate-total; and 500 at
  generic ratio-scale. All mathematical latent pools passed.
- **Rationale:** A lower headline size cannot be selected without a complete schedule. Exhausting
  the approved list under unchanged controls is stronger evidence than another aggregate capacity
  estimate.
- **Alternatives rejected:** Test below 500, change the 1.10 multiplier, loosen reuse/identity
  rules, add templates, change mathematical ranges, or infer feasibility from latent capacity.
- **Consequence:** No pilot size, schedule, smoke, packet, dataset, or training run is authorized.
  The next decision is whether to end the synthetic-data line or choose a new research direction.

## 2026-07-20: fast-track matched-template signal experiment

- **Status:** selected and fixture-calibrated before generation
- **Context:** Repeated exact scheduling searches showed sufficient unique mathematics but failed
  because self-imposed surface identities treated reviewed worksheet skeletons as one-use capacity.
- **Decision:** Permanently close capacity-search milestones and use `matched-template-signal-v1`.
  Exact rendered questions, normalized exact questions, complete latent programs, example IDs, and
  targeted/generic full examples remain globally unique. Reviewed sentence plans, semantic frames,
  number-neutral worksheet structures, reasoning structures, scenarios, and target types may repeat
  under deterministic balancing and frozen concentration caps.
- **Rationale:** The research question compares targeted versus generic curricula under one language
  system; it does not require every training question to have a unique sentence skeleton. The
  immediate scientific priority is measuring that controlled signal.
- **Evidence:** Eight original fixtures pass 8/8. Fixture SHA-256 is `63b9eec6...2bbe`; policy
  SHA-256 is `7e56acfa...3518`; configuration SHA-256 is `563a52c4...582a`; calibration SHA-256 is
  `2e1e7915...aaa1`.
- **Consequence:** No further pilot-size, multiplier, template-cap, surface-cap, number-neutral-cap,
  difficulty, capacity-preflight, template-expansion, or realization-model experiment is authorized.

## 2026-07-20: freeze the matched 500-by-2 signal datasets

- **Status:** dataset-generation and automatic-quality gates passed; human sample review pending
- **Decision:** Freeze the 1,000 accepted examples produced from schedule SHA-256
  `a70cb62c...5eb`: targeted family counts `275/117/108`, generic counts `167/167/166`, with
  `450/50` deterministic training/validation splits and exactly 100 output-contract examples in
  each dataset. Do not remove isolated examples after freeze.
- **Evidence:** Exactly 1,100 attempts yielded 1,000 acceptances and 100 quota-filled reserves.
  There were zero false labels, verifier disagreements, language defects, target mismatches, exact
  or latent duplicates, cross-dataset/split overlaps, or development-contamination findings.
  Deterministic decision SHA-256 is `4574c969...ea93` on both executions.
- **Language review:** The blind Codex advisory audit covered all 1,000 rendered questions and found
  no defect spanning three sentence plans. The ignored 100-question stratified human packet is
  pending and does not become official without explicit user decisions.
- **Consequence:** The frozen datasets may enter only the approved 32-step QLoRA compatibility
  gate after this stage is committed and pushed. Results remain provisional pending human review.

## 2026-07-20: freeze and admit the one-seed QLoRA recipe

- **Status:** selected; 32-step compatibility gate passed
- **Decision:** Use `foundry-qwen2.5-1.5b-signal-qlora-v1` at recipe SHA-256
  `4a9c6043...0590` for both final adapters. Preserve the exact base revision, dependency lock,
  SFT format, NF4/double-quant settings, LoRA modules, 200 optimizer steps, effective batch eight,
  evaluation cadence, seed, and final-adapter-only rule.
- **Evidence:** Native Windows NF4 and paged-AdamW probes passed. The counted smoke completed 32
  steps, finite loss, step-25 validation, save, offline reload, and deterministic inference using
  3,741,319,168 bytes peak reserved VRAM. Only LoRA parameters were trainable.
- **Rationale:** The compatibility result proves the approved stack can execute the exact recipe on
  the RTX 3080 without WSL2, CPU offload, ordinary LoRA substitution, quantization changes, or cloud
  compute. It does not establish downstream quality.
- **Consequence:** After the setup commit is verified and pushed, train generic first and targeted
  second with the same recipe. Do not tune from training loss or expose development data during
  training.

## 2026-07-20: stop the one-seed comparison at training-token parity

- **Status:** mandatory gate failure before benchmark evaluation
- **Decision:** Do not evaluate either trained adapter. The generic run processed 271,396
  non-padding loss tokens and the targeted run 306,766; the 11.5299% relative difference exceeds
  the frozen 2% maximum even though padded input tensors, steps, examples, and every recipe field
  match.
- **Evidence:** Both ignored adapters hash exactly to their summaries, load offline on CUDA, and
  offload zero parameters. Parity evidence SHA-256 is recorded with the final commit.
- **Rationale:** Padding tokens are masked from loss, so equal 512-token tensor shapes do not equalize
  learning exposure. Evaluating now would confound curriculum targeting with token volume.
- **Consequence:** The frozen 814-example evaluator remains unused for both adapters; no one-seed
  result or category effect exists. Do not tune or retrain without a separately approved design that
  freezes equal loss-bearing token budgets before optimizer execution.

## 2026-07-20: select whole-example token-budgeted QLoRA v2

- **Status:** selected; four-step parity smoke passed; full retraining pending protocol publication
- **Decision:** Preserve the parity-failed adapters as negative controls and use
  `foundry-token-matched-qlora-v2` at SHA-256 `df7c7b8d...fa54`. Method A is rejected because its
  exact minimum gap is 9.4343%. Method B schedules 271,292 generic and 271,150 targeted
  loss-bearing tokens across 200 updates, scales every microexample mean loss by its fraction of
  the step's loss-bearing tokens, and keeps whole-example boundaries.
- **Evidence:** The complete 900-record census replays exactly and has zero truncation/masked-only
  records. Schedule hashes are `38c030d7...d7f0` and `76f43825...cc44`; scheduled parity is
  0.05234%. Numerical fixture gradients match a combined-token reference. Fresh four-step GPU
  smokes processed 5,464 and 5,440 actual tokens (0.43924%), with finite losses/gradients and
  successful offline reload.
- **Rationale:** Variable whole-example accumulation equalizes the actual supervised token volume
  without editing data, splitting examples, packing boundaries, or changing the optimizer, LoRA,
  learning rate, base checkpoint, seed, or 200-step budget.
- **Consequence:** Publish the verified protocol before retraining. Then train generic first and
  targeted second from the untouched base. The frozen development evaluator remains blocked until
  final actual-token parity is at most 0.5%.

## 2026-07-20: stop after the token-matched one-seed signal failure

- **Status:** final Milestone 8D signal gate failed
- **Decision:** Preserve the fresh v2 adapters and evaluation evidence, but do not tune, retrain,
  run a second seed, or access sealed-final. Generic scored 15/814 and targeted 14/814 after final
  actual-token parity passed at 0.05234%. Targeted is one answer below generic, its extractability
  is 22.11%, and it misses the frozen 529-correct, +4-over-generic, and 91.38%-extractability gates.
- **Evidence:** Both 814-example runs completed with zero backend failures under the same frozen
  evaluator. The paired targeted-minus-generic estimate is -0.12285 percentage points; its
  fixed-seed 10,000-replicate 95% bootstrap interval is [-1.22850, +0.98280] points. On the frozen
  taxonomy's 293 base-failure rows, neither arm fixes a selected-category example.
- **Rationale:** The shared collapse from 92.38% base extractability to roughly 21% after either
  adapter is evidence of a common instruction/output-contract retention failure. It prevents this
  run from measuring the intended curriculum effect and is not a reason to adjust the signal gate.
- **Consequence:** The narrowest next decision is whether to approve a read-only/focused audit of
  SFT label scope, completion format, and adapter behavior before any new training. Stratified
  human language review remains pending. This result is provisional and no later milestone begins
  without explicit approval.
## 2026-07-20: stop after both assistant-only retention recipes fail

- **Status:** mandatory Milestone 8E retention-smoke gate failure
- **Decision:** Preserve `foundry-assistant-only-sft-v3`, its diagnostic evidence, and all four
  temporary adapters as diagnostic artifacts, but select no learning rate and perform no full
  retraining or new development evaluation.
- **Evidence:** All 900 prior rows incorrectly supervised system/user content, while only 20% of
  targets used the evaluator-aligned terminal line; adapter loading itself was correct. V3 blocks
  those defects. At `2e-4`, generic/targeted arithmetic was 25/30 and 22/30, format 10/15 and 13/15,
  and instruction 10/15 and 8/15. At `5e-5`, generic missed instruction at 13/15 and targeted missed
  arithmetic at 25/30. Both fallback arms otherwise passed format, extractability, echo,
  question-generation, backend, finite-loss, and exact token-parity requirements.
- **Rationale:** A correction can be structurally correct without being retention-safe. Selecting a
  recipe that fails even one frozen arm-level threshold would reintroduce a shared training-method
  confound and violate the predeclared stop rule.
- **Consequence:** Stages J through P are blocked. Any next experiment requires explicit approval
  and a new design; it may not silently add a learning rate, change LoRA settings, rerun GSM1K, or
  treat the collapsed Milestone 8D comparison as a curriculum result.

## 2026-07-20: stop the retention-safe ladder at disjoint validation

- **Status:** Fast-Track 8F-H validation gate failed
- **Decision:** Preserve all four predeclared ladder variants as diagnostic-only artifacts. Select
  Variant A step 32 exactly as prescribed by the calibration hierarchy, but do not promote it to a
  full-training protocol because both arms fail the unseen validation instruction threshold.
- **Evidence:** Every ladder pair received 14,400 assistant tokens with exact 8/16/24/32 prefix
  parity. Variant A passed calibration at all four common checkpoints; concise-v4 Variants B-D did
  not. On `foundry-retention-validation-v1`, generic scored 45/45 arithmetic, 20/20 format, 21/25
  instruction, and 90/90 extractable; targeted scored 44/45, 20/20, 21/25, and 89/90. Echo,
  question generation, and backend failures were zero. Gate SHA-256 is `0dc0a92d...19e4`.
- **Rationale:** Calibration-only checkpoint selection is valid only if it transfers to the
  separately frozen validation suite. Selecting another variant or checkpoint after this failure
  would turn validation into a tuning set and violate the predeclared protocol.
- **Consequence:** No protocol commit claiming retention safety, 200-step retraining, final-holdout
  adapter evaluation, GSM1K evaluation, paired signal analysis, second seed, or sealed-final access
  is authorized. The result remains a training-method blocker, not evidence about targeted versus
  generic curriculum quality.

## 2026-07-20: stop powered adjudication at the untouched-base usability gate

- **Status:** Fast-Track 8I–8K Stage D gate failed before adapter exposure
- **Decision:** Preserve the old 25-item failure audit and the three newly frozen original
  artifacts, but do not use the 300-item adjudication suite for adapter noninferiority and do not
  proceed to the holdout, fallback training, or GSM1K.
- **Evidence:** The old instruction transitions are 21 pass/pass/pass, two fail/fail/fail, and two
  pass/fail/fail; all four adapter failures are genuine instruction noncompliance. The new artifacts
  have 720 unique IDs and normalized prompts with zero exact or 12-token overlap against prior
  retention, synthetic, or development prompts. On adjudication, the untouched base scored
  arithmetic `84/100`, format `48/100`, instruction `55/100`, and extractable `268/300`, below every
  fixed usability threshold. Backend failures, prompt echo, question generation, ambiguous prompts,
  reference defects, and scorer defects were zero. Base-gate evidence SHA-256 is
  `fa1fec57e87a03f390eb4944427f10c4cf0a716c2938447c922070a451067d48`.
- **Rationale:** A noninferiority instrument must first be demonstrably usable by the untouched base.
  The failures are direct noncompliance with unambiguous objective contracts, not repairable prompt
  defects. Rewriting valid prompts after observing base behavior would calibrate the gate to the
  model and violate the predeclared stop rule.
- **Consequence:** No adapter has seen either new evaluation suite. Variant A remains
  `diagnostic_only_pending_powered_retention_adjudication`; no shared-anchor adapter, retention
  decision, benchmark comparison, paired interval, or one-seed signal result exists from this fast
  track. Further work requires a separately approved decision.

## 2026-07-20: stop the SFT line after base-conditioned retention failure

- **Status:** Milestone 8L retention matrix completed; GSM1K gate did not open
- **Decision:** Adopt `foundry-base-conditioned-retention-v1` as the approved conditional
  preservation instrument, freeze both base-correct subsets, and mark the existing A/32 adapter
  pair `failed_base_conditioned_retention`. Do not run GSM1K or train another adapter.
- **Evidence:** Adjudication contains 187 base-correct items (`84/48/55`); holdout contains 210
  (`96/60/54`). Generic and targeted each preserve `181/187` adjudication items with Wilson lower
  bound `93.18%`, but format is `43/48` and question generation is one. Generic preserves
  `197/210` holdout items (Wilson `89.70%`) and targeted `200/210` (Wilson `91.46%`); both have
  format `53/60`, while generic has one question-generation output. Backend failures and prompt
  echo are zero throughout. Pair-decision SHA-256 is
  `433c911d89925b2359c1aeb2bca03bc48eac10842034c1ea9cfb846c1fe0a237`.
- **Rationale:** Conditional preservation asks whether adaptation retained demonstrated base
  behavior without rewarding the adapter for tasks the base failed. The four-cell matrix is large
  enough for the predeclared bounds, and its consistent exact-format regressions are genuine gate
  failures. Lowering category thresholds or ignoring question generation after inference would be
  post hoc relaxation.
- **Consequence:** The SFT adaptation line stops. The frozen 521/814 GSM1K base result is not rerun;
  neither adapter is evaluated on GSM1K in this milestone. Human language review remains pending,
  and any next work requires explicit approval for project interpretation or stop.

## 2026-07-21: approve a common runtime LoRA scale of 0.50 by retention

- **Status:** Milestone 8M independent retention gate passed; GSM1K becomes eligible only after the
  retention decision is published
- **Decision:** Keep both Variant A step-32 adapters byte-identical and unmerged; apply the same
  reversible runtime factor `0.50` to every active LoRA module. Approve the pair as
  `retention_approved_common_scaled_short_run_adapters`. Skip scale 0.25 under the frozen
  first-passing selection rule.
- **Evidence:** Scale 0.0/1.0 reproduce base/unscaled outputs exactly. Scale 0.75 passes both
  adjudication cells but fails the holdout zero-question-generation clause (3 generic, 4 targeted).
  Scale 0.50 passes all four selection cells: adjudication generic `182/187`, targeted `183/187`;
  holdout `205/210` for each. The new 318-item base-correct final holdout yields generic `314/318`
  and targeted `315/318`, with both at `127/127` format and zero forbidden output behavior.
  Selection SHA-256 is `d7455a57001acf68f37369503cce9ef4e3fc30755b2f7f3fd8b7c7055a0b986c`;
  final validation SHA-256 is `6f3e7a29dfbb184f5b6b5eb09fd52060c3c2465c5da2343f85d62d05f8589cc7`.
- **Rationale:** A common inference-strength factor tests whether the learned signal can coexist
  with demonstrated base behavior without another training or checkpoint search. Selection used
  only the two existing subsets; the newly frozen holdout was a one-way independent validation.
- **Consequence:** After an atomic verified retention commit is pushed, evaluate generic then
  targeted once on the frozen 814-item development evaluator at scale 0.50. Do not rerun the base,
  tune the factor, train another adapter, access sealed-final, or treat the result as confirmed
  before human review and a separately approved second seed.

## 2026-07-21: stop the common-scaled one-seed line after absolute signal failure

- **Status:** frozen development comparison complete; one-seed signal gate failed.
- **Decision:** Record common-scale `0.50` targeted-versus-generic results as a provisional negative
  adaptation result. Do not tune scale, retrain, run a second seed, access sealed-final, or begin
  another SFT/GRPO method automatically.
- **Evidence:** Generic is `387/814`; targeted is `414/814`; frozen base is `521/814`. Targeted is
  +27 versus generic, with paired 95% interval `[+1.3514, +5.2826]` points, but remains -107 versus
  base and fails the frozen `>=529` absolute floor. All other signal clauses pass. Decision summary
  SHA-256 is `2b4f39b542ebe16a4cdfd4835856b9965de9dc04c2384fffaf12a064d736a0ed`.
- **Rationale:** The positive paired curriculum contrast does not compensate for the large absolute
  regression. Retention suites measured demonstrated capability preservation on their own domains;
  they did not guarantee broad GSM1K retention.
- **Next user decision:** Complete the pending stratified human language review for dataset
  provenance, then decide whether to stop the adaptation project or separately design a materially
  different retention-preserving training architecture. A second seed is not justified by the
  frozen gate failure.

## 2026-07-21: close adapter arithmetic after contrastive retention failure

- **Status:** Milestone 8N exact construction passed; retention-only scale selection failed.
- **Decision:** Preserve the exact rank-32 `targeted_minus_generic_v1` adapter and content-free
  evidence as a negative result, select no runtime scale, and close the current adapter-arithmetic
  route. Do not use the independent final holdout, run GSM1K, tune another scale, try another merge
  algorithm, retrain, or access sealed-final.
- **Evidence:** Generic and targeted dense-update norms are `1.6918784364` and `1.6980775191`,
  their cosine similarity is `0.9399098552`, and the contrastive norm is `0.5876302228`
  (`34.6056%` of targeted). Dense-analysis SHA-256 is
  `36ce1b90beee7499aa33e11dacbe163e107a98bda5f1065e3f7841fbd85fbaa2`. The adapter SHA-256 is
  `84f02df1cbc5ec1015d096164dbfe3833e166a14eda9ffadf62b5d2d2527c961`; all 196 module-level
  comparisons passed with maximum dense error `1.7462298274e-10`, relative error
  `2.9353350496e-7`, and functional maximum logit error `5.6266784668e-5`. Construction summary
  SHA-256 is `07a99bde03339494cc1ce9cf8428d7ecf7ad35aef58b55038389a3888d2c586c`.
- **Retention result:** Scales `1.00`, `0.75`, `0.50`, and `0.25` respectively preserved
  `181`, `182`, `183`, and `184` of 187 adjudication items; all four cells passed. They preserved
  `204`, `207`, `207`, and `208` of 210 anchor items, but emitted `1`, `2`, `1`, and `2`
  question-generation outputs. Thus every anchor cell failed only the frozen zero-question clause.
  No scale passed both subsets; selection SHA-256 is
  `b41d975f342820ac34ca693d599677994e3f272243c114c313605beb020ad49a`.
- **Rationale:** Exact low-rank subtraction can isolate a small differential update algebraically,
  but exact arithmetic is not evidence of behavioral safety. Allowing a forbidden question-like
  response after observing it, or selecting against a later holdout or GSM1K result, would relax the
  predeclared retention firewall post hoc.
- **Consequence:** No independent-final-holdout or GSM1K contrastive result exists. Human language
  review remains pending at the frozen local review page, and the result retains its provisional
  one-seed label. Any continuation requires separate approval for KL/replay-regularized adaptation
  or verifier-reward GRPO; neither begins automatically.

## 2026-07-21: preserve base behavior during adaptation with shared replay

- **Status:** Approved protocol stopped at the independent-instrument gate before training.
- **Decision:** Conventional unregularized SFT, post-training scale search, adapter arithmetic, and
  additional merge/composition methods remain closed. The next and only approved conventional
  adaptation architecture is `foundry-base-replay-kl-v1`: both curriculum arms begin from the
  untouched base, receive identical frozen base-behavior replay evidence, and use one of exactly
  three predeclared interventions (`R20`, `R20-KL`, or `R40`).
- **Evidence:** Common LoRA scaling improved retention but left generic/targeted GSM1K at
  `387/814` and `414/814`, below the untouched base's `521/814`. Exact rank-32
  targeted-minus-generic arithmetic passed numerical equivalence but every approved contrastive
  scale failed the frozen anchor gate on question generation. The targeted curriculum nevertheless
  retained a paired `+27` advantage over generic, with a 95% interval of
  `[+1.3514,+5.2826]` percentage points.
- **Controls:** The replay source is the existing original `shared_retention_anchor_v1`; replay
  targets are untouched-base deterministic outputs on base-correct items, never benchmark or gold
  answers. Synthetic questions, labels, membership, splits, learning rate, LoRA recipe, seed,
  total CE-token budget, retention thresholds, and GSM1K evaluator remain frozen. Method selection
  uses retention only.
- **Stop rule:** If none of the three variants passes both calibration subsets, or the selected pair
  fails the newly frozen independent holdout, stop conventional SFT adaptation and do not inspect
  GSM1K. Any later continuation requires a separately approved verifier-reward GRPO design or
  project stop.

## 2026-07-21: stop base-replay adaptation at the independent-instrument gate

- **Status:** Stopped before schedule construction or adapter training.
- **Decision:** Preserve the valid 83-record base-behavior replay corpus, but do not implement or
  execute `R20`, `R20-KL`, or `R40` under this milestone because the pre-training independent
  holdout failed its frozen untouched-base usability gate.
- **Evidence:** The replay source passed with arithmetic/format/instruction counts `40/20/23` and
  83 total. The independent 450-item holdout passed reference, scorer, uniqueness, and 3,314-prompt
  exact/12-token disjointness checks, but the untouched base scored only `84/27/30` by category and
  `141/450` overall. Gate-summary SHA-256 is
  `e1bdc1cc14f2e126b8fb43f310b009b47bfef32d31795686259d49c8913d3f8a`.
- **Rationale:** A base-conditioned preservation instrument cannot support the planned decision if
  its frozen base-correct population is below the declared coverage floor. Relaxing thresholds or
  rewriting exact-output prompts after the base result would turn the holdout into a tuning set.
- **Consequence:** No base-correct subset was frozen, no schedules or adapters are approved, no
  retention selection or independent adapter validation exists, and GSM1K remains uninspected for
  this architecture. A continuation requires a separate user decision; this failed suite remains
  immutable evidence.

## 2026-07-21: stop verifier-reward GRPO at the compatibility gate

- **Status:** Milestone 10 compatibility gate failed before the first completion.
- **Decision:** Preserve the frozen 141-item base-correct final-retention subset, prompt-only paired
  schedules, verifier reward, official PEFT reference contract, and runtime implementation, but do
  not run G1 or G2 training, retention selection, independent final retention, or GSM1K.
- **Evidence:** The subset contains `84/27/30 = 141` arithmetic/format/instruction items and has
  SHA-256 `f56845076a1a59e5ca1a95466541339b56f026e945f86118caec307a690ee4ec`. Both schedules have
  64 groups, 52 synthetic and 12 shared replay groups, four planned completions per group, and exact
  `6,702/6,702` prompt-token parity. During the first G1 sampled generation, Transformers' frozen
  top-p `0.95` path called CUDA cumulative summation. PyTorch 2.5.1+cu121 rejected that operation
  because strict deterministic algorithms were active and `cumsum_cuda_kernel` has no deterministic
  implementation. Failure-summary SHA-256 is
  `8b57b6284c1e7dccd978379162de9519b7af30addbbfb9eb4d5a95a7f2b439a6`.
- **Rationale:** Treating the exception as a warning, disabling deterministic enforcement around
  sampling, changing decoding, moving sampling to CPU, or upgrading the stack would change a frozen
  reproducibility or runtime decision. None is a mechanical retry under the approved protocol.
- **Consequence:** Zero completions, rewards, reference-KL passes, backward passes, optimizer steps,
  adapters, retention results, or new benchmark results exist. The next action requires explicit
  user approval either to freeze a scientifically defensible reconciliation of deterministic
  execution with stochastic sampling or to stop verifier-GRPO. No setting may be weakened silently.
  The prior benchmark result retains only its provisional one-seed label, and stratified human
  review remains pending at the frozen local review URL.

## 2026-07-21: close verifier-GRPO after exact warning-only replay failure

- **Status:** Final Milestone 10E compatibility decision; verifier-GRPO closed for this project
  version.
- **Decision:** Enforce the predeclared rule that any exact same-process replay difference closes
  the route. Do not reinterpret diagnostic model-output equality as an exact replay pass, and do not
  run fresh-process replay, the two-step smoke, G1/G2, retention, or GSM1K.
- **Evidence:** All three official same-process runs completed the same frozen `3 x 4` generation
  workload (`36` completions total). The diagnostic payloads were equal and only the approved CUDA
  cumsum warning occurred, but the shared compatibility source changed after replay 1 and the
  replay-evidence source changed after replay 2. Exact packet hashes were therefore
  `68ae4849...d8c`, `80ad3251...a7`, and `be3c8aa8...504e`. Warning-contract summary SHA-256 is
  `eff84b9ec92715eeb74a6c74bcad5980dded9c4b5482012fd8e2438857f24598`; failure-summary SHA-256 is
  `8501b7681262ceca002659978c07c688a6f7baa45923ebb3c06e6134adabebe4`.
- **Rationale:** Source identity is part of the exact packet and cannot change during a controlled
  replay. The gate did not authorize an equivalence exception for concurrent source edits, even
  when model/evidence payloads otherwise matched.
- **Consequence:** No optimizer step, adapter, checkpoint, retention result, or new benchmark
  result exists. This is a project stop, not an invitation to rerun after the source settles.

## 2026-07-21: decouple immutable GRPO source, interpreter, and artifact roots

- **Status:** Milestone 10G orchestration correction authorized; scientific contract unchanged.
- **Decision:** Replace the two invalid same-root assumptions with one typed, hashable
  `GrpoRuntimePaths` contract. Bind imports to a detached source worktree, process launch to the
  approved primary-repository CPython executable, all writable state to a disjoint external artifact
  root, and model reads to a frozen read-only cache manifest.
- **Rationale:** A detached immutable worktree intentionally has neither the primary training
  environment nor a writable in-repository output area. Interpreter selection and output safety are
  separate contracts from source identity; inferring either from `source_root` makes the approved
  topology impossible without improving scientific reproducibility.
- **Evidence:** All original `165` focused tests and `17` new path-contract tests pass with hash seed
  `20260720`; the expanded slice is `182`. Canonical traversal, case, symlink/junction, import-origin,
  binary-identity, root-replacement, command-hash, and environment-hash failures are fail-closed.
  Protected scientific and dependency paths have zero diff from commit
  `8f67e46262b7edafc57861aaf185efa345228179`.
- **Consequence:** The orchestration patch may be committed once as
  `fix: decouple GRPO runtime roots`. Only after that commit is pushed may a new detached V2
  experiment freeze the complete runtime contract and source manifest. This decision authorizes no
  sampling, reward, optimizer, schedule, retention, evaluator, dependency, or benchmark change.

## 2026-07-21: stop Milestone 10G after immutable replay validation failure

- **Status:** Official same-process replay failed; downstream gates permanently closed.
- **Decision:** Do not patch or retry the immutable experiment. Publish the failure and skip
  fresh-process replay, both two-step smokes, G1/G2, retention, GSM1K, paired analysis, and the signal
  gate.
- **Evidence:** Commit `b647a3d` and tree `099a9987` froze cleanly. Runtime contract
  `2400654e...d953`, complete source manifest `72cd61b5...2fab`, and model manifest
  `5173393f...4006` validated before execution. The first 12-completion generation replay returned
  internally, but its wrapper's final validation rejected `CUBLAS_WORKSPACE_CONFIG=:16:8` because
  the new path module expected the launch value `:4096:8`. Packet writing occurs after that wrapper,
  so zero raw packet or summary files were written and no equality comparison occurred.
- **Rationale:** The `:4096:8` to `:16:8` transition is already part of the frozen Transformers
  full-determinism contract, not source, model, or scientific drift. The orchestration patch
  incorrectly conflated a required process-entry value with a required lifetime value. Although no
  model-side mismatch was observed, the authorization's no-retry rule applies once official replay
  has started and failed.
- **Consequence:** Optimizer steps, adapters, checkpoints, retention results, and new benchmark
  results remain zero. Failure-summary SHA-256 is
  `0a1c7085a95fef8138c06b17faaa8e0b5c0af195148012ca9a88c7a07a6d1eeb`. The next action is project
  stop unless a future explicit project-level authorization opens a new experiment.

## 2026-07-21: standardize the Milestone 10H GRPO environment before launch

- **Status:** Explicit project-level authorization supersedes the Milestone 10G no-retry decision
  for one new V3 experiment; the historical V1/V2 directories remain sealed.
- **Decision:** Every replay, smoke, training, retention, and evaluation process must begin with the
  exact five environment values written by Transformers 4.51.3, including
  `CUBLAS_WORKSPACE_CONFIG=:16:8`. The stock helper may assign those same values, but no effective
  environment transition is permitted. Child environments use an explicit allowlist and never an
  uncontrolled parent copy.
- **Evidence:** The installed `trainer_utils.py` SHA-256 is
  `33561736fc04ae94729a513845b9bb900637a5eb6a768aabe018494cf631a95e`; the helper source SHA-256 is
  `1893964197a05bfd07d1477815b58e42e883b9e64985f0e795b4562fc9f84834`. The new V3 environment
  contract SHA-256 is `1f80b1415fc189488b50d04fc69bb0c0ab098ab4f66d03efebac2b6b95b738af`.
  All `198` focused GRPO tests and all `709` repository tests pass.
- **Consequence:** Publish exactly one source correction as
  `fix: standardize GRPO deterministic environment`, then create a new detached V3 worktree. This
  decision changes no sampling, reward, reference, KL, optimizer, schedule, LoRA, retention,
  evaluator, dataset, or dependency setting.

## 2026-07-21: stop Milestone 10H after V3 NVML allowlist failure

- **Status:** Official V3 same-process gate failed before model loading; all downstream gates are
  closed.
- **Decision:** Preserve the detached V3 worktree and external runtime evidence. Do not retry the
  replay, add a newly discovered parent field to the allowlist, run either two-step smoke, train
  G1/G2, evaluate retention, or access GSM1K under this authorization.
- **Evidence:** V3 froze commit `2254b22a`, tree `da9939e`, runtime contract `6154aecd...761a`, source
  manifest `f9f48118...c729`, and model manifest `5173393f...4006`. The exact 30-field child
  environment passed Python/import/determinism validation, but `nvidia-smi` exited `255` with
  `Failed to initialize NVML: Unknown Error` during pre-model CUDA validation. The same query
  succeeds under the parent environment with driver `610.47`.
- **Rationale:** The official replay proved that the environment minimization omits an input needed
  by NVML. Discovering and adding that field after the gate would create a new launch contract and
  a new experiment, not complete the authorized immutable V3 run.
- **Consequence:** No model-side replay mismatch was observed, but compatibility did not pass.
  Model generations, replay packets, optimizer steps, adapters, checkpoints, retention results,
  GSM1K predictions, and sealed-final access remain zero. Failure-summary SHA-256 is
  `b5f0e4b21b496b47a9ae5a93a42d9d9c39bb81b5e2fa7b4ddd36c7432464c2bf`.

## 2026-07-21: validate Milestone 10I child GPU execution through PyTorch CUDA

- **Status:** A new explicit project-level authorization opens one V4 experiment; V1-V3 remain immutable.
- **Decision:** `nvidia-smi` is normal-parent monitoring evidence only and has no child-gate authority. The minimized deterministic child must not invoke NVML or pynvml. It must prove the expected RTX 3080 path by allocating fixed FP32 CUDA tensors, running deterministic elementwise arithmetic and matrix multiplication three times, synchronizing, and producing one exact result hash through the frozen PyTorch runtime.
- **Evidence:** The host and child typed contracts hash to `18a87a86...1f68` and `ead57033...20ec`; the fixed probe configuration hashes to `1fc17a20...6775`. All `213` focused and `724` repository tests pass, with zero protected scientific or dependency changes.
- **Consequence:** Publish the orchestration-only correction as `fix: validate GRPO GPU through CUDA runtime`, then create a new detached V4 worktree. A successful child CUDA probe may proceed to unchanged replay and training gates even if an observational parent `nvidia-smi` query fails. Any actual child CUDA or model replay failure still stops the route.

## 2026-07-21: stop Milestone 10I at the first two-step warning-contract failure

- **Status:** Child CUDA, same-process generation replay, and three fresh-process generation replays passed; the first complete two-step smoke failed before optimizer step 1.
- **Decision:** Do not run the duplicate two-step smoke, counted G1/G2, retention, GSM1K, paired analysis, or the one-seed signal gate. Preserve V4 and publish `analysis: stop verifier GRPO after V4 replay failure`.
- **Evidence:** The direct child CUDA result was `f8850fe4...e5af6` three times. All six generation replays matched packet `084515f9...ee2f`. Under the gradient-checkpointed training path, the first generation warning audit raised `generation emitted multiple distinct normalized warning classes`; stderr also records use-cache disabling and a DynamicCache/PyTorch-version message. No two-step packet or metadata was written, progress remained `0/2`, optimizer steps are zero, and no adapter/checkpoint exists.
- **Rationale:** The NVML orchestration defect is fixed, but complete two-step compatibility requires the frozen single-warning-class generation boundary as well as exact generation packets. That gate failed on the actual training path. Changing warning handling, cache behavior, gradient checkpointing, or the whitelist would alter the frozen contract and is not authorized.
- **Consequence:** Compatibility is failed closed without an exact packet mismatch. Source manifest `dda8cf58...a8b8`, environment `0a5bd3bb...e55d`, and model manifest `5173393f...4006` remain unchanged. Failure summary is `164d3e35...c6f91`.

## 2026-07-22: stop Milestone 10J after the immutable V4 warning audit

- **Status:** The primary repository and detached V4 source are clean at their expected commits;
  all six successful replay packets and both summaries reconstruct. No model process was rerun.
- **Decision:** Classify the four recoverable stderr warning classes as one Class B, one unresolved
  Class C, and two Class E warnings. Classify the discarded Python warning set as UNKNOWN. Publish
  `analysis: stop verifier GRPO after training-warning audit`; do not create the phase-warning
  contract or either V5 directory.
- **Evidence:** The four stderr classes each occur once at lines `1`, `2`, `4`, and `5`, with
  normalized hashes `ad97f015...5ce2`, `5f068b26...c2a5`, `594cd40f...42c4`, and
  `e35718a7...e270`. The V4 auditor proves at least two captured Python classes existed but the
  process persisted no raw text, category, source, count, normalized hash, or class ID for that
  set. V4 explicitly records `two_step_failure_warning_class_ids_persisted=false`.
- **Rationale:** The authorization makes an unsupported/ambiguous runtime notice Class E and any
  warning that cannot be directly classified from frozen evidence fatal. Reconstructing the
  missing Python class from likely call paths or rerunning the model would be inference, not an
  immutable audit.
- **Consequence:** Warning-audit SHA-256 is
  `a3e4d1ca40c3fb3f9fe984d3a019ed064a6ba96394a69b009257a248eebf1602`; classification SHA-256 is
  `37f564cf5a73e91a196496c00c31b0822c44a2b4c84e519b5628d5135479ad74`. V5 replay, G1/G2,
  retention, GSM1K, paired analysis, and the signal gate remain unrun.

## 2026-07-22: close Phase 1 with an evidence-bound public research package

- **Status:** The Phase 1 evidence chain is complete at repository commit
  `20409ba41dc99bb1e6300b53d9ad9b3db1431722`. Genuine stratified human language review remains
  pending, and sealed-final GSM1K evaluation has never been accessed.
- **Decision:** Publish one consolidated Phase 1 report, reproducibility guide, architecture and
  milestone indexes, machine-readable summary and consistency tables, deterministic figures, and
  separately scoped Phase 2 research directions. Preserve every historical result and classify
  the conclusion as provisional pending genuine human review.
- **Evidence:** The untouched base scored `521/814`; matched generic and targeted SFT scored
  `387/814` and `414/814`. Targeted exceeded generic by `27` questions (`3.316953` percentage
  points; paired 95% interval `[1.351351, 5.282555]`) but remained `107` questions below base.
  Final retention lower bounds were `0.968109` and `0.972635`. Six verifier-GRPO generation
  replays matched packet `084515f9...ee2f`, but the training-warning audit failed closed before
  any optimizer step.
- **Rationale:** Phase 1 established a reproducible research system and a statistically supported
  targeted-over-generic contrast without establishing an improvement over the base model. A
  single evidence-linked closeout makes that mixed result legible while retaining the negative
  contrastive and GRPO findings as first-class results.
- **Consequence:** Phase 1 ends without a production-training recommendation, GSM transfer claim,
  or sealed-final score. Any new training stack, human-review completion, or sealed evaluation
  requires separately scoped authorization.

## 2026-07-22: open Phase 2 with vetted human-written source wording

- **Status:** Phase 1 is frozen at synchronized release commit
  `f4ee93afa4c2be52ca21aef8ca16dbf5827b4a99`. The untouched base scored `521/814`; targeted
  synthetic SFT exceeded generic synthetic SFT by `27` questions, while both remained below base.
- **Decision:** Replace generated question wording with verified, externally published
  human-written wording. Use ASDiv as the primary source and MathQA train only as the explicitly
  gated fallback. Foundry performs deterministic verification, family classification, selection,
  matching, target construction, and training without paraphrasing corpus questions.
- **Evidence boundary:** GSM1K development content is permitted only for contamination screening
  and the final frozen evaluation after retention. It cannot influence family assignment,
  curriculum selection, matching, targets, checkpoint choice, or hyperparameters. Sealed-final
  content remains prohibited.
- **Consequence:** Proceed through Milestones 12A-12D only while every predeclared gate passes.
  Stop at the first source, verification, contamination, capacity, matching, training, or
  retention failure; do not lower thresholds or introduce another dataset, seed, or recipe.

## 2026-07-22: pin official ASDiv V1.0 as the Phase 2 primary source

- **Decision:** Use only official repository commit `883f90a9a65bf00304ba8f37423910fe743abc47`
  and tree `2c3e8723c68436a2a6697329edfdf7fbd44e52ac`. Treat raw XML SHA-256
  `ef8904068482919ac48c8eeaaf6df344b8a308ba66d048c2d4d87eab82dc4929` as the canonical
  primary-data identity.
- **Evidence:** The pinned README and XML establish `2,305` problems, the required schema, the ACL
  2020 citation, and CC BY-NC 4.0. The repository has no separate license file.
- **Consequence:** Restrict ASDiv to non-commercial research, preserve attribution, and keep all
  raw text ignored. MathQA remains a non-active fallback unless the later evaluated-capacity gate
  proves ASDiv cannot support the 200-per-arm minimum.

## 2026-07-22: admit only ASDiv rows supported by the restricted exact verifier

- **Decision:** Accept only one-equality formulas composed of exact numeric literals, addition,
  subtraction, multiplication, exact division, parentheses, bounded non-negative integer powers,
  unary signs, and explicit postfix percentages. Do not use `eval`, variables, named functions,
  implicit operations, multiple equations, or ambiguous answers.
- **Evidence:** `1,497/2,305` rows pass formula execution, independent answer extraction, exact
  equality, unit compatibility, and replay. `1,452` belong to a supported Foundry family. Whole-run
  replay reproduced all hashes; accepted disagreements, duplicate IDs, and nondeterminism are zero.
- **Consequence:** Carry only those `1,452` rows into contamination screening. Preserve all `808`
  rejection reasons locally; do not broaden parsing to recover additional capacity.

## 2026-07-22: reject every ASDiv candidate at development similarity 0.75 or above

- **Decision:** Apply the pinned local MiniLM encoder to every supported row and reject candidate
  similarity `>=0.75` against the approved 904-question development inventory. Use no manual-review
  band. Also reject exact, 12-token, number-neutral, operation-structure, source, duplicate, and
  Phase 1 synthetic overlaps.
- **Evidence:** `73` semantic candidates were rejected and `1,379` remain. All other overlap counts
  and unresolved candidates are zero. Complete replay reproduced all output hashes.
- **Consequence:** Freeze clean ASDiv hash `8d99a1de...eaac`. The clean rate-family count is `111`,
  three below the combined 200-per-arm quota; evaluate the base pool before activating fallback,
  exactly as predeclared.

## 2026-07-22: activate the pinned MathQA fallback after measured ASDiv failure capacity

- **Decision:** Activate only official `allenai/math_qa` train Parquet revision
  `fafb9f7ee5b9ec4da9499f9c4177a4c91389f2d6`; never access validation, test, rationales, or remote
  code. Select at most `5,000` verified rows before model inference using the frozen stable fields.
- **Evidence:** ASDiv base failures are only `152/22/38` for bookkeeping/rate/discrete and cannot
  satisfy the combined smallest quotas. MathQA verification accepted `15,468/29,837`, selected
  `5,000` at hash `02fb19a8...3d45`, and contamination retained `4,929` at hash
  `93d4d250...f418`. License metadata is Apache-2.0 and upstream provenance is AQuA-RAT.
- **Consequence:** Combine the clean, verified base-failed rows from ASDiv and MathQA for the fixed
  matched-size test. Preserve ASDiv's CC BY-NC restriction and MathQA's Apache-2.0 attribution.

## 2026-07-22: stop Phase 2 at the Stage H matching-quality gate

- **Decision:** Stop after the fixed selector tested `300`, `250`, and `200` examples per arm and
  none passed every non-curriculum matching threshold. Do not revise selection, quotas, gates,
  corpora, or matching fields after observing the result.
- **Evidence:** At size `200`, source composition was exact and every categorical per-level
  difference was at most five points. Formula-depth SMD was `0.1138945925` and operation-count SMD
  was `0.1087652881`, exceeding the fixed `0.10` limit. The size-200 attempt summary is
  `d240b0b5...9874`; the stop result is `1b169ab5...650f`.
- **Consequence:** Stages I-W are closed. No assistant targets, split freeze, optimizer steps,
  adapters, checkpoints, retention evaluation, GSM1K adapter evaluation, bootstrap, signal-gate
  result, commit, or push is authorized by this stopped route. Sealed-final remains untouched.

## 2026-07-23: accept the deterministic Milestone 12E matching-only repair

- **Decision:** Preserve the original Stage H failure, then accept only the lexicographically
  optimal passing legal single-row replacement under the separately authorized unchanged gates.
- **Evidence:** Exhaustive search selected generic `mathqa-train-26455` to
  `mathqa-train-28853`. Formula-depth SMD is `0.0870898715`, operation-count SMD is
  `0.0680561998`, the other SMDs are below `0.01`, and categorical maximum is `0.05`. Matching
  evidence `004d338b...d5b5` and dataset identity `ee18f7f9...dc31` replay byte-identically.
- **Consequence:** Freeze the repaired 200-per-arm datasets and exact 180/20 splits. Proceed only
  to the authorized retention-first V1 training protocol; do not run V2 unless V1 lacks a common
  passing checkpoint, and do not expose either adapter to GSM1K before final retention passes.

## 2026-07-23: stop Milestone 12E before V1 when the required environment lacks QLoRA

- **Decision:** Do not install packages, modify the project `.venv`, or substitute another
  interpreter after the V1 preflight finds PEFT, bitsandbytes, and TRL absent.
- **Evidence:** The required CPython 3.12.10 environment passes `pip check`; CUDA PyTorch
  `2.5.1+cu121` sees the RTX 3080, but the three required training packages are unavailable.
- **Consequence:** Preserve the passed matching and dataset evidence. Record zero model loads,
  optimizer steps, adapters, checkpoints, retention runs, and adapter GSM1K evaluations. Resume
  only after explicit authorization freezes a compatible training environment.

## 2026-07-23: stop Milestone 12F-A at the authorized training-environment gate

- **Decision:** Do not construct schedules, train adapters, or run retention because the authorized
  `.venv-training` interpreter fails `pip check` when the repository source root is visible.
- **Evidence:** All required runtime versions and the RTX 3080 identity match, but installed Foundry
  metadata requires `PyYAML==6.0.2` while the environment contains `PyYAML==6.0.3`.
- **Boundary:** Do not install, remove, upgrade, or downgrade packages under this authorization.
  GSM1K and sealed-final evaluation remain unaccessed.

## 2026-07-23: grant the narrow PyYAML exception and stop at the native CUDA probe

- **Decision:** Grant `foundry-pyyaml-metadata-exception-v1` only for the exact project
  `PyYAML==6.0.2` versus training-environment PyYAML 6.0.3 mismatch. All 31 tracked YAML files,
  real training-loader projections, and repeat audits are identical.
- **Environment result:** Stop the QLoRA gate because the first deterministic CuBLAS forward
  rejected a process launched without `CUBLAS_WORKSPACE_CONFIG` set before Python started.
- **Consequence:** Do not retry under changed launch conditions in this milestone turn. No schedule,
  backward pass, optimizer state/update, adapter, checkpoint, retention evaluation, or GSM1K
  evaluation is authorized after the failed probe.

## 2026-07-23: freeze the CuBLAS prelaunch contract and stop on wrapper path escaping

- **Decision:** Freeze `foundry-vetted-qlora-deterministic-launch-v1` with
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` and the five previously frozen process variables.
- **Result:** Pre-import and post-import launch validation passed, but the single child command
  encoded Windows backslashes as Python string escapes. The replay path therefore contained a
  backspace control character and failed before fixture loading or model loading.
- **Consequence:** Do not patch or retry in this milestone. Preserve zero model loads, generations,
  optimizer steps, schedules, adapters, checkpoints, retention runs, GSM1K runs, and sealed-final
  access.
