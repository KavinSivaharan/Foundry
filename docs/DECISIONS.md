# Foundry Decision Log

This log separates proposals from approved decisions. A proposal does not authorize implementation.

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
