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
