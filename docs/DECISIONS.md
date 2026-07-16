# Foundry Decision Log

This log separates proposals from approved decisions. A proposal does not authorize implementation.

## D-001 — Recommend arithmetic reasoning for Phase 1

- **Date:** 2026-07-16
- **Status:** proposed; awaiting user approval
- **Decision:** Use grade-school arithmetic reasoning with a pinned public GSM1K revision and `Qwen/Qwen2.5-1.5B-Instruct` as the Phase 1 experiment.
- **Alternatives considered:**
  - Python function synthesis with EvalPlus HumanEval+ and `Qwen/Qwen2.5-Coder-1.5B-Instruct`.
  - Structured function calling with a pinned BFCL V4 single-turn subset and `Qwen/Qwen2.5-1.5B-Instruct`.
- **Rationale:** GSM1K has objective integer labels, was created as a contamination-aware counterpart to GSM8K, and supports synthetic examples generated from executable arithmetic programs. A 1.54B model should permit a complete local QLoRA loop on an RTX 3080. This combination tests Foundry's core data/evaluation loop with fewer confounding infrastructure problems than sandboxed code or function-call equivalence.
- **Expected consequences:** Phase 1 will be narrower but easier to audit. It should require no paid API or cloud GPU. If successful, it provides reusable evaluation, synthesis, verification, training, and comparison interfaces for a more agentic Phase 2 task.
- **Reconsider if:** the model has too little measurable headroom; the public dataset cannot be pinned or safely split; a smoke test cannot fit the target GPU; exact-answer parsing is unstable; or the user prioritizes product-facing tool use over minimizing research risk.

## D-002 — Prefer the general Qwen checkpoint over a math-specialized checkpoint

- **Date:** 2026-07-16
- **Status:** proposed; awaiting user approval
- **Decision:** Start from `Qwen/Qwen2.5-1.5B-Instruct`, not a Qwen2.5-Math checkpoint.
- **Alternatives considered:** Qwen2.5-Math-1.5B-Instruct, Qwen3-class small models, and models near 3B parameters.
- **Rationale:** The general 1.5B checkpoint has an Apache-2.0 license, documented 32,768-token context, likely local feasibility, and more likely headroom. A math-specialized baseline could obscure whether Foundry's loop or prior domain post-training caused the result. Newer reasoning-mode models also introduce output-mode and evaluation complexity that is not needed for the first loop.
- **Expected consequences:** The base score may be lower, but a targeted SFT gain should be easier to detect and attribute. The work will demonstrate adaptation rather than incremental polishing of an already-specialized model.
- **Reconsider if:** the baseline is effectively random, cannot follow the answer format, or requires more synthetic data than the local training budget permits.

## D-003 — Put a benchmark firewall between evaluation and synthesis

- **Date:** 2026-07-16
- **Status:** proposed; awaiting user approval
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
