# Foundry Project Plan

Last updated: 2026-07-20

## Current milestone status: retention ladder failed disjoint validation; full retraining blocked

Fast-Track 8F-8H compared four predeclared, token-matched adaptation variants without consulting
GSM1K. Variant A (assistant-only v3 at `5e-5`) was the only variant with common calibration passes,
so the frozen hierarchy selected step 32. On the separately frozen validation suite, however, both
arms scored 21/25 (84%) on instruction following, below the required 90%. The mandatory stop rule
therefore blocks protocol promotion, full retraining, final-holdout adapter evaluation, GSM1K,
paired analysis, and a second seed. The next decision must address the training method without
reusing validation for model or checkpoint selection; the pending human language review remains
separate and unchanged.

## Project objective

Foundry is an autonomous post-training system that discovers a model's weaknesses, creates targeted training data, verifies that data without relying on an LLM judge alone, fine-tunes the model, and measures whether the same model improved under a controlled evaluation.

Phase 1 will prove the smallest credible version of that loop on one local GPU. It will not attempt to be a general training platform, a distributed service, or a production agent framework.

## Precise research question

Can a failure-targeted, automatically verified synthetic SFT dataset produce a reproducible improvement in exact-answer arithmetic reasoning for a pinned revision of `Qwen/Qwen2.5-1.5B-Instruct`, compared with the identical base model on a locked GSM1K holdout under identical prompts and decoding settings, when training is limited to QLoRA on an RTX 3080?

The proposed Phase 1 success bar is:

1. At least a 3 percentage-point absolute accuracy improvement on the locked final holdout.
2. A 95% paired bootstrap confidence interval for the accuracy difference whose lower bound is greater than zero.
3. A positive improvement from at least two approved SFT training seeds, with the selected checkpoint rule fixed before viewing final-holdout results.
4. No benchmark example or answer used for training or synthetic-example generation.
5. Fully recorded model, dataset, package, prompt, decoding, hardware, and seed versions.

This threshold is a proposal. It must be confirmed after a baseline establishes the amount of headroom and the exact public dataset revision is inspected.

## Repository state at discovery

On 2026-07-16 the repository was clean on `main`, tracking `origin/main`, with one commit (`def688b`, `Initial commit`). It contained only `README.md` and a standard Python `.gitignore`. No application code, environment definition, configuration, dataset, result, or experiment had been created.

## Candidate comparison

Scores are engineering judgments from 1 (poor) to 5 (strong), not measured experiment results. For contamination risk, 5 means the lowest risk.

| Criterion | A: Arithmetic / GSM1K / Qwen2.5-1.5B | B: Python / HumanEval+ / Qwen2.5-Coder-1.5B | C: Function calling / BFCL V4 / Qwen2.5-1.5B |
| --- | ---: | ---: | ---: |
| Technical importance | 4 | 5 | 5 |
| Benchmark credibility | 5 | 4 | 5 |
| Potential wow factor | 4 | 5 | 5 |
| Automatically verifiable labels | 5 | 4 | 4 |
| RTX 3080 feasibility | 5 | 5 | 5 |
| Low API/cloud cost | 5 | 4 | 4 |
| Low contamination risk | 4 | 1 | 4 |
| Likelihood of measurable improvement | 4 | 3 | 3 |
| Later SFT and GRPO suitability | 5 | 5 | 5 |
| **Total** | **41** | **36** | **40** |

### Option A — Grade-school arithmetic reasoning

- **Task domain:** multi-step grade-school arithmetic word problems.
- **Benchmark:** [GSM1K](https://proceedings.neurips.cc/paper_files/paper/2024/hash/53384f2090c6a5cac952c598fd67992f-Abstract-Datasets_and_Benchmarks_Track.html), with the public dataset obtained from the [ScaleAI dataset repository](https://huggingface.co/datasets/ScaleAI/gsm1k) at a pinned revision.
- **Base model:** [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), pinned to an exact repository commit before the first run.
- **Verification:** normalize the model's final integer, compare it exactly with the benchmark answer, and verify synthetic examples by independently executing their structured arithmetic program with exact rational arithmetic. Reject examples that fail constraints, disagree across solver paths, or overlap the benchmark.

Why it is credible: GSM1K appeared in the NeurIPS 2024 Datasets and Benchmarks Track and was created to investigate contamination and overfitting on GSM8K. Its integer answers make the primary metric objective. Synthetic examples can be created from executable arithmetic templates rather than benchmark answers.

Main limitation: grade-school math is narrower and less product-like than code or tool use. Public release means future models may eventually contain GSM1K, so the dataset revision and model provenance still matter. Natural-language rendering can introduce ambiguity even when the underlying arithmetic program is correct; the verifier must reject such cases conservatively.

### Option B — Python function synthesis

- **Task domain:** generate short Python functions from docstrings and signatures.
- **Benchmark:** [EvalPlus HumanEval+](https://github.com/evalplus/evalplus), which extends HumanEval with substantially more tests.
- **Base model:** [`Qwen/Qwen2.5-Coder-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct), an Apache-2.0 1.54B-parameter code model with a 32,768-token context window.
- **Verification:** run generated code in an isolated, resource-limited process against public plus generated property tests; accept synthetic tasks only when a reference implementation passes and mutations are caught.

Why it is compelling: code correctness is concrete, demonstrations are visually impressive, and both SFT and outcome-based GRPO fit naturally.

Why it is not first: HumanEval is old, small, and at high contamination risk. Tests are only an approximation of a specification, so weak tests can accept incorrect code. Safe execution, timeouts, dependency control, and anti-cheating checks add infrastructure before the core Foundry loop is proven. A newer LiveCodeBench slice could reduce contamination but would likely be too difficult and noisy for the first 1.5B-model improvement experiment.

### Option C — Structured function calling

- **Task domain:** choose a tool and emit valid arguments from a user request and typed function schemas.
- **Benchmark:** a pinned single-turn, non-live subset of [Berkeley Function Calling Leaderboard V4](https://gorilla.cs.berkeley.edu/leaderboard), whose methodology is described in the [ICML 2025 BFCL paper](https://proceedings.mlr.press/v267/patil25a.html).
- **Base model:** [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
- **Verification:** parse the output, validate it against JSON Schema, compare the function and typed arguments using BFCL's AST rules, and execute only synthetic functions in a deterministic sandbox where outcome checking is available.

Why it is compelling: function calling is central to agents, BFCL is a strong and current benchmark, and deterministic schemas make large-scale synthesis possible. It has the highest product-facing wow factor.

Why it is not first: semantic equivalence between calls is harder than numeric equality, format behavior can dominate the score, and the benchmark is actively versioned. Building schema generation, AST comparison, execution fixtures, and robust abstention cases would combine several research problems at once. It is an excellent Phase 2 domain after the arithmetic loop is reliable.

## Recommendation

**Recommend Option A: GSM1K arithmetic reasoning with `Qwen/Qwen2.5-1.5B-Instruct`.**

It is the strongest first choice because it isolates Foundry's core research claim. The benchmark is credible and contamination-aware, the score is objective, synthetic labels can be derived from executable programs, and the complete QLoRA loop should fit a single RTX 3080 without an API or cloud GPU. It still has enough difficulty and visible outcome value: a small local model improving on previously failed multi-step problems through verified targeted data is a meaningful demonstration.

The general Qwen instruction model is preferred over a math-specialized checkpoint because a specialized model may leave too little headroom and make the result harder to attribute to Foundry. The Qwen model card reports 1.54B parameters, an Apache-2.0 license, a 32,768-token context window, and up to 8,192 generated tokens. Phase 1 would cap training sequences at 512 or 1,024 tokens because GSM1K does not need the full context and activations, not stored weights, are likely to be the RTX 3080 memory constraint.

### Feasibility estimates for the recommended model

These are planning estimates, not measurements:

- **License:** Apache-2.0 according to the model card; suitable for modification and redistribution subject to the license terms.
- **Raw weight memory:** 1.54B parameters require about 3.08 GB at 16 bits or 0.77 GB at 4 bits before quantization metadata and runtime buffers.
- **Expected QLoRA peak:** approximately 6–9 GB with 4-bit NF4 weights, batch size 1, gradient accumulation, gradient checkpointing, and a 512–1,024-token training cap. This must be measured because RTX 3080 cards exist with different VRAM capacities and software kernels vary.
- **Expected training throughput:** roughly 150–500 processed tokens/second. A deliberately small pilot totaling 1–3 million processed tokens is therefore estimated at about 1–6 hours, excluding setup, evaluation, and failed runs. This range is intentionally broad and will be replaced by a smoke-test measurement.
- **Context limitation:** the model supports 32,768 input tokens, but Phase 1 intentionally uses at most 1,024 training tokens. Longer context would consume memory without helping this benchmark.
- **Cloud/API cost:** $0 is the target. Template-based synthesis, verification, evaluation, and training can run locally. Electricity and download bandwidth remain real local costs. Any optional paid paraphrasing or cloud run requires a separate approval.

[Hugging Face TRL documents direct SFT integration with LoRA/QLoRA](https://huggingface.co/docs/trl/peft_integration), and [PEFT documents QLoRA-style adapters over linear layers](https://huggingface.co/docs/peft/main/package_reference/lora). These interfaces will be pinned to tested package versions rather than followed from `latest` during an experiment.

## Current proposed architecture

This is a plan, not implemented code.

1. **Experiment configuration:** immutable YAML records model and dataset revisions, prompt, decoding, split seed, training hyperparameters, and output locations.
2. **Benchmark firewall:** loads the pinned benchmark and creates a stable development/final split. Benchmark answers are visible only to the scorer. The synthetic generator receives aggregate failure categories, never benchmark answers or example text.
3. **Evaluator:** renders one fixed prompt, runs deterministic generation, extracts the final answer, and stores per-example predictions plus summary metrics.
4. **Failure analyzer:** maps incorrect development predictions into a small, predefined taxonomy such as operation chain, rate/ratio, percentage, unit conversion, and distractor handling. It emits counts and anonymized category statistics.
5. **Synthetic generator:** samples new structured arithmetic programs and renders them through independent language templates targeted to the weak categories.
6. **Verification and filtering:** recomputes solutions through a separate exact-arithmetic path, validates constraints, filters malformed examples, deduplicates exact and near-semantic matches, and rejects overlap with all benchmark prompts.
7. **SFT trainer:** fine-tunes a QLoRA adapter only; the base weights remain frozen. Smoke training precedes any full run.
8. **Comparator:** evaluates base and candidate with the same final-holdout configuration, calculates paired differences and uncertainty, and writes auditable result summaries.
9. **Experiment ledger:** connects configuration, code commit, environment, hardware, artifacts, runtime, cost, and interpretation.

The Phase 1 implementation should be a typed Python CLI and local files. It does not need a web UI, database server, distributed queue, multi-agent framework, or cloud orchestration.

## Data and training loop

1. Pin the model, GSM1K dataset, evaluator code, and package revisions.
2. Before model evaluation, create a stable development/final split from benchmark IDs using a recorded seed or stable hash. Exact counts will be recorded after the pinned dataset is inspected.
3. Use the development portion only for failure discovery and iteration. It never enters training.
4. Keep the final portion sealed until the SFT candidate and checkpoint-selection rule are frozen.
5. Evaluate the base model on development data and record every prediction.
6. Categorize failures. Pass only category-level targets to synthesis.
7. Generate new examples from executable arithmetic specifications that were not derived from benchmark examples or labels.
8. Verify labels independently; filter ambiguity, invalid values, duplicates, and exact/semantic benchmark overlap.
9. Split synthetic data by generator template family before augmentation into training and synthetic validation sets. This prevents closely related variants from appearing on both sides.
10. Run an inexpensive QLoRA smoke train, then an approved full SFT run.
11. Freeze the candidate. Evaluate both base and candidate once on the identical locked final split and configuration.
12. Report paired accuracy, category results, confidence intervals, runtime, peak VRAM, and any regressions.
13. Iterate using development failures only. Do not reopen the final holdout for tuning.

## Milestones

### Milestone 0 — Discovery and plan

Status: complete, awaiting approval of the recommendation.

- Inspect repository state.
- Compare three task/model/benchmark/verifier combinations.
- Recommend one combination.
- Create project governance and learning documents.

### Milestone 1 — Reproducible evaluation foundation

Status: complete, including the deferred real CUDA smoke on the RTX 3080 desktop.

**What it would build:** a small typed CLI that loads a validated experiment configuration, creates a stable benchmark development/final manifest, formats prompts, extracts and scores exact integer answers, and writes auditable prediction records. It would include a fake-model integration path and, if the local GPU environment is compatible, a maximum 10-example Qwen smoke evaluation.

**Why it is necessary:** Foundry cannot trust a baseline or later improvement until model revision, dataset revision, split membership, prompt, decoding, parsing, and output schema are reproducible. Parser and split bugs can create fake gains even when the model did not improve.

**Planned files:**

- `pyproject.toml`
- `uv.lock` if `uv` is approved as the environment manager; otherwise an equivalent committed dependency lock
- `src/foundry/__init__.py`
- `src/foundry/cli.py`
- `src/foundry/config.py`
- `src/foundry/evaluation/__init__.py`
- `src/foundry/evaluation/benchmark.py`
- `src/foundry/evaluation/prompting.py`
- `src/foundry/evaluation/scoring.py`
- `src/foundry/evaluation/runner.py`
- `configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml`
- `tests/unit/test_config.py`
- `tests/unit/test_split.py`
- `tests/unit/test_scoring.py`
- `tests/integration/test_evaluation_smoke.py`
- `docs/PROJECT_PLAN.md`, `docs/DECISIONS.md`, `docs/DEVLOG.md`, and `experiments/EXPERIMENTS.md` for the verified milestone record
- `results/` summary files produced by the smoke run; per-example raw output remains under ignored `results/raw/`

The names may change only with an explanation and approval if environment inspection reveals an incompatibility.

**Verification:**

1. Unit tests prove configuration validation rejects unpinned revisions and invalid decoding values.
2. Split tests prove the same seed/revision produces the same disjoint example IDs and that final IDs cannot enter development output.
3. Adversarial answer-parser tests cover commas, signs, extra numbers, malformed answers, and refusal text.
4. A fake-model integration test runs end to end without downloading a real model.
5. Formatting, linting, type checking, and tests pass.
6. If GPU inspection passes, a maximum 10-example real-model smoke run records prompt hash, predictions, score, runtime, throughput, and peak VRAM. Its score is a plumbing check, not a benchmark claim.
7. `git diff --check` and a secret/large-artifact review pass before a local milestone commit. Nothing is pushed.

**Costs, risks, assumptions, and unresolved decisions:** no API or cloud cost is planned. A real-model smoke run would download roughly a few gigabytes and consume local GPU time and electricity. CUDA, `bitsandbytes`, PyTorch, and the GPU driver may be incompatible; the card may have 10 or 12 GB; the host OS is unknown; and `uv` versus another lock mechanism is not approved. The milestone must stop after environment inspection if the real smoke run is unsafe, while unit/fake-model verification can still complete.

**Completion result:**

- Detected macOS 15.6.1 on Apple M2 with 8 GB unified memory, no NVIDIA GPU, no `nvidia-smi`, no CUDA toolkit, Python 3.9.6 as the system Python, Python 3.12.11 available separately, no `uv`, and approximately 14 GiB free disk.
- Used an isolated Python 3.12 `.venv`, exact direct pins in `pyproject.toml`, and pip-compiled development/smoke lock files.
- Pinned the Qwen model to `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Pinned GSM1K to `bc09569d09a614b9b530edc7f076fb214ac10493`.
- Created a deterministic 904-example development manifest and 301-example sealed-final manifest. Both contain identifiers and row indices only.
- Added config validation, prompt hashing/rendering, strict integer scoring, benchmark loading, fake and CUDA model backends, auditable raw/summary output, and sealed-final access controls.
- Passed formatting, linting, strict type checking, 29 unit/integration tests, and deterministic manifest validation.
- Invoked the smoke command only to exercise its preflight. It refused before model or dataset download because the CUDA dependencies/environment were unavailable. Therefore no benchmark score, throughput, or VRAM result exists.
- Later completed the deferred Windows RTX 3080 smoke in commit `c1ef561`: 10 development examples, 2 correct, 7 invalid-format outputs, no generation/OOM failure, 36.900 seconds evaluation time, 0.271 examples/second, and 3.088 GiB peak reserved VRAM.

**Exact deferred RTX smoke command:**

After creating the locked environment on a CUDA-capable RTX machine, run:

```text
HF_HOME=data/huggingface .venv/bin/foundry smoke \
  --config configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml \
  --manifest configs/eval/manifests/gsm1k_development.json \
  --output-dir results/smoke/qwen2_5_1_5b \
  --limit 10
```

On Windows PowerShell, replace `.venv/bin/foundry` with `.venv\Scripts\foundry.exe` and set `$env:HF_HOME = "data/huggingface"` first. Before either command, `nvidia-smi` must confirm the RTX 3080 and PyTorch must report `torch.cuda.is_available() == True`. The pinned PyTorch 2.5.1 CUDA 12.1 wheel is installed from the [official PyTorch wheel index](https://pytorch.org/get-started/previous-versions/).

### Milestone 1.5 — Output-format diagnosis and calibration

Status: complete; the 90% valid-output admission gate was not met, so no prompt was selected and Milestone 2 remains blocked.

- Diagnosed the seven original smoke invalids: all omitted the literal `Final answer:` line and instead used boxed, prose, unit/currency, inline-LaTeX, or bold conclusions. The chat template preserved the instruction, the responses were not truncated, and the strict parser behaved as designed.
- Deterministically reserved 30 identifiers from the 904-example development partition for prompt-format calibration using seed `foundry-gsm1k-prompt-format-calibration-v1`.
- Reserved the remaining 874 identifiers as the disjoint future main-development baseline. The calibration and future-baseline manifests contain only stable IDs and row indices; the sealed-final manifest was not read or modified.
- Evaluated three prompt variants on the identical 30 calibration IDs with the same pinned model/dataset, strict parser, greedy decoding, 512-token limit, and RTX 3080. Exactly 90 generations ran.
- Current prompt: 16.67% valid, 83.33% invalid, 305.97 average output tokens, 124.190 seconds.
- Minimal `format_v1`: 10.00% valid, 90.00% invalid, 287.37 average output tokens, 114.161 seconds.
- Explicit-contract `format_v2`: 43.33% valid, 56.67% invalid, 230.60 average output tokens, 91.467 seconds.
- No prompt reached the predeclared 90% validity threshold. No parser, model, dataset, generation setting, or research direction was changed, and no prompt was frozen.

### Milestone 1.6 — Deterministic answer-extraction calibration

Status: complete; the fresh 90% extractability gate was not met, so no evaluator/prompt configuration is admitted for Milestone 2.

- Preserved the strict `Final answer:` parser as an exact-format compliance metric and added separate deterministic terminal-integer extraction for benchmark accuracy.
- Re-scored the existing 90 outputs without generation. Current prompt: 96.67% extractable, 16.67% exact compliant, 50.00% correct. `format_v1`: 93.33% extractable, 10.00% exact compliant, 50.00% correct. `format_v2`: 90.00% extractable, 43.33% exact compliant, 43.33% correct.
- Manually audited all 63 calibration outputs newly accepted beyond the strict parser and found zero false extractions.
- Selected the current prompt for fresh validation because it led the primary calibration extraction metric; math accuracy did not drive selection.
- Deterministically reserved 30 fresh IDs from the 874-ID pool and left 844 IDs as a disjoint candidate main baseline. Both new manifests are identifier-only and parent-hashed.
- Fresh validation reached 23/30 extractable (76.67%), 3/30 exact compliant (10.00%), and 14/30 correct (46.67%), with zero generation failures and zero audited false extractions. Three outputs hit the 512-token limit, two ended in non-integral decimals, and two clear integer conclusions were conservatively rejected.
- The gate failed even under the post-hoc upper bound of 25/30 (83.33%) that would accept both conservative false rejections. No validation-driven extractor change, second run, prompt freeze, or Milestone 2 work occurred.

### Milestone 1.7 — Final evaluator blocker resolution

Status: complete; the final fresh 90% extractability gate failed at 83.33%, so the evaluator and 814-ID candidate baseline are not admitted for Milestone 2.

- Diagnosed all seven Milestone 1.6 rejections: two explicit terminal non-integral decimals, two clear integers in unsupported terminal wrappers, and three confirmed 512-token truncations.
- Preserved the strict `Final answer:` compliance parser unchanged. Versioned `foundry-terminal-number-v2` (SHA-256 `e099d1c247968fed982cb849022ec3137b1694c15f23a65663a127b8158c06df`) separately accepts explicit terminal decimals and fractions through exact `Fraction`/`Decimal` normalization, plus narrowly constrained currency, percentage, unit, boxed, bold, inline-LaTeX, text, conclusion, and equation wrappers. It rejects truncation, conflicting candidates, malformed values, unfinished expressions, arbitrary intermediate numbers, and ambiguous prose.
- Re-scored the unchanged Milestone 1.6 validation outputs: 27/30 extractable (90.00%), 3/30 exact compliant (10.00%), and 15/30 correct (50.00%). Four newly accepted outputs were audited with zero false extractions; two explicit non-integral answers were valid-but-wrong.
- Confirmed three original 512-token truncations and created a hash-guarded final configuration that changes only `max_new_tokens` to 768 (config SHA-256 `5f315d5de645f9563b8d1e61bc8e02c3513c453238ad9e1d6f9473489b5a622b`). A three-record diagnostic resolved two completions while one still reached 768; diagnostic results did not replace validation evidence.
- Deterministically reserved one untouched 30-ID final evaluator set (SHA-256 `2234e5ee82cf57e8fb74839a21f7f0ca0d2ff02ddd0fb0e42d93934415b2db93`) from the 844-ID pool and left 814 candidate baseline IDs (SHA-256 `5e810d3ab644bef1d43c598a14a6164ba6464b27fde50e92a2f241816ce87897`). The 30/30/30/814 development partitions are pairwise disjoint and exhaustive.
- Final fresh validation reached 25/30 extractable (83.33%), 5/30 exact compliant (16.67%), and 13/30 correct (43.33%). All 30 outputs were audited with zero false extractions and zero backend generation failures. Four complete human-clear answers remained outside the frozen grammar and one response reached 768 tokens.
- The final gate failed. Per the approved stop rule, no additional prompt/parser iteration, evaluator freeze, main baseline, training, synthesis, or sealed-final access occurred.

### Milestone 2 — Base development benchmark

Status: complete under the one-time D-011 exception; one frozen 814-example development run, a bounded 100-example failure audit, and the complete label-blind Milestone 2.1 correct-response audit completed.

- Freeze the exact Milestone 1.7 prompt, strict parser, `foundry-terminal-number-v2` extractor, greedy 768-token generation config, pinned Qwen/GSM1K revisions, and 814-ID development-baseline manifest.
- Count every unextractable response as incorrect in end-to-end accuracy while reporting extractability and exact-format compliance separately.
- Evaluate the untouched base model once on exactly the 814 development identifiers without retries.
- Record accuracy, coverage, category counts, runtime, tokens, peak VRAM, parsing failures, and a provisional taxonomy from a deterministic sample of at most 100 incorrect development outputs.
- Do not access sealed-final examples, generate synthetic data, or train a model.
- Result: 521/814 correct (64.00% end-to-end); 752/814 extractable (92.38%); 69.28% accuracy among extractable answers; 130/814 exact-format compliant (15.97%); 62 unextractable; three truncated; zero backend failures.
- A deterministic audit of 100/231 extractable-but-wrong records found recurring bookkeeping/omission, target interpretation, constraint/discrete, time/unit/sequence, arithmetic, and rate/ratio/percentage weaknesses. It also found two false extractions that remained scored wrong, so the aggregate score is reproducible under the frozen evaluator but extractor precision across the full baseline is not established.
- Milestone 2.1 then audited all 521 correct-scored completions without benchmark labels. It confirmed every extracted value as the model's intended terminal answer, found zero false-positive correct answers and zero ambiguous cases, and left the audited lower bound, upper bound, and adjusted exact accuracy equal at 521/814 (64.0049%). The frozen evaluator score remains separately preserved.

### Milestone 3 — Failure taxonomy and targeted synthetic-data design

Status: complete as a design-only milestone; no synthetic examples or training artifacts were created.

- Exhaustively reviewed all 293 development failures: 231 extractable-but-wrong and 62 unextractable. Complete primary counts are output format/extraction 69, multi-step bookkeeping/omission 68, target/language interpretation 53, rate/ratio/percentage/average 28, constraint/distribution/discrete 27, time/unit/sequence 24, arithmetic execution 22, and benchmark ambiguity/annotation risk 2.
- Kept all questions, labels, and completions in ignored raw artifacts. The tracked aggregate contains only counts, content-free definitions, hashes, and stable identifier prefixes. No sealed-final content was accessed.
- Selected exactly three first-pilot reasoning categories: multi-step bookkeeping/omission, rate/ratio/percentage/average, and constraint/distribution/discrete reasoning. A separate terminal-answer output-contract track is shared by targeted and generic curricula.
- Selected fully procedural latent programs with controlled templates. Labels come from exact execution and must pass a different independent verifier; verifier disagreement always rejects.
- Froze typed schema, contamination stages and thresholds, matched 4,000-example targeted/generic pilot budgets, a 120-candidate generator-smoke stage, compute estimates, and predeclared generation/training success gates in `docs/SYNTHETIC_DATA_DESIGN.md`.
- Added only design scaffolding and original unit fixtures. No full generator, model paraphrasing, large semantic scan, synthetic dataset, training code execution, or evaluator change occurred.

### Milestone 4 — Synthetic data and verification

Status: complete; the bounded smoke failed its readiness gate, so full pilot generation remains
blocked.

- Pinned `sentence-transformers/all-MiniLM-L6-v2` at immutable revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` for CPU-only local semantic screening without a new
  dependency.
- Implemented the three approved procedural families, the shared 20% terminal-answer track,
  distinct exact verifiers, ordered duplicate/contamination gates, stable raw records, and dry
  deterministic replay.
- Processed exactly 120 candidates: 60 targeted and 60 generic. Accepted 24 and rejected 96;
  accepted by category was 4 bookkeeping, 16 rate/ratio, and 4 discrete.
- All 120 were manually audited. Labels and verifier agreement were correct in every case, but five
  accepted renderings were invalid and controlled-template diversity caused 75 early duplicate
  rejections. The finalized replay matched exactly and left zero unresolved contamination cases.
- The fixed 75% acceptance and 15-per-family gates failed. Do not generate the 4,000 + 4,000 pilots
  or begin training until a separately approved rendering/diversity blocker-resolution smoke
  passes the unchanged gates.

### Milestone 4.1 — Generator rendering and diversity blocker resolution

Status: complete; the fresh bounded smoke still failed the unchanged readiness gate, so full pilot
generation remains blocked.

- Added typed object, unit, location, quantity, and ledger-operation contracts and a deterministic
  rule-based post-render quality stage. The five sanitized Milestone 4 defects now fail explicit
  tests.
- Expanded bookkeeping to two mathematical families, eight renderers, and 24 scenario domains;
  rate/ratio to five families, six renderers, and 20 domains; and discrete reasoning to four
  families, six renderers, and 20 domains. All 120 fresh attempts had distinct number-neutral and
  latent-structure hashes.
- Processed exactly 120 fresh attempts under a new seed and unchanged semantic artifact and
  thresholds. Accepted 86 and rejected 34: 19 automatic semantic, nine manual semantic, and six
  five-token-overlap rejections. Accepted counts were 30 bookkeeping, 29 rate/ratio, and 27
  discrete.
- Deterministic replay matched exactly and every arithmetic label/verifier decision was correct.
  Manual review nevertheless found 11 invalid accepted renderings across residual noun grammar,
  weighted-average rendering/target consistency, one omitted rate denominator, and discrete noun
  or plural rendering.
- The gate failed because acceptance was 86/120 rather than at least 90, invalid acceptances were
  nonzero, and systematic renderer defects remained. No threshold was lowered, no candidate was
  replaced, and no second smoke, full dataset, or training run followed.

### Milestone 4.2 — Typed natural-language realization compiler

Status: complete with a failed renderer stress gate; the procedural-only lineage is stopped and no
fresh 120-candidate pipeline smoke was permitted.

- Replaced the live ad hoc prose path with typed problem and sentence representations, explicit
  lexemes and irregular morphology, typed answer targets, explicit rate denominators, exact
  semantic-node coverage, and stable render signatures. All eleven prior defect classes have
  sanitized regressions.
- Ran the predeclared maximum 900 in-memory render attempts: 300 per family. All passed typed
  morphology, target, coverage, and grammar-metadata validation; there were zero exact or structural
  duplicates and 900 distinct render signatures.
- The scale-diversity evidence failed: 99 number-neutral templates repeated and 899/900 renders had
  a nearest generated semantic neighbor at or above the frozen 0.82 rejection threshold.
- The required 60-render audit found zero false mathematical labels but 13 unnatural question
  surfaces caused by imperative request clauses receiving direct-question punctuation. Because a
  systematic defect remained, the stress gate prohibited the counted 120-candidate smoke.
- Per the final procedural-lineage stop rule, no Milestone 4.3, full dataset, or training run may
  follow. The next discussable architecture retains exact procedural programs and dual verifiers,
  uses constrained local-model surface realization, and requires round-trip semantic validation
  plus the unchanged contamination pipeline.

### Milestone 5A — Constrained local-model realization design

Status: design complete; no model download or inference was performed.

- Retain exact procedural programs, typed semantic IR, targets, labels, traces, dual verifiers, and
  benchmark-contamination controls. Give the local model only a wording role.
- Select `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` as primary and the
  already proven Qwen2.5-1.5B-Instruct revision as fallback. Both are Apache-2.0 and use standard
  Transformers without remote code; the primary needs a separately approved dependency lock.
- Freeze opaque placeholders, strict JSON, semantic-node maps, target/intent equality,
  deterministic filling, dual mathematical verification, answer-blind naturalness audit, and a
  fixed three-beam/no-retry strategy.
- Preserve MiniLM 0.75/0.82 for generated-to-development contamination. Calibrate a separate
  generated-to-generated semantic-diversity policy on original fixtures before any future smoke.
- Proposed Milestone 5B is a maximum 120-IR, 360-candidate implementation smoke with unchanged
  zero-defect gates and at least 90 clean IR acceptances. It requires separate approval.

### Milestone 5B — Bounded local-model realization smoke

Status: complete; exact replay passed, but the readiness gate failed at 0/120 clean IRs. Full pilot
generation and training remain blocked.

- Calibrated the internal generated-peer policy before seeing Qwen output. The selected
  `evidence-gated-balanced-v1` policy rejects exact, number-neutral, latent-structure, five-token,
  and supported high-semantic duplicates while preserving the development firewall's unchanged
  MiniLM 0.75 review / 0.82 rejection bands.
- Created an isolated CPython 3.12.10 environment with PyTorch 2.5.1 CUDA 12.1, Transformers
  4.51.3, and tokenizers 0.21.4. Downloaded only
  `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; offline FP16 reload succeeded.
- Processed exactly 120 fresh IRs and 360 fixed beams. The procedural programs and two independent
  verifiers agreed throughout, with zero false labels, backend failures, or verifier disagreements.
- Only 181/360 beams parsed as the exact JSON schema. Of 179 parse failures, 160 consumed the full
  256-token budget. Every parsed beam still failed semantic-node coverage and clause/discourse
  validation; model templates usually emitted only the target question or echoed the semantic
  instructions rather than composing a complete problem.
- The all-beam hidden-label audit found 63 natural-but-drifted outputs and 59
  semantics-preserving-but-unnatural outputs. In total, 301/360 drifted and 297/360 were unnatural.
  Every automatic rejection was correct; invalid acceptances and incorrect rejections were zero.
- Exact replay reproduced all 360 beam texts, ordering, validation decisions, and aggregate hash
  `a2e6fb565da817ec5e2e6e3c87ba8a54643b2b5ec294dd8f5d24204083d06dcf`.
- The next architecture must not generate the 4,000 + 4,000 pilot. The narrowest evidence-based
  option is a separately approved compact realization-protocol redesign that removes verbose echoed
  inventories/maps, uses declarative rather than imperative event input, and proves semantic
  coverage on original fixtures before another bounded model run.

### Milestone 5 — SFT smoke train

- Train a tiny QLoRA adapter for a few steps.
- Confirm loss, checkpoints, resume behavior, VRAM, and deterministic evaluation plumbing.

### Milestone 6 — Approved SFT experiment

- Freeze an experiment configuration.
- Train the planned adapter locally.
- Select a checkpoint using synthetic validation data, never the final benchmark.

### Milestone 7 — Locked comparison

- Freeze the candidate.
- Evaluate base and candidate on the identical final holdout.
- Calculate paired uncertainty and decide whether the hypothesis was supported.

### Milestone 8 — One controlled iteration

- Use remaining development failures to produce a second verified dataset.
- Repeat SFT only if the first loop is sound and the change is approved.

### Milestone 9 — GRPO admission review

GRPO is not automatically approved. It is considered only if SFT produces a reproducible gain but then plateaus on a behavior that an exact, exploit-resistant reward can target. The review must specify reward behavior, exploitation paths, reward verification, extra compute, and a result threshold that justifies the complexity.

## Benchmark and success metrics

Primary metric:

- Canonical deterministic terminal-number accuracy on the locked GSM1K final split, using an evaluator frozen before final access.

Required separately reported format metric:

- Exact compliance with the literal terminal `Final answer: <integer>` contract.

Required comparison controls:

- Same model base revision.
- Same dataset revision and example IDs.
- Same chat template and prompt text.
- Same answer parser and evaluator commit.
- Same greedy decoding settings and token limit.
- Same hardware class and software environment where practical.

Secondary metrics:

- Accuracy by predefined failure category.
- Invalid or unparseable output rate.
- Base-to-candidate paired win/loss/tie counts.
- 95% paired bootstrap confidence interval and McNemar test.
- Training loss and synthetic-validation accuracy.
- Peak VRAM, tokens/second, wall time, energy/cost estimate, and artifact size.
- Exact and semantic dedup rejection rates.
- Regression rate on categories not targeted by training.

Measured base main-development baseline: **521/814 correct (64.00% end-to-end)**, with **92.38% extractability**, **69.28% accuracy among extractable answers**, and **15.97% exact-format compliance**. This is a frozen development result, not a sealed-final score. No candidate or trained-model score exists.

## Current project phase

Milestone 1 and its deferred RTX smoke, Milestones 1.5–1.7, the frozen Milestone 2
base-development baseline, the bounded Milestone 2.1 correct-response audit, design-only Milestone
3, bounded Milestones 4 through 4.2, design-only Milestone 5A, bounded Milestone 5B, and the
Milestone 5C compact-protocol micro-smoke are complete.
The one approved 814-example run used the D-011 exception without changing the frozen evaluation
stack.

The repository records deterministic, pairwise-disjoint development partitions of 30
prompt-calibration IDs, 30 answer-extraction-validation IDs, 30 final-evaluator-validation IDs, and
814 baseline IDs. The completed baseline counts every unextractable output wrong and reports
coverage separately. Milestone 2.1 audited all 521 correct-scored responses label-blind: 521 intended
answers, zero false acceptances, and zero ambiguity. Milestone 3 classified all 293 failures and
froze the content-free generator design. Milestone 4.2 demonstrated correct typed semantics but
failed the renderer stress gate. Milestone 5B proved that the pinned Qwen3 runtime is local,
reproducible, and mathematically firewalled, but the verbose strict-JSON protocol produced zero
clean IRs. Milestone 5C removed redundant JSON metadata and reached 90/90 tag parses, but Qwen3
still echoed ordered token lists rather than realizing natural predicate-argument clauses; all 90
beams were unnatural and semantically drifted, with zero clean IRs. The final Qwen3 prompt-patching
stop rule is active. Full dataset generation and every training stage remain blocked; no complete
synthetic dataset or adapter exists.

## Unresolved questions

1. Should the project approve a separately designed bounded test of a stronger local realization
   model using the now-frozen compact tagged protocol? No further Qwen3-1.7B prompt patch or
   120-IR Qwen3 smoke is allowed under the Milestone 5C stop rule.
2. Should the cross-platform dependency locks explicitly pin Windows-only `colorama` and `tzdata` in a separately approved lock-maintenance task?
3. Any future comparison must preserve the exact 814-ID manifest and frozen prompt/extractor/generation configuration unless the user explicitly authorizes a new evaluator lineage and complete reruns.
4. The pure procedural renderer and Qwen3-1.7B prompt-engineering lineages are closed. Any stronger
   local model comparison needs an explicit model/revision/license/dependency/compute design and
   separate approval before download or inference.
5. The pinned MiniLM encoder behaved acceptably on original fixtures; future work must retain its exact revision/configuration unless a separate design decision replaces the semantic lineage.
6. Is a 3-point final improvement statistically realistic after the development baseline, or should the success threshold be revised before training?

## Next approved milestone

No further milestone is approved. Milestone 5C ends after its verified evidence commit is pushed.
Its 0/30 clean result fails the fixed micro-gate, so a 120-IR compact Qwen3 smoke is not justified.
The exact next decision is whether to approve a design-and-bounded-smoke proposal for one stronger
local realization model using the frozen compact protocol. The alternative approved by the stop
rule was a manually vetted template-bank architecture, but Milestone 5C selects the stronger-model
test as the single recommendation because it isolates model capacity while preserving the tested
protocol and every deterministic safety layer. The 4,000 + 4,000 pilot, training, SFT, QLoRA, GRPO,
paid services, benchmark inference, fallback inference, and sealed-final access remain unapproved.

### Milestone 5D: stronger-model substitution micro-smoke (complete; gate failed)

Milestone 5D changed exactly one experimental variable: Qwen3-1.7B was replaced by pinned
`Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`. The same 30 M5C plans,
semantic IRs, latent programs, compact requests, prompt hashes, three-beam decoding, validators,
verifiers, screens, audit rules, and gates were proven identical before inference. The official
Apache-2.0 snapshot loaded offline in FP16 without CPU offload and passed the fixed memory probe.

The counted run produced exactly 90 beams but accepted 0/30 IRs. Only 71/90 outputs tag-parsed;
47/90 preserved placeholders, 50/90 preserved anchors, and 47/90 preserved targets. All 90 audited
outputs were unnatural and semantically drifted, while false labels, invalid acceptances, verifier
disagreements, timeouts, and backend failures remained zero. Exact replay passed. The 22/30 gate and
all family minima failed, so a 120-IR run is not justified.

The final local-model substitution stop rule is active. No further local model, compact-prompt
experiment, full dataset generation, or training is approved. The exact next decision is whether to
authorize design and bounded validation of an offline, manually vetted natural-language template
bank layered over the retained exact procedural IR, labels, dual verifiers, and contamination stack.

### Milestone 6A: offline template-bank implementation and smoke (complete; technical gate failed)

The offline bank contains 58 original semantic frames and 232 versioned sentence plans. It consumed
the existing typed IR without changing generator or verifier mathematics and ran exactly 120 new
attempts. Automatic gates accepted 118, both verifiers agreed throughout, contamination remained
clear, and exact replay passed. The ignored user packet contains all 120 rendered questions.

Codex inspection found 13 clearly invalid or unnatural surfaces despite zero rule-based language
failures. Repeated frame nouns, invalid ordinal forms, malformed noun compounds, and literal frame
labels are systematic, so the technical gate failed. Full pilot generation and training remain
blocked. The next decision is whether to approve one architecture-level bank-composition blocker
resolution after user review, or stop synthesis realization.

### Milestone 6B: offline template-composition compiler (complete; human review pending)

Milestone 6B retained the 58-frame/232-plan bank and replaced identifier-derived prose with approved
surface lexemes, typed single-head noun phrases, correct ordinals, and token-level provenance. All
13 Milestone 6A defects are deterministic regressions. A non-dataset expansion rendered ten fixtures
per plan: 2,320/2,320 passed, all 232 signatures were exercised, and a stratified 90-render Codex
inspection (30 per family, not human review) found no remaining deterministic surface defect.

The one fresh 120-attempt smoke accepted 116 automatically: targeted 58/60, generic 58/60;
bookkeeping/rates/discrete 53/31/32; easy/medium/hard 38/39/39. Three latent-program copies and one
number-neutral copy were safely rejected. False labels, verifier disagreements, deterministic
language defects, target mismatches, benchmark rejections, unresolved contamination, exact accepted
duplicates, and reused render signatures were zero. Replay exactly matched SHA-256
`f5caa7e811cbf257c752a15059e25cc20b2f978fb60e8ad0890c64186095a254`.

The technical status is **TECHNICALLY READY — HUMAN REVIEW PENDING**. Full 4,000 + 4,000
generation and training remain blocked. The exact next decision is the user's review of the ignored
HTML packet and explicit approval or rejection of template-bank language quality.

### Milestone 6C-R: human-review-driven bank revision (complete; technical gate failed)

The genuine Milestone 6B review was verified at SHA-256
`564a8ca584984ee7a0b997eec4a6a6f377308c869b62cf65ebeef5375cef0791`: 120 unique decisions,
60 Approve, 60 Reject, and no Unsure. The rejected surfaces concentrated in indirect update-log,
transfer-record, register/tally, vague ratio/sample, awkward average, and indirect discrete wording.
The review-derived manifest freezes 60 approved historical plan uses and 60 quarantined uses without
tracking any question text.

The v3 bank replaces the 12 affected plan families with direct, original worksheet-style language.
Every one of 232 plans passed ten static fixtures (2,320/2,320), exact and number-neutral sentence-plan
identity checks were zero, and a 90-render Codex inspection found no remaining surface defect after
one shared duplicated-quantifier rule was corrected.

The one permitted fresh smoke accepted only 104/120 automatically. All mathematics, dual verification,
language validation, contamination controls, and exact replay passed; 15 number-neutral template
copies and one latent-program copy were safely rejected. Because the fixed gate requires 110, no v3
review packet was created. Full generation and training remain blocked. The next decision is whether
to authorize a narrowly bounded plan-allocation/diversity correction and one new smoke; the current
milestone does not authorize that work.

### Milestone 6D: runtime-diversity capacity audit (complete; capacity gate failed)

Milestone 6D diagnosed all 16 v3 duplicate rejections before changing allocation. The 15
number-neutral collisions reused a sentence-plan identity under different frame/template metadata
and collapsed to an earlier normalized surface; the one latent-program collision was a repeated
discrete structure from the seed schedule. Neither pattern was a label, verifier, language, or
benchmark-contamination failure.

The precommitted capacity audit then applied the 125% attempt budget to both future 4,000-example
groups. The accepted quota of 8,000 requires 10,003 fixed attempts after per-stratum rounding:
4,418 bookkeeping, 2,834 rate/ratio, and 2,751 discrete. The current active plan-level capacity is
only 72/80/80, while existing domain-aware signatures provide 1,728/400/1,600 and actual
number-neutral surfaces provide 768/88/320. Every category, difficulty, and output-contract stratum
fails. Therefore the mandatory stop rule prevented allocator implementation, latent scheduling,
schedule freezing, a fresh 120-attempt smoke, and packet creation.

Full generation and training remain blocked. The exact next decision is whether to authorize a
separate expansion milestone for independently authored and human-reviewed sentence plans, scenario
domains, and structural surfaces. Any expansion must first raise the measured number-neutral pools
by at least 3,650 bookkeeping, 2,746 rate/ratio, and 2,431 discrete identities and then repeat this
capacity audit under the unchanged duplicate and contamination policies.

### Milestone 6E: bounded template reuse and revised capacity (complete; capacity gate failed)

Milestone 6E corrects the scientific interpretation of a sentence plan. Exact rendered questions
and complete latent programs remain globally unique, while structural skills and reviewed language
plans may recur under deterministic quota-derived caps. The selected policy passed 14/14 original
fixtures; the one-use policy and an uncapped permissive policy did not. Development-contamination
screening remains unchanged.

The correction removes language identity as the limiting layer, but the unchanged mathematical
generators still cannot construct the full balanced pool. Bookkeeping supplies 5,524 bounded unique
programs for 4,418 required attempts. Rates supply 1,632 for 2,834 (shortfall 1,202), and discrete
reasoning supplies 2,073 for 2,751 (shortfall 678). The 10,003-attempt gate therefore still fails.
No allocator, schedule, fresh smoke, replay, or second review packet was created.

Full generation and training remain blocked. The exact next decision is whether to authorize a
narrow mathematical-program capacity expansion for the finite rate and discrete modes, with
unchanged labels, dual verifiers, language bank, benchmark firewall, and contamination thresholds,
followed by the same capacity audit before any allocation or generation.

### Milestone 7A: signal-first capacity preflight (stopped; capacity gate failed)

The proposed first experiment is frozen at 1,000 accepted targeted and 1,000 accepted generic
examples, each split 900/100 for training/synthetic validation with 200 output-contract examples.
Its fixed attempt pool is 2,504: 1,106 bookkeeping, 709 rates, and 689 discrete problems.

An initial aggregate audit appeared to pass, but allocator preflight exposed a missing compatibility
constraint. Under the unchanged quota-derived policy, modes sharing one target type must share that
target-type cap, and modes also share the per-frame and finite global supply across both datasets.
The exact compatibility graph supports 1,384 bookkeeping attempts, 695 rate attempts, and 598
discrete attempts. Rates are short by 14; discrete is short by 91; generic discrete also fails its
own 417-attempt quota at 399. The reduced-pilot gate is therefore false.

Per the approved stop rule, no allocator, 2,504-slot schedule, 120-question smoke, replay, or second
review packet was created. Full generation and training remain blocked. The next decision is whether
to authorize a separate correction to the frozen target-type/semantic-frame cap derivation so it
reflects predeclared curriculum compatibility, or to abandon/redefine the signal-first quotas. No
benchmark-contamination control needs or receives a change.

### Milestone 7D: submode-local runtime-surface caps (stopped; difficulty stratum failed)

The approved correction replaces only the family-level runtime number-neutral cap with
`submode-local-balanced-surface-reuse-v1`. Exact rendered questions and latent programs remain
globally unique, while the same reviewed wording may be reused by distinct verified programs within
a quota-derived dataset/family/submode cap. Ten original fixtures select this policy over both the
family-level and uncapped alternatives.

All eleven aggregate submode quotas fit. Weighted-average, however, has four identities shared by
easy/medium wording and four different identities used by hard wording. The frozen allocations need
47 targeted and 66 generic easy/medium attempts. Their local caps provide only 44 and 64; combined
capacity is 108/113. The complete schedule gate therefore remains false. No new 2,504-slot schedule,
120-question smoke, replay, or review packet is authorized. The next decision must address this
five-attempt weighted-average difficulty compatibility shortfall without weakening exact, latent,
runtime-identity, or benchmark-contamination controls.

## Milestone 7E stopped result: weighted correction passed, complete schedule blocked

`minimal-compatible-difficulty-reallocation-v1` moved exactly three targeted and two generic
weighted-average attempts from easy/medium to hard and made five inverse shifts in ratio-scale and
combined-rate. Dataset difficulty totals, rate-submode totals, output-track totals, and future split
totals remain exact. The weighted compatibility proof now passes at targeted 44/44, generic 64/64,
and combined 108/108. Exact 2,504-slot construction then exposed a different joint constraint:
generic complete-packages cannot assign its 121 attempts under the unchanged sentence-plan,
plan-plus-scenario, frame, and canonical runtime-identity caps. The schedule gate therefore failed;
no fresh smoke, replay, or review packet was created. The next permitted decision is whether to
reduce the signal-pilot fixed attempt pool. Complete dataset generation and training remain blocked.

## Milestone 7F stopped result: every predeclared reduced attempt pool failed exact scheduling

The accepted objective stayed fixed at 1,000 targeted and 1,000 generic examples. Foundry evaluated
only the approved descending buffers. `1.15` derives 2,302 attempts, `1.125` derives 2,253, and
`1.10` derives 2,203. Each pool has sufficient unique verified mathematical programs and preserves
the frozen submode, difficulty, output, split, template-reuse, and contamination policies.

The final uniform exact scheduler cannot assign the generic percentage submode for any candidate
under the joint reviewed-plan, plan/scenario, semantic-frame, canonical runtime-identity, and exact
surface controls. No multiplier was selected. Therefore no complete schedule, new 120-question
smoke, replay, or assisted review packet was created. The next architectural decision is whether to
reduce the accepted signal pilot itself; no further attempt multiplier may be invented from these
results.

### Milestone 7G: reduced signal-pilot selection (complete; architecture stop)

Milestone 7G evaluated exactly `900`, `800`, `700`, `600`, and `500` accepted examples per
dataset at the frozen `1.10` multiplier. Stable largest remainder preserved the targeted
`550/233/217` and generic `334/333/333` family weights; every candidate preserved 90/10
train/validation, 20% output-contract, frozen submode/difficulty policies, exact uniqueness,
runtime identities, and verification. The corresponding fixed pools were `1,981`, `1,762`,
`1,544`, `1,320`, and `1,102` attempts.

All five candidates had sufficient verified latent mathematics but failed exact joint surface
scheduling. No accepted size was selected, so the complete schedule, review smoke, deterministic
replay, and assisted packet were not created. Per the approved final stop rule, the current
synthetic-data architecture ends here; the next decision must concern project direction rather
than another scheduling, cap, template, generator-range, or pilot-size patch.

### Fast-Track 8A: matched 500-example signal datasets (dataset gate passed)

The fast-track decision replaced one-use worksheet-structure scheduling with
`matched-template-signal-v1` while retaining globally unique exact/normalized questions, latent
programs, synthetic IDs, and targeted/generic examples. One predeclared pool supplied exactly 550
attempts per dataset. It produced all 500 targeted and 500 generic acceptances, exact `450/50`
training/validation splits, exact family quotas, and exactly 100 output-contract examples per
dataset. All candidates replay exactly, and automatic quality, verification, overlap, and
development-contamination gates pass.

The ignored stratified human-review packet is pending. Under the approved fast track, this does not
block the provisional one-seed experiment, but a later false label, systematic wording defect, or
greater-than-5% clean-sample rejection rate invalidates promotion. Next is a separately isolated
QLoRA environment and 32-step compatibility smoke; no optimizer step is permitted before the
dataset-stage commit is verified and pushed.

### Fast-Track 8B: native Windows RTX 3080 QLoRA compatibility (gate passed)

An isolated `.venv-training` uses CPython 3.12.10, PyTorch 2.5.1+cu121, Transformers
4.51.3, PEFT 0.15.2, TRL 0.17.0, bitsandbytes 0.49.2, and Accelerate 1.7.0. The
immutable recipe hash is `4a9c6043...0590`; it uses NF4 double quantization, rank-16 LoRA on
all seven approved projections, 512-token unpacked examples, effective batch eight, paged AdamW
8-bit, cosine learning rate, and a 200-step final-adapter-only rule.

The exact 32-step compatibility smoke passed on 128 frozen targeted records: forward, backward,
optimizer, finite loss, evaluation, adapter save, offline reload, and deterministic inference all
succeeded. Peak reserved VRAM was 3.741 GB, below the 9.6 GiB gate. Next is repository-wide
verification and publication of the training setup; final generic and targeted runs remain blocked
until that commit is pushed.

### Fast-Track 8C: matched adapter training (stopped at parity gate)

Generic and targeted adapters each completed the frozen 200-step recipe and reload offline on CUDA.
Every metadata and padded-input field matches. The actual loss-bearing token exposure does not:
generic processed 271,396 non-padding tokens and targeted processed 306,766, an 11.5299% relative
difference. The approved maximum was 2%.

The training-parity gate therefore failed before development evaluation. Neither adapter was run on
the frozen 814 examples; there is no generic, targeted, category-level, or one-seed signal result.
The next plan decision is whether to approve a fresh token-budget-matched experimental design and
retraining. These adapters cannot answer the research question.

### Milestone 8D: token-matched retraining protocol (GPU smoke passed)

The frozen 900-record census confirms that sequence length is the only material exposure mismatch:
generic records contain 77,348 loss-bearing tokens once each and targeted records contain 87,317;
no record truncates or masks every label. Fixed 1,600-occurrence Method A cannot close the gap: its
exact favorable boundary remains 278,167 versus 307,144 tokens (9.4343%).

The selected `foundry-token-matched-qlora-v2` Method B schedule uses whole examples, variable
microexample counts, and a token-weighted mean within each of 200 optimizer steps. Generic schedules
271,292 tokens over 1,578 occurrences; targeted schedules 271,150 over 1,398, a 0.05234% difference.
Recipe SHA-256 is `df7c7b8d...fa54`. Both four-step fresh-adapter smokes passed at 5,464 versus
5,440 actual tokens, finite loss/gradients, four scheduler updates, and offline reload. Next is to
publish this protocol, then retrain generic followed by targeted without development exposure.

#### Completed token-matched comparison

Protocol commit `02a7a3f1...2638` was published before either full run. Fresh generic and targeted
adapters then processed 271,292 and 271,150 actual loss-bearing tokens over the same 200 updates, a
0.05234% difference; both saved, hashed, and reloaded offline. The frozen 814-example development
evaluation produced 15 correct / 20.52% extractable for generic and 14 correct / 22.11%
extractable for targeted, versus the frozen base's 521 correct / 92.38% extractable. Targeted's
paired difference from generic is -1 answer, with a fixed-seed 95% bootstrap interval of -1.2285
to +0.9828 percentage points.

The one-seed signal gate therefore failed. No second seed, tuning, retraining, sealed-final
evaluation, or GRPO is planned automatically. The next decision is whether to approve a narrow
training-format/instruction-retention diagnosis; stratified human language review also remains
pending. The result is provisional and cannot support the targeted-data hypothesis.
## Milestone 8E outcome: assistant-only correction is not retention-safe

Milestone 8E identified two concrete shared SFT protocol defects: the prior formatter made system
and user tokens loss-bearing in all 900 training records, and only 200 of 1,000 targets used the
development evaluator's terminal-answer contract. Adapter application and base restoration were
correct. `foundry-assistant-only-sft-v3` removes prompt/header/padding loss, retains only assistant
content plus final EOS, and normalizes every completion to one `Final answer:` line without changing
questions, reasoning, answers, membership, or splits.

The original 60-prompt retention suite was frozen before corrected training. Base performance was
30/30 arithmetic, 14/15 format, 14/15 instruction, and 59/60 extractability. The `2e-4` 32-step
recipe failed multiple retention gates in both arms. The only authorized fallback, `5e-5`, also
failed: generic instruction following was 13/15 and targeted arithmetic was 25/30. Each paired run
had exact 14,404-token parity, finite losses, zero echo/question generation/backend failures, and no
development exposure.

**Current stop:** no common retention-safe recipe exists among the two predeclared options. No
200-step retraining, retention checkpoint, final training-parity gate, corrected GSM1K evaluation,
second seed, or sealed-final evaluation is authorized. The next decision must be a separately
approved training-method investigation, not automatic tuning. Stratified human language review
remains pending at the existing ignored local review page.

## Fast-Track 8F-8H outcome: calibration success did not generalize

The approved retention-safe ladder audited all 1,000 v3 targets, froze concise equation-grounded
v4, and added two original ignored 90-prompt retention suites. The untouched base passed both new
suites. Four predeclared 32-step pairs then received exactly 14,400 assistant loss tokens per arm,
with exact checkpoint-prefix parity at steps 8, 16, 24, and 32.

Only Variant A (v3, `5e-5`) passed calibration for both arms; its latest common checkpoint was 32.
Variants B-D (concise-v4 at `5e-5`, `2e-5`, and `1e-5`) all missed calibration instruction
retention. On the disjoint validation suite, selected Variant A scored 21/25 instruction prompts
for both generic and targeted, below the required 90%. Generic otherwise scored 45/45 arithmetic,
20/20 format, and 90/90 extractable; targeted scored 44/45, 20/20, and 89/90 extractable.

**Current stop:** the disjoint validation gate failed. No 200-step schedule, full retraining, final
holdout adapter evaluation, GSM1K evaluation, paired analysis, second seed, or sealed-final access
was run. The narrowest next decision is an evidence-based training-method diagnosis; the pending
human language review remains separate and unchanged.

## Fast-Track 8I–8K outcome: powered instrument failed its base-usability gate

The prior 25-item instruction failure was audited without benchmark access. Base/generic/targeted
remains `23/25`, `21/25`, and `21/25`; the two base-only successes are genuine shared adapter
regressions, while two other instructions fail in all three systems. No prompt, reference, or
scorer defect was found, so the original failed gate remains unchanged.

Three original disjoint artifacts were then frozen before any new adapter evaluation: a 300-item
adjudication suite, a 120-item shared anchor, and a 300-item anchor holdout. The untouched base
scored only `84/100` arithmetic, `48/100` format, and `55/100` instruction on adjudication, with
`268/300` extractable. All zero-error backend, echo, question-generation, and ambiguity checks
passed, but every required score threshold failed. Direct blind inspection classified all 113
failures as genuine terminal-contract, format, or deterministic-instruction noncompliance; no
original prompt was objectively defective.

**Current stop:** the powered suite cannot serve as the approved noninferiority instrument for this
base model. The holdout was not evaluated, neither adapter saw adjudication, shared-anchor training
did not run, and GSM1K was not run. A new explicit decision is required; the present fast track
cannot continue by tuning the suite to the model.

## Milestone 8L outcome: base-conditioned retention failed

Milestone 8L replaced the unsuitable absolute-capability question with a conditional preservation
instrument. It froze every adjudication item the untouched base answered correctly (187 items:
84 arithmetic, 48 format, 55 instruction), then evaluated the untouched base once on the already
frozen anchor holdout. The holdout passed its sample-size gate at 210/300 correct (96 arithmetic,
60 format, 54 instruction), so those 210 IDs were frozen before adapter exposure.

Both existing A/32 adapters were evaluated on both immutable subsets. Generic and targeted each
preserved 181/187 adjudication items, but each preserved only 43/48 format items (89.58%) and emitted
one question-generation output. On holdout, generic preserved 197/210 and targeted 200/210; both
preserved only 53/60 format items (88.33%). Generic also emitted one question-generation output.
Every overall Wilson lower bound exceeded 85%, but the fixed category and output-behavior clauses
did not all pass.

**Current stop:** the pair is `failed_base_conditioned_retention`, and the current SFT adaptation
line is closed. GSM1K was not run. The exact next decision is whether to end the project or approve
an interpretation-only milestone; no further SFT method, tuning, or second seed is recommended by
this result. The stratified human language review remains pending independently.

## Milestone 8M retention outcome: common scale 0.50 approved

Milestone 8M tested a reversible inference-only reduction in LoRA contribution without changing or
merging either trained adapter. Scale 0.0 reproduced the untouched base exactly, scale 1.0
reproduced each unscaled adapter exactly, and every evaluation restored all 196 per-module scaling
values plus unchanged adapter/base state hashes.

The descending retention-only search reconstructed the failed 1.00 decision, rejected 0.75 because
the two anchor-holdout cells generated 3 and 4 questions, and selected 0.50 after all four existing
subset cells passed. At 0.50, generic/targeted preserve `182/187` and `183/187` on adjudication;
both preserve `205/210` on anchor holdout. A separately authored 450-item final holdout yielded 318
base-correct IDs (`112/127/79`). Generic preserves `314/318` and targeted `315/318`, including
`127/127` format for both, with zero question generation, prompt echo, or backend failure.

**Current gate:** retention is approved as
`retention_approved_common_scaled_short_run_adapters` at common scale 0.50. The next authorized
action is the frozen generic-then-targeted 814-item development evaluation, but only after this
retention decision is verified, committed, pushed, and synchronized. Human language review remains
pending, and any later benchmark result remains provisional pending that review and a second seed.

## Milestone 8M final outcome: targeted beats generic but signal gate fails

> **Provisional one-seed result pending stratified human language review and second-seed
> confirmation.**

After the retention decision was independently committed and pushed, the unchanged frozen
development evaluator ran generic then targeted at common scale 0.50. Generic scored `387/814`
(`47.5430%`) and targeted `414/814` (`50.8600%`), versus the frozen base `521/814` (`64.0049%`).
Targeted therefore gains 27 correct answers over generic, and the paired 10,000-replicate interval
for targeted minus generic is `[+1.3514, +5.2826]` percentage points. Nevertheless, generic is 134
below base and targeted is 107 below base.

**Final gate:** failed only the absolute clause requiring targeted at least `529/814`; the
targeted-over-generic, extractability, backend, untargeted-taxonomy, token-parity, and all-three-
subset retention clauses pass. This is evidence that targeted curriculum allocation outperformed
the matched generic curriculum within this adaptation method, but the method still caused a large
absolute capability regression. Do not run a second seed, sealed-final evaluation, another scale,
or further SFT automatically. Human language review remains pending at the frozen local review URL.

## Milestone 8N outcome: exact task vector failed retention selection

> **Provisional one-seed result pending stratified human language review and second-seed
> confirmation.**

Milestone 8N constructed the frozen targeted-minus-generic update without retraining. The generic
and targeted dense LoRA updates have Frobenius norms `1.6918784364` and `1.6980775191`, with cosine
similarity `0.9399098552`. Their exact difference has norm `0.5876302228`, or `34.7324%` of the
generic norm and `34.6056%` of the targeted norm; its largest absolute element is
`0.0007056529`. Attention and feed-forward contrastive norms are `0.2377137337` and
`0.5374025117`. The dense-analysis SHA-256 is
`36ce1b90beee7499aa33e11dacbe163e107a98bda5f1065e3f7841fbd85fbaa2`; compatibility and
contrastive-definition hashes are `1c921ad51219131857475c569f52492977eaf8dcf0f3ab6aca305f4df48d3092`
and `c711c3f97ec750f1dd3471822b1be76edca4668ec9ed50bab828c749b843f3e6`.

PEFT `cat` composition produced one unmerged, reversible rank-32, alpha-32 adapter with SHA-256
`84f02df1cbc5ec1015d096164dbfe3833e166a14eda9ffadf62b5d2d2527c961` under final protocol
`b4914d5a95bb46a52374b9a390038634f01df99f69a4ef6f79c5bfe4f8d983fa`. Across all 196 modules,
the maximum dense error was `1.7462298274e-10` and relative Frobenius error was
`2.9353350496e-7`; the functional logit comparison measured `5.6266784668e-5` maximum absolute
error and `1.9593861237e-6` relative error. Scale-zero/base and scale-one/unscaled sanity checks,
source-adapter immutability, and base-state restoration all passed. Construction summary SHA-256 is
`07a99bde03339494cc1ce9cf8428d7ecf7ad35aef58b55038389a3888d2c586c`.

Retention-only selection then tested every predeclared scale:

| Scale | Adjudication preservation | Anchor preservation | Anchor question generation | Result |
| --- | ---: | ---: | ---: | --- |
| `1.00` | `181/187` | `204/210` | `1` | fail |
| `0.75` | `182/187` | `207/210` | `2` | fail |
| `0.50` | `183/187` | `207/210` | `1` | fail |
| `0.25` | `184/187` | `208/210` | `2` | fail |

Every adjudication cell passed, and every anchor cell met the overall, category, Wilson-bound,
extractability, prompt-echo, backend, and failure-concentration clauses. Each anchor cell failed the
unchanged requirement of zero question generation, so no contrastive scale was selected. Selection
summary SHA-256 is `b41d975f342820ac34ca693d599677994e3f272243c114c313605beb020ad49a`.

**Current stop:** the contrastive adapter-arithmetic route is closed for this project version. The
independent final retention holdout was not evaluated, GSM1K development was not run, no paired
benchmark analysis exists, and sealed-final remained untouched. The stratified human language
review remains pending at
`file:///C:/Users/Admin/Projects/Foundry/results/raw/foundry_500x2_signal_review/codex_assisted_review.html`.

Any next adaptation experiment requires separate approval and a materially different retention
architecture: either KL/replay-regularized adaptation or verifier-reward GRPO.

## Milestone 9 outcome: replay source passed; independent instrument failed before training

The frozen shared anchor yielded 83 scorer-correct untouched-base behaviors: 40 arithmetic, 20
format, and 23 instruction. Their actual deterministic base outputs were frozen as replay targets,
with replay-corpus SHA-256
`b511129f89ce450014b78698e9e439bdaa0947657f301c3e99b2a9955b7ab4d1` and format SHA-256
`758dc1f35020e88e04c425b6106e54ea2f577f547afa4762ade9923762af6d66`.

Before training, Foundry froze a new 450-item independent retention holdout and proved zero exact or
12-token overlap against 3,314 prior retention, synthetic, and development prompts. The untouched
base then scored 84/150 arithmetic, 27/150 format, and 30/150 instruction, or 141/450 overall. The
predeclared usability minimums were 60 in every category and 250 overall. Format, instruction, and
overall therefore failed despite zero backend failures and valid deterministic references/scorers.

**Current stop:** no replay/KL schedule was frozen, no adapter was trained, no retention method was
selected, and GSM1K was not evaluated. The next project decision is whether to stop conventional
adaptation or separately approve a new architecture and independently frozen retention instrument;
the failed holdout may not be edited post hoc. Human language review remains pending at the existing
local review URL.

## Milestone 10 outcome: verifier-GRPO stopped at CUDA deterministic sampling

Foundry froze the 141 untouched-base-correct rows from the unused final retention holdout
(`84/27/30` arithmetic/format/instruction). The content-free subset SHA-256 is
`f56845076a1a59e5ca1a95466541339b56f026e945f86118caec307a690ee4ec`. It also froze paired
prompt-only GRPO schedules: each arm has 64 groups, comprising 52 synthetic and 12 identical replay
groups, with four planned completions per group. Generic and targeted each contain exactly 6,702
model-visible prompt tokens. Their manifest hashes are
`5848ed6640dda21752ab9692c8e531d9175314a7d5a472616dc19ad834a6351e` and
`cb13d4d522746bdfa829c9a405defdb0eff0acbd23859dc7fe49457318cc1ccf`; the paired schedule-summary
hash is `23fede9132f53b7d32f354056c728fc68faa20586a9162e101834db34f71ca64`.

The deterministic verifier reward, official adapter-disabled base-reference path, exact truncation
hook, and strict GRPO runtime were implemented and passed static/unit verification. The G1
compatibility probe nevertheless stopped during the first sampled generation: with top-p `0.95`
and strict deterministic algorithms enabled, PyTorch 2.5.1+cu121 reported that CUDA
`cumsum_cuda_kernel` has no deterministic implementation. No completion, reward, reference-KL
pass, backward pass, optimizer step, or adapter artifact was produced. Failure-summary SHA-256 is
`8b57b6284c1e7dccd978379162de9519b7af30addbbfb9eb4d5a95a7f2b439a6`.

**Current stop:** the compatibility gate failed mechanically. G1 and G2 training, checkpoint
retention, independent final retention, and GSM1K were not run. Continuing requires explicit
approval of a revised, predeclared contract that reconciles stochastic top-p sampling with the
available deterministic CUDA kernels, or a decision to stop verifier-GRPO. Foundry must not
silently weaken determinism, alter sampling, change the dependency stack, or retry. The existing
benchmark label remains **Provisional one-seed result pending stratified human language review and
second-seed confirmation**; no new benchmark result was created. Human review remains pending at
`file:///C:/Users/Admin/Projects/Foundry/results/raw/foundry_500x2_signal_review/codex_assisted_review.html`.

## Milestone 10E outcome: exact warning-only replay failed; verifier-GRPO closed

The approved `foundry-warning-only-top-p-replay-v1` contract preserved temperature `0.8`, top-p
`0.95`, top-k `50`, four generations, and strict deterministic enforcement outside the actual
`GenerationMixin.generate` call. Its final warning-contract summary SHA-256 is
`eff84b9ec92715eeb74a6c74bcad5980dded9c4b5482012fd8e2438857f24598`; the tracked artifact file
SHA-256 is `f473149d963b8a81bef69f4d13ce9f22ccfbf6b965a9b44fafad89eba84c90af`.

The official same-process replay completed three runs of the exact frozen three-group schedule,
producing `36` total completions. All warnings were the approved CUDA-cumsum class, warning-only
state did not leak, and the diagnostic projection of tokens, completions, rewards, log probabilities,
KL, RNG transitions, base state, and LoRA state was equal. Exact replay nevertheless failed because
shared source changed while the run was active: compatibility-source SHA-256 changed from
`da03f405...e6704` to `58358c39...a939`, and replay-evidence source SHA-256 changed from
`4f5bd8ec...c9ce0` to `7cc32841...77e1`. The three exact packet SHA-256 values therefore differed.
Failure-summary SHA-256 is `8501b7681262ceca002659978c07c688a6f7baa45923ebb3c06e6134adabebe4`;
the tracked failure-artifact file SHA-256 is
`ea9b7323e9565d2f2514c53849d53ebb503bb924d9b5e857923ef4676437e05b`.

**Final project stop:** the exact same-process gate failed, so fresh-process replay, both two-step
smokes, G1/G2 training, retention selection, independent retention, GSM1K evaluation, bootstrap,
and signal adjudication were not run. The Milestone 10E stop rule closes verifier-GRPO for this
project version. Do not retry the replay, train an adapter, or open another GRPO configuration.

## Milestone 10G authorization: immutable runtime-root decoupling

Milestone 10G explicitly authorizes one orchestration-only correction before replaying the unchanged
scientific contract. `GrpoRuntimePaths` now freezes five independent identities: the detached source
root, mutable primary repository, exact training interpreter, external writable artifact root, and
read-only model cache. Fresh processes use the configured absolute interpreter; all generated state
must resolve beneath the external artifact root; imports must resolve beneath the detached source;
and source, executable, cache, environment, and command identities are revalidated before and after
execution.

The original `165` focused GRPO tests still pass under `PYTHONHASHSEED=20260720`; `17` new path and
orchestration cases bring the focused total to `182`. The patch changes no generator, sampler,
reward, reference-policy, optimizer, schedule, retention, evaluator, synthesis, or dependency file.
The approved CPython executable remains SHA-256
`0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14`, and the planned V2 process
command-template SHA-256 is
`6680c2c4d713882877d1c7e2ab1c47211ec07f2c84cee0464964e4de7b1d3498`.

After the atomic orchestration commit is pushed, the only authorized continuation is a detached V2
worktree plus external V2 runtime manifest. The actual runtime-path-contract hash is frozen there
because it binds the new commit and complete tracked-source manifest. No model generation,
backward pass, optimizer step, checkpoint, adapter, retention evaluation, or GSM1K evaluation has
occurred during this patch phase.

## Milestone 10G outcome: immutable replay stopped on environment validation

The orchestration patch was committed and pushed as
`b647a3dcadcab941359fbecab2b11c8f9f63cb8d`. The detached V2 worktree froze tree
`099a9987df1b0a2d4da85eba33b4e22694ef2ab6`, runtime-path contract
`2400654e155ba7be36aba99ffc4cf7588f80d726ffed59074f0f9955b948d953`, and complete 491-file
source manifest `72cd61b5f374f95bc7b0dbc1e51c0cafa81ca2cf3979d3d51421f2a1af4e2fab` outside Git.

The first official same-process generation replay completed its frozen three-group, 12-completion
in-memory workload, then failed before its packet could be written. The new path contract treated
the launch-time cuBLAS value `:4096:8` as a lifetime invariant, while the already frozen stock
Transformers full-determinism transition correctly changed it to `:16:8` before model operations.
Post-run validation therefore raised `RuntimeError`. No packet comparison, fresh-process replay,
two-step smoke, backward pass, optimizer step, adapter, checkpoint, retention evaluation, or GSM1K
evaluation occurred.

**Final Milestone 10G stop:** this is an orchestration-contract failure, not evidence of a model-side
replay mismatch. Nevertheless, the authorization forbids patching or retrying after an official
replay failure. The source manifest, model manifest, detached worktree, and primary repository all
remained unchanged. Failure-summary SHA-256 is
`0a1c7085a95fef8138c06b17faaa8e0b5c0af195148012ca9a88c7a07a6d1eeb`; its tracked file SHA-256
is `d38741f5e24c63279994b2cfd983cb2005c8a5e7d141a30d84dde96585163bb4`.

## Milestone 10H plan: one source-immutable V3 experiment

An explicit correction authorization supersedes the Milestone 10G project stop for one new V3
experiment only. The orchestration patch now constructs a typed, secret-free deterministic
environment before Python starts. cuBLAS begins and remains at `:16:8`; the other four values
written by the frozen Transformers helper are likewise predeclared and immutable.

Patch verification is complete: the installed helper and file hashes match, all `198` focused GRPO
tests and all `709` repository tests pass, strict Mypy and Ruff pass, and protected scientific and
dependency paths have zero diff. After the exact environment-fix commit is pushed, create only
`Foundry-grpo-frozen-v3` and `Foundry-grpo-runtime-v3`, freeze the external path/environment/source
manifests, and proceed automatically through replay, duplicate two-step compatibility, conditional
G1/G2, retention, and GSM1K gates while each preceding gate passes.

## Milestone 10H outcome: V3 failed before model loading

The environment fix was committed as `2254b22aa10c9f024eebd56c1f1b98b9a3cf16ab`. V3 froze tree
`da9939e50adb11d523fc00dec53a8350df5866d2`, runtime-path contract
`6154aecda902d6a4f9a9773a68f4da873d52e3474acb6cced10aee3a4291761a`, complete 495-file source
manifest `f9f481186f3fb2e4e1c2c44b1d281069910f302a99738fd1a420930977e4c729`, and unchanged model
manifest `5173393ff459ebe94d4019bf76e129a88022af448e1f24e954a8b9d291184006`.

The official same-process process received only the explicit 30-field allowlist. Its pre-model CUDA
contract check invoked `nvidia-smi --query-gpu=driver_version`; NVML initialization failed with exit
`255`. The query succeeds under the parent environment, proving that removal of a non-allowlisted
field changes NVML behavior. No model load or generation occurred, and no replay packet or summary
was written.

**Final Milestone 10H stop:** do not retry V3, patch the allowlist, run fresh-process or two-step
replay, train G1/G2, select retention checkpoints, or evaluate GSM1K. Failure-summary SHA-256 is
`b5f0e4b21b496b47a9ae5a93a42d9d9c39bb81b5e2fa7b4ddd36c7432464c2bf`. A future experiment would
require new explicit project-level authorization and a predeclared GPU-process environment.

## Milestone 10I plan: one source-immutable V4 experiment

The explicit V4 authorization replaces the child-process NVML prerequisite with a direct frozen
PyTorch CUDA compute gate. Normal-parent `nvidia-smi` evidence is observational only. The child must
use the exact deterministic launch environment, import Foundry from the detached V4 source tree,
identify the expected RTX 3080, allocate and compute on `cuda:0`, synchronize, and reproduce one
fixed result hash across three runs without NVML, CPU fallback, or model loading.

Patch verification is complete: `213/213` focused GRPO tests and `724/724` repository tests pass;
Ruff and strict Mypy pass; and generation, sampling, reward, reference, KL, optimizer, schedules,
LoRA, retention, evaluation, datasets, and dependencies are unchanged. After publishing
`fix: validate GRPO GPU through CUDA runtime`, create only `Foundry-grpo-frozen-v4` and
`Foundry-grpo-runtime-v4`, freeze all source/environment/interpreter/cache/host/child evidence, and
continue through the unchanged replay, duplicate two-step, conditional G1/G2, retention, and GSM1K
gates only while every preceding gate passes.

## Milestone 10I outcome: V4 stopped at first complete two-step smoke

V4 froze commit `a13c31b43a72c3bec205e440aaf7c424ac487d47`, tree
`b938e97308d3e73493cf066ed7a656363657f4cb`, runtime contract `a8543712...bc0a`, source
manifest `dda8cf58...a8b8`, environment `0a5bd3bb...e55d`, and unchanged model manifest
`5173393f...4006`. Parent GPU monitoring succeeded, and the child direct-CUDA result hash
`f8850fe4...e5af6` repeated three times without NVML or CPU fallback.

Three same-process and three fresh-process generation replays matched exactly at packet
`084515f9...ee2f`. The first two-step G1 process reached model generation under gradient
checkpointing, then its strict warning audit found multiple normalized warning classes and raised
before backward or optimizer step 1. No two-step packet, metadata, adapter, or checkpoint exists.

**Final Milestone 10I stop:** do not run the duplicate smoke, G1/G2, retention, GSM1K, category
analysis, bootstrap, or signal gate. Preserve V4 and publish failure summary
`164d3e35828758d4eff77b21919b9b3b28dee6238135478fcd2b2e5e024c6f91` as
`analysis: stop verifier GRPO after V4 replay failure`.

## Milestone 10J outcome: immutable warning audit closed the V5 route

Milestone 10J reopened only an evidence-first audit of V4. Primary and V4 identities, stderr hash,
all six replay packets, and both replay summaries revalidated exactly without loading the model.
The complete stderr contains four warning-level classes with one occurrence each. Source-pinned
classification yields B for the PEFT label-name notice, C for the automatic call-local cache
transition, and E for both the unsupported SDPA/sliding-window notice and the DynamicCache runtime
version uncertainty.

Separately, the V4 auditor proves that at least two Python warning classes were captured inside its
generation-only `catch_warnings` scope. Their raw text, categories, source locations, counts,
normalized hashes, and class IDs were never serialized. Because the authorization makes any
unclassifiable warning fatal and forbids benignity inference, no phase-aware contract or
equivalence fixture may be created.

**Final Milestone 10J stop:** preserve V1-V4; do not create V5, rerun replay, train G1/G2, evaluate
retention or GSM1K, or run the signal gate. Publish warning audit
`a3e4d1ca40c3fb3f9fe984d3a019ed064a6ba96394a69b009257a248eebf1602` as
`analysis: stop verifier GRPO after training-warning audit`.

## Fast-Track Phase 2 Milestones 12A-12D: vetted human-written curricula

Phase 2 opens from the clean synchronized Phase 1 release at
`f4ee93afa4c2be52ca21aef8ca16dbf5827b4a99`. Phase 1 established a positive curriculum-selection
contrastâ€”targeted synthetic SFT exceeded matched generic synthetic SFT by `27/814`â€”but both
adapters remained below the untouched base. The next experiment therefore isolates selection
quality from generated question wording.

The primary source is a pinned official ASDiv revision stored only under ignored external-data
paths. Foundry may normalize whitespace, verify formulas and answers, classify supported families,
screen contamination, evaluate the untouched base, select disjoint targeted and generic pools,
construct deterministic formula-grounded targets, and train the exact authorized retention-first
LoRA protocol. Foundry must not rewrite corpus wording, use GSM1K for selection, access sealed-final
content, add an unapproved corpus, or change a failed gate after observing results.

Execution is sequential and fail-closed: source/provenance, formula verification, contamination,
capacity, base-pool evaluation, matching, dataset freeze, training, retention, and only then the
frozen GSM1K development evaluation. Each completed publication boundary receives one atomic
commit and push. No Phase 1 summary, evaluator, manifest, figure, report, or result may change.

**Stage B passed:** ASDiv is pinned at commit `883f90a9...abc47`, tree `2c3e8723...e52ac`, with
`2,305` source rows and raw XML SHA-256 `ef890406...c4929`. The official README states CC BY-NC
4.0 and the ACL 2020 attribution. MathQA remains inactive. Next, implement and test the restricted
exact ASDiv parser before any model evaluation.

**Stage C passed:** the exact verifier accepted `1,497` rows and classified `1,452` into a
supported Foundry family. It rejected `808` rows without weakening the grammar. Supported counts
are `1,126` bookkeeping, `118` rate/ratio, and `208` discrete; the full run and replay share
summary hash `6c45b435...895d`. Next, screen all `1,452` supported rows against the frozen
development inventory, Phase 1 synthetic questions, and each other.

**Stages D-E passed:** the fixed contamination screen rejected `73` candidates at development
semantic similarity `>=0.75` and left `1,379` clean rows. Exact, 12-token, number-neutral,
structure, source, duplicate, Phase 1 overlap, and unresolved counts are zero. Clean families are
`1,076/111/192` for bookkeeping/rate/discrete. ASDiv alone misses the smallest combined rate quota
by three rows, but the authorization requires the untouched-base failure census before fallback.
Next, evaluate all clean ASDiv candidates with the pinned base and frozen pool-inference contract.

**Stages F-G passed:** the untouched base processed all `1,379` clean ASDiv rows with zero backend
failures and exact 30-row replay. It scored `1,167/1,379` (`84.6265%`) and failed only
`152/22/38` bookkeeping/rate/discrete rows, activating the predeclared MathQA fallback. The pinned
official MathQA train artifact produced `15,468` exactly verified rows; the deterministic
pre-inference subset retained `5,000`, contamination screening retained `4,929`, and the base
processed all `4,929` with zero backend failures and exact replay. It scored `2,363/4,929`
(`47.9408%`) and failed `1,214/1,136/216` rows by family.

**Stage H failed; Phase 2 stopped:** the fixed selector tested `300`, `250`, then `200` per arm.
The two larger designs exceeded numerical and categorical balance limits. At `200`, all
categorical levels and exact source composition passed, but formula-depth SMD was `0.113895` and
operation-count SMD was `0.108765`, above the frozen `0.10` maximum. Do not alter the selector,
quotas, sources, or thresholds; do not construct targets or splits, train adapters, run retention,
or evaluate GSM1K. A future continuation requires new explicit authorization.

## Fast-Track Milestone 12E: matching repair and vetted dataset freeze

The separately authorized matching-only repair preserved the Stage H stop, froze all eligible
failures and covariates, and exhaustively evaluated every legal single-row replacement. Generic
`mathqa-train-26455` was replaced by `mathqa-train-28853`; all four SMDs are now at most `0.10`,
the categorical maximum is `0.05`, source composition and family quotas are unchanged, and all
duplicate and contamination gates pass. Two-row and global fallback stages were not run.

Deterministic formula/program-derived targets and 180/20 splits now replay byte-identically under
dataset identity `ee18f7f9...dc31`. The next authorized stage is V1 REPLAY25 training for the
generic and targeted arms, followed only by retention-based checkpoint selection. GSM1K adapter
evaluation remains closed until independent final retention passes.

**Training environment stopped before model load:** the recovery-required `.venv` has CUDA
PyTorch but lacks PEFT, bitsandbytes, and TRL. Package installation/modification and switching
interpreters are outside the authorization. V1 therefore has zero model loads and optimizer steps.
Resume only after explicit authorization identifies an immutable compatible training environment.
