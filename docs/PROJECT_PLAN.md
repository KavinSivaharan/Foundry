# Foundry Project Plan

Last updated: 2026-07-18

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
3, bounded Milestones 4 through 4.2, design-only Milestone 5A, and bounded Milestone 5B are complete.
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
clean IRs. Full dataset generation and every training stage remain blocked; no complete synthetic
dataset or adapter exists.

## Unresolved questions

1. Should the project stop local-model realization, or approve a design-only compact-protocol
   blocker-resolution milestone that preserves Qwen3, the procedural labels/verifiers, and all
   contamination gates while replacing the verbose response contract before any new inference?
2. Should the cross-platform dependency locks explicitly pin Windows-only `colorama` and `tzdata` in a separately approved lock-maintenance task?
3. Any future comparison must preserve the exact 814-ID manifest and frozen prompt/extractor/generation configuration unless the user explicitly authorizes a new evaluator lineage and complete reruns.
4. The pure procedural renderer is closed, and the first constrained local-model protocol failed
   its fixed readiness gate. Any compact-protocol design or later inference needs explicit approval.
5. The pinned MiniLM encoder behaved acceptably on original fixtures; future work must retain its exact revision/configuration unless a separate design decision replaces the semantic lineage.
6. Is a 3-point final improvement statistically realistic after the development baseline, or should the success threshold be revised before training?

## Next approved milestone

No further milestone is approved. Milestone 5B ends after its verified evidence commit is pushed.
The exact next decision is whether to approve a design-only compact realization-protocol milestone
or stop this synthesis route. The proposed design work would not download another model or run
inference; it would replace verbose placeholder-inventory/clause-map echoing with a shorter
deterministically recoverable contract and freeze original fixtures before any later bounded smoke.
The 4,000 + 4,000 pilot, training, SFT, QLoRA, GRPO, paid services, benchmark inference, fallback
model, and sealed-final access remain unapproved.
