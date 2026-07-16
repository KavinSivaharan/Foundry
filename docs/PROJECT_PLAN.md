# Foundry Project Plan

Last updated: 2026-07-16

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

Status: complete on the available machine; real CUDA smoke deferred because this is not the RTX 3080 desktop.

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

### Milestone 2 — Base development benchmark

- Freeze prompt and decoding settings.
- Evaluate the base model on the development split.
- Record accuracy, category counts, runtime, peak VRAM, and parsing failures.

### Milestone 3 — Failure taxonomy

- Implement deterministic, reviewable failure categories.
- Audit a sample manually without exposing benchmark labels or examples to synthesis.

### Milestone 4 — Synthetic data and verification

- Generate a small targeted batch from structured arithmetic programs.
- Add independent solution verification, constraint checks, labeling, exact deduplication, semantic overlap checks, and benchmark-overlap rejection.
- Audit acceptance/rejection reasons.

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

- Exact normalized final-integer accuracy on the locked GSM1K final split.

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

Measured baseline and candidate scores: **not yet available; no experiment has run.**

## Current project phase

Milestone 1 is complete for the GPU-independent evaluation foundation. The repository now has pinned configurations, identifier-only manifests, stable prompting, strict scoring, fake-model integration coverage, an optional CUDA backend, and reproducible development/smoke dependency locks.

No model or benchmark was downloaded, no real-model evaluation ran, and no training occurred. The approved 10-example CUDA smoke remains deferred until the repository is opened on the RTX 3080 desktop.

## Unresolved questions

1. Is the target RTX 3080 the 10 GB or 12 GB model, and what OS/CUDA/driver versions are available?
2. Is local disk sufficient for model cache, quantized runtime, adapters, and raw predictions?
3. Does PyTorch 2.5.1 with its CUDA 12.1 wheel work with the target driver, and what peak VRAM does the 10-example float16 smoke consume?
4. What generation throughput and invalid-answer rate does the pinned base model produce on the RTX 3080?
5. Should the Milestone 2 development benchmark process all 904 examples in one approved run or begin with a staged subset after the smoke measurement?
6. What maximum sequence length and generation limit fit the observed prompt/solution distribution?
7. Should the first synthetic generator use templates only, or later allow an approved local/paid paraphraser behind the same verifier?
8. Which small embedding model and threshold should implement semantic-overlap rejection without introducing an excessive dependency or false positives?
9. Is a 3-point final improvement statistically realistic after the development baseline, or should the success threshold be revised before training?

## Next approved milestone

None. Before Milestone 2, the recommended next action is to open this repository on the RTX 3080 desktop and explicitly authorize the already-defined 10-example deferred CUDA smoke. After that result is reviewed, Milestone 2 would run the pinned base model over the approved development scope, record the untouched baseline predictions and performance, and create the first evidence-backed failure inventory. No Milestone 2 work is authorized yet.
