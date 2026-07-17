# Foundry Experiment Log

Two bounded real-model experiment groups have completed: the ten-example Milestone 1 CUDA smoke and the 30-example/three-prompt Milestone 1.5 format calibration recorded below. They are software, hardware, and format-control evidence—not a full benchmark result or a training experiment.

Every future experiment must be registered here before a costly run begins. Its machine-readable configuration must be saved under `configs/`, raw evaluation outputs under `results/raw/`, and reviewable summaries under `results/`.

## Experiment template

### EXP-YYYYMMDD-NNN — Short name

- **Status:** proposed | approved | running | completed | failed | cancelled
- **Date:** YYYY-MM-DD
- **Hypothesis:** A falsifiable statement written before the run.
- **Model:** repository/model identifier.
- **Exact model revision:** immutable commit hash; never `main` alone.
- **Dataset:** repository/dataset identifier and configuration.
- **Exact dataset revision:** immutable commit hash plus split manifest hash.
- **Training configuration:** path to the machine-readable config under `configs/`; include method, sequence length, batch size, gradient accumulation, optimizer, learning rate, schedule, epochs/steps, precision/quantization, LoRA settings, checkpoint rule, and all seeds.
- **Evaluation configuration:** prompt/template hash, decoding values, answer parser version, evaluator commit, example IDs or split-manifest hash.
- **Hardware:** GPU model and VRAM, CPU, RAM, OS, driver, CUDA, and relevant kernel/backend details.
- **Random seed:** data, training, and generation seeds.
- **Baseline score:** metric, value, sample count, and uncertainty under the identical evaluation configuration.
- **Resulting score:** metric, value, sample count, uncertainty, and paired difference from baseline.
- **Cost and runtime:** wall time, GPU time, peak VRAM, processed tokens, throughput, API/cloud spend, and local cost estimate when available.
- **Artifacts/checkpoint:** local path or approved artifact reference; do not commit large artifacts.
- **Code commit:** Git commit used for the run and worktree-cleanliness status.
- **Interpretation:** what the result supports, what it does not prove, regressions, and anomalies.
- **Hypothesis supported:** yes | no | inconclusive, with reason.
- **Failures/blockers:** failed commands, parser failures, OOMs, data issues, and deviations from plan.
- **Next experiment:** the smallest justified follow-up; it requires approval if it is a new milestone or costly run.

## EXP-20260716-001 — Qwen GSM1K 10-example CUDA smoke

- **Status:** completed on the approved RTX 3080 desktop; evaluation only
- **Date:** 2026-07-16
- **Hypothesis:** The pinned base model and evaluation pipeline can process 10 GSM1K development examples on an RTX 3080 without an out-of-memory error while recording deterministic predictions, runtime, throughput, token counts, and peak VRAM.
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Exact model revision:** `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- **Dataset:** `ScaleAI/gsm1k`, configuration `default`, source split `test`
- **Exact dataset revision:** `bc09569d09a614b9b530edc7f076fb214ac10493`
- **Training configuration:** none; this is evaluation only.
- **Evaluation configuration:** `configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml`; config SHA-256 `2a6e737cf3376ae081fd17600e31937824830ecdbb624644e729f6b5752f8eba`; development manifest SHA-256 `d2c895f43a1e76a12796d6a263b60dc230a9abab58f9624674ec925f37319fae`; prompt SHA-256 `738ea5a3b94e7c75ac0bd50a229bbf04f3fc5d773e14658bc6728bc7a4b18350`; greedy decoding; maximum 512 new tokens.
- **Hardware:** Windows 11 Pro 64-bit, version 10.0.26200 build 26200; AMD Ryzen 7 9700X (8 cores, 16 logical processors); 31.11 GiB system RAM; NVIDIA GeForce RTX 3080 with 10,240 MiB VRAM; NVIDIA driver/KMD 610.47 reporting CUDA UMD 13.3; CPython 3.12.10; PyTorch 2.5.1+cu121 reporting CUDA runtime 12.1 and `torch.cuda.is_available() == True`. Initial free `C:` space was 218.19 GiB and 207.57 GiB remained after environment/cache creation.
- **Random seed:** no generation sampling; manifest seed `foundry-gsm1k-v1`.
- **Baseline score:** smoke-only base-model result: 2/10 correct (20% accuracy) on ten development identifiers. Seven outputs were invalid under the strict final-answer parser (70% invalid rate), one additional output was validly parsed but incorrect, and no generation failed.
- **Resulting score:** not applicable; no adapter, candidate model, training, or base-versus-candidate comparison occurred.
- **Cost and runtime:** $0 API/cloud cost. First-run model download/load time was 120.6527 seconds. Recorded evaluation time was 36.8997 seconds; load-plus-evaluation time was 157.5524 seconds; throughput was 0.2710 examples/second and 73.74 generated tokens/second. The run processed 1,387 input tokens and generated 2,721 output tokens. Peak CUDA memory was 3,116,510,720 bytes (2,972.14 MiB, 2.902 GiB) allocated and 3,315,597,312 bytes (3,162 MiB, 3.088 GiB) reserved. Local model cache size was 3,098,955,668 bytes (2.886 GiB); dataset Hub plus materialized caches totaled about 778,026 bytes (0.74 MiB).
- **Artifacts/checkpoint:** aggregate summary at `results/smoke/qwen2_5_1_5b/summary.json`; ignored raw predictions at `results/raw/smoke/qwen2_5_1_5b/predictions.jsonl`; Hugging Face cache under ignored `data/huggingface`. No checkpoint was created.
- **Code commit:** evaluator/configuration source was commit `f9f579f`; the worktree contained only in-progress smoke documentation when the run began, with no source, config, prompt, manifest, or lock changes.
- **Interpretation:** The approved CUDA 12.1 PyTorch wheel works with driver 610.47, the pinned model/dataset/evaluator completed exactly ten development examples without CUDA, dependency, generation, or out-of-memory failure, and float16 evaluation fits comfortably within 10 GiB. The memory result makes a conservative short-sequence QLoRA pilot plausible, but it does not measure training memory and cannot approve or prove QLoRA feasibility by itself. Seven format failures reveal that the base model often did not follow the strict final-answer contract. The 2/10 score is a software/hardware smoke observation, not a meaningful benchmark conclusion.
- **Hypothesis supported:** yes for the narrow smoke hypothesis: all ten approved development examples completed on the RTX 3080 and runtime, tokens, failures, and peak VRAM were recorded without OOM.
- **Failures/blockers:** no blocking failure. Non-fatal warnings: Hugging Face could not use cache symlinks on Windows; Transformers reported sampling defaults that were ignored because `do_sample=False`; the macOS-generated lock omitted Windows-only `colorama==0.4.6` and `tzdata==2026.3`, which pip resolved without changing a pinned package. Seven responses were invalid because they did not contain exactly one `Final answer:` line.
- **Next experiment:** Milestone 2 would evaluate the approved development scope and establish a base failure inventory, but it remains unapproved and must not begin without explicit user authorization.

## EXP-20260717-002 — Qwen GSM1K 30-example prompt-format calibration

- **Status:** completed; admission hypothesis not supported and no prompt selected
- **Date:** 2026-07-17
- **Hypothesis:** At least one of the current prompt and at most two minimal format-only revisions will produce at least 90% valid outputs on 30 predeclared development calibration identifiers, with zero generation failures and no unreasonable output-length increase.
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Exact model revision:** `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- **Dataset:** `ScaleAI/gsm1k`, configuration `default`, source split `test`
- **Exact dataset revision:** `bc09569d09a614b9b530edc7f076fb214ac10493`
- **Training configuration:** none; prompt-format evaluation only.
- **Evaluation configuration:** fixed 30-ID calibration manifest SHA-256 `a020b74b626e75c1197abc35942e85d929463cfe2bfaac1364806bcab1743ee4`, selected deterministically with seed `foundry-gsm1k-prompt-format-calibration-v1` from canonical development manifest `d2c895f43a1e76a12796d6a263b60dc230a9abab58f9624674ec925f37319fae`. The disjoint future-baseline manifest contains 874 IDs and has SHA-256 `d6bb412367a44b6c9fc1695bfa856650c42f90a8ee942223c010511c10f7e1eb`. All variants used greedy decoding, 512 maximum new tokens, and the unchanged strict parser. Prompt hashes: current `738ea5a3b94e7c75ac0bd50a229bbf04f3fc5d773e14658bc6728bc7a4b18350`; `format_v1` `de85fb299156e284d34f51f74983a19b564fd5725d000bc0dac10186e274fcbc`; `format_v2` `a17f10b85f491b865b2c9cc8e4b0b9f2550eae13259f308502591023f1fa9324`.
- **Hardware:** Windows 11 Pro build 26200; AMD Ryzen 7 9700X; 31.11 GiB RAM; NVIDIA GeForce RTX 3080 with 10,240 MiB; driver 610.47; CPython 3.12.10; PyTorch 2.5.1+cu121 reporting CUDA runtime 12.1.
- **Random seed:** deterministic greedy generation; calibration selection seed `foundry-gsm1k-prompt-format-calibration-v1`.
- **Baseline score:** current prompt: 5/30 valid (16.67%), 25/30 invalid (83.33%), 1/30 correct (3.33%), 305.97 average output tokens, 124.190 seconds, 0.2416 examples/second, 2,983.43 MiB peak allocated and 3,192 MiB peak reserved VRAM.
- **Resulting score:** `format_v1`: 3/30 valid (10.00%), 27/30 invalid (90.00%), 2/30 correct (6.67%), 287.37 average output tokens, 114.161 seconds, 0.2628 examples/second, 2,983.60 MiB allocated and 3,192 MiB reserved. `format_v2`: 13/30 valid (43.33%), 17/30 invalid (56.67%), 3/30 correct (10.00%), 230.60 average output tokens, 91.467 seconds, 0.3280 examples/second, 2,990.10 MiB allocated and 3,192 MiB reserved. Every run had zero generation failures.
- **Cost and runtime:** $0 API/cloud cost. Exactly 90 real-model generations; 14,637 total input tokens and 24,718 total output tokens. Aggregate recorded evaluation time was 329.817 seconds and load-plus-evaluation time was 335.402 seconds. Maximum observed CUDA reservation was 3,192 MiB.
- **Artifacts/checkpoint:** aggregate summaries under `results/calibration/{current,format_v1,format_v2}/summary.json`; ignored raw predictions under `results/raw/calibration/{current,format_v1,format_v2}/predictions.jsonl`; identifier-only manifests under `configs/eval/manifests/`. No checkpoint.
- **Code commit:** runs began from published commit `c1ef561`; Milestone 1.5 adds only bounded calibration selection/evaluation plumbing, prompt alternatives, tests, summaries, and documentation.
- **Interpretation:** `format_v2` materially improved validity and reduced average output length, but 43.33% remains far below the predeclared 90% gate. The result does not justify selecting a prompt, loosening the parser, or beginning the main baseline. Accuracy was secondary and did not determine selection.
- **Hypothesis supported:** no; none of the three prompts reached 90% valid output. No generation failure occurred, but compliance remained inadequate.
- **Failures/blockers:** current failure categories: 7 boxed, 15 prose/inline, 1 alternate label, 1 alternate wording, 1 incomplete at 512 tokens. `format_v1`: 4 boxed, 15 prose/inline, 4 alternate labels, 3 alternate wording, 1 incomplete. `format_v2`: 1 boxed, 10 prose/inline, 2 alternate labels, 4 alternate wording, no token-limit hit. One read-only PowerShell categorization command had a syntax error and was corrected without affecting artifacts.
- **Next experiment:** none approved. A new format-control proposal is required before Milestone 2; it must preserve the strict parser and the 30/874 calibration/baseline separation.
