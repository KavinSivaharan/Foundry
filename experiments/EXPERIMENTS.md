# Foundry Experiment Log

One real-model experiment has completed: the approved ten-example Milestone 1 CUDA smoke recorded below. It is software/hardware integration evidence, not a full benchmark result or a training experiment.

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
