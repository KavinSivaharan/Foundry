# Foundry Experiment Log

No model experiment has completed yet. Milestone 1 ran only unit tests and a synthetic fake-model integration test; those are software verification, not benchmark evidence.

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

- **Status:** approved but deferred; not executed because the detected machine has no NVIDIA GPU or CUDA
- **Date:** 2026-07-16
- **Hypothesis:** The pinned base model and evaluation pipeline can process 10 GSM1K development examples on an RTX 3080 without an out-of-memory error while recording deterministic predictions, runtime, throughput, token counts, and peak VRAM.
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Exact model revision:** `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- **Dataset:** `ScaleAI/gsm1k`, configuration `default`, source split `test`
- **Exact dataset revision:** `bc09569d09a614b9b530edc7f076fb214ac10493`
- **Training configuration:** none; this is evaluation only.
- **Evaluation configuration:** `configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml`; config SHA-256 `2a6e737cf3376ae081fd17600e31937824830ecdbb624644e729f6b5752f8eba`; development manifest SHA-256 `d2c895f43a1e76a12796d6a263b60dc230a9abab58f9624674ec925f37319fae`; prompt SHA-256 `738ea5a3b94e7c75ac0bd50a229bbf04f3fc5d773e14658bc6728bc7a4b18350`; greedy decoding; maximum 512 new tokens.
- **Hardware:** target RTX 3080 desktop details not yet detected. The available machine was macOS 15.6.1, Apple M2 with 8 GB unified memory, and is ineligible.
- **Random seed:** no generation sampling; manifest seed `foundry-gsm1k-v1`.
- **Baseline score:** not measured.
- **Resulting score:** not measured.
- **Cost and runtime:** $0 API/cloud cost; no model runtime or VRAM measurement because execution was refused before download.
- **Artifacts/checkpoint:** planned summary under `results/smoke/qwen2_5_1_5b/summary.json`; ignored raw predictions under the corresponding `raw/` directory. No checkpoint.
- **Code commit:** the local Milestone 1 commit containing this record.
- **Interpretation:** No claim about model accuracy, throughput, or RTX feasibility can be made yet. The CUDA preflight correctly prevented an ineligible machine from downloading the model.
- **Hypothesis supported:** inconclusive; the experiment did not run.
- **Failures/blockers:** no NVIDIA GPU, `nvidia-smi`, CUDA toolkit, or smoke dependencies on the detected Apple Silicon machine; only about 14 GiB free disk.
- **Next experiment:** run this exact 10-example smoke on the RTX 3080 before requesting the full Milestone 2 development benchmark.
