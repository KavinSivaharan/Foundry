# Foundry Experiment Log

Five bounded real-model experiment groups have completed: the ten-example Milestone 1 CUDA smoke, the 30-example/three-prompt Milestone 1.5 format calibration, the 30-example fresh Milestone 1.6 answer-validation run, the final 30-example Milestone 1.7 evaluator validation, and the frozen 814-example Milestone 2 base-development baseline recorded below. No training experiment or sealed-final benchmark result exists.

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

## EXP-20260717-003 — Deterministic answer extraction and fresh validation

- **Status:** completed; fresh admission hypothesis not supported and evaluator not frozen
- **Date:** 2026-07-17
- **Hypothesis:** A deterministic generic terminal-integer extractor, calibrated on the existing 90 outputs and manually audited, will produce at least 90% extractable answers with zero false extractions and zero generation failures when the best existing prompt is run on 30 fresh development identifiers.
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Exact model revision:** `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- **Dataset:** `ScaleAI/gsm1k`, configuration `default`, source split `test`
- **Exact dataset revision:** `bc09569d09a614b9b530edc7f076fb214ac10493`
- **Training configuration:** none; deterministic re-scoring plus one bounded fresh evaluation.
- **Evaluation configuration:** strict parser retained for exact compliance; canonical extractor `foundry-terminal-integer-v1` SHA-256 `ffce6538526f9aa21e05ce4d9d6830ec71d3a6334a23fa1e9c7beef3c2053946`; selected current prompt SHA-256 `738ea5a3b94e7c75ac0bd50a229bbf04f3fc5d773e14658bc6728bc7a4b18350`; config SHA-256 `2a6e737cf3376ae081fd17600e31937824830ecdbb624644e729f6b5752f8eba`; greedy decoding; 512 maximum new tokens. Fresh validation manifest: 30 IDs, SHA-256 `9582e4b07706a391f00dcfa0d9c68ee86a70cebca6c35f10daa3f3f66c9063f6`, seed `foundry-gsm1k-answer-extraction-validation-v1`. Candidate main-baseline manifest: 844 IDs, SHA-256 `4f80bbe2f6de4fb33e57ed5463a1c393716a3b64d3b98a08767a7f8056648d79`.
- **Hardware:** Windows 11 Pro build 26200; AMD Ryzen 7 9700X; 31.11 GiB RAM; NVIDIA GeForce RTX 3080 with 10,240 MiB; driver 610.47; CPython 3.12.10; PyTorch 2.5.1+cu121 with CUDA runtime 12.1.
- **Random seed:** deterministic greedy generation; validation selection seed `foundry-gsm1k-answer-extraction-validation-v1`.
- **Baseline score:** Existing-output re-score: current 29/30 extractable (96.67%), 5/30 exact compliant (16.67%), 15/30 correct (50.00%); `format_v1` 28/30 extractable (93.33%), 3/30 exact compliant (10.00%), 15/30 correct (50.00%); `format_v2` 27/30 extractable (90.00%), 13/30 exact compliant (43.33%), 13/30 correct (43.33%). All 63 newly accepted calibration outputs were audited with zero false extractions.
- **Resulting score:** Fresh current-prompt validation: 23/30 extractable (76.67%), 3/30 exact compliant (10.00%), 14/30 correct (46.67%), 7 rejected, zero generation failures, and zero false extractions after auditing all 30 outputs. Rejections: 3 token-limit, 2 non-integral decimal, and 2 clear-but-unsupported terminal phrasings.
- **Cost and runtime:** $0 API/cloud cost. Exactly 30 new generations; 3,948 input tokens and 8,159 output tokens. Evaluation time 107.996 seconds; total load-plus-evaluation time 110.269 seconds; 0.2778 examples/second; 271.97 average output tokens. Peak CUDA memory: 2,978.35 MiB allocated and 3,172 MiB reserved.
- **Artifacts/checkpoint:** aggregate re-score summaries under `results/extraction_calibration/`; aggregate fresh summary and content-free audit under `results/answer_validation/current/`; ignored fresh raw predictions under `results/raw/answer_validation/current/predictions.jsonl`; identifier-only validation/baseline manifests under `configs/eval/manifests/`. No checkpoint.
- **Code commit:** re-scoring used predictions generated from published commit `5e873f1`; fresh validation ran in the Milestone 1.6 worktree before the final local commit, with prompt/model/dataset/generation configuration unchanged.
- **Interpretation:** The extractor correctly separates exact-format compliance from intended integer extraction and made no audited false extraction, but its calibrated 96.67% rate did not generalize. Three truncations and two non-integral outputs mean accepting the two unsupported clear integer phrasings would still reach only 83.33%. The evaluator is not reliable enough for the 844-ID baseline.
- **Hypothesis supported:** no; fresh extractability was 76.67%, below 90%, despite zero false extractions and generation failures.
- **Failures/blockers:** three outputs reached the 512-token limit; two produced non-integral decimals; two clear integer conclusions were outside the frozen grammar. Transformers emitted non-fatal warnings for sampling defaults ignored under greedy decoding. No CUDA, OOM, dependency, backend, or raw-artifact containment failure occurred.
- **Next experiment:** none approved. A blocker-resolution proposal would need predeclared grammar/generation changes and a new untouched admission set before Milestone 2 can be reconsidered.

## EXP-20260717-004 — Final deterministic evaluator blocker resolution

- **Status:** completed; final fresh admission hypothesis not supported and evaluator not frozen
- **Date:** 2026-07-17
- **Hypothesis:** After narrowly supporting explicit terminal numeric formats and conservatively increasing the confirmed truncation limit from 512 to 768, the frozen current-prompt evaluator will extract at least 90% of answers on one final untouched 30-ID development set with zero false extractions, zero backend generation failures, and reasonable length/runtime.
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Exact model revision:** `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- **Dataset:** `ScaleAI/gsm1k`, configuration `default`, source split `test`
- **Exact dataset revision:** `bc09569d09a614b9b530edc7f076fb214ac10493`
- **Training configuration:** none; deterministic re-scoring, three truncation diagnostics, and one bounded final evaluation only.
- **Evaluation configuration:** current prompt SHA-256 `738ea5a3b94e7c75ac0bd50a229bbf04f3fc5d773e14658bc6728bc7a4b18350`; unchanged strict parser for exact compliance; canonical extractor `foundry-terminal-number-v2` SHA-256 `e099d1c247968fed982cb849022ec3137b1694c15f23a65663a127b8158c06df`; final config `configs/eval/gsm1k_qwen2_5_1_5b_final_evaluator.yaml` SHA-256 `5f315d5de645f9563b8d1e61bc8e02c3513c453238ad9e1d6f9473489b5a622b`; greedy decoding; 768 maximum new tokens and no other generation change. Final validation manifest: 30 IDs, SHA-256 `2234e5ee82cf57e8fb74839a21f7f0ca0d2ff02ddd0fb0e42d93934415b2db93`. Candidate main baseline: 814 IDs, SHA-256 `5e810d3ab644bef1d43c598a14a6164ba6464b27fde50e92a2f241816ce87897`.
- **Hardware:** Windows 11 Pro build 26200; AMD Ryzen 7 9700X; 31.11 GiB RAM; NVIDIA GeForce RTX 3080 with 10,240 MiB VRAM; driver 610.47; CPython 3.12.10; PyTorch 2.5.1+cu121 with CUDA runtime 12.1.
- **Random seed:** deterministic greedy generation; final-set selection seed `foundry-gsm1k-final-evaluator-validation-v1`.
- **Baseline score:** Re-score of the unchanged 30 Milestone 1.6 validation outputs: 27/30 extractable (90.00%), 3/30 exact compliant (10.00%), 15/30 correct (50.00%), three historical 512-token rejections, and zero false extractions after auditing all four newly accepted outputs.
- **Resulting score:** Final untouched validation: 25/30 extractable (83.33%), 5/30 exact compliant (16.67%), 13/30 correct (43.33%), 12 extractable-but-wrong, five rejected, zero backend generation failures, and zero false extractions after manually auditing all 30 outputs. Rejections were four complete clear terminal prose answers outside the frozen grammar and one 768-token truncation.
- **Cost and runtime:** $0 API/cloud cost. Exactly 30 final generations plus three separately labeled truncation diagnostics. Final run: 4,491 input tokens, 9,451 output tokens, 315.03 average output tokens, 125.095 seconds evaluation, 126.885 seconds including model load, 0.2398 examples/second, and 75.55 generated tokens/second. Peak CUDA memory was 2,985.59 MiB allocated and 3,196 MiB reserved. The three-record diagnostic generated 1,942 tokens in 27.129 seconds and is not part of the final result.
- **Artifacts/checkpoint:** final aggregate summary and content-free manual audit under `results/final_evaluator_validation/current/`; old-output re-score/audit under `results/final_evaluator_calibration/`; ignored raw final predictions under `results/raw/final_evaluator_validation/current/`; ignored diagnostic outputs under `results/raw/truncation_diagnostic/current_768/`; identifier-only final manifests under `configs/eval/manifests/`. No checkpoint.
- **Code commit:** run began from clean published commit `e1d0576`; Milestone 1.7 code, config, identifier manifests, tests, aggregate records, and documentation were uncommitted during the run and are intended for one local atomic commit.
- **Interpretation:** Exact rational normalization and narrow wrappers safely separated clear wrong answers from unextractable responses, and 768 tokens resolved two known truncations. However, the final untouched 83.33% result failed the 90% coverage gate. Zero false extractions demonstrates conservative precision but does not compensate for four clear false rejections and one truncation. Post-hoc grammar changes are prohibited, so the evaluator and 814-ID baseline are not admitted for Milestone 2.
- **Hypothesis supported:** no; fresh extractability was 83.33%, below 90%, despite zero false extractions and zero backend generation failures.
- **Failures/blockers:** one final output reached 768 tokens and four complete clear answers used unsupported terminal prose. Transformers emitted non-fatal warnings for sampling defaults ignored under greedy decoding. No CUDA, OOM, dependency, dataset, backend, manifest, or raw-artifact containment failure occurred.
- **Next experiment:** none approved. The user must choose either (1) proceed to Milestone 2 while scoring all unextractable outputs incorrect and reporting coverage/exact compliance separately, or (2) reconsider the base model or benchmark. No further evaluator-calibration milestone is proposed.

## EXP-20260718-005 — Frozen Qwen GSM1K base-development baseline

- **Status:** completed; one-time D-011 exception applied; no training or synthesis
- **Date:** 2026-07-18
- **Hypothesis:** The untouched base model can complete exactly the frozen 814-ID development baseline once on the RTX 3080 under the immutable Milestone 1.7 stack, producing a reproducible end-to-end score and an initial bounded failure inventory without a backend or memory failure.
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Exact model revision:** `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- **Dataset:** `ScaleAI/gsm1k`, configuration `default`, source split `test`
- **Exact dataset revision:** `bc09569d09a614b9b530edc7f076fb214ac10493`
- **Training configuration:** none; untouched base-model evaluation only.
- **Evaluation configuration:** current prompt SHA-256 `738ea5a3b94e7c75ac0bd50a229bbf04f3fc5d773e14658bc6728bc7a4b18350`; unchanged strict parser; canonical extractor `foundry-terminal-number-v2` SHA-256 `e099d1c247968fed982cb849022ec3137b1694c15f23a65663a127b8158c06df`; final config SHA-256 `5f315d5de645f9563b8d1e61bc8e02c3513c453238ad9e1d6f9473489b5a622b`; greedy decoding; 768 maximum new tokens; 814-ID development manifest SHA-256 `5e810d3ab644bef1d43c598a14a6164ba6464b27fde50e92a2f241816ce87897`.
- **Hardware:** Windows 11 Pro build 26200; AMD Ryzen 7 9700X; 31.11 GiB RAM; NVIDIA GeForce RTX 3080 with 10,240 MiB VRAM; driver 610.47; CPython 3.12.10; PyTorch 2.5.1+cu121 with CUDA runtime 12.1.
- **Random seed:** deterministic greedy generation; fixed identifier manifest and no stochastic generation.
- **Baseline score:** 521/814 correct, or 64.0049% end-to-end accuracy with every unextractable response counted wrong. The evaluator classified 752/814 answers extractable (92.3833%); 521/752 extracted answers were correct (69.2819%); 130/814 outputs were exact-format compliant (15.9705%). There were 231 extractable-but-wrong outputs, 62 unextractable outputs, three truncations, and zero backend failures.
- **Resulting score:** not applicable; no candidate model or adapter was evaluated.
- **Cost and runtime:** $0 API/cloud cost. Evaluation time 3,160.074 seconds (52:40.074); total runtime including 1.847 seconds model loading was 3,161.921 seconds. Throughput was 0.2576 examples/second and 74.08 generated tokens/second. The run processed 113,720 input tokens and generated 234,106 output tokens, averaging 287.60 output tokens/example. Peak RTX 3080 memory was 3,133,062,144 bytes (2,987.92 MiB) allocated and 3,353,346,048 bytes (3,198 MiB; 3.123 GiB) reserved.
- **Artifacts/checkpoint:** aggregate summary and content-free failure inventory under `results/development_baseline/qwen2_5_1_5b/`; complete 814-record output under ignored `results/raw/development_baseline/qwen2_5_1_5b/`; no checkpoint.
- **Code commit:** run began from clean published Milestone 1.7 commit `7786fe7`; Milestone 2 execution/summary plumbing, tests, aggregate records, and documentation were uncommitted during the run and are intended for one local atomic commit.
- **Interpretation:** The base model completed the exact frozen development scope without CUDA, OOM, dataset, or backend failure. A deterministic SHA-256-ranked audit sampled 100 of 231 extractable-but-wrong records. Primary categories were bookkeeping/omission 28, target/language interpretation 18, constraint/discrete reasoning 15, time/unit/sequence 14, arithmetic execution 12, rate/ratio/percentage/average 12, and benchmark ambiguity/annotation risk 1. These are provisional sampled interpretations, not exhaustive population counts.
- **Hypothesis supported:** yes for execution and reproducible aggregate measurement. The broader premise that the extractor makes only conservative errors was not supported: the 100-record audit found two false extractions. Both remained mathematically wrong, so the 521-correct count did not change, but precision among the 521 scored-correct records remains unaudited.
- **Failures/blockers:** 42 ambiguous terminal answers, eight conflicting answers, seven missing terminal answers, two malformed terminal answers, and three truncated generations. Two local read-only inspection commands failed before correction—one quoting syntax error and one incorrect dataset configuration—and one later inspection batch hit a Windows console-encoding error after returning 19 records; none changed files or evaluation artifacts. Transformers emitted expected warnings about sampling defaults ignored under greedy decoding. No run retry occurred.
- **Next experiment:** none approved. The user must decide whether to authorize a bounded audit of records scored correct before candidate comparisons or accept the documented extractor risk and separately scope targeted synthetic-data design. No generation or training begins automatically.
