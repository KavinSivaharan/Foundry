# Foundry Experiment Log

Five bounded real-model experiment groups have completed: the ten-example Milestone 1 CUDA smoke, the 30-example/three-prompt Milestone 1.5 format calibration, the 30-example fresh Milestone 1.6 answer-validation run, the final 30-example Milestone 1.7 evaluator validation, and the frozen 814-example Milestone 2 base-development baseline recorded below. A read-only Milestone 2.1 audit and design-only Milestone 3 record are also complete. No training experiment or sealed-final benchmark result exists.

Every future experiment must be registered here before a costly run begins. Its machine-readable configuration must be saved under `configs/`, raw evaluation outputs under `results/raw/`, and reviewable summaries under `results/`.

## EXP-20260719-007B — Signal-first feasible allocation and review smoke

- **Status:** completed with technical-gate failure
- **Scope:** content-free 2,504-slot allocation plus one 120-question offline-template smoke; no dataset generation or training
- **Dry schedule:** 2,504/2,504 unique latent programs and semantic IRs; zero targeted/generic or future train/validation overlap; exact replay
- **Smoke:** 120 attempted, 115 automatically accepted, five rejected for runtime number-neutral collisions
- **Other failures:** zero false labels, verifier disagreements, target mismatches, deterministic language defects, exact duplicates, latent duplicates, benchmark findings, or unresolved contamination cases
- **Decision:** preserve the runtime normalizer and block review-packet creation until scheduler/runtime identity is unified

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

## EXP-20260718-006 — Label-blind audit of correct-scored development responses

- **Status:** completed; `BASELINE TRUSTED`; no generation, synthesis, or training
- **Date:** 2026-07-18
- **Hypothesis:** Exhaustively auditing all 521 correct-scored development completions without benchmark labels will determine whether accidental extractor matches materially inflate the frozen 64.00% score.
- **Model/data execution:** none. This experiment read only the existing ignored Milestone 2 outputs from the frozen Qwen/GSM1K development run; it did not generate, retry, rescore, or access sealed-final examples.
- **Audit configuration:** label-blind view schema v1; raw predictions SHA-256 `73d52dace0f27577b1177bdfa81dfbb4c88252107c9b04e2ff49dbbd93da6cc0`; audit configuration SHA-256 `e50df38364b88d4900dfecc948cd56d0d552e050971ca47a7a21264699ee4122`; view SHA-256 `b0fa85ffa26413137e05992a2b368ba4e059d04a649b47605fb70cbbe7e63dee`. Views exposed stable identifier, completion, extraction status/value/rule/span, exact compliance, completion/token metadata, terminal competing values, and generic flags—never the benchmark question or label.
- **Freeze protocol:** All 521 classifications were completed before any score join. Frozen classification SHA-256 is `669a866e984c35908bdb9e5443cb989733fd762d11bf62456387a25a5c12e14c`; counts were internally verified as 521 confirmed intended answers, zero confirmed false acceptances, and zero ambiguous cases.
- **Baseline score:** Frozen evaluator score 521/814 = 64.0049%; 90 strict-parser correct and 431 canonical-only correct.
- **Resulting score:** Audited lower bound 521/814 = 64.0049%; upper bound 521/814 = 64.0049%; adjusted exact accuracy 521/814 = 64.0049%; confirmed false-positive rate 0/521 = 0%.
- **Pattern review:** 41 responses had multiple values in the terminal context, 337 used broader conclusion rules, and four used unsigned negative-intent language. Manual review confirmed explicit terminal intent in every case. The prior percentage/currency collision occurred zero times; the four negative-language cases all stated positive magnitudes and produced zero false acceptances.
- **Artifacts/checkpoint:** aggregate-only `results/development_baseline/qwen2_5_1_5b/correct_response_audit.json`; detailed views, rationales, working decisions, final classifications, and freeze summary under ignored `results/raw/correct_response_audit/current/`; no checkpoint.
- **Interpretation:** The correct numerator was not inflated by an accidental extractor match. The two known false acceptances in the wrong-output sample remain disclosed, but they are isolated from the 521 correct-scored population. The 231 extractable-but-wrong pool and 62 unextractable records remain useful for provisional development failure analysis, subject to their existing limitations.
- **Hypothesis supported:** yes. No false-positive correct answer or unresolved systematic extraction risk was found, so the predeclared decision is `BASELINE TRUSTED`.
- **Limitations:** Intent was assessed from completion/extraction evidence, not by independently solving questions. The audit covers development responses scored correct, not the complete behavior of the extractor, sealed-final performance, or an exhaustive failure taxonomy.
- **Next experiment:** none approved. The next proposed milestone is a bounded targeted synthetic-data design that freezes category, generator, verification, and contamination controls before any examples are created.

## EXP-20260718-007 — Complete development-failure taxonomy and synthesis design

- **Status:** completed design record; no synthetic generation or training
- **Date:** 2026-07-18
- **Hypothesis:** An exhaustive development-failure review can identify three independently generatable and deterministically verifiable curriculum targets, and a matched targeted-versus-generic experiment can be frozen before creating examples.
- **Model/data execution:** none. This record inspected only the existing 814-example ignored development baseline; it did not generate model outputs, retry examples, alter the evaluator, load sealed-final content, create synthetic questions, or train a model.
- **Evidence scope:** All 231 extractable-but-wrong and 62 unextractable development records were directly reviewed. Detailed content-bearing views and classifications remain ignored. Raw prediction SHA-256 is `73d52dace0f27577b1177bdfa81dfbb4c88252107c9b04e2ff49dbbd93da6cc0`; frozen detailed taxonomy SHA-256 is `964d0c18b60d4f0262f0ec711b2f13b396ca4fa1c921f0dc2e91205d393cb692`.
- **Complete taxonomy:** output format/extraction 69; multi-step bookkeeping/omission 68; target/language interpretation 53; rate/ratio/percentage/average 28; constraint/distribution/discrete 27; time/unit/sequence 24; arithmetic execution 22; benchmark ambiguity/annotation risk 2. Failure kind was 224 reasoning and 69 output; confidence was 274 high, 17 medium, and two low.
- **Selected targets:** multi-step bookkeeping/omission, rate/ratio/percentage/average, and constraint/distribution/discrete reasoning. The separate shared output track is `terminal-final-answer-contract-v1`. Target/language interpretation was deferred because reliable controlled rendering and low contamination risk are not yet established.
- **Architecture:** selected fully procedural latent programs plus controlled templates. Exact execution supplies the label, and a distinct independent method must agree. Local-model paraphrasing is deferred; frontier-model generation is rejected for the first pilot.
- **Matched control:** proposed targeted and generic datasets each contain 4,000 accepted examples, including 400 synthetic-validation and 800 output-contract examples, with about 1.4 million tokens. Generator framework, quality gates, length/difficulty range, later QLoRA settings, optimizer steps, token budget, and seeds are matched; curriculum selection is the meaningful difference.
- **Planning estimates:** 5–15 minutes generation and 10–30 minutes verification per dataset, 40–100 MB combined accepted data/metadata, and 2–4 hours with approximately 7.5–9.5 GiB peak VRAM per future QLoRA run. These are estimates, not measurements.
- **Contracts:** content-free taxonomy contract SHA-256 `021837a1f1a3bb5a189b1f39c808bb907e415e28d8fa722a8a03c3114717cf28`; architecture/curriculum/gate contract SHA-256 `910bf21dba7cef833fd9f7bd83842034e9e7261cf93979d7cdddc0479094d347`; synthesis config SHA-256 `7c087ac45c9027ab872cfecbc0dbf6123b60ec088b35e6d6ddc4dfd9094a99d5`.
- **Safety:** no benchmark question, answer, response, or structure is permitted as generator input. Exact text, number-neutral template, latent structure, five-gram, semantic similarity, and manual closest-match gates precede acceptance. Missing semantic screening escalates to manual review. No embedding model was downloaded in this milestone.
- **Hypothesis supported:** yes at the design level. Exactly three credible targets and a separate output track were selected, and typed schemas, dual-verifier contracts, contamination controls, matched-control budgets, staged execution, and success gates were frozen. No performance improvement has been measured.
- **Failures/limitations:** Category labels remain human interpretations with overlapping secondary causes. Controlled templates may have limited linguistic diversity. The local semantic encoder and protocol still require a pinned future decision. QLoRA runtime/VRAM estimates have not been measured under training.
- **Next experiment:** none approved. The proposed Milestone 4 would pin one local semantic-screening artifact, implement only the three approved procedural families plus output track, and run at most a 120-candidate generator smoke. The full 4,000 + 4,000 pilot and all training require later approval.

## EXP-20260718-008 — Bounded procedural-generator and contamination smoke

- **Status:** completed; readiness gate failed; full pilot generation blocked
- **Date:** 2026-07-18
- **Hypothesis:** The three approved deterministic generator families can produce at least 90 accepted examples from 120 fixed attempts, including at least 15 per family, with dual-verifier agreement, zero false labels, zero invalid acceptances, zero unresolved contamination, and exact replay.
- **Execution:** Exactly 120 attempts under master seed `foundry-phase1-procedural-smoke-master-v1`: 60 targeted (33 bookkeeping, 14 rate/ratio, 13 discrete) and 60 generic (20/20/20), with 12 output-contract attempts per group. No rejected candidate was replaced.
- **Semantic artifact:** `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, Apache-2.0, 91,577,897 required bytes, 384 dimensions, CPU float32 attention-mask mean pooling, L2 normalization, local-only loading, no remote code, and no new dependency.
- **Result:** 24 accepted and 96 rejected. Accepted by family: bookkeeping 4, rate/ratio 16, discrete 4. Rejections: 25 numeric-template copies, 50 five-token overlaps, seven semantic automatic rejections, and 14 manual semantic rejections. There were zero primary/independent verifier failures, disagreements, ambiguous targets, generator exceptions, unresolved contamination cases, or false labels.
- **Manual audit:** All 120 attempts were reviewed. Five accepted examples were invalid: four bookkeeping renderings lacked a coherent common inventory unit across heterogeneous nouns, and one discrete capacity example had grammar and tied-constraint difficulty defects. No incorrect rejection decision or suspicious benchmark resemblance was found.
- **Determinism:** Final replay matched decision SHA-256 `661410933e90680d34a06c1836c7aca6fecfd5bba507c2dfaf3d8ecd5340c8b9` and aggregate SHA-256 `eb85cf9efe130d34164bca20badb9b3dce8f050493abf0e014614332b68f8771` exactly.
- **Runtime/resources:** The counted process used 34.356 seconds wall-clock including its required 60-attempt progress pause. Uninterrupted final reconstruction took 9.693 seconds plus 194 seconds recorded manual audit; semantic screening 8.828 seconds, deduplication 0.790 seconds, generation 0.009 seconds, verification 0.005 seconds. Peak process RAM was 927,363,072 bytes; GPU use was zero; semantic cache was 91,578,751 bytes; ignored raw artifacts were 1,690,598 bytes.
- **Safety:** The contamination scanner loaded only 904 development questions and no benchmark answers. Benchmark content never entered a generator interface or tracked artifact. Sealed-final content, Qwen inference, paid/cloud services, full dataset generation, and all training were absent.
- **Hypothesis supported:** no. Acceptance was 20%, two families were below 15 accepted, and five invalid accepted examples plus systematic template defects violated fixed gates despite perfect label verification.
- **Next experiment:** none approved. The proposed blocker-resolution smoke would repair only entity/unit rendering, grammar, discrete-constraint difficulty, and hand-authored template diversity, use a new version/seed, and repeat exactly 120 attempts under the unchanged thresholds and readiness gates. It would not authorize full pilot generation or training.

## EXP-20260718-009 — Generator rendering-and-diversity blocker resolution

- **Status:** completed; readiness gate failed; full pilot generation remains blocked
- **Date:** 2026-07-18
- **Hypothesis:** Typed object/unit contracts, broader hand-authored renderers, explicit discrete grammar, and measurable difficulty controls can raise a fresh fixed 120-attempt smoke to at least 90 accepted examples with zero invalid acceptances under unchanged contamination gates.
- **Execution:** Exactly 120 counted attempts under master seed `foundry-phase1-procedural-smoke-master-v2-rendering-diversity`: 60 targeted (33 bookkeeping, 14 rate/ratio, 13 discrete) and 60 generic (20/20/20), with 12 output-contract attempts per group and no replacements. A deterministic replay reconstructed the same candidates and decisions without counting another smoke.
- **Implementation:** Added typed objects, units, locations, quantities, transfers, noun forms, and renderer-quality evidence. Template inventory is bookkeeping 2 families × 8 renderers × 24 domains; rate/ratio 5 × 6 × 20; discrete 4 × 6 × 20. Discrete search-space bands are easy 9–35, medium 36–80, and hard 81–200.
- **Frozen contamination:** `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; CPU float32, 384 dimensions, attention-mask mean pooling, L2 normalization, 0.75 manual-review and 0.82 automatic-rejection thresholds, five-token Jaccard 0.35.
- **Result:** 86 accepted and 34 rejected. Targeted accepted 52/60 and generic 34/60. Accepted by family: bookkeeping 30/53, rate/ratio 29/34, discrete 27/33. By difficulty: easy 28/40, medium 30/40, hard 28/40. Output-contract attempts accepted 18/24. Rejections were 19 semantic automatic, nine manually confirmed generated-to-generated near matches, and six five-token overlaps; exact, number-neutral, and latent-structure rejections were zero.
- **Verification:** Zero false labels, primary/independent failures, verifier disagreements, generator exceptions, ambiguous targets, unresolved contamination cases, incorrect rejections, or overlooked benchmark resemblance. Decision SHA-256 `84bd6c622b30034a5932a4098c166b8710e39bbf4756e74b1c7c51cf54ce84a3`; aggregate SHA-256 `0e2e20a3516beacb651dfafea96be9b3e95760fbede8804ae6bea76eb6657ed6`; replay matched both.
- **Manual audit:** All 120 candidates were reviewed. Eleven accepted renderings remained invalid because of residual noun grammar, a duplicated weighted group, weighted-average target inconsistency, an omitted rate denominator, an elided discrete noun, and an irregular container plural. All rejection decisions were appropriate.
- **Runtime/resources:** Counted wall time 30.911 seconds including the 60-attempt pause; semantic screening 8.578 seconds, deduplication 2.301 seconds, generation 0.013 seconds, verification 0.011 seconds, and recorded manual audit 67 seconds. Peak process RAM 929,832,960 bytes; GPU use zero; semantic artifact 91,578,751 bytes; ignored raw artifacts 2,545,296 bytes after replay/audit.
- **Hypothesis supported:** no. Yield improved from 20% to 71.67% and every family exceeded 15 accepts, but 86 < 90 and 11 invalid acceptances plus systematic renderer defects violate unchanged gates.
- **Safety:** The generator received no benchmark content. The contamination scanner returned only the approved 904 development questions and no answers; no benchmark text, raw candidate, embedding weights, cache, dataset, model output, or sealed-final artifact is tracked. Qwen inference, full generation, paid/cloud services, and training did not occur.
- **Next experiment:** none approved. The user must decide whether to stop this procedural lineage or explicitly scope another response to the 11 audited renderer defects. Full pilot generation and every training stage remain blocked.

## EXP-20260718-010 — Typed natural-language realizer stress

- **Status:** completed; renderer stress gate failed; fresh full-pipeline smoke not run
- **Date:** 2026-07-18
- **Hypothesis:** Typed semantic/sentence representations, centralized explicit morphology, target contracts, and exact semantic coverage can eliminate the eleven audited defects while providing natural, sufficiently diverse procedural surfaces for a final 120-candidate readiness smoke.
- **Implementation:** Added typed problem/entity/lexeme/quantity/unit/group/rate/target/clause/render-signature/coverage representations and one compiler consumed by all three mathematical generators. All eleven defect classes have sanitized regressions. Generator/verifier mathematics, curriculum, MiniLM artifact, and contamination thresholds were unchanged.
- **Stress execution:** Exactly 900 in-memory render attempts under master seed `foundry-m4.2-typed-renderer-stress-v1`: 300 bookkeeping, 300 rate/ratio, and 300 discrete. No benchmark comparison, question-corpus persistence, model text generation, full dataset, or training occurred.
- **Automated result:** 900/900 passed typed validation; zero morphology, target, missing-node, duplicated-node, grammar-metadata, exact-duplicate, or latent-structure failures; 900 distinct render signatures. There were 99 number-neutral template duplicates. Nearest generated semantic neighbor was at or above 0.82 for 899 renders, in 0.75–0.82 for one, with median 0.936379 and maximum 0.996438.
- **Manual audit:** A deterministic 20-render sample per family found zero false mathematical labels and 13 unnatural questions. The systematic defect was imperative request clauses normalized with question punctuation; two of those also used awkward hyphenated count compounds.
- **Gate decision:** failed. Naturalness and scale-diversity requirements were not met, so the prompt-authorized fresh 120-candidate smoke did not run. No acceptance-by-group/category/output-track result exists for Milestone 4.2.
- **Safety:** No sealed-final access, benchmark inference, benchmark generator input, paid/cloud service, LLM-generated question/label, full synthetic dataset, SFT, QLoRA, or GRPO occurred.
- **Next experiment:** none approved. The pure procedural-renderer lineage is closed. The exact next decision is whether to approve a design-only pivot to constrained local-model surface realization with exact typed round-trip validation and the unchanged contamination pipeline.

## EXP-20260718-011 — Constrained local-model realization design

- **Status:** completed design record; no model download, inference, generation, or training
- **Date:** 2026-07-18
- **Hypothesis:** A local language model can be restricted to surface wording while deterministic
  slots, node coverage, procedural execution, and independent verification retain control of every
  mathematical fact and label.
- **Candidate comparison:** Compared only the approved Qwen2.5-1.5B-Instruct, Qwen3-1.7B, and
  SmolLM3-3B official metadata. Selected
  `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` as primary and the existing pinned
  Qwen2.5 revision as fallback. No weights were downloaded. Qwen3 requires a future dedicated
  Transformers >=4.51 lock; no dependency changed here.
- **Contract:** The model sees typed semantic roles and opaque placeholders, not values, answers, or
  benchmark content. Strict JSON returns a question template, placeholder inventory,
  clause-to-node map, target/intent echoes, and no answer. Deterministic checks reject slot, node,
  target, intent, rate, unit, discourse, structure, or answer-content deviations before filling.
- **Round trip:** Existing typed quality checks, exact program execution, independent mathematical
  verification, output-contract validation, contamination screening, and answer-blind human audit
  all remain mandatory. A model reverse parse can only add rejection evidence.
- **Semantic policy:** Preserve MiniLM 0.75/0.82 for generated-to-development contamination. Keep
  exact/template/latent/five-token controls internally and calibrate a separate semantic-diversity
  policy on original fixtures before any future generation.
- **Future smoke:** Proposed 120 IRs and at most 360 fixed three-beam candidates under the unchanged
  category/output-track allocation, no retries or replacements, complete audit, at least 90 clean
  IRs and 15 per family, and zero false labels, semantic-drift accepts, invalid accepts, verifier
  disagreements, or unresolved contamination.
- **Planning estimate:** Qwen3 needs about 4.08 GB of repository download and an estimated 4.5–5.5
  GiB FP16 VRAM. A 120-IR generation phase is estimated at 15–45 minutes plus 2–4 hours of complete
  audit. The estimates are not measurements.
- **Hypothesis supported:** supported as an auditable design, not as an empirical realization result.
  The typed contracts can make model output reject-only, but placeholder fidelity, naturalness,
  yield, deterministic replay, and scale remain unmeasured.
- **Safety:** No benchmark or sealed content, model weights, generated examples, model inference,
  dependency installation, dataset, paid/cloud service, QLoRA, SFT, or GRPO was used.
- **Next experiment:** none approved. The exact proposed Milestone 5B is a bounded implementation
  smoke that first freezes the internal-diversity policy, then downloads only the pinned Qwen3
  artifact and processes 120 IRs/360 candidates maximum. Full dataset generation and training remain
  separately blocked.

## EXP-20260718-012 — Bounded Qwen3 local-realization smoke

- **Status:** completed; exact replay passed; readiness gate failed; full pilot generation blocked
- **Date:** 2026-07-18
- **Hypothesis:** A pinned local Qwen3 model can convert at least 90 of 120 fixed, value-blind
  procedural IRs into clean natural questions while preserving every placeholder, event, unit,
  constraint, and target, with zero label, semantic, contamination, or replay defects.
- **Internal policy calibration:** Compared exactly three predeclared policies on 24 original fixture
  pairs before Qwen generation. Selected `evidence-gated-balanced-v1` (SHA-256
  `26c030e8497c4727e286ff3e89d4720cee1c2681a224b8a93b8c515ef521cc90`): 22/24 exact
  fixture outcomes, zero duplicate escapes, zero distinct automatic rejections, and two documented
  ambiguous outcomes. Development MiniLM 0.75/0.82 controls were unchanged.
- **Environment/model:** CPython 3.12.10; PyTorch 2.5.1+cu121; Transformers 4.51.3;
  tokenizers 0.21.4; `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`,
  Apache-2.0, 4,079,450,110 bytes, FP16, no remote code. Fresh offline reload succeeded.
- **Execution:** Exactly 120 fresh IRs under `foundry-m5b-ir-master-20260718-v1`: targeted
  33/14/13 and generic 20/20/20 across bookkeeping/rates/discrete, with 12 output-contract IRs per
  group. Exactly three deterministic beams per IR produced 360 outputs; there were no retries,
  replacement IRs, fallback inference, or benchmark generation.
- **Automatic result:** 181/360 JSON parses (50.28%); 41/181 parsed outputs preserved the complete
  placeholder set; 0/181 passed semantic-node coverage; 171/181 preserved target/intent. Of 179
  unparsed outputs, 160 reached 256 tokens. No IR had a passing beam. Primary and independent
  verifiers agreed throughout; false labels, verifier disagreements, backend failures, and per-IR
  timeouts were zero. No beam reached semantic contamination screening because earlier required
  layers failed.
- **Manual audit:** All 360 beams were reviewed through 37 exact-template groups with answers and
  verifier evidence hidden on the first pass. Naturalness was 63 natural and 297 unnatural;
  semantic preservation was 59 preserved and 301 drifted. Every automatic rejection was correct;
  invalid acceptances and incorrect rejections were zero. The systematic defects were target-only
  event omission, instruction echo, run-on/all-caps surfaces, incomplete JSON, and invalid clause
  maps.
- **Determinism:** Exact replay reproduced beam text, ordering, validation decisions, and final
  SHA-256 `a2e6fb565da817ec5e2e6e3c87ba8a54643b2b5ec294dd8f5d24204083d06dcf`.
- **Runtime/resources:** Counted generation 802.093 seconds; end to end 812.465 seconds; 134,148 input
  and 86,199 output tokens; 107.47 returned output tokens/s; 6.684 generation seconds/IR. Peak
  allocated/reserved VRAM 4,289,053,184/4,716,494,848 bytes; peak process RAM 7,531,028,480 bytes;
  ignored raw/audit/replay artifacts 967,542 bytes.
- **Hypothesis supported:** no. Clean yield was 0/120 rather than at least 90. Zero defects among
  accepted outputs is vacuous because nothing was accepted; the strict boundary protected labels
  but did not produce usable wording.
- **Safety:** No accepted/full dataset, training, QLoRA, SFT, GRPO, paid/cloud service, benchmark
  inference, fallback model, or sealed-final access occurred. Complete outputs, question exports,
  and audit decisions remain ignored.
- **Next experiment:** none approved. The narrowest proposed step is a design-only compact-protocol
  milestone that retains Qwen3 and every safety gate but removes verbose echoed inventories/maps and
  imperative event input before any separately approved new inference. The alternative is to stop
  local surface realization.

## EXP-20260718-013 — Compact tagged-realization micro-smoke

- **Status:** completed; exact replay passed; micro-gate failed; Qwen3 prompt lineage closed
- **Date:** 2026-07-18
- **Hypothesis:** Removing redundant JSON metadata and asking Qwen3 only for tagged natural clauses
  can yield at least 22 clean IRs from 30 while deterministic code retains all semantics and labels.
- **Protocol:** Consecutive `<E…>` fact tags plus one `<Q>` tag; immutable value and semantic-anchor
  tokens; no JSON, answer, calculation, mapping echo, target echo, repair, retry, or replacement.
  Fixed three-beam deterministic decoding, seed 5172026, 384-token hard cap, and per-beam `</Q>` stop.
- **Execution:** Exactly 30 new IRs under `foundry-m5c-compact-ir-master-20260718-v1`: targeted
  8/4/3 and generic 5/5/5 across bookkeeping/rates/discrete, with three output-contract IRs in each
  group. Exactly 90 beams were returned from the existing offline Qwen3 snapshot.
- **Automatic result:** 90/90 tags parsed; 87/90 preserved placeholders, semantic anchors, and target
  tokens; 60 failed before deterministic filling; all 90 failed language quality; 0/30 IRs had a
  selected beam. Dual verifiers agreed throughout and no backend failure or timeout occurred.
- **Manual audit:** Every beam was reviewed with answers and verifier evidence hidden. All 90 were
  unnatural and semantically drifted because Qwen copied token lists with postfixed predicates and
  inadequate connecting syntax. Every rejection was correct; false labels, invalid acceptances, and
  incorrect rejections were zero.
- **Determinism:** Exact replay reproduced all beam text, order, decisions, and SHA-256
  `b9b1a7bc8214c2656b6cd45cb089252f63fbe572c52f910e1148a34cd6a4358a`.
- **Runtime/resources:** Counted generation 83.474 seconds; total 92.869 seconds; 11,530 input and
  11,472 output tokens; 137.43 returned output tokens/s. Peak allocated/reserved VRAM
  3,728,026,112/3,825,205,248 bytes; peak process RAM 7,532,572,672 bytes; counted raw beam file
  160,872 bytes.
- **Hypothesis supported:** no. Structural compliance improved, but clean yield remained 0/30. The
  required 22 total, 8 bookkeeping, 6 rate, and 5 discrete acceptances all failed.
- **Safety:** No accepted/full dataset, fallback inference, training, QLoRA, SFT, GRPO, benchmark
  inference, paid/cloud service, or sealed-final access occurred. Raw beams and audits remain ignored.
- **Next experiment:** none approved. The final Qwen3 prompt-patching stop rule is active. The one
  recommended pivot is a separately approved stronger local realization model using the same frozen
  compact protocol and unchanged deterministic gates.

## EXP-20260718-014 — Stronger-model compact realization comparison

- **Status:** complete; fixed gate failed; final local-model stop rule active.
- **Question:** Does changing only Qwen3-1.7B to Qwen3-4B-Instruct-2507 make the frozen compact
  protocol a viable natural-language realization layer?
- **Control:** Exact M5C 30-IR plan and hashes; same compact prompt, three beams, 384-token limit,
  seed, event tags, anchors, validation, verifiers, screens, audit, and 22/30 gate.
- **Model:** `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`,
  Apache-2.0, offline FP16 CUDA, no offload or quantization.
- **Probe:** 3/3 decoded and tag-parsed; 8,489,271,296 peak reserved VRAM; 946,094,080 bytes free;
  3.575 seconds generation.
- **Counted result:** 30 IRs, 90 beams, 71 tag parses, 47 placeholder passes, 50 anchor passes,
  47 target passes, 19 language-quality passes, 10 filled-consistency passes, 0 automatic selections,
  and 0 clean IRs after audit. All 90 beams were unnatural and semantically drifted. False labels,
  invalid acceptances, incorrect rejections, verifier disagreements, timeouts, and backend failures
  were zero.
- **Runtime/resources:** 108.983 seconds generation; 121.937 seconds total; 11,410 input and 8,838
  output tokens; 8,553,572,352/8,703,180,800 peak allocated/reserved VRAM; 12,593,078,272 peak RAM;
  160,116 raw beam bytes.
- **Replay:** exact pass at `7043fb5f94cbd95fe76391fa167ba766acf5080f77b0fede7197c00b8b9a9f01`.
- **Conclusion:** Capacity substitution did not rescue the protocol. No 120-IR run is justified.
- **Next experiment:** none approved. Recommend a separately approved manually vetted offline
  template-bank design and bounded smoke; do not test another live realization model.

## EXP-20260719-015 — Offline template-bank bounded smoke

- **Status:** complete; technical gate failed; human review pending.
- **Question:** Can a 58-frame/232-plan offline bank produce at least 90 clean automatic candidates
  while preserving exact IR, labels, dual verification, and frozen contamination controls?
- **Budget:** exactly 120 attempts, no replacements; targeted 33/14/13 and generic 20/20/20; 24
  output-contract attempts; no LLM or GPU inference.
- **Automatic result:** 118 accepted and two rejected (one latent-program copy, one number-neutral
  copy). Bookkeeping/rate/discrete accepted 53/33/32. Both verifiers agreed on all 120; false labels,
  deterministic language defects, target mismatches, benchmark rejections, and unresolved cases were
  zero.
- **Replay:** exact match at
  `bf87e7af166f5dca107c9777337216e6da7a656b4eec3efb372dc98d1bfa5487`.
- **Resources:** 1.579 seconds counted, 1.585 seconds replay, 914,022,400 peak process bytes, 440,657
  ignored raw bytes, no GPU.
- **Codex inspection:** all 120 reviewed; not a human audit. Thirteen invalid/unnatural surfaces show
  systematic frame-label, grouping-noun, compound-noun, and ordinal-inflection defects.
- **Conclusion:** automatic count gates pass, but the systematic-template-defect gate fails. Human
  review packet is pending at `results/raw/template_bank_smoke/human_review.md`.
- **Next experiment:** none approved. Await user packet review and an explicit decision on one
  offline architecture-level bank-composition blocker resolution or stopping realization.

## EXP-20260719-016 — Offline template-composition compiler and final review smoke

- **Status:** complete; technical gate passed; genuine human review pending.
- **Question:** Can typed lexical, ordinal, noun-phrase, and provenance composition eliminate the 13
  M6A defects broadly enough to justify user review of a fresh bounded smoke?
- **Static expansion:** 232 plans x 10 deterministic fixtures = 2,320 attempts; 2,320 valid; all 232
  render signatures exercised; zero final noun, internal-label, ordinal, morphology, target, or
  coverage failures. A stratified 90-render Codex sample had zero findings. Expansion SHA-256 is
  `78802a61a421ed060aeeab9841c8dd139b97b0ddf971a9b5dd85f5e4766d8e99`.
- **Budget:** exactly 120 fresh attempts, no replacement; targeted 33/14/13 and generic 20/20/20;
  24 output-contract attempts; no language model or GPU inference.
- **Automatic result:** 116 accepted, four rejected (three latent copies and one number-neutral
  copy). Targeted/generic accepted 58/58; bookkeeping/rates/discrete accepted 53/31/32. Verifiers
  agreed throughout; false labels, deterministic language defects, target mismatches, benchmark
  rejections, unresolved contamination, exact duplicates, and signature reuse were zero.
- **Replay:** exact match at
  `f5caa7e811cbf257c752a15059e25cc20b2f978fb60e8ad0890c64186095a254`.
- **Resources:** 1.670 seconds counted, 1.612 seconds replay, 912,838,656 peak process RSS, 539,386
  ignored artifact bytes, no GPU.
- **Conclusion:** **TECHNICALLY READY — HUMAN REVIEW PENDING**. The ignored HTML packet is
  `results/raw/template_bank_smoke_v2/human_review.html`; Codex inspection is not human approval.
- **Next experiment:** none approved. Await the user's complete packet review and explicit decision;
  do not generate the 4,000 + 4,000 pilot or train.

## EXP-20260719-017 — Human-review-driven template-bank revision

- **Status:** complete; technical gate failed; no second-review packet created.
- **Question:** Can replacements derived from 60 genuine approvals and 60 genuine rejections pass
  static language validation and the fixed 110/120 runtime gate?
- **Review evidence:** 120 unique matching decisions, 60 Approve/60 Reject/0 Unsure, SHA-256
  `564a8ca584984ee7a0b997eec4a6a6f377308c869b62cf65ebeef5375cef0791`.
- **Static result:** 232 plans x 10 fixtures = 2,320/2,320 valid; zero exact or number-neutral duplicate
  sentence plans; clean 90-render Codex sample after one shared quantifier regression repair;
  expansion SHA-256 `fc3c6a16...cc6c`.
- **Budget:** exactly 120 fresh attempts, no replacement; targeted 33/14/13 and generic 20/20/20;
  24 output-contract attempts; no language model or GPU inference.
- **Automatic result:** 104 accepted and 16 rejected. Targeted/generic accepted 56/48;
  bookkeeping/rates/discrete 39/33/32. Fifteen number-neutral rendered-template copies and one latent
  copy were rejected. False labels, verifier disagreements, deterministic language defects, target
  mismatches, benchmark rejections, unresolved contamination, exact duplicates, and signature reuse
  were zero.
- **Replay/resources:** exact match at `44cd5265...1e0f`; counted/replay runtime 1.612/1.530 seconds;
  923,930,624 peak RSS; 380,785 ignored raw bytes; no GPU.
- **Conclusion:** The fixed 110 threshold failed, so no v3 human/Codex-assisted packet was created.
  The language repair is retained as evidence, but full generation and training remain blocked.
- **Next experiment:** none approved. A future decision may authorize a narrow runtime
  template-selection/diversity correction and one new bounded smoke without lowering any gate.

## EXP-20260719-018 — Runtime-diversity capacity preflight

- **Status:** complete; capacity gate failed; mandatory stop applied.
- **Question:** Can the frozen v3 bank support collision-free cross-dataset allocation for 4,000
  targeted and 4,000 generic-control acceptances with fixed 125% attempt pools under every unchanged
  uniqueness and contamination rule?
- **Source evidence:** The existing 120-record v3 smoke was read locally and remained ignored. All 16
  duplicate rejections were mapped to deterministic earlier partners: 15 number-neutral surface
  copies and one latent-program copy. Source-attempt SHA-256 is
  `773e2bfad21374cc6ee403b486c1af1f72215360fc647e8c1c96e7c861651828`.
- **Budget:** 8,000 accepted examples imply 10,003 fixed attempts after applying `ceil(1.25 * quota)`
  to each group/category stratum: 4,418 bookkeeping, 2,834 rates, and 2,751 discrete.
- **Capacity result:** Active plan signatures are 72/80/80; domain-aware signatures are
  1,728/400/1,600; number-neutral surfaces are 768/88/320 for bookkeeping/rates/discrete. Every
  category, difficulty, and output-contract stratum fails. Audit SHA-256 is
  `8b921822bf10da964cf357cf3851084a2e0bd15ffc5dc549a85e04f84c9ccd7b`.
- **Scope result:** No allocator, latent schedule, candidate schedule, master seed, fresh smoke,
  replay, raw candidate evidence, or review packet was created. No language, mathematics, verifier,
  evaluator, duplicate rule, contamination threshold, or benchmark artifact changed.
- **Conclusion:** A scheduler cannot satisfy cross-dataset uniqueness with the current finite bank.
  Full generation and training remain blocked.
- **Next experiment:** none approved. Decide whether to expand independently authored and
  human-reviewed plan/scenario capacity, then repeat the same preflight before allocator work.

## EXP-20260719-019 — Bounded template-reuse policy and revised capacity

- **Status:** complete; selected policy passed fixtures; full capacity failed; mandatory stop
  applied.
- **Question:** Can quota-derived reuse of reviewed sentence plans support the fixed 10,003-attempt
  pool while exact rendered questions and latent programs remain globally unique and benchmark
  contamination remains unchanged?
- **Policy calibration:** Three predeclared policies were tested on 14 original fixtures. Bounded
  balanced reuse matched 14/14; legacy one-use matched 12/14; permissive exact/latent-only matched
  11/14. Fixture, selected-policy, and calibration SHA-256 values are `2a829eea...419b`,
  `66443bc8...25f0`, and `fd731501...2693`.
- **Method:** Caps were mechanically derived per dataset/family for plan, number-neutral signature,
  plan plus domain, frame, target, difficulty, and output-contract use. A fixed 20,000-seed pool per
  family constructed unchanged generator programs; each available program counted toward a cap was
  checked with both existing verifiers. No program corpus or question dataset was persisted.
- **Capacity result:** Surface identity layers pass. Bookkeeping provides 5,524 programs for 4,418
  attempts. Rates provide 1,632/2,834 (shortfall 1,202); discrete provides 2,073/2,751 (shortfall
  678). Internal capacity-audit SHA-256 is `1a40db7b...1129`.
- **Scope result:** Development MiniLM revision and 0.75/0.82 thresholds are unchanged. No allocator,
  schedule, new seed, 120-candidate smoke, replay, second review packet, dataset, inference, training,
  or sealed-final access occurred.
- **Conclusion:** The one-use surface rule was scientifically over-conservative, but 10,003 fixed
  attempts remain infeasible because finite rate and discrete program domains fail the balanced
  latent-capacity gate.
- **Next experiment:** none approved. Decide whether to expand only the deficient exact rate and
  discrete program ranges/modes, preserving labels and dual verification, then rerun capacity before
  any allocation.

## EXP-20260719-020 — Signal-first pilot compatibility audit

- **Status:** complete; reduced quotas frozen; exact compatibility gate failed; mandatory stop
  applied.
- **Question:** Can the unchanged bounded-reuse policy and mathematical inventory support 1,000
  accepted targeted plus 1,000 accepted generic examples through the complete 2,504 fixed attempts?
- **Method:** Froze exact dataset/family/split/output/mode totals, then joined per-dataset target-type
  and semantic-frame caps to their compatible modes through a deterministic integer maximum-flow
  graph. Globally finite mode capacity was shared across datasets; no candidate schedule or question
  corpus was created.
- **Capacity result:** Bookkeeping passes at 1,384/1,106. Rates fail at 695/709 (shortfall 14).
  Discrete fails at 598/689 (shortfall 91), with generic discrete also failing independently at
  399/417. Internal audit SHA-256 is `522b5b4e...7aaf`; serialized evidence SHA-256 is
  `b18b7e44...6995`.
- **Diagnostic correction:** A preliminary aggregate audit reported a pass because it did not join
  target identities to compatible modes. The exact graph supersedes that result before allocator
  implementation or smoke output.
- **Scope result:** No allocator, 2,504-slot schedule, 120-question smoke, deterministic replay,
  second packet, dataset, inference, training, or sealed-final access occurred. Templates,
  generators, verifiers, evaluator, and benchmark-contamination controls remain unchanged.
- **Next experiment:** none approved. Decide whether to redesign only target-type/semantic-frame cap
  derivation around the predeclared curriculum compatibility graph, or redefine the signal-first
  quotas, before capacity is audited again.

## EXP-20260719-021 — Runtime-exact signal-pilot scheduling correction

- **Status:** complete; canonical identity integrated; complete schedule gate failed; mandatory stop applied.
- **Question:** Can the feasible 2,504-slot allocation be reconstructed when scheduler uniqueness uses the exact unchanged runtime number-neutral identity rather than semantic metadata?
- **Collision evidence:** All five prior pairs differed in frame/template metadata; some also differed in difficulty, output track, scenario, or lexical metadata. Each pair nevertheless had an identical runtime-normalized SHA-256.
- **Method:** Added one versioned canonical identity service, rendered every schedule option before identity assignment, required schedule/runtime equality, and preserved the production normalizer's NFKC, lowercase, numeric replacement, tokenization, punctuation, and hashing behavior. Mathematical generators, labels, verifiers, templates, and all benchmark-contamination controls remained unchanged.
- **Capacity result:** The fixed weighted-average pool has 9,265 unique latent programs and 10,512 difficulty realizations but only eight runtime number-neutral identities. Frozen caps give targeted capacity 40/70 and generic capacity 48/100, shortfalls of 30 and 52. The full 2,504-slot reconstruction is therefore mathematically impossible under current controls.
- **Scope result:** No fresh 120-slot schedule, candidate smoke, replay, assisted packet, complete dataset, inference, training, or sealed-final access occurred. Existing Milestone 7B evidence remains preserved at commit `23c2cdb5ab2a931df6b711b1f3fc571c748035c3`.
- **Next experiment:** none approved. Decide whether to authorize genuinely distinct reviewed weighted-average surface forms or a separately justified cap-policy revision; do not weaken runtime normalization after observing this result.

## EXP-20260719-022 — Submode-local balanced surface-reuse calibration

- **Status:** complete; selected policy calibrated; full capacity gate failed; mandatory stop applied.
- **Question:** Does deriving canonical runtime-surface caps at dataset/family/submode granularity make
  the fixed 2,504-attempt signal pilot schedulable without changing wording or duplicate controls?
- **Method:** Compared exactly three policies on ten original fixtures, enumerated active identities
  from fixed latent pools and every compatible reviewed plan, and applied the approved 1.25 cap
  formula to all eleven submodes. Exact questions and latent programs remained globally unique.
- **Calibration:** The selected policy matches 10/10 fixtures; family-level and permissive alternatives
  each match 6/10. Fixture SHA-256 is `7f32574c...15be5`; policy `acb01f6d...4babc`.
- **Capacity result:** Aggregate mode capacity passes. The weighted-average easy/medium identity group
  fails: targeted 47/44, generic 66/64, combined 113/108. Corrected audit SHA-256 is
  `7c0f2913...10f5`.
- **Scope result:** No complete dry schedule, new 120-question smoke, replay, assisted packet, dataset,
  inference, training, or sealed-final access occurred. Benchmark-contamination controls were unchanged.
- **Next experiment:** none approved. Decide whether to make a separately frozen correction to the
  weighted-average difficulty/surface compatibility allocation or stop the pilot.

## Milestone 7E - Minimal difficulty correction and exact scheduling stop

- **Policy:** `minimal-compatible-difficulty-reallocation-v1`; 9/9 original fixtures pass. Policy
  SHA-256 `edd80f67...93d7`, corrected allocation `1f9abb51...d332`, and calibration
  `ec00bfe5...6a11`.
- **Corrected capacity:** weighted-average easy/medium is targeted 44/44, generic 64/64, combined
  108/108; hard is 26/44, 36/64, and 62/108. Arithmetic capacity SHA-256 is
  `4ab358b7...04cb`.
- **Exact schedule result:** failed in generic complete-packages (121 required) under the unchanged
  joint plan, plan/scenario, frame, and runtime-identity caps. Content-free blocker SHA-256 is
  `a387b5ce...41ab`.
- **Downstream status:** No complete 2,504-slot schedule, fresh review schedule, 120-question smoke,
  deterministic smoke replay, assisted review packet, dataset, inference, training, or sealed-final
  access occurred.
- **Next experiment:** none approved. The only relevant permitted choice is to reduce the fixed
  signal-pilot attempt pool; weighted-average plan expansion does not solve the discrete blocker.

## EXP-20260719-023 — Fixed-attempt-pool exact preflight

- **Status:** complete; all approved multipliers failed; mandatory stop applied.
- **Question:** Can the 1,000 + 1,000 accepted pilot retain a smaller predeclared fixed candidate
  buffer while satisfying every frozen exact scheduling control?
- **Method:** Evaluated only `1.15`, `1.125`, and `1.10` in descending order. Derived family attempts
  by one ceiling operation, reused the frozen allocation policies, rendered actual candidate
  surfaces, and applied bounded exact matching across every rate and discrete submode.
- **Results:** 2,302, 2,253, and 2,203 attempts respectively; all fail exact scheduling in generic
  percentage. Selection configuration SHA-256 is `c5840a94...707e`; content-free evidence SHA-256
  is `df31ac17...7ad7`.
- **Scope result:** No multiplier selected; no complete schedule, smoke, replay, packet, dataset,
  inference, training, or sealed-final access occurred. Templates, generators, labels, verifiers,
  runtime normalizer, reuse rules, and benchmark-contamination controls remained unchanged.
- **Next experiment:** none approved. Decide whether to reduce the accepted signal-pilot size itself.

### EXP-SYN-016: reduced matched-size exact selection

- **Status:** completed; selection gate failed; architecture stop invoked
- **Question:** What is the largest exactly schedulable matched pilot among 900, 800, 700, 600,
  and 500 accepted examples per dataset at a fixed 1.10 attempt multiplier?
- **Frozen method:** Stable largest-remainder family quotas; 90/10 split; 20% output track; frozen
  submode, difficulty, template/surface reuse, runtime identity, exact uniqueness, dual verification,
  contamination, and sealed-final controls. Actual runtime surfaces—not metadata proxies—determined
  schedulability.
- **Results:** All candidates failed. Fixed pools were 1,981, 1,762, 1,544, 1,320, and 1,102.
  Exact blockers were respectively generic ratio-scale, rate-total, two-type allocation, rate-total,
  and ratio-scale. Every latent mode met its unique mathematical requirement.
- **Evidence:** Selection config SHA-256 `073f03bb...bd49`; aggregate evidence SHA-256
  `793c0276...407f`. Complete candidate configurations remain in the ignored raw evidence boundary.
- **Scope result:** No selected size, complete schedule, smoke, replay, assisted packet, dataset,
  training, benchmark evaluation, or sealed-final access.
- **Next experiment:** none in this architecture. A new project-direction decision is required.

### EXP-SYN-017: matched 500-by-2 signal dataset generation

- **Status:** dataset and automatic-quality gates passed; stratified human review pending
- **Question:** Can one frozen 550+550 attempt pool produce matched 500-example targeted and
  generic datasets when reviewed worksheet structures may repeat under bounded balancing?
- **Frozen method:** `matched-template-signal-v1`; exact targeted family quota `275/117/108` and
  generic `167/167/166`; 450/50 split; 100 output-track examples per dataset; unchanged generators,
  dual verifiers, language checks, development contamination model/thresholds, and sealed-final
  prohibition.
- **Result:** Exactly 1,100 attempts produced all 1,000 required acceptances; the 100 remaining
  candidates were quota-filled reserves. False labels, verifier disagreements, language/target
  defects, exact/latent duplicates, cross-dataset/split overlaps, and contamination findings are
  all zero. Replay matched exactly. The blind Codex audit recommended all 1,000 at high confidence;
  genuine human review of the ignored stratified 100-question packet is pending.
- **Evidence:** Schedule `a70cb62c...5eb`; decision/replay `4574c969...ea93`; targeted dataset
  `987712f6...2876`; generic `49294282...2e7e`; language audit `e148e8fd...e99d`; packet
  `ca5a3e01...31ab`.
- **Next experiment:** after the dataset-stage commit is published, create the isolated pinned
  training environment and run the predeclared 32-step QLoRA compatibility smoke.

### EXP-TRAIN-001: RTX 3080 QLoRA compatibility smoke

- **Status:** passed; final matched training not yet run
- **Question:** Can native Windows execute the exact Qwen2.5-1.5B NF4 QLoRA/SFT recipe through
  backward, paged optimization, validation, save, reload, and inference within 9.6 GiB reserved
  VRAM?
- **Frozen method:** Recipe SHA-256 `4a9c6043...0590`; 128 targeted training records; exactly 32
  optimizer steps; 512-token unpacked inputs; effective batch eight; step-25 validation; one ignored
  adapter; no benchmark access or tuning.
- **Result:** Passed. Logged loss moved 2.5810 to 0.5413; validation loss was 0.633282; runtime was
  102.395 seconds; peak allocated/reserved VRAM was 3,343,800,832/3,741,319,168 bytes. The
  89,796,953-byte adapter saved, hashed to `11159bd5...849`, reloaded offline, and generated one
  deterministic original-fixture response.
- **Limitations:** This tests execution compatibility, not generalization. The final comparison is
  provisional pending the stratified human language review and, if positive, a separately approved
  second seed.
- **Next experiment:** publish the frozen setup, then train generic followed by targeted using the
  same 200-step recipe without development-benchmark exposure.

### EXP-TRAIN-002: matched adapter training and parity gate

- **Status:** both adapters trained; parity failed; development evaluation not run
- **Method:** Generic then targeted, each from the pinned base with recipe `4a9c6043...0590`, seed
  `20260720`, 450/50 split, 200 optimizer steps, effective batch eight, and final-adapter-only save.
- **Generic result:** 641.366 seconds; loss 3.1699 to 0.1179; final validation loss 0.153627;
  271,396 non-padding tokens; adapter `36b19165...e3ac`.
- **Targeted result:** 645.737 seconds; loss 2.7859 to 0.1199; final validation loss 0.144995;
  306,766 non-padding tokens; adapter `217a9bcf...406e`.
- **Parity:** All metadata and 819,200 padded tokens per run match, both adapters load offline on
  CUDA, but the non-padding difference is 11.5299% versus the required maximum 2%.
- **Gate:** failed before benchmark access. No generic, targeted, category-level, or one-seed signal
  score exists. This is an experimental-control blocker, not a model-performance result.
- **Next experiment:** none automatically. A separately approved token-budget-matched retraining
  design is required before the research question can be evaluated.

### EXP-TRAIN-003: token-matching protocol and parity smoke

- **Status:** protocol frozen; four-step parity smoke passed; full retraining pending publication
- **Census:** 450 generic records total 77,348 tokens (111-323; mean 171.8844); 450 targeted total
  87,317 (111-320; mean 194.0378). Zero truncation and zero fully masked records; census hashes are
  `eee9b961...e6a3` and `3782412c...9f8`.
- **Method A:** failed. Exact stratified 3/4-repeat boundary was 278,167 generic versus 307,144
  targeted, a 9.4343% difference.
- **Method B:** selected. Whole-example balanced cycles schedule 271,292 generic versus 271,150
  targeted tokens, a 0.05234% difference, across 200 steps. Recipe is
  `foundry-token-matched-qlora-v2` at `df7c7b8d...fa54`.
- **Parity smoke:** fresh adapters processed 5,464 and 5,440 scheduled/actual tokens over four
  steps, a 0.43924% difference. Losses and gradients were finite; scheduler counts, adapter hashes,
  and offline reload passed. Peak reserved VRAM was 3,577,741,312 bytes in both runs.
- **Next experiment:** publish the protocol; then retrain generic followed by targeted from the
  untouched pinned base. Development evaluation remains prohibited until full-run parity passes.

### EXP-TRAIN-004: token-matched one-seed targeted-versus-generic comparison

- **Status:** complete; final one-seed signal gate failed
- **Protocol:** `foundry-token-matched-qlora-v2` (`df7c7b8d...fa54`), Method B, seed
  `20260720`, identical pinned base and software, 200 optimizer/scheduler steps per arm.
- **Training parity:** Generic processed 1,578 occurrences / 271,292 actual loss tokens; targeted
  processed 1,398 / 271,150. Difference: 142 tokens, or 0.052342%. Both adapters load offline and
  all final parity fields pass.
- **Generic evaluation:** 15/814 (1.8428%); 167 extractable (20.5160%); 137 exact-format; zero
  backend failures; 1,357.107 seconds. Adapter `c039612d...5df1`.
- **Targeted evaluation:** 14/814 (1.7199%); 180 extractable (22.1130%); 157 exact-format; zero
  backend failures; 1,359.252 seconds. Adapter `b4a2e55d...b02e`.
- **Paired result:** Targeted wins 11 rows and generic wins 12 (net -1). Targeted-minus-generic is
  -0.12285 percentage points with a fixed-seed 10,000-replicate paired-bootstrap 95% interval of
  [-1.22850, +0.98280] points. The frozen 293-row failure taxonomy shows 0 selected-category fixes
  for either arm; targeted fixes two untargeted rows and generic fixes one.
- **Gate:** Failed the targeted >=529, targeted >=generic+4, and targeted extractability >=91.38%
  clauses. It passed actual-token parity, zero backend failures, and the frozen taxonomy's
  untargeted-decline clause. The result remains provisional pending stratified human language
  review and second-seed confirmation, but a second seed is not recommended from this failure.
- **Interpretation:** The shared extractability collapse is a common training-format or
  instruction-retention failure, not positive or negative evidence about targeted curriculum.
- **Next experiment:** none approved. The narrowest possible next approval would diagnose SFT label
  scope/completion behavior without tuning, retraining, or sealed-final access.

### EXP-TRAIN-005: assistant-only retention diagnosis and bounded recipe gate

- **Status:** both predeclared recipes failed; full retraining and GSM1K evaluation not run
- **Diagnosis:** All 900 prior training rows supervised system/user tokens. Only 200/1,000 assistant
  targets used the evaluator-required terminal line. Adapter application was correct and disabling
  either collapsed adapter restored the untouched base exactly. Causes 1 and 3 are directly
  evidenced; no sole-cause claim is made.
- **Frozen instrument:** 60 original non-benchmark prompts: 30 arithmetic, 15 format, and 15 general
  instruction tasks. Base scored 30/30, 14/15, 14/15, and 59/60 extractable. Suite SHA-256 is
  `0f0b73d8...3eb9`; prompt SHA-256 is `451ed6c7...2fe`.
- **Corrected format:** `foundry-assistant-only-sft-v3`, format SHA-256 `3ffba986...35329`, recipe
  SHA-256 `9a968154...e1df7`. System/user/header/padding/post-EOS labels are all masked; decoded
  labels contain exactly the assistant completion plus final EOS and one terminal answer line.
- **Recipe 1 (`2e-4`):** Exact 14,404-token parity. Generic scored arithmetic/format/instruction
  25/30, 10/15, 10/15; targeted scored 22/30, 13/15, 8/15. Both failed.
- **Recipe 2 (`5e-5`):** Exact 14,404-token parity. Generic scored 28/30, 15/15, 13/15 and failed
  instruction retention. Targeted scored 25/30, 15/15, 14/15 and failed arithmetic retention.
  Extractability was 58/60 and 57/60; echo, question generation, and backend failures were zero.
- **Gate:** Failed. Summary SHA-256 `5d1d5f0d...1201`. No recipe, full adapter, common checkpoint,
  final parity result, corrected GSM1K result, bootstrap interval, or new one-seed signal decision
  exists. A separately approved training-method design is required before further model work.

### EXP-TRAIN-006: retention-safe adaptation ladder

- **Status:** selected calibration variant failed disjoint retention validation; mandatory stop
- **Target audit:** v3 procedural/program-trace style in 376/500 generic and 419/500 targeted;
  concise natural classification in 83 and 54. Blind Codex sample: 100/100 procedural or terse
  trace, explicitly AI-assisted rather than human review.
- **Concise-v4:** All 1,000 records reconstruct, replay, verify, and tokenize with one terminal line;
  maximum target length 41. Format SHA-256 `0d7b8fbd...415a1`.
- **New instruments:** Validation base 45/45 arithmetic, 20/20 format, 23/25 instruction, 90/90
  extractable. Final-holdout base 44/45, 20/20, 23/25 instruction, 89/90 extractable. Suite hashes
  are `96e88c82...1d10` and `3af7b87c...227d`; neither suite overlaps training/development text.
- **Ladder:** Every arm/checkpoint uses exact 3,600-token increments to 14,400. Variant A (v3,
  `5e-5`) passes calibration at 8/16/24/32. Variants B-D (v4 at `5e-5`, `2e-5`, `1e-5`) fail
  instruction retention at all checkpoints. Ladder result SHA-256 `bb180848...7739` selects A/32.
- **Disjoint validation:** Generic 45/45, 20/20, 21/25, 90/90 extractable; targeted 44/45, 20/20,
  21/25, 89/90. Both fail the >=90% instruction requirement. Gate SHA-256 `0dc0a92d...19e4`.
- **Stop:** Full schedules/training, final-holdout adapter evaluation, GSM1K, paired analysis, second
  seed, and sealed-final access were not run. No alternative was selected after validation.

### EXP-TRAIN-007: powered retention adjudication instrument

- **Status:** stopped at untouched-base usability gate; no adapter adjudication or benchmark run
- **Old-slice audit:** Base/generic/targeted instruction scores remain `23/25`, `21/25`, `21/25`.
  Transitions are 21 pass/pass/pass, two fail/fail/fail, and two pass/fail/fail; all four adapter
  failures are genuine instruction noncompliance and prompt/reference/scorer defects are zero.
- **Frozen artifacts:** Adjudication/anchor/holdout contain `300/120/300` items with section layouts
  `100/100/100`, `40/40/40`, and `100/100/100`. All 720 IDs and normalized prompts are unique;
  exact and 12-token overlap with prior retention, synthetic, and development prompts is zero.
  Artifact evidence SHA-256 is `5f19ec76...29e06`.
- **Untouched base:** Adjudication arithmetic `84/100`, format `48/100`, instruction `55/100`,
  overall `187/300`, extractable `268/300`; 32 malformed outputs, zero backend failures, prompt
  echoes, or question generation. Blind failure inspection found 16 terminal-contract failures, 52
  format failures, and 45 deterministic-instruction failures, with zero defective prompts or
  scorers. Gate SHA-256 is `fa1fec57...67d48`.
- **Gate:** Failed. The untouched holdout, both selected A/32 adapters, shared-anchor fallback,
  parity gate, GSM1K, paired analysis, second seed, and sealed-final partition were not run. A new
  explicit experimental decision is required.

### EXP-TRAIN-008: base-conditioned retention adjudication

- **Status:** both existing adapters failed; current SFT line stopped before GSM1K
- **Instrument:** `foundry-base-conditioned-retention-v1` freezes scorer-correct untouched-base IDs
  before any adapter exposure. Adjudication has 187 IDs (`84/48/55`) with subset SHA-256
  `c76df74b...99e1`. Holdout base scored `96/100`, `60/100`, `54/100`, 210/300 overall and 283/300
  extractable; its 210-ID subset (`96/60/54`) has SHA-256 `36be91d0...c7a9`.
- **Generic A/32:** Adjudication `181/187`, Wilson lower `93.18%`, sections `84/84`, `43/48`,
  `54/55`, question generation one. Holdout `197/210`, Wilson lower `89.70%`, sections `90/96`,
  `53/60`, `54/54`, question generation one.
- **Targeted A/32:** Adjudication `181/187`, Wilson lower `93.18%`, sections `84/84`, `43/48`,
  `54/55`, question generation one. Holdout `200/210`, Wilson lower `91.46%`, sections `93/96`,
  `53/60`, `54/54`, question generation zero.
- **Safety:** All four cells have zero prompt echo and backend failures; instruction-family failure
  concentration is at most one on adjudication and zero on holdout. Adapter hashes remain
  `faa4b72b...8f35` and `c4e45543...bb5b`; exact 14,400-token parity is unchanged.
- **Gate:** Failed. Both arms miss >=90% format preservation on both independent subsets, and three
  cells have nonzero question generation. Pair-decision SHA-256 is `433c911d...a237`. GSM1K,
  paired bootstrap, category signal analysis, second seed, and sealed-final access were not run.

### EXP-TRAIN-009: common runtime LoRA-scale calibration

- **Status:** common scale 0.50 passed three retention subsets; frozen GSM1K comparison pending
  publication of the retention decision
- **Mechanism:** `foundry-common-lora-runtime-scaling-v1`, identical common scale for both arms,
  no adapter/base tensor mutation or merge. Scale 0/1 sanity exactly reproduced base/unscaled
  outputs. Source/config/sanity hashes are `1e0506ce...eff`, `938ec15a...dfc`, and
  `9f7605fe...dea`.
- **New instrument:** 450 original prompts (`150/150/150`), suite `b856c8ce...353e`, zero exact or
  12-token overlap with prior retention, synthetic training, or development prompts. Untouched base
  scored `318/450` (`112/127/79`); subset `0884923c...2ded` contains those 318 correct IDs.
- **Search:** Historical 1.00 failed. Scale 0.75 adjudication is `182/187` for each arm and passes,
  but generic/targeted holdout generate 3/4 questions and fail. Scale 0.50 passes all cells:
  adjudication generic `182/187`, targeted `183/187`; holdout `205/210` for both. Scale 0.25 was
  skipped. Selection SHA-256 is `d7455a57...986c`.
- **Independent validation:** Generic `314/318` (`110/112`, `127/127`, `77/79`); targeted
  `315/318` (`111/112`, `127/127`, `77/79`). Both have zero malformed output, question generation,
  echo, and backend failures. Final validation SHA-256 is `6f3e7a29...9cc7`.
- **Gate:** Passed as `retention_approved_common_scaled_short_run_adapters`. Generic then targeted
  frozen 814-item development evaluation is eligible only after this evidence is verified,
  committed, pushed, and synchronized. Human review and second-seed confirmation remain pending.

### EXP-TRAIN-010: common-scale frozen one-seed comparison

- **Status:** complete; frozen signal gate failed on absolute targeted accuracy.
- **Label:** Provisional one-seed result pending stratified human language review and second-seed
  confirmation.
- **Arms:** unchanged Variant A step-32 generic/targeted adapters at common runtime scale `0.50`;
  actual assistant-token exposure `14,400/14,400`; frozen base not rerun.
- **Results:** base `521/814`; generic `387/814` with 94.3489% extractability; targeted `414/814`
  with 94.2260% extractability. Generic/targeted deltas versus base are -134/-107; targeted is +27
  versus generic.
- **Paired analysis:** targeted wins 47, generic wins 20, net +27; point estimate +3.3170 points;
  10,000-replicate paired 95% interval `[+1.3514, +5.2826]` points, seed `20260720`.
- **Gate:** six of seven clauses pass. Targeted fails only the `>=529/814` absolute floor at 414.
  Paired-analysis SHA-256 is `8cd2e7c9556e08850345166b89ed5c1d2c932b96f7ed203e59ef43f50cfcb9ed`;
  decision SHA-256 is `2b4f39b542ebe16a4cdfd4835856b9965de9dc04c2384fffaf12a064d736a0ed`.
- **Next experiment:** none approved. Complete the pending stratified human review, then decide
  whether to stop adaptation or separately authorize a materially different retention-preserving
  architecture. Do not run a second seed from this failed frozen gate.

### EXP-TRAIN-011: contrastive curriculum task vector

- **Status:** exact adapter arithmetic passed; all predeclared scales failed retention; no GSM1K
  evaluation was authorized.
- **Label:** Provisional one-seed result pending stratified human language review and second-seed
  confirmation.
- **Sources:** unchanged Variant A step-32 generic and targeted adapters with exact 14,400-token
  parity. Dense norms are generic `1.6918784364`, targeted `1.6980775191`, and contrastive
  `0.5876302228`; generic-targeted cosine similarity is `0.9399098552`. The differential is
  `34.6056%` of the targeted norm.
- **Construction:** exact `Delta_targeted - Delta_generic` PEFT concatenation, rank 32, unmerged
  and reversible. Ignored adapter SHA-256 is
  `84f02df1cbc5ec1015d096164dbfe3833e166a14eda9ffadf62b5d2d2527c961`.
- **Equivalence:** layerwise maximum absolute error `1.7462298274e-10`; maximum relative
  Frobenius error `2.935335e-7`. FP32 functional maximum/relative logit error
  `5.626678e-5`/`1.959386e-6`. Both equivalence gates passed, with source and base state unchanged.
- **Retention ladder:** Scale 1.00 scored adjudication/anchor `181/187` and `204/210`; 0.75 scored
  `182/187` and `207/210`; 0.50 scored `183/187` and `207/210`; 0.25 scored `184/187` and
  `208/210`. Every adjudication cell passed. Anchor failed solely on nonzero question generation,
  with counts `1`, `2`, `1`, and `2` in descending scale order.
- **Gate:** failed. No scale was selected, no independent final-holdout adapter evaluation occurred,
  and GSM1K development remained untouched. Selection SHA-256 is
  `b41d975f342820ac34ca693d599677994e3f272243c114c313605beb020ad49a`.
- **Decision:** close adapter arithmetic for this project version. Do not tune on GSM1K, test
  another merge method, retrain, or run a second seed automatically. Complete the pending human
  review, then decide between project stop and a separately approved KL/replay-regularized or
  verifier-reward architecture.

### EXP-TRAIN-012: base-behavior replay pre-training gates

- **Status:** stopped at the independent holdout's untouched-base usability gate; no adapter
  training and no GSM1K evaluation.
- **Replay source:** Shared anchor base result `40/40` arithmetic, `20/40` format, `23/40`
  instruction, 83/120 total, zero backend failures. The scorer-correct actual base outputs form an
  83-item replay corpus with SHA-256
  `b511129f89ce450014b78698e9e439bdaa0947657f301c3e99b2a9955b7ab4d1`.
- **Independent instrument:** 450 original prompts, 150 per category; suite SHA-256
  `4f49c42cbae8ce7b5029192786f8ff493a4cc445f940063298e0bd22392b6ef9`. Exact and 12-token audit
  against 3,314 prior prompts found zero overlaps; references self-score with zero known defects.
- **Base result:** arithmetic `84/150`, format `27/150`, instruction `30/150`, overall `141/450`;
  405 extractable, zero prompt echo, zero question generation, and zero backend failures.
- **Gate:** failed format (`27 < 60`), instruction (`30 < 60`), and overall (`141 < 250`).
  Gate-summary SHA-256 is
  `e1bdc1cc14f2e126b8fb43f310b009b47bfef32d31795686259d49c8913d3f8a`.
- **Decision:** do not freeze a base-correct final subset, schedule R20/R20-KL/R40, train adapters,
  run retention selection, or inspect GSM1K. Preserve the failed instrument and request an explicit
  project-stop or new-architecture decision.

### EXP-TRAIN-013: verifier-reward GRPO compatibility

- **Status:** stopped at the RTX 3080 compatibility gate before the first completion; no training,
  retention evaluation, or GSM1K evaluation.
- **Final retention subset:** `84/27/30 = 141` untouched-base-correct
  arithmetic/format/instruction rows; subset SHA-256
  `f56845076a1a59e5ca1a95466541339b56f026e945f86118caec307a690ee4ec`.
- **Schedules:** Each arm freezes 64 prompt groups (52 synthetic, 12 identical replay), four planned
  completions per group, and 6,702 prompt tokens. Generic/targeted manifest SHA-256 values are
  `5848ed6640dda21752ab9692c8e531d9175314a7d5a472616dc19ad834a6351e` and
  `cb13d4d522746bdfa829c9a405defdb0eff0acbd23859dc7fe49457318cc1ccf`;
  schedule-summary SHA-256 is
  `23fede9132f53b7d32f354056c728fc68faa20586a9162e101834db34f71ca64`.
- **Reward/reference contracts:** `foundry-verifier-grpo-reward-v1` uses only frozen arithmetic or
  replay scorers, exact-format bonuses, and additive safety penalties. Implementation/configuration
  hashes are `089650105e29ead3c4ad62f1e0e41263e6c2af5fb8a12cb2851644aca3599616`
  and `4a47359fa3129b1bfd79dd158ecb609177e9b1642a95368c106e016a1554a965`.
  The audited TRL/PEFT path uses the same base with the active adapter disabled and no second model.
- **Frozen runtime:** Configuration SHA-256 is
  `01515d186f2485662ea20ef0b444902bdf368a2b4a8cde335f34bfe1b9222bda`;
  G1/G2 execution hashes are `d7023bf6705702a39dfe8d8718db264f6b2c0e2e211753145ad71e2368f4f4c0`
  and `e31d814fc4bcd9fa94e6b74f48992bb79ec70bcf678e56e620277ea19dbe7bd8`.
- **Compatibility result:** Model and fresh adapter loaded on CUDA without CPU offload or a second
  reference model. The first top-p generation failed because PyTorch 2.5.1+cu121 has no
  deterministic CUDA `cumsum` implementation under strict deterministic algorithms. Counts are
  zero completions, rewards, reference-KL passes, backward passes, optimizer steps, and adapters.
  Failure-summary SHA-256 is
  `8b57b6284c1e7dccd978379162de9519b7af30addbbfb9eb4d5a95a7f2b439a6`.
- **Gate:** failed and hard stop enforced. G1/G2 training, checkpoint retention, independent final
  retention, and GSM1K were not run. No new benchmark result exists.
- **Next decision:** Explicitly approve and freeze a scientifically defensible reconciliation of
  stochastic top-p sampling and deterministic execution, or stop verifier-GRPO. Do not silently
  weaken determinism or alter decoding. Human review remains pending; the existing benchmark label
  stays **Provisional one-seed result pending stratified human language review and second-seed
  confirmation.**

### EXP-TRAIN-014: warning-only top-p exact replay

- **Status:** official same-process replay completed all three runs but failed exact identity;
  verifier-GRPO closed before fresh-process replay or training.
- **Contract:** `foundry-warning-only-top-p-replay-v1`; temperature `0.8`, top-p `0.95`, top-k `50`,
  four generations, maximum completion length `256`, and seed `20260720` remained unchanged.
  Warning-only enforcement was confined to stock generation; all other operations remained strict.
- **Contract evidence:** Implementation SHA-256
  `58358c3960c0a26f28caad2694fcd86f721c5b89490463976cabc46607f9a939`; whitelist
  `79ff68714c1143eca80d368e9432a080e89d2dfcd36de4dde77e951e339caf11`; summary
  `eff84b9ec92715eeb74a6c74bcad5980dded9c4b5482012fd8e2438857f24598`.
- **Run:** Three same-process replays each processed two synthetic groups and one base-replay group,
  four completions per group: `36` completions total, zero optimizer steps. The diagnostic projection
  of all model/evidence fields was equal; every run saw only the approved normalized CUDA-cumsum
  warning, and no warning-only state leaked.
- **Exact failure:** Compatibility source changed between runs 1 and 2; replay-evidence source
  changed between runs 2 and 3. Packet hashes were `68ae4849...d8c`, `80ad3251...a7`, and
  `be3c8aa8...504e`. Failure-summary SHA-256 is
  `8501b7681262ceca002659978c07c688a6f7baa45923ebb3c06e6134adabebe4`.
- **Gate:** failed. Fresh-process replay, both two-step smokes, G1/G2 training, retention, GSM1K,
  category analysis, and bootstrap were not run. No adapter exists.
- **Decision:** Enforce the predeclared hard stop. Do not retry after source stabilization or open
  another verifier-GRPO variant for this project version.

### EXP-TRAIN-015: source-immutable GRPO runtime-root decoupling

- **Status:** orchestration patch verified; immutable V2 replay not yet started.
- **Authorization:** One new experiment may replay the unchanged Milestone 10F scientific contract
  from a detached V2 worktree after an atomic orchestration commit and push.
- **Mechanical correction:** A typed path contract separates immutable source, mutable primary
  repository, exact external interpreter, writable external artifact root, and read-only model
  cache. Canonical traversal, symlink/junction escapes, wrong imports, alternate interpreters, root
  replacement, source drift, cache drift, environment drift, and command-template drift fail closed.
- **Verification:** The prior `165` focused GRPO cases and `17` new path-contract cases pass under
  `PYTHONHASHSEED=20260720` (`182` total). Protected scientific, evaluator, synthesis, and dependency
  files have zero diff from the prior commit.
- **Frozen mechanical identities:** CPython executable SHA-256
  `0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14`;
  planned V2 command-template SHA-256
  `6680c2c4d713882877d1c7e2ab1c47211ec07f2c84cee0464964e4de7b1d3498`.
- **Scientific activity:** zero generation, completions, rewards, backward passes, optimizer steps,
  adapters, checkpoints, retention evaluations, or GSM1K evaluations in this patch phase.
- **Next gate:** Push the single orchestration commit, create the new detached V2 worktree and
  external runtime root, then freeze the complete contract/source manifests before any model load.

#### EXP-TRAIN-015 outcome

- **Status:** failed closed during official same-process replay; project stop.
- **Frozen identities:** commit `b647a3dcadcab941359fbecab2b11c8f9f63cb8d`, tree
  `099a9987df1b0a2d4da85eba33b4e22694ef2ab6`, runtime contract `2400654e...d953`, source manifest
  `72cd61b5...2fab`, and model manifest `5173393f...4006`.
- **Run:** The first generation-only iteration completed 3 groups x 4 completions in memory. Its
  final path-contract validation rejected active cuBLAS `:16:8` after the frozen stock transition
  from launch value `:4096:8`. The error happened before packet or summary persistence.
- **Interpretation:** Orchestration validation failure; no model-side replay comparison or mismatch.
  The source worktree, primary repository, source manifest, and model artifacts remained unchanged.
- **Downstream work:** Fresh-process replay, both two-step runs, G1/G2, retention, GSM1K, paired
  analysis, and the signal gate were not run. Optimizer steps, adapters, and checkpoints are zero.
- **Decision:** Enforce the authorized no-retry rule and publish failure summary
  `0a1c7085a95fef8138c06b17faaa8e0b5c0af195148012ca9a88c7a07a6d1eeb`.

### EXP-TRAIN-016: source-immutable verifier-GRPO V3

- **Status:** orchestration correction verified; V3 execution not yet started.
- **Authorization:** One new V3 experiment explicitly supersedes the V2 no-retry outcome while all
  historical source/runtime directories remain immutable.
- **Mechanical correction:** Predeclare all five Transformers 4.51.3 deterministic environment
  writes before Python launch, including cuBLAS `:16:8`; propagate only an explicit parent
  allowlist; bind the exact interpreter, command, import root, artifact root, and model cache; and
  fail closed on any boundary mutation.
- **Frozen patch evidence:** environment contract `1f80b141...38af`, Transformers file
  `33561736...a95e`, function `18939641...4834`, command template `511500a2...a58b`.
- **Verification:** `198/198` focused GRPO tests, `709/709` repository tests, Ruff, and strict Mypy
  pass. Protected scientific and dependency paths changed: zero.
- **Scientific activity:** zero model generations, backward/optimizer steps, adapters,
  checkpoints, retention evaluations, GSM1K evaluations, or sealed-final access in this phase.
- **Next gate:** Publish `fix: standardize GRPO deterministic environment`, then create and freeze
  only the V3 worktree/runtime roots before same-process replay.

#### EXP-TRAIN-016 outcome

- **Status:** failed closed before model loading; project stop.
- **Frozen identities:** commit `2254b22aa10c9f024eebd56c1f1b98b9a3cf16ab`, tree
  `da9939e50adb11d523fc00dec53a8350df5866d2`, runtime contract `6154aecd...761a`, source manifest
  `f9f48118...c729`, model manifest `5173393f...4006`, and environment `1f80b141...38af`.
- **Run:** The official same-process command used the explicit 30-field child allowlist. Before any
  model load, its CUDA contract query called `nvidia-smi`; NVML initialization failed with exit
  `255`. The same query returned driver `610.47` under the parent environment.
- **Interpretation:** Orchestration allowlist failure; no model-side replay mismatch and no
  scientific result. Zero model generations, packets, rewards, optimizer steps, adapters, or
  checkpoints exist.
- **Downstream work:** Fresh-process replay, both two-step runs, G1/G2, retention, GSM1K, paired
  analysis, and the signal gate were not run.
- **Decision:** Enforce the no-retry rule and publish content-free failure summary
  `b5f0e4b21b496b47a9ae5a93a42d9d9c39bb81b5e2fa7b4ddd36c7432464c2bf`.

### EXP-TRAIN-017: source-immutable verifier-GRPO V4

- **Status:** direct CUDA-runtime orchestration correction verified; V4 execution not yet started.
- **Authorization:** One new V4 experiment supersedes the V3 orchestration stop while preserving all historical worktree/runtime directories.
- **Mechanical correction:** Collect `nvidia-smi` only from the normal parent as non-gating evidence. In the minimized child, prohibit NVML/pynvml and require three identical fixed-tensor `torch.cuda` allocation/arithmetic/matmul/synchronization result hashes on the exact RTX 3080. Use public PyTorch APIs for replay, smoke, and training memory evidence.
- **Frozen patch evidence:** host contract `18a87a86...1f68`, child contract `ead57033...20ec`, probe configuration `1fc17a20...6775`, planned V4 environment `0a5bd3bb...e55d`, and patch summary `712ab82e...e8b5`.
- **Verification:** `213/213` focused GRPO tests, `724/724` repository tests, Ruff, and strict Mypy pass. Protected scientific and dependency paths changed: zero.
- **Scientific activity:** zero model loads, generations, optimizer steps, adapters, checkpoints, retention evaluations, GSM1K evaluations, or sealed-final access in this phase.
- **Next gate:** Publish `fix: validate GRPO GPU through CUDA runtime`, create/freeze only V4, collect host evidence, then run the authoritative fresh child CUDA probe before any model load.

#### EXP-TRAIN-017 outcome

- **Status:** failed closed in the first complete two-step G1 smoke; project stop.
- **Frozen identities:** orchestration commit `a13c31b43a72c3bec205e440aaf7c424ac487d47`, tree `b938e973...f4cb`, runtime contract `a8543712...bc0a`, source manifest `dda8cf58...a8b8`, environment `0a5bd3bb...e55d`, and model manifest `5173393f...4006`.
- **CUDA gate:** Parent RTX 3080/driver `610.47` evidence was monitoring-only. The child direct-PyTorch gate passed without NVML or CPU fallback and repeated `f8850fe4...e5af6` three times.
- **Generation replay:** Three same-process and three distinct fresh-process runs exactly matched common packet `084515f9...ee2f`; summaries are `319be850...043` and `de8ca110...dbb9`.
- **Two-step run:** The first gradient-checkpointed generation raised `generation emitted multiple distinct normalized warning classes` before backward at `0/2`. No two-step packet/metadata, optimizer step, adapter, or checkpoint exists. The duplicate run was not launched.
- **Downstream work:** G1/G2 counted training, retention, GSM1K, category analysis, bootstrap, and signal gate were not run. Sealed-final remained untouched.
- **Decision:** Enforce the no-retry compatibility stop and publish content-free failure summary `164d3e35828758d4eff77b21919b9b3b28dee6238135478fcd2b2e5e024c6f91`.

### EXP-TRAIN-018: immutable V4 training-warning audit

- **Status:** failed closed during evidence-only audit; V5 not created.
- **Scope:** Reconstruct existing V4 identities, packets, summaries, stderr, and warning sources
  without loading or generating from the model.
- **Recoverable classes:** Four logging warnings, each with count one: one B informational class,
  one C automatic-state-transition class that was not accepted, and two E unsupported/ambiguous
  runtime classes. Normalized class hashes are `ad97f015...5ce2`, `5f068b26...c2a5`,
  `594cd40f...42c4`, and `e35718a7...e270`.
- **Fatal evidence gap:** The failed generation captured at least two distinct Python warning
  classes, but V4 persisted none of their class IDs, normalized hashes, categories, sources, or
  counts. The set is UNKNOWN and cannot be inferred from static call paths.
- **Scientific activity:** Zero model reruns, optimizer steps, adapters, checkpoints, retention
  evaluations, GSM1K evaluations, or sealed-final access. Source, dependencies, and scientific
  settings are unchanged.
- **Decision:** Do not create `foundry-grpo-phase-warning-contract-v2` or V5. Publish audit
  `a3e4d1ca40c3fb3f9fe984d3a019ed064a6ba96394a69b009257a248eebf1602` as
  `analysis: stop verifier GRPO after training-warning audit`.

### EXP-TRAIN-019: vetted-corpus targeted versus generic adaptation

- **Status:** authorized; Stage A repository gate passed; source verification pending.
- **Question:** Can Foundry select a more useful curriculum from a vetted human-written problem
  pool than a matched generic selector while preserving the untouched base model's capabilities?
- **Frozen baseline:** Phase 1 release `f4ee93afa4c2be52ca21aef8ca16dbf5827b4a99`; Qwen revision
  `989aa798...a306`; GSM1K revision `bc09569d...0493`; base `521/814` with `752/814`
  extractable; sealed-final untouched.
- **Primary source:** Official ASDiv repository, exact commit/tree/license/citation to be verified
  before download. MathQA train is a fallback only if verified ASDiv base failures cannot support
  `200` disjoint examples per arm under the frozen quotas and matching gates.
- **Design:** Both arms draw only from mathematically verified, uncontaminated base failures.
  Targeted and generic differ in frozen family quotas but match non-curriculum covariates, source
  composition, assistant-token budget, replay, LoRA recipe, seed, steps, and checkpoint rule.
- **Stop rule:** Fail closed at the first unmet gate. Do not rewrite questions, use GSM1K for
  selection, change quotas or training settings, add a dataset/seed/variant, run GRPO/DPO, or access
  sealed-final content.
- **Stage B source freeze:** Official ASDiv commit `883f90a9...abc47`, tree
  `2c3e8723...e52ac`, raw XML `ef890406...c4929`, `2,305` problems, CC BY-NC 4.0, ACL 2020
  attribution. Detached ignored clone is clean; MathQA is inactive; no model process has run.
- **Stage C verification:** `1,497` exact verified, `1,452` supported, `45` verified unsupported,
  and `808` rejected. Supported family counts are `1,126/118/208` for bookkeeping/rate/discrete.
  Duplicate IDs, accepted disagreements, and nondeterminism are zero. Complete replay reproduced
  summary `6c45b435...895d` and supported rows `6478aa3e...c016`; no model process has run.
- **Stages D-E:** MiniLM/lexical screening rejected `73` and retained `1,379` at clean hash
  `8d99a1de...eaac`; all non-semantic overlap and unresolved counts are zero. Clean families are
  `1,076/111/192`. ASDiv-only fails the frozen sizes because rate deficits are `59/30/3` for
  `300/250/200` per arm. Fallback remains inactive until actual base-failure evaluation.
- **Stages F-G:** ASDiv base accuracy was `1,167/1,379`; exact replay and zero backend failures
  passed, but failures `152/22/38` required fallback. Pinned MathQA train verification accepted
  `15,468/29,837`; stable pre-inference selection and contamination left `4,929`. MathQA base
  accuracy was `2,363/4,929`, replay exact, with failure counts `1,214/1,136/216`.
- **Stage H outcome:** **failed closed; experiment stopped.** Sizes `300`, `250`, and `200` were
  tested in order. At `200`, source composition and every categorical gate passed, but
  formula-depth SMD `0.113895` and operation-count SMD `0.108765` exceeded `0.10`. Stop result is
  `1b169ab5...650f`.
- **Scientific activity after stop:** zero targets, splits, training schedules, optimizer steps,
  adapters, checkpoints, retention runs, GSM1K adapter evaluations, bootstrap replicates, or
  signal-gate results. The frozen base was not rerun on GSM1K; sealed-final remained untouched.
- **Decision:** Do not tune matching, lower the gate, train, or evaluate GSM1K. No dataset-freeze
  commit/push exists because Stage K was not reached. New authorization is required for any
  continuation.
