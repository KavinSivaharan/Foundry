# Reproducibility

This guide separates checks anyone can run from model executions that require Foundry's exact frozen
Windows machine state. Phase 1 closeout reran only source, evidence, documentation, and test checks;
it did not rerun inference, training, generation, or sealed evaluation.

## Evidence boundary

Foundry uses three storage layers:

| Layer | Tracked | Contents |
| --- | --- | --- |
| Public release | yes | source, configs, identifier-only manifests, aggregate JSON/CSV, hashes, tests, docs, SVG figures |
| Runtime evidence | no | predictions, synthetic records, review packets, schedules with prompts, adapters, checkpoints, logs |
| Local caches/environments | no | pinned Hugging Face snapshots, virtual environments, CUDA/package state |

The public release can reconstruct result-table arithmetic, the paired interval already recorded in
the committed analysis, retention decisions, zero-step GRPO status, timeline consistency, and every
figure byte. Exact dataset/split and adapter hashes can be recomputed only when the ignored runtime
evidence exists. Model outputs can be replayed only with the pinned local snapshots and environment.

## Frozen machine environment

The QLoRA and final comparison environment was:

- Windows;
- NVIDIA GeForce RTX 3080;
- CPython 3.12.10;
- CUDA runtime 12.1;
- PyTorch 2.5.1+cu121;
- Transformers 4.51.3;
- TRL 0.17.0;
- PEFT 0.15.2;
- bitsandbytes 0.49.2;
- Accelerate 1.7.0;
- dependency lock `requirements-training.lock.txt`, SHA-256
  `fc158cd278124af82406a110afb5efcde2346776dce79875fe3cf6aa5ccb4755`.

The general development environment is `.venv`; the training environment is `.venv-training`. Both
directories are ignored. No permanent `PATH` modification is required.

## Deterministic process environment

Before every Python process used for the final GRPO replay contract, the launching PowerShell process
set:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONHASHSEED = "20260720"
$env:PYTHONNOUSERSITE = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:CUDA_LAUNCH_BLOCKING = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":16:8"
$env:ASCEND_LAUNCH_BLOCKING = "1"
$env:HCCL_DETERMINISTIC = "1"
$env:FLASH_ATTENTION_DETERMINISTIC = "1"
```

`PYTHONHASHSEED` must be present before the interpreter starts. The value `0` is not part of the
frozen contract. The CUDA and accelerator variables are retained even when a particular backend does
not consume them so every launched process is compared under one exact environment record.

## Pinned model and benchmark

| Artifact | Identity |
| --- | --- |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Model revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Benchmark | `ScaleAI/gsm1k` |
| Benchmark revision | `bc09569d09a614b9b530edc7f076fb214ac10493` |
| Development manifest | `5e810d3ab644bef1d43c598a14a6164ba6464b27fde50e92a2f241816ce87897` |
| Evaluator config | `5f315d5de645f9563b8d1e61bc8e02c3513c453238ad9e1d6f9473489b5a622b` |
| Prompt | `738ea5a3b94e7c75ac0bd50a229bbf04f3fc5d773e14658bc6728bc7a4b18350` |
| Canonical extractor | `e099d1c247968fed982cb849022ec3137b1694c15f23a65663a127b8158c06df` |

The sealed-final manifest is isolated from the development workflow. Do not open it as part of public
release verification. All Phase 1 evidence records `sealed_final_accessed: false`.

## Public release verification

After setting the process environment above, run from the repository root:

```powershell
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\mypy.exe
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\render_phase1_figures.py --check
.\.venv\Scripts\python.exe -m pytest tests\unit\test_phase1_release.py
.\.venv\Scripts\python.exe -m pip check
.\.venv-training\Scripts\python.exe -m pip check
git diff --check
git status --short --branch
```

The Phase 1 release test checks summary arithmetic, source hashes, paired confidence values, sealed
status, GRPO zero-step status, README values, CSV schemas, deterministic SVGs, and forbidden tracked
artifact paths.

## Historical evaluator command

The frozen 814-example base run was executed once with this command shape after the model and dataset
were available under the pinned offline cache:

```powershell
$env:HF_HOME = "data/huggingface"
.\.venv\Scripts\python.exe -m foundry.cli development-baseline `
  --base-config configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml `
  --config configs/eval/gsm1k_qwen2_5_1_5b_final_evaluator.yaml `
  --development-manifest configs/eval/manifests/gsm1k_development.json `
  --source-pool-manifest configs/eval/manifests/gsm1k_development_baseline.json `
  --source-baseline-manifest configs/eval/manifests/gsm1k_development_baseline_844.json `
  --baseline-manifest configs/eval/manifests/gsm1k_development_baseline_814.json `
  --output-dir results/raw/development_baseline/qwen2_5_1_5b
```

This is a content-generating GPU command. It is documented for provenance, not required for checking
the public release and not authorized by Milestone 11.

## Dataset reconstruction

The matched dataset contract is frozen by:

- `configs/synthesis/matched_signal_dataset.yaml`;
- `configs/synthesis/matched_signal_schedule.json`;
- `results/synthesis_smoke/matched_signal_dataset_manifest.json`;
- source under `src/foundry/synthesis/template_bank/`.

The ignored runtime root is `results/raw/foundry_500x2_signal_data/`. It contains the 1,100 attempt
records, 1,000 accepted split records, and deterministic replay packet. Canonical JSON hashing of the
accepted records reproduces:

- generic dataset `49294282...2e7e`;
- targeted dataset `987712f6...876`;
- generic training/validation `52276f04...41dd` / `42f51218...58c`;
- targeted training/validation `9f8fef80...e464` / `1ec20743...0ac`.

The public manifest carries only stable IDs, categories, provenance hashes, verifier hashes, and split
assignments. It does not publish questions, solutions, or training completions.

## QLoRA smoke and token-matched training

The historical compatibility smoke used `foundry.training.qlora` with the pinned recipe, model path,
lock, targeted training and validation splits, a fresh ignored output directory, `--max-steps 32`, and
`--compatibility-inference`. It completed forward, backward, optimizer, save, and offline reload.

The full token-matched runs used:

```text
python -m foundry.training.token_matched_qlora
  --recipe configs/training/qwen2_5_1_5b_token_matched_qlora_v2.yaml
  --model-path <PINNED_MODEL_SNAPSHOT>
  --lock requirements-training.lock.txt
  --train <IGNORED_ARM_TRAINING_JSONL>
  --validation <IGNORED_ARM_VALIDATION_JSONL>
  --group <generic_control|targeted>
  --output-dir <FRESH_IGNORED_OUTPUT_DIRECTORY>
  --summary <CONTENT_FREE_SUMMARY_PATH>
  --max-steps 200
```

The generic and targeted processes were independent and both started from the untouched base. Their
actual loss-bearing totals were 271,292 and 271,150, within 0.5%. Those adapters are preserved as
diagnostic evidence of shared SFT collapse; they are not the adapters used for the headline scores.

The final comparison source was retention-safe ladder Variant A checkpoint 32, exactly 14,400
loss-bearing tokens per arm, followed by a uniform runtime scale of 0.50.

## Retention evaluation and scaling

Retention suites are stored as ignored prompt/reference packets with tracked content-free suite and
gate summaries. The workflow was:

1. freeze the suite and scorer identities before adapter outputs;
2. evaluate the untouched base and freeze base-correct subsets where required;
3. evaluate generic then targeted in the predeclared order;
4. compute per-section preservation, Wilson lower bounds, malformed/echo/generation counts, and state
   restoration evidence;
5. apply the gate without consulting GSM1K;
6. select the first common descending scale that passes every required cell;
7. validate the selected scale on a separately frozen final holdout.

The selected scale was 0.50. The final 318-item base-correct holdout preserved 314 generic and 315
targeted behaviors. Adapter scaling was applied in memory and the original adapter/base state was
restored after each evaluation.

## Paired bootstrap reconstruction

The paired analysis command accepts ignored aligned base, generic, and targeted prediction files plus
tracked summaries, taxonomy, parity, and retention evidence:

```text
python -m foundry.training.paired_analysis
  --base-predictions <IGNORED_BASE_PREDICTIONS>
  --generic-predictions <IGNORED_GENERIC_PREDICTIONS>
  --targeted-predictions <IGNORED_TARGETED_PREDICTIONS>
  --generic-summary results/training/common_scale_0_50_generic_development.json
  --targeted-summary results/training/common_scale_0_50_targeted_development.json
  --taxonomy results/development_baseline/qwen2_5_1_5b/complete_failure_taxonomy.json
  --final-parity results/training/common_scale_training_parity.json
  --retention-decision results/training/common_lora_scale_final_validation.json
  --output results/training/common_scale_0_50_paired_analysis.json
```

The implementation sorts by stable ID and uses 10,000 paired nonparametric percentile bootstrap
replicates with seed `20260720`. Reproduction from the ignored aligned predictions is byte-identical.
The public release test independently checks the point-estimate arithmetic and recorded interval.

## Source-immutable GRPO replay

The final GRPO replay separated three roots:

- primary orchestration repository: `C:\Users\Admin\Projects\Foundry`;
- detached immutable source worktree: `C:\Users\Admin\Projects\Foundry-grpo-frozen`;
- external runtime-artifact directory: `C:\Users\Admin\Projects\Foundry-grpo-runtime`.

The V4 source worktree was detached at commit
`a13c31b43a72c3bec205e440aaf7c424ac487d47`, tree
`b938e97308d3e73493cf066ed7a656363657f4cb`. A source manifest froze tracked paths and hashes before
model execution. The interpreter and model cache were external to the source tree; generated packets,
logs, and trainer scratch lived only under the runtime-artifact root.

Three same-process generation runs and three distinct fresh Python processes produced the same common
packet hash `084515f9...ee2f`. The same-process and fresh-process summaries were
`319be850...043` and `de8ca110...dbb9`. The exact replay claim is limited to that frozen source,
interpreter, dependency inventory, GPU/driver, model cache, process command, and launch environment.
It is not a cross-platform determinism claim.

## Why GRPO optimizer training stopped

After the six generation replays passed, the first complete two-step G1 smoke entered its first
generation warning audit. It did not complete the first two-step process and did not reach backward.
Milestone 10J audited the already captured warning set without rerunning the model. Fatal or unresolved
classes remained, including attention-implementation support, dynamic-cache version support, and an
unknown Python-warning identity/category/source/count. Under the frozen contract, incomplete evidence
could not be whitelisted.

The final state is therefore:

- generation replay passed;
- completed generation replays: 6;
- counted GRPO training started: false;
- backward passes completed: 0;
- optimizer steps completed: 0;
- GRPO adapters or checkpoints written: 0;
- route decision: failed closed.

Changing warning policy, dependencies, source, sampling parameters, or determinism settings would
create a new experiment rather than reproduce Phase 1.

## Ignored artifact locations

Important ignored roots include:

- `data/` — model and dataset caches;
- `results/raw/` — predictions, generated datasets, prompt-bearing schedules, audits, and logs;
- `results/raw/training/` — adapters, checkpoints, retention packets, and training diagnostics;
- `.venv/`, `.venv-training/`, `.venv-realization/` — machine-local environments;
- `C:\Users\Admin\Projects\Foundry-grpo-runtime` — external immutable-replay artifacts.

The expected human-review export filename is `foundry-500x2-signal-review.json`. The complete export
must remain ignored even if a future genuine review validates.
