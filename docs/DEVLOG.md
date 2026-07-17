# Foundry Development Log

## 2026-07-16 — Milestone 0: discovery and Phase 1 proposal

### Work completed

- Inspected the repository, branch, remote, history, tracked files, README, and existing ignore rules.
- Confirmed that the repository started clean with no application code or experiment artifacts.
- Researched three model/domain/benchmark/verifier combinations using official model cards, benchmark repositories, and benchmark papers.
- Recommended GSM1K arithmetic reasoning with `Qwen/Qwen2.5-1.5B-Instruct`, pending user approval.
- Documented the proposed architecture, evaluation firewall, data loop, milestones, metrics, risks, unresolved questions, learning notes, and experiment-record template.
- Extended the existing Python `.gitignore` with the requested local data and large-artifact paths.

### Files changed

- `.gitignore`
- `docs/PROJECT_PLAN.md`
- `docs/DECISIONS.md`
- `docs/DEVLOG.md`
- `docs/LEARNING_NOTES.md`
- `experiments/EXPERIMENTS.md`

### Commands run

Repository inspection:

```text
git rev-parse --show-toplevel
git status --short --branch
git remote -v
git log --oneline --decorate -5
find . -maxdepth 3 -type f -not -path './.git/*' -print | sort
sed -n '1,320p' README.md
sed -n '1,360p' .gitignore
```

Documentation setup:

```text
mkdir -p docs experiments
```

Research was performed through the public official pages linked from `docs/PROJECT_PLAN.md`; no API generation job was run.

Verification commands:

```text
test -s <each required documentation file>
grep -Fqx <each required .gitignore rule> .gitignore
grep -Fqx <each required heading> <documentation file>
git diff --check
rg -n <credential and private-key patterns> .gitignore docs experiments README.md
git status --short
git diff --stat
git config user.name
git config user.email
git log -1 --format='%an <%ae>'
git add .gitignore docs experiments/EXPERIMENTS.md
git diff --cached --check
git diff --cached --name-status
git -c user.name=KavinSivaharan -c user.email=140466612+KavinSivaharan@users.noreply.github.com commit -m "docs: define Foundry phase 1 research plan"
```

The configured email was the placeholder `your.email@example.com`. The commit therefore uses the GitHub noreply identity already present on the repository's initial commit, scoped to this one command. Repository and global Git configuration are not changed.

### Tests performed

- No model, dataset, training, or application test was run because no implementation exists and the initial task forbids training.
- Required files: passed; every requested file exists and is non-empty.
- Required `.gitignore` rules: passed.
- Required `PROJECT_PLAN.md` and `LEARNING_NOTES.md` sections: passed.
- `git diff --check`: passed with no whitespace errors.
- Secret-pattern scan: passed with no credential or private-key patterns found.
- Model, dataset, training, and application tests: not run because no implementation exists and the initial task forbids training.

### Outputs and results

- Measured benchmark score: not available.
- Training result: not available.
- API/cloud cost: $0.
- Proposed combination: GSM1K + `Qwen/Qwen2.5-1.5B-Instruct` + exact integer scoring and independent executable verification for synthetic labels.

### Failures or blockers

- The target RTX 3080 VRAM capacity, OS, CUDA version, driver, and free disk are not yet confirmed.
- Exact model, dataset, and package revisions are intentionally unpinned until the proposed evaluation-foundation milestone is approved.
- Memory and runtime figures are estimates, not measurements.

### Next action

Stop and wait for explicit approval or revision of the recommendation. If approved, propose and execute only Milestone 1: the reproducible evaluation foundation and a maximum 10-example smoke evaluation, with no training or full benchmark run.

## 2026-07-16 — Milestone 1: reproducible evaluation foundation

### Work completed

- Inspected the current machine before implementation.
- Confirmed that the machine is an Apple Silicon Mac, not the RTX 3080 desktop, and ruled out a real CUDA smoke run.
- Used the existing Python 3.12.11 installation to create an isolated `.venv`; did not modify system Python, Homebrew, or Conda.
- Used pip-tools because `uv` was not installed. Created exact development and optional smoke dependency locks.
- Verified the current immutable Hugging Face Git revisions for the selected Qwen model and GSM1K dataset.
- Implemented typed YAML configuration loading with strict keys, immutable revision validation, deterministic decoding enforcement, and stable configuration hashing.
- Implemented deterministic manifest construction using hash-ranked pinned row identities.
- Created:
  - 904-example development manifest, SHA-256 `d2c895f43a1e76a12796d6a263b60dc230a9abab58f9624674ec925f37319fae`;
  - 301-example sealed-final manifest, SHA-256 `b0dd1077ad443fc2f9ef31e8ea3d95faa604dff52b3071eaf552827a674308b2`.
- Verified that manifests are exhaustive, disjoint, deterministic, internally signed, configuration-bound, and free of questions and answers.
- Implemented stable two-message prompt rendering and prompt SHA-256 `738ea5a3b94e7c75ac0bd50a229bbf04f3fc5d773e14658bc6728bc7a4b18350`.
- Implemented strict final-integer extraction and scoring.
- Implemented a synthetic fixture loader, fake model backend, pinned Hugging Face dataset loader, optional CUDA model backend, evaluation runner, raw JSONL records, aggregate JSON summaries, runtime/throughput/token accounting, and peak CUDA VRAM measurement hooks.
- Implemented CLI commands:
  - `foundry validate-config`
  - `foundry build-manifests`
  - `foundry evaluate-fixture`
  - `foundry smoke`
- Added unit and integration coverage, including a complete fake-model run with no GPU or download.
- Exercised the CUDA preflight; it refused before network/model access on the ineligible Mac.
- Did not train, generate synthetic data, implement GRPO, run a full benchmark, use paid APIs/clouds, or push.

### Detected environment

- **Operating system:** macOS 15.6.1, build 24G90; Darwin 24.6.0 arm64.
- **CPU/GPU platform:** Apple M2; built-in 10-core Apple GPU; no NVIDIA GPU.
- **Memory:** 8 GB unified memory.
- **System Python:** 3.9.6 at `/usr/bin/python3`.
- **Project Python:** 3.12.11 at `/opt/homebrew/bin/python3.12`.
- **NVIDIA driver:** unavailable; `nvidia-smi` not installed.
- **CUDA toolkit/runtime:** unavailable; `nvcc` not installed.
- **Free disk at inspection:** approximately 14 GiB on `/System/Volumes/Data`.
- **uv:** not installed.
- **Real-model smoke suitability:** unsuitable. No NVIDIA CUDA device is present, memory is limited, and the remaining disk margin is poor for an unnecessary local model cache.

### Files changed

- `pyproject.toml`
- `requirements-dev.lock.txt`
- `requirements-smoke.lock.txt`
- `configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml`
- `configs/eval/manifests/gsm1k_development.json`
- `configs/eval/manifests/gsm1k_sealed_final.json`
- `src/foundry/__init__.py`
- `src/foundry/cli.py`
- `src/foundry/config.py`
- `src/foundry/evaluation/__init__.py`
- `src/foundry/evaluation/backends.py`
- `src/foundry/evaluation/benchmark.py`
- `src/foundry/evaluation/manifests.py`
- `src/foundry/evaluation/prompting.py`
- `src/foundry/evaluation/runner.py`
- `src/foundry/evaluation/scoring.py`
- `tests/unit/test_config.py`
- `tests/unit/test_manifests.py`
- `tests/unit/test_prompting.py`
- `tests/unit/test_scoring.py`
- `tests/integration/test_evaluation_smoke.py`
- `docs/PROJECT_PLAN.md`
- `docs/DECISIONS.md`
- `docs/DEVLOG.md`
- `docs/LEARNING_NOTES.md`
- `experiments/EXPERIMENTS.md`

### Commands run

Machine and repository inspection:

```text
git status --short --branch
git log -3 --oneline --decorate
uname -a
sw_vers
python3 --version
command -v uv
uv --version
nvidia-smi
nvcc --version
system_profiler SPDisplaysDataType
df -h /Users/kavins/Foundry
sysctl -n hw.memsize
```

Revision verification and dataset metadata inspection:

```text
git ls-remote https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct.git HEAD
git ls-remote https://huggingface.co/datasets/ScaleAI/gsm1k.git HEAD
curl -fsSL https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B-Instruct/revision/<revision>
curl -fsSL https://huggingface.co/api/datasets/ScaleAI/gsm1k/revision/<revision>
curl -fsSL https://huggingface.co/api/datasets/ScaleAI/gsm1k/tree/<revision>?recursive=true
```

Environment and dependency locking:

```text
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -e '.[dev]'
.venv/bin/pip-compile --extra dev --strip-extras --no-emit-index-url --no-emit-trusted-host --output-file requirements-dev.lock.txt pyproject.toml
.venv/bin/pip-compile --extra smoke --strip-extras --no-emit-index-url --no-emit-trusted-host --output-file requirements-smoke.lock.txt pyproject.toml
```

Framework and manifest verification:

```text
.venv/bin/ruff format src tests
.venv/bin/ruff format --check src tests
.venv/bin/ruff check src tests
.venv/bin/mypy src
.venv/bin/pytest
.venv/bin/foundry validate-config --config configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml
.venv/bin/foundry build-manifests --config configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml
rg -n 'question|answer|Natalia|Shera' configs/eval/manifests
.venv/bin/foundry smoke --config configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml --manifest configs/eval/manifests/gsm1k_development.json --output-dir results/smoke/qwen2_5_1_5b --limit 10
rm -rf /tmp/foundry-lock-check
/opt/homebrew/bin/python3.12 -m venv /tmp/foundry-lock-check
/tmp/foundry-lock-check/bin/python -m pip install -r requirements-dev.lock.txt
/tmp/foundry-lock-check/bin/python -m pip install --no-deps -e .
/tmp/foundry-lock-check/bin/python -m pip check
/tmp/foundry-lock-check/bin/pytest tests/unit tests/integration
```

The smoke command returned exit code 2 with `CUDA smoke evaluation requires the pinned 'smoke' optional dependencies`. This is the intended pre-download refusal on a non-CUDA machine; no results directory or model/dataset cache was produced by Foundry.

### Tests performed

- Initial check: 29 tests collected, 23 passed and 6 failed.
- Initial failures:
  - manifest construction accidentally converted typed entries into dictionaries;
  - one test string contained a literal backslash-n;
  - strict typing required the PyYAML stub package;
  - formatting and an import modernization were pending.
- All failures were fixed without changing the milestone architecture or weakening validation.
- Final framework check before documentation closeout:
  - Ruff formatting: passed.
  - Ruff linting: passed.
  - Mypy strict type checking: passed for 10 source files.
  - Pytest: 29 passed.
  - Fake-model integration: passed with 3 synthetic examples, 2 correct, 1 intentionally invalid, and no GPU/download.
- Final closeout:
  - Unit tests: 28 passed.
  - Integration tests: 1 passed.
  - Dependency integrity with `pip check`: passed.
  - Fresh temporary Python 3.12 environment installed from `requirements-dev.lock.txt`: passed; all 29 tests passed again.
  - `git diff --check`: passed.
  - Secrets scan: passed.
  - Manifest-content scan for `question` or `answer` fields: passed.
  - Candidate repository file-size review: passed; no non-ignored milestone file exceeds 1 MiB.
  - Repository status review: only the approved Milestone 1 source, tests, configuration, manifests, locks, and documentation were changed.

### Outputs and results

- **Evaluation config SHA-256:** `2a6e737cf3376ae081fd17600e31937824830ecdbb624644e729f6b5752f8eba`.
- **Development examples:** 904.
- **Sealed-final examples:** 301.
- **Fake integration score:** 2/3 on intentionally synthetic test records; this is a software assertion, not a model benchmark.
- **Real-model smoke score:** not measured.
- **Real-model throughput:** not measured.
- **Peak GPU VRAM:** not measured.
- **API/cloud cost:** $0.
- **Downloaded model/dataset artifacts:** none.

### Exact RTX 3080 setup and smoke command

For Linux Bash with Python 3.12 and a driver compatible with CUDA 12.1:

```text
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip==25.1.1
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -r requirements-smoke.lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
HF_HOME=data/huggingface .venv/bin/foundry smoke --config configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml --manifest configs/eval/manifests/gsm1k_development.json --output-dir results/smoke/qwen2_5_1_5b --limit 10
```

For Windows PowerShell, use `py -3.12 -m venv .venv`, replace `.venv/bin/python` with `.venv\Scripts\python.exe`, replace `.venv/bin/foundry` with `.venv\Scripts\foundry.exe`, and set `$env:HF_HOME = "data/huggingface"` before the same smoke arguments.

Do not run the smoke if `nvidia-smi` does not show the RTX 3080, the CUDA assertion fails, or disk space is insufficient for the roughly 3 GB model plus cache/runtime overhead.

### Failures or blockers

- The RTX 3080 desktop's OS, VRAM size, NVIDIA driver, CUDA compatibility, RAM, and free disk remain unknown because that machine is not connected to this task.
- The real CUDA code path and target-platform smoke lock have not been executed on NVIDIA hardware.
- The benchmark loader depends on the public pinned Hub revision remaining retrievable.
- No accuracy, throughput, or peak-VRAM claim is possible until the deferred smoke runs.

### Next action

Stop after the local Milestone 1 commit. The recommended next decision is to open this repository on the RTX 3080 desktop and explicitly authorize the deferred 10-example CUDA smoke. Review that result before approving Milestone 2, which would establish the base model's development benchmark and failure inventory.

## 2026-07-16 22:59:53 -04:00 — Deferred RTX smoke: Step 1 context verification blocked

- **Current step:** Step 1 — verify repository, host, Python, CUDA, and smoke-test prerequisites.
- **Action performed:** Read the project plan, decisions, development log, learning notes, and experiment log; inspected repository state, Windows, RAM, disk, NVIDIA GPU/driver, available Python commands, Python registry entries, and standard Python installation paths.
- **Reason:** The approved smoke evaluation requires a clean pinned repository, the RTX 3080, sufficient local resources, Python 3.12, and CUDA-enabled PyTorch before any dependency or model download.
- **Important commands run:** `git rev-parse --show-toplevel`, `git status --short --branch`, `git log --oneline -2`, `git remote -v`, `git rev-list --left-right --count main...origin/main`, `Get-CimInstance Win32_OperatingSystem`, `py -0p`, `py -3.12 --version`, `where.exe python`, `python --version`, `Get-CimInstance Win32_ComputerSystem`, `Get-CimInstance Win32_LogicalDisk`, `nvidia-smi`, a Python/PyTorch CUDA probe, and read-only checks of Windows Python registry keys and standard installation paths.
- **Result:** Repository root `C:\Users\Admin\Projects\Foundry` is on clean `main`; the existing tracking refs report `0` ahead and `0` behind `origin/main`; commits `f9f579f` and `f1e4dd9` are present; origin is `https://github.com/KavinSivaharan/Foundry.git`. Host is Windows 11 Pro build 26200 with 31.11 GiB RAM and 218.19 GiB free on `C:`. NVIDIA driver 610.47 detects an RTX 3080 with 10,240 MiB VRAM. No compatible Python executable was found, so PyTorch CUDA access could not be tested and no environment or smoke run was started.
- **Files changed:** `docs/DEVLOG.md` only.
- **Errors or uncertainty:** `py` is not installed; `python` resolves only to the Microsoft Store app-execution alias and reports that Python was not found. Windows Python registry keys and the approved standard Python 3.10–3.12, Conda, and Scoop paths returned no installation. Remote synchronization is based on the existing local `origin/main` tracking ref; no fetch, pull, or push occurred.
- **Next action:** Stop and request approval for a compatible Python installation or an explicitly identified existing Python executable. Do not create `.venv`, install dependencies, download the model/dataset, or run the smoke test until that dependency decision is provided.

## 2026-07-16 23:08:31 -04:00 — Deferred RTX smoke resumed: Python 3.12 installation started

- **Current step:** Step 1 — install only CPython 3.12 and verify the real interpreter.
- **Action performed:** Received explicit approval to install Python 3.12 and resume only the deferred Milestone 1 ten-example smoke evaluation. Prepared the exact approved `winget` installation command.
- **Reason:** The earlier context check found no compatible Python, and the pinned smoke environment requires Python `>=3.12,<3.13`.
- **Important command:** `winget install --id Python.Python.3.12 -e --source winget --accept-package-agreements --accept-source-agreements`.
- **Result:** Installation is starting; no result is recorded yet.
- **Files changed:** `docs/DEVLOG.md`.
- **Errors or uncertainty:** Windows may require a UAC or installer confirmation. If it does, execution will pause for the user.
- **Next action:** Run the approved installer, then verify `py -3.12`, its executable and version, and the Python launcher paths from a fresh PowerShell process.

## 2026-07-16 23:10:28 -04:00 — Python 3.12 installation completed

- **Current step:** Step 1 — verify the newly installed CPython 3.12 interpreter.
- **Action performed:** Installed `Python.Python.3.12` version 3.12.10 through `winget`, opened a fresh PowerShell process, tested launcher discovery, located the normal per-user installation when inherited `PATH` did not expose it, and verified both the launcher and interpreter by full path.
- **Reason:** The smoke environment must use a real compatible CPython installation, not the Microsoft Store app-execution alias.
- **Important commands run:** the approved `winget install` command; `py -3.12 --version`; `py -3.12 -c "import sys; ..."`; `where.exe py`; `where.exe python`; and full-path checks of `C:\Users\Admin\AppData\Local\Programs\Python\Launcher\py.exe` and `C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe`.
- **Result:** `winget` installed Python 3.12.10 successfully. The real interpreter is `C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe`, reporting CPython 3.12.10, 64-bit AMD64. The installed launcher at `C:\Users\Admin\AppData\Local\Programs\Python\Launcher\py.exe` selects the same interpreter with `-3.12`.
- **Files changed:** `docs/DEVLOG.md`; Python 3.12 was installed under the normal per-user programs directory by `winget`.
- **Errors or uncertainty:** Fresh inherited PowerShell sessions did not yet resolve bare `py`; bare `python` still resolved to the Microsoft Store alias. This does not block the run because the verified full interpreter path will be used. No permanent PATH change was made.
- **Next action:** Confirm the pinned smoke locks and sealed-final access guard, create `.venv` from the verified interpreter, and install only approved pinned dependencies.

## 2026-07-16 23:11:35 -04:00 — Smoke environment setup started

- **Current step:** Step 2 — validate guardrails and create the reproducible Windows CUDA environment.
- **Action performed:** Read `pyproject.toml`, `requirements-smoke.lock.txt`, the smoke configuration, the development manifest, and the CLI/backend/benchmark/runner/manifest code relevant to dependency pins, CUDA preflight, manifest access, the ten-example cap, and recorded metrics.
- **Reason:** Dependency and data-boundary checks must precede package or model downloads so the run cannot silently change versions, use the sealed-final partition, or exceed ten examples.
- **Important commands planned:** create `.venv` with the verified CPython 3.12.10 executable; install the documented pinned pip version, approved `torch==2.5.1` CUDA 12.1 wheel, `requirements-smoke.lock.txt`, and the project itself without dependency resolution; run `pip check` and a CUDA probe.
- **Result:** Pins match the approved protocol: model `Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, dataset `ScaleAI/gsm1k@bc09569d09a614b9b530edc7f076fb214ac10493`, CUDA float16, greedy decoding, and at most ten examples. The selected manifest is explicitly labeled `development`; the smoke CLI rejects any other partition and rejects limits outside 1–10. The sealed-final manifest file was not read.
- **Files changed:** `docs/DEVLOG.md`; `.venv` will be created next and is ignored by Git.
- **Errors or uncertainty:** No blocker found in the existing pins. CUDA compatibility remains unverified until the wheel is installed and probed against driver 610.47.
- **Next action:** Create `.venv`, install the exact approved packages, verify package integrity, and test CUDA without downloading the model or dataset.

## 2026-07-16 23:17:42 -04:00 — Windows CUDA smoke environment verified

- **Current step:** Step 2 — complete the reproducible environment and CUDA preflight.
- **Action performed:** Created `.venv` with CPython 3.12.10; installed pip 25.1.1, the official PyTorch 2.5.1 CUDA 12.1 wheel, the exact listed smoke lock, and Foundry in editable mode without dependency resolution; ran `pip check` and a direct PyTorch CUDA/device-memory probe.
- **Reason:** The real-model command must fail before network model access unless the pinned runtime can use the target CUDA GPU, and package integrity must be known before attributing later failures to the model pipeline.
- **Important commands run:** `python.exe -m venv .venv`; `.venv\Scripts\python.exe -m pip install --upgrade pip==25.1.1`; `.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121`; `.venv\Scripts\python.exe -m pip install -r requirements-smoke.lock.txt`; `.venv\Scripts\python.exe -m pip install --no-deps -e .`; `.venv\Scripts\python.exe -m pip check`; and the Python/PyTorch CUDA probe.
- **Result:** Environment interpreter is `C:\Users\Admin\Projects\Foundry\.venv\Scripts\python.exe`, CPython 3.12.10. PyTorch is `2.5.1+cu121`, reports CUDA runtime 12.1, returns `torch.cuda.is_available() == True`, and detects `NVIDIA GeForce RTX 3080` with 10,736,893,952 bytes (10,240 MiB) total memory. `pip check` reports no broken requirements. The approved PyTorch wheel download was 2,449.3 MB.
- **Files changed:** `docs/DEVLOG.md`; ignored `.venv`; normal pip download/build caches. No source, config, prompt, manifest, or lock file changed.
- **Errors or uncertainty:** The macOS-generated smoke lock omitted Windows-only transitive packages, so pip resolved `colorama==0.4.6` and `tzdata==2026.3`; no pinned package changed and no lock was edited. Pip advertised a newer release, which was intentionally not installed.
- **Next action:** Set local `HF_HOME=data/huggingface` and run exactly the approved smoke command with `configs/eval/manifests/gsm1k_development.json` and `--limit 10`; do not access the sealed-final manifest.

## 2026-07-16 23:18:08 -04:00 — Ten-example real-model smoke started

- **Current step:** Step 3 — run only the approved pinned real-model smoke evaluation.
- **Action performed:** Reconfirmed that the command names `configs/eval/manifests/gsm1k_development.json`, passes `--limit 10`, and has no sealed-final override. Prepared a process-local `HF_HOME=data/huggingface` and the existing CLI command.
- **Reason:** This run tests model/dataset retrieval, CUDA model loading, prompt/decoding integration, strict parsing, metrics, and RTX 3080 capacity without making a benchmark-scale claim.
- **Important command:** `.venv\Scripts\python.exe -m foundry.cli smoke --config configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml --manifest configs/eval/manifests/gsm1k_development.json --output-dir results/smoke/qwen2_5_1_5b --limit 10` with process-local `HF_HOME=data/huggingface`.
- **Result:** Run is starting; no score is recorded yet. The sealed-final manifest file has not been read or passed to the command.
- **Files changed:** `docs/DEVLOG.md`; ignored `data/huggingface` and `results/smoke/qwen2_5_1_5b` may be created by the approved run.
- **Errors or uncertainty:** Model and dataset downloads depend on the pinned Hub revisions remaining retrievable; float16 model loading may still fail from dependency, CUDA, or VRAM pressure.
- **Next action:** Monitor the single smoke process through download, load, and exactly ten development examples; record any failure without expanding scope.

## 2026-07-16 23:21:41 -04:00 — Ten-example real-model smoke completed

- **Current step:** Step 3 — complete the approved pinned real-model smoke evaluation.
- **Action performed:** Downloaded and loaded the pinned Qwen model and GSM1K dataset revision into ignored local cache storage, then evaluated exactly ten identifiers from the development manifest using the existing CUDA float16 backend, prompt, greedy decoding, parser, and result writer.
- **Reason:** This is the deferred Milestone 1 software/hardware integration check on the target RTX 3080, not a benchmark-scale evaluation.
- **Important command run:** `.venv\Scripts\python.exe -m foundry.cli smoke --config configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml --manifest configs/eval/manifests/gsm1k_development.json --output-dir results/smoke/qwen2_5_1_5b --limit 10`, with process-local `HF_HOME=data/huggingface`.
- **Result:** Exit code 0. Processed 10 development examples: 2 correct, 7 invalid, 0 generation failures, accuracy 0.20. Model download/load time was 120.6527 seconds; evaluation time was 36.8997 seconds; recorded total model-load-plus-evaluation time was 157.5524 seconds; throughput was 0.2710 examples/second. Token counts were 1,387 input and 2,721 generated. Peak CUDA allocation was 3,116,510,720 bytes and peak reservation was 3,315,597,312 bytes. Backend identified the RTX 3080. The sealed-final manifest was not passed to or read by the run.
- **Files changed:** `docs/DEVLOG.md`; ignored model/dataset cache under `data/huggingface`; initial raw and aggregate summary outputs under `results/smoke/qwen2_5_1_5b` (the raw JSONL was moved into ignored `results/raw/` during validation).
- **Errors or uncertainty:** No CUDA, OOM, generation, or dependency failure occurred. Hugging Face warned that symlink caching is unavailable on this Windows setup, which can consume extra disk. Transformers warned about model-default sampling fields that are ignored because `do_sample=False`; the run remained greedy. Seven outputs failed the strict final-answer contract and require result-record inspection before interpretation.
- **Next action:** Validate the ignored result files, calculate invalid rate and human-readable VRAM values, summarize parser failures, measure local cache sizes, and document what the ten-example result does and does not mean.

## 2026-07-16 23:23:51 -04:00 — Smoke results validated and raw records secured

- **Current step:** Step 4 — validate and interpret the measured smoke artifacts.
- **Action performed:** Read the aggregate summary and ten raw prediction records, grouped parser outcomes, recalculated rates and human-readable CUDA memory, measured Hugging Face cache sizes, checked Git ignore behavior, moved the raw JSONL into the existing ignored `results/raw/` tree, and recorded CPU/current disk details.
- **Reason:** The milestone record must distinguish model accuracy from format failures, keep benchmark-level raw records out of Git, quantify local resource use, and avoid overstating a ten-example sample.
- **Important commands run:** PowerShell JSON parsing of `summary.json` and `predictions.jsonl`; recursive file-size measurements under `data/huggingface`; `git check-ignore -v`; `git status --short --branch`; safe single-file `Move-Item` into `results/raw/smoke/qwen2_5_1_5b/predictions.jsonl`; and `Get-CimInstance` CPU/disk queries.
- **Result:** Accuracy is 20% (2/10). Invalid-output rate is 70% (7/10), all from failure to produce exactly one `Final answer:` line; there were no generation failures. One additional response was validly parsed but incorrect. Peak allocated CUDA memory was 2,972.14 MiB (2.902 GiB); peak reserved was 3,162 MiB (3.088 GiB). Generated-token throughput was 73.74 tokens/second over the recorded evaluation interval. Local model cache size is 3,098,955,668 bytes (2.886 GiB). Dataset Hub plus materialized caches total approximately 778,026 bytes (0.74 MiB). The raw JSONL is confirmed ignored under `results/raw/`; the aggregate summary contains no benchmark questions, labels, or model response text and remains a reviewable commit candidate.
- **Files changed:** `docs/DEVLOG.md`; raw artifact moved from unignored `results/smoke/qwen2_5_1_5b/raw/predictions.jsonl` to ignored `results/raw/smoke/qwen2_5_1_5b/predictions.jsonl`. Aggregate `results/smoke/qwen2_5_1_5b/summary.json` remains untracked pending the final atomic commit.
- **Errors or uncertainty:** The 10-example score has very high sampling uncertainty and is not a benchmark conclusion. Seven parser failures show a format-following problem under the fixed prompt, but no prompt or parser change is authorized. The recorded `backend_load_seconds` includes first-run model download/loading, while the CLI's total excludes dataset materialization time. Current free `C:` space is 207.57 GiB after environment and cache creation.
- **Next action:** Update `docs/LEARNING_NOTES.md` and `experiments/EXPERIMENTS.md` with the measured environment, results, warnings, and cautious QLoRA feasibility assessment; then run the required pinned development checks and repository safety review.

## 2026-07-16 23:25:21 -04:00 — Smoke documentation updated; verification started

- **Current step:** Step 5 — finalize durable documentation and run repository verification.
- **Action performed:** Added measured CUDA compatibility, VRAM, parsing, cache, and QLoRA-feasibility lessons to `docs/LEARNING_NOTES.md`; converted `EXP-20260716-001` from deferred to completed with exact environment, result, artifacts, warnings, interpretation, and next-approval boundary; retained the earlier Python blocker and subsequent recovery entries in this DEVLOG.
- **Reason:** The console result must be auditable from the repository while clearly separating a ten-example plumbing test from benchmark or training evidence.
- **Important commands planned:** install the existing pinned development lock needed for requested checks; run Ruff format check and lint, strict Mypy, unit tests, integration tests, `pip check`, `git diff --check`, the existing secret scan, manifest-content/safety checks, candidate file-size review, and repository status/diff review.
- **Result:** Documentation now records the completed smoke and states that 10 GiB is promising but not proven for QLoRA training. No source, configuration, prompt, scoring, benchmark manifest, or lock file has changed.
- **Files changed:** `docs/DEVLOG.md`, `docs/LEARNING_NOTES.md`, `experiments/EXPERIMENTS.md`, and aggregate `results/smoke/qwen2_5_1_5b/summary.json`. Raw predictions remain ignored.
- **Errors or uncertainty:** The aggregate score is not statistically meaningful; the high invalid-format rate is recorded without changing the fixed evaluation behavior. The development tools are not in the smoke lock and must now be installed from the existing pinned development lock solely to run the explicitly requested checks.
- **Next action:** Install `requirements-dev.lock.txt` into the same isolated `.venv`, run all requested checks, fix only smoke-caused documentation or artifact issues, then create one local atomic commit without pushing.

## 2026-07-16 23:28:27 -04:00 — Final checks passed; local commit prepared

- **Current step:** Step 5 — complete quality, safety, and repository review before the atomic local commit.
- **Action performed:** Installed the existing pinned development lock, ran formatting, linting, strict type checking, all tests, dependency integrity, whitespace, secret-pattern, development-manifest content, aggregate-summary content, raw-record count/ignore, candidate-size, diff, and status checks. Fixed the only check failure with a narrow cross-platform Mypy rule for the external untyped `datasets` package and mechanically reformatted the touched import.
- **Reason:** The user required the complete verification suite and authorized fixes only for issues directly caused by or required for this smoke. A combined dev+smoke environment exposes `datasets` as installed but untyped, unlike the earlier dev-only environment where it was absent.
- **Important commands run:** `.venv\Scripts\python.exe -m pip install -r requirements-dev.lock.txt`; `.venv\Scripts\python.exe -m pip check`; Ruff format/check; `mypy src`; `pytest tests/unit tests/integration`; `git diff --check`; targeted `rg` safety scans; `git check-ignore -v`; candidate file-size enumeration; `git diff --stat`; and `git status --short --branch`.
- **Result:** Ruff format check: 15 files already formatted. Ruff lint: passed. Mypy strict: no issues in 10 source files. Pytest: 29 passed (28 unit, 1 integration). `pip check`: no broken requirements. Whitespace, secret, content-leak, raw-count/ignore, and file-size checks passed. The raw prediction file has exactly ten lines and is ignored. No candidate exceeds 0.03 MiB. Repository changes are limited to smoke documentation/evidence plus the directly required Mypy portability fix.
- **Files changed:** `docs/DEVLOG.md`, `docs/LEARNING_NOTES.md`, `experiments/EXPERIMENTS.md`, `pyproject.toml`, `src/foundry/evaluation/benchmark.py`, and `results/smoke/qwen2_5_1_5b/summary.json`. Ignored `.venv`, `data/huggingface`, and `results/raw` artifacts remain local.
- **Errors or uncertainty:** The first Mypy run failed because the inline `import-not-found` ignore became an `import-untyped` error when the smoke dependency was installed. The final targeted module override preserves the prior intended `Any` boundary without changing runtime behavior. Git reports expected LF-to-CRLF conversion warnings on Windows. Repository Git user name/email are not configured, so the final commit will use the repository's existing GitHub noreply identity only for that command, without modifying Git configuration.
- **Next action:** Stage exactly the six listed reviewable files, run staged diff/safety checks, create one local commit, verify the clean branch state and commit hash, do not push, and stop before Milestone 2.
