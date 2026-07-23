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

## 2026-07-17 00:01:02 -04:00 — Milestone 1.5 Step 1: RTX smoke commit published

- **Current step:** Step 1 — push only the completed RTX smoke commit before calibration work.
- **Action performed:** Verified repository root, branch, clean worktree, exact local/remote hashes, and ahead/behind count; pushed explicit ref `c1ef561:main`; reverified local and remote state.
- **Reason:** Milestone 1.5 must begin from the already approved and published RTX smoke result without mixing later calibration changes into that push.
- **Important commands run:** `git rev-parse --show-toplevel`, `git branch --show-current`, `git status --short --branch`, `git rev-parse HEAD`, `git rev-parse origin/main`, `git rev-list --left-right --count main...origin/main`, and `git push origin c1ef561:main`.
- **Result:** `C:\Users\Admin\Projects\Foundry` is on clean `main`; both `main` and `origin/main` resolve to `c1ef56184723c092a39937009fa2565aa1841c3c`; branch is 0 ahead and 0 behind. Only `c1ef561` was pushed.
- **Files changed:** `docs/DEVLOG.md` after push confirmation; the push itself changed no local files.
- **Errors or uncertainty:** The first credential-manager dialog was cancelled and the first push failed without remote change. A retry completed GitHub authentication and pushed successfully.
- **Next action:** Inspect only the seven invalid model responses in the existing ignored raw prediction file, categorize their formatting deviations, and identify verified causes without changing the parser.

## 2026-07-17 00:02:10 -04:00 — Milestone 1.5 Step 2: seven invalid outputs diagnosed

- **Current step:** Step 2 — inspect and categorize only the seven invalid RTX smoke responses.
- **Action performed:** Parsed the ignored raw JSONL in memory, selected only non-generation failures, and inspected only `row_index`, output-token count, parser error, and model response. Rendered the current prompt through the locally cached Qwen chat template with `[QUESTION REDACTED]`, and probed the strict parser with sanitized boxed-only and required-line responses.
- **Reason:** Prompt calibration must address a verified output-format failure rather than guessing, loosening the scorer, or exposing benchmark questions and labels in tracked files.
- **Important commands run:** PowerShell JSON filtering over `results/raw/smoke/qwen2_5_1_5b/predictions.jsonl`; local-only `AutoTokenizer.apply_chat_template`; sanitized `score_response` probes.
- **Result:** All seven invalid responses omitted the literal `Final answer:` line. Aggregate categories: 1 boxed-answer-only conclusion; 4 prose conclusions with units or currency; 2 prose conclusions using inline LaTeX or bold numeric emphasis. Sanitized endings include `\[\boxed{<integer>}\]`, `Therefore ... <integer> minutes.`, `Therefore ... **$<integer>**.`, and `... is \(<integer>\).` No response contained multiple candidate final lines or trailing commentary after a valid contract line. All were coherent completions of 239–350 tokens, far below the 512-token limit.
- **Files changed:** `docs/DEVLOG.md` only. Raw predictions remain ignored and unchanged; no benchmark question, reference label, or complete raw response was copied into tracked files.
- **Errors or uncertainty:** The first sanitized parser probe passed an integer reference where the scoring helper expects the dataset's string reference, producing an `AttributeError`; the corrected probe passed. Verified cause: the model followed its learned prose/boxed-answer habits instead of the required terminal contract. The chat template preserved the full instruction, generation did not truncate, and the parser correctly accepted `Final answer: 42` while rejecting boxed-only prose by design. Therefore there is no verified parser bug or chat-template bug. The existing instruction is understandable but not sufficiently forceful/salient for this model.
- **Next action:** Create and test a deterministic, identifier-only 30-example prompt-format calibration manifest selected solely from development identifiers, plus an explicitly disjoint future-baseline manifest; do not read or modify the sealed-final manifest.

## 2026-07-17 00:07:37 -04:00 — Milestone 1.5 Step 3: calibration IDs reserved

- **Current step:** Step 3 — create deterministic prompt-format calibration and future-baseline development subsets.
- **Action performed:** Added a typed, hashed development-subset format; deterministic hash-ranked selection; pair validation; load/save integrity checks; a 30-example-only calibration CLI; prompt-only variant enforcement; and unit tests. Generated the 30-ID prompt-format calibration manifest and the disjoint 874-ID future-baseline manifest twice.
- **Reason:** Prompt selection must use a small predeclared development subset that will never be counted in the later main baseline, while retaining only stable identifiers and preserving the benchmark firewall.
- **Important commands run:** Ruff format/lint; strict Mypy; targeted Pytest; `foundry build-format-calibration` with size 30 and seed `foundry-gsm1k-prompt-format-calibration-v1`; SHA-256 file comparisons; identifier-only content scan.
- **Result:** Calibration purpose is `prompt_format_calibration`, 30 IDs, semantic manifest SHA-256 `a020b74b626e75c1197abc35942e85d929463cfe2bfaac1364806bcab1743ee4`. Future-baseline purpose is `main_development_baseline`, 874 IDs, semantic manifest SHA-256 `d6bb412367a44b6c9fc1695bfa856650c42f90a8ee942223c010511c10f7e1eb`. The subsets are disjoint, their union is the canonical 904-ID development manifest, repeated generation was byte-identical, and scans found no question or answer fields. Three new calibration unit tests passed; Ruff and strict Mypy passed.
- **Files changed:** `src/foundry/evaluation/calibration.py`, `src/foundry/cli.py`, `tests/unit/test_calibration.py`, `configs/eval/manifests/gsm1k_prompt_format_calibration.json`, `configs/eval/manifests/gsm1k_development_baseline.json`, initial prompt variant `configs/eval/gsm1k_qwen2_5_1_5b_format_v1.yaml`, and `docs/DEVLOG.md`.
- **Errors or uncertainty:** Ruff initially found only CLI import ordering, which its pinned mechanical fix resolved. The future Milestone 2 command must use the new 874-ID baseline manifest rather than the original 904-ID development superset. The selected prompt is not yet known; `format_v1` is only a candidate until measured.
- **Next action:** Run the current prompt on exactly the 30 calibration IDs, record validity/accuracy/tokens/runtime/VRAM/failure categories, then run the minimal `format_v1` prompt on the identical identifiers with unchanged model, dataset, parser, and generation controls.

## 2026-07-17 00:17:16 -04:00 — Milestone 1.5 Step 4: three-prompt calibration completed

- **Current step:** Step 4 — compare at most three prompt variants on the same 30 calibration identifiers.
- **Action performed:** Ran the current prompt, minimal `format_v1`, and explicit-contract `format_v2` through the prompt-only calibration CLI. Verified identical stable-ID/row-index sequences, moved all raw JSONL files into ignored `results/raw/calibration/<variant>/`, and compared aggregate metrics and response-only failure categories.
- **Reason:** The milestone must optimize format compliance without changing model, data, parser, decoding, or example membership, and it must stop at 90 real-model generations.
- **Important commands run:** three `foundry format-calibrate` commands with the same base config, canonical development source, 30-ID calibration manifest, CUDA runtime, and output controls; PowerShell summary/category analysis; identity-sequence comparisons; `git check-ignore -v` for each raw artifact.
- **Result:** Exactly 90 generations ran (30 per prompt), with zero generation failures and identical identifiers. Current prompt: 5/30 valid (16.67%), 25/30 invalid (83.33%), 1/30 correct (3.33%), 305.97 average output tokens, 124.190 seconds, 0.2416 examples/second, 2,983.43 MiB allocated and 3,192 MiB reserved. `format_v1`: 3/30 valid (10.00%), 27/30 invalid (90.00%), 2/30 correct (6.67%), 287.37 average output tokens, 114.161 seconds, 0.2628 examples/second, 2,983.60 MiB allocated and 3,192 MiB reserved. `format_v2`: 13/30 valid (43.33%), 17/30 invalid (56.67%), 3/30 correct (10.00%), 230.60 average output tokens, 91.467 seconds, 0.3280 examples/second, 2,990.10 MiB allocated and 3,192 MiB reserved.
- **Files changed:** aggregate summaries under `results/calibration/current`, `results/calibration/format_v1`, and `results/calibration/format_v2`; ignored raw files under `results/raw/calibration/`; candidate prompt config `configs/eval/gsm1k_qwen2_5_1_5b_format_v2.yaml`; `docs/DEVLOG.md`.
- **Errors or uncertainty:** A read-only PowerShell report initially had an empty-pipe syntax error and was corrected without changing results. One current and one `format_v1` response reached the 512-token limit; `format_v2` had none, so no generation-setting change was made. Failure categories were: current — 7 boxed, 15 prose/inline, 1 alternate label, 1 alternate wording, 1 incomplete; `format_v1` — 4 boxed, 15 prose/inline, 4 alternate labels, 3 alternate wording, 1 incomplete; `format_v2` — 1 boxed, 10 prose/inline, 2 alternate labels, 4 alternate wording. Math accuracy is recorded only as secondary diagnostic information.
- **Next action:** Do not select or freeze any prompt because none reached the required 90% valid-output gate. Document the failed gate and alternatives across the plan, decisions, learning notes, and experiment log; add exact rendering/hash tests for the tried variants as audit evidence, run the full quality/safety suite, and commit Milestone 1.5 locally without pushing.

## 2026-07-17 00:20:25 -04:00 — Milestone 1.5 Step 5: admission gate failed and documented

- **Current step:** Step 5 — apply the predeclared selection rule and update durable project records.
- **Action performed:** Declined to select/freeze any prompt; added decision D-008; updated the project plan/current phase/next gate, learning notes, and experiment ledger; pinned exact rendering and prompt hashes for the current and two tried variants in tests; retained all aggregate summaries and kept raw predictions ignored.
- **Reason:** The best relative prompt still had a majority-invalid output rate. The project must preserve the 90% absolute gate instead of choosing a winner post hoc or loosening the strict parser.
- **Important actions:** compared each variant against the admission criteria; recorded alternative prompts, failure categories, token/runtime/VRAM measurements, and the permanent 30/874 development separation; added exact prompt-hash assertions.
- **Result:** No prompt configuration is selected for Milestone 2. D-008 records that current (16.67%), `format_v1` (10.00%), and `format_v2` (43.33%) all failed the 90% validity threshold. The main-development baseline remains untouched and must later use only the reserved 874-ID manifest. Documentation explicitly states that Milestone 2 is not ready.
- **Files changed:** `docs/PROJECT_PLAN.md`, `docs/DECISIONS.md`, `docs/DEVLOG.md`, `docs/LEARNING_NOTES.md`, `experiments/EXPERIMENTS.md`, `tests/unit/test_prompting.py`, both candidate prompt configs, calibration code/tests/manifests, and three aggregate result summaries.
- **Errors or uncertainty:** No tested prompt qualifies. `format_v2` is evidence for improved compliance and shorter output, not a frozen evaluation prompt. Any further prompt, parser, constrained-decoding, or generation-control change requires a new explicit approval; material scoring changes remain prohibited.
- **Next action:** Run the complete formatting, linting, strict typing, unit/integration, dependency, whitespace, secret, benchmark-leak, raw-ignore/count, artifact-size, diff, and status checks; fix only Milestone 1.5 issues, then create one local atomic commit without pushing.

## 2026-07-17 00:23:45 -04:00 — Milestone 1.5 Step 6: verification passed

- **Current step:** Step 6 — verify the bounded calibration implementation, evidence, documentation, and repository safety before the local atomic commit.
- **Action performed:** Mechanically formatted Python files, ran the full static-analysis and test suite, checked installed-package consistency and whitespace, scanned candidate files for credential patterns and benchmark-content leakage, verified aggregate summaries contain no raw prompts or responses, confirmed every raw variant has exactly 30 ignored records, reviewed candidate sizes, and inspected the complete status/diff and artifact schemas.
- **Reason:** Milestone 1.5 must leave reproducible identifier-only calibration evidence without leaking benchmark content, committing raw generations, weakening quality gates, or changing unrelated project behavior.
- **Important commands run:** Ruff format and format check; Ruff lint; strict `mypy src`; `pytest tests/unit tests/integration`; `python -m pip check`; `git diff --check`; targeted `rg` secret/content scans; `git check-ignore`; JSON manifest/summary inspection; candidate-size enumeration; `git diff --stat`; and `git status --short --branch`.
- **Result:** Ruff reports 17 files formatted and all lint checks passed. Strict Mypy found no issues in 11 source files. Pytest passed all 34 tests (33 unit and 1 integration) in 0.22 seconds. `pip check`, whitespace, secret-pattern, manifest-content, summary-content, raw-count/ignore, and candidate-size checks passed. Each raw calibration file has exactly 30 records and is ignored; no candidate file exceeds 1 MiB. The two identifier manifests retain the expected 30/874 disjoint split and hashes, and all three aggregate summaries match the documented prompt hashes and measurements.
- **Files changed:** one Milestone 1.5 Python file was mechanically reformatted; this final verification entry was added to `docs/DEVLOG.md`. No dependency, raw result, project data, or sealed-final artifact changed.
- **Errors or uncertainty:** No check failed. Git emitted expected LF-to-CRLF conversion warnings for this Windows worktree. The scientific blocker remains unchanged: the best tested prompt reached only 43.33% validity, below the required 90%, so no prompt is selected and Milestone 2 remains blocked.
- **Next action:** Stage exactly the Milestone 1.5 code, configs, identifier manifests, aggregate summaries, tests, and documentation; run staged whitespace/name/size/secret/content reviews; create one local atomic commit; verify it is one commit ahead of `origin/main`; do not push or begin Milestone 2.

## 2026-07-17 00:25:09 -04:00 — Milestone 1.5 Step 6: staged review passed

- **Current step:** Step 6 — final staged-scope and safety review.
- **Action performed:** Staged the exact approved 16-file Milestone 1.5 set and checked staged filenames, whitespace, credential patterns, identifier-manifest content, aggregate-summary content, prohibited paths, unstaged changes, and diff statistics.
- **Reason:** The atomic commit must contain reproducible calibration evidence and documentation only, with no raw predictions, model/cache data, virtual environment, or sealed-final artifact.
- **Important commands run:** explicit `git add -- <16 paths>`; `git diff --cached --check`; `git diff --quiet`; staged `git grep`; staged manifest/summary scans via `git show`; `git diff --cached --stat`; `git diff --cached --name-status`; and `git status --short --branch`.
- **Result:** All staged checks passed. Exactly 16 expected files are staged; no prohibited path is present; no unstaged tracked change remains; the staged secret/content scans found no match.
- **Files changed:** this final entry extends `docs/DEVLOG.md`; it will be restaged into the same atomic commit.
- **Errors or uncertainty:** The first read-only staged-check script treated the empty output from successful `git diff --quiet` as a failure because it tested PowerShell output truthiness. The corrected script checked `$LASTEXITCODE` and passed. No repository file or result was changed by the script error. The only remaining scientific uncertainty is the documented failed 90% validity gate.
- **Next action:** Restage `docs/DEVLOG.md`, repeat staged whitespace/scope checks, create one local Milestone 1.5 commit, verify a clean branch exactly one commit ahead of `origin/main`, and stop without pushing.

## 2026-07-17 00:32:20 -04:00 — Milestone 1.6 Step 1: calibration commit published

- **Current step:** Step 1 — publish only the completed Milestone 1.5 calibration commit before answer-extraction work.
- **Action performed:** Verified repository root, branch, clean worktree, exact local/remote hashes, and ahead/behind count; pushed explicit ref `5e873f1:main`; fetched and reverified both refs.
- **Reason:** Milestone 1.6 must begin from a clean, published calibration checkpoint without mixing new extractor work into the previous commit.
- **Important commands run:** `git rev-parse --show-toplevel`, `git branch --show-current`, `git status --short --branch`, `git rev-parse HEAD`, `git rev-parse origin/main`, `git rev-list --left-right --count main...origin/main`, `git push origin 5e873f1:main`, and a read-only fetch/reverification.
- **Result:** Repository is `C:\Users\Admin\Projects\Foundry` on `main`; local `main` and `origin/main` both resolve to `5e873f10dfca2b9ad4ebd2dc30bee62c72008aa3`; branch is 0 ahead and 0 behind; worktree was clean after the push. Only the existing Milestone 1.5 commit was published.
- **Files changed:** `docs/DEVLOG.md` after push confirmation; the push itself changed no local file.
- **Errors or uncertainty:** None. Git authentication was already available and the explicit push completed normally.
- **Next action:** Inspect the existing strict parser, scoring records, summaries, and tests; design a separate deterministic terminal-answer extractor and its failure taxonomy without changing exact-format compliance behavior or generating model output.

## 2026-07-17 00:39:08 -04:00 — Milestone 1.6 Step 2: separate canonical extractor implemented

- **Current step:** Step 2 — build and test a deterministic answer extractor without weakening the exact-format parser.
- **Action performed:** Added a separate terminal-answer grammar and scorer, versioned failure taxonomy, conflict and truncation rejection, stable specification hash, aggregate re-scoring utility/CLI, and dual-metric evaluation records/summaries. Kept `extract_final_integer` and its strict literal-line contract unchanged.
- **Reason:** Mathematical benchmark accuracy must be measured independently from whether a model obeys the requested `Final answer:` syntax, while remaining deterministic, generic, auditable, and conservative about ambiguity.
- **Important commands run:** targeted Ruff format/lint, strict Mypy, extractor and re-scoring unit tests, integration test, and a local extractor identity/hash query.
- **Result:** Extractor ID is `foundry-terminal-integer-v1`; specification SHA-256 is `7745fdfebfa7d4d791fa29aff116084298090a64475ea3dd182c65eb99397900`. It accepts explicit terminal markers, standalone boxed/bold values, answer/result statements, conclusion-cued statements with optional units/currency, integral decimals, signs, and valid commas. It rejects truncation, conflicts, non-integral decimals, malformed numbers/expressions, ambiguous answer-like endings, empty responses, and responses with no clear terminal answer. Ruff and strict Mypy passed; 21 extractor tests, one re-scoring test, and one integration test passed.
- **Files changed:** `src/foundry/evaluation/answer_extraction.py`, `src/foundry/evaluation/rescoring.py`, `src/foundry/evaluation/runner.py`, `src/foundry/cli.py`, `tests/unit/test_answer_extraction.py`, `tests/unit/test_rescoring.py`, `tests/integration/test_evaluation_smoke.py`, and `docs/DEVLOG.md`.
- **Errors or uncertainty:** Two initial test fixtures accidentally used literal `\\n` text rather than newline characters; both were corrected. Ruff also mechanically reordered one import. These were test/tooling issues and did not require relaxing extractor rules. Real-output behavior has not yet been measured.
- **Next action:** Re-score the existing three 30-record calibration prediction files without generation, write content-free aggregate summaries, enumerate every newly accepted response, and manually verify that each extraction matches the model's clear terminal intent rather than an arbitrary last number.

## 2026-07-17 00:46:18 -04:00 — Milestone 1.6 Step 3: 90 outputs re-scored and audited

- **Current step:** Step 3 — calibrate the extractor on existing outputs, compare prompts, and manually audit every newly accepted result.
- **Action performed:** Re-scored the three existing ignored 30-record prediction files without generation; reviewed rejected terminal excerpts; tightened overly broad plain-number, line-crossing conclusion, comma, malformed-wrapper, and contextual-word rules; added generic Markdown/LaTeX answer-label forms; reran tests and summaries after each correction; manually reviewed all 63 outputs accepted by the canonical extractor but rejected by the strict parser.
- **Reason:** The extractor must recover clear intended integer answers without guessing from arbitrary intermediate values, and every broadened acceptance rule must be supported by manual evidence before fresh validation.
- **Important commands run:** three `foundry rescore-answers` commands with `--max-new-tokens 512`; response-only audit scripts over ignored JSONL; targeted Ruff, strict Mypy, extractor/re-scoring/integration tests; and extractor hash queries. No backend or model-generation command ran.
- **Result:** Final extractor ID is `foundry-terminal-integer-v1`, SHA-256 `ffce6538526f9aa21e05ce4d9d6830ec71d3a6334a23fa1e9c7beef3c2053946`. Current prompt: 29/30 extractable (96.67%), 5/30 exact compliant (16.67%), 15/30 correct (50.00%), one token-limit rejection. `format_v1`: 28/30 extractable (93.33%), 3/30 exact compliant (10.00%), 15/30 correct (50.00%), one token-limit and one non-integral rejection. `format_v2`: 27/30 extractable (90.00%), 13/30 exact compliant (43.33%), 13/30 correct (43.33%), one malformed-number and two non-integral rejections. All runs had zero generation failures because these were existing outputs. Manual audit covered 63 newly accepted outputs (24 current, 25 `format_v1`, 14 `format_v2`) and found zero false extractions.
- **Files changed:** extractor/scoring integration and tests from Step 2; aggregate summaries under `results/extraction_calibration/{current,format_v1,format_v2}/summary.json`; content-free `results/extraction_calibration/manual_audit.json`; `docs/DEVLOG.md`. Existing raw predictions remain ignored and unchanged.
- **Errors or uncertainty:** Calibration exposed and corrected several conservative or overbroad grammar edges; each final rule is generic and syntax-based. A bare integer is accepted only when it is the entire response. The current prompt has the highest extractability and is selected for fresh validation; its one 512-token calibration output remains a known risk. Math accuracy did not break a prompt-selection tie or drive extractor rules.
- **Next action:** Deterministically reserve 30 identifiers from the existing 874-ID future-baseline pool, leave a disjoint 844-ID main-baseline pool, add integrity/overlap tests, verify the sealed-final path is untouched, then run only the current prompt on those 30 fresh validation identifiers.

## 2026-07-17 00:50:03 -04:00 — Milestone 1.6 Step 4: fresh validation IDs reserved

- **Current step:** Step 4 — create the isolated fresh format-validation set and freeze the remaining candidate baseline IDs before model use.
- **Action performed:** Added a parent-hashed answer-validation manifest schema, deterministic selector, loader/saver, pair validator, evaluation adapter, bounded CLI commands, and tests. Generated the manifests twice with seed `foundry-gsm1k-answer-extraction-validation-v1`; verified byte identity, identifier-only content, source binding, overlap, and complete 874-pool coverage.
- **Reason:** The fresh admission result must not reuse the 30 prompt-calibration examples or contaminate the eventual main-development baseline.
- **Important commands run:** Ruff format/lint; strict Mypy; six validation/calibration unit tests; two identical `foundry build-answer-validation` runs; SHA-256 comparisons; targeted content scan; in-memory overlap/coverage validation; repository changed-path review.
- **Result:** Fresh answer-validation manifest contains 30 IDs with semantic SHA-256 `9582e4b07706a391f00dcfa0d9c68ee86a70cebca6c35f10daa3f3f66c9063f6`. Remaining main-baseline manifest contains 844 IDs with semantic SHA-256 `4f80bbe2f6de4fb33e57ed5463a1c393716a3b64d3b98a08767a7f8056648d79`. Original calibration (30), fresh validation (30), and main baseline (844) are pairwise disjoint; validation plus baseline exactly equals the reserved 874-ID pool. Repeated file hashes were identical and content scans found no question/answer fields.
- **Files changed:** `src/foundry/evaluation/validation.py`, `src/foundry/cli.py`, `tests/unit/test_validation.py`, `configs/eval/manifests/gsm1k_answer_extraction_validation.json`, `configs/eval/manifests/gsm1k_development_baseline_844.json`, and `docs/DEVLOG.md`.
- **Errors or uncertainty:** Strict Mypy initially found one cross-module `Literal` annotation mismatch for the shared main-baseline purpose string; a validation-local typed constant fixed it without changing data. The sealed-final manifest was not opened or modified; safety follows because every new ID is proven to be a member of the already isolated development-only 874-ID pool, and no sealed-final path appears in changes or the validation command.
- **Next action:** Run the selected existing current prompt once on exactly the 30 fresh validation IDs with the pinned model/dataset, greedy 512-token generation, and final canonical extractor; then move the raw JSONL into `results/raw/`, manually audit every extracted answer, and apply the admission gate. No other generation is authorized.

## 2026-07-17 00:53:15 -04:00 — Milestone 1.6 Step 4: fresh admission gate failed

- **Current step:** Step 4 — run and manually audit the single fresh 30-example answer-validation set, then apply the predeclared gate.
- **Action performed:** Ran the selected existing current prompt once on the 30 fresh identifiers; moved its 30-record JSONL into ignored `results/raw/answer_validation/current/`; inspected every extracted terminal answer and every rejection; compared the audited metrics with the 90% admission rule. No extractor or prompt change was made after seeing validation output.
- **Reason:** A fresh held-out set is required to determine whether the calibrated extractor generalizes rather than merely fitting the original 90 outputs.
- **Important command run:** `foundry answer-validate` with current config `configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml`, development source, 874-ID source pool, 30-ID validation manifest, pinned Qwen/GSM1K revisions, CUDA float16, greedy decoding, and 512 maximum new tokens; subsequent response-only audit and `git check-ignore` verification.
- **Result:** 30/30 attempted; 23/30 extractable (76.67%); 7/30 rejected (23.33%); 3/30 exact-format compliant (10.00%); 14/30 correct (46.67% benchmark accuracy); zero generation failures; zero confirmed false extractions. Rejections: 3 generation-truncated at the 512-token limit, 2 non-integral decimals, and 2 clear integer conclusions outside the frozen grammar. Even accepting both clear-but-unsupported conclusions post hoc would yield only 25/30 (83.33%), so the 90% gate definitively fails. Evaluation took 107.996 seconds at 0.2778 examples/second; total runtime including load was 110.269 seconds; 3,948 input and 8,159 output tokens; average output 271.97 tokens; peak GPU memory was 2,978.35 MiB allocated and 3,172 MiB reserved.
- **Files changed:** aggregate `results/answer_validation/current/summary.json`; content-free `results/answer_validation/current/manual_audit.json`; ignored `results/raw/answer_validation/current/predictions.jsonl`; `docs/DEVLOG.md`. Existing model cache was reused.
- **Errors or uncertainty:** Transformers repeated non-fatal warnings that sampling defaults are ignored because `do_sample=False`; decoding remained greedy. The two conservative false rejections identify possible future grammar coverage work, but modifying the extractor now would tune on validation and would not overcome the three truncations plus two non-integral outputs. No false extraction, CUDA failure, OOM, or backend generation failure occurred.
- **Next action:** Do not freeze a prompt/extractor configuration for Milestone 2 and do not run further generations. Record the failed gate and alternatives in project documentation, preserve the 30/844 manifests as bounded evidence rather than an admitted baseline split, run the complete quality/safety suite, and create one local Milestone 1.6 commit without pushing.

## 2026-07-17 00:55:30 -04:00 — Milestone 1.6 Step 5: failed gate documented

- **Current step:** Step 5 — preserve the scientific interpretation and stop decision before final verification.
- **Action performed:** Added decision D-009; updated the phase plan, metric definitions, current status, unresolved questions, and next gate; added an educational note explaining compliance versus extraction and held-out evaluator validation; registered EXP-20260717-003 with exact revisions, hashes, hardware, calibration/fresh metrics, audits, runtime, VRAM, failures, and artifact paths.
- **Reason:** The repository must make clear that a conceptually sound dual-metric evaluator still failed its fresh reliability gate, and that the 844-ID pool is not authorization to begin Milestone 2.
- **Important actions:** compared calibration and fresh results; recorded the post-hoc 83.33% upper bound; documented zero false extractions; preserved extractor/prompt hashes as evidence without declaring them admitted; explicitly prohibited reuse of the observed validation set as a fresh gate.
- **Result:** Project documentation now reports canonical benchmark accuracy separately from exact-format compliance, records the deterministic 30 prompt-calibration / 30 fresh-validation / 844 candidate-baseline partition, and states that no evaluator is frozen for Milestone 2. The next decision is a separately approved blocker-resolution design with a new untouched admission set.
- **Files changed:** `docs/PROJECT_PLAN.md`, `docs/DECISIONS.md`, `docs/DEVLOG.md`, `docs/LEARNING_NOTES.md`, and `experiments/EXPERIMENTS.md`.
- **Errors or uncertainty:** No documentation error occurred. The remaining technical uncertainties are generation truncation, two unsupported terminal phrasings, and how to construct a genuinely untouched next admission set without changing research direction; none is resolved or approved here.
- **Next action:** Run the full Ruff, strict Mypy, unit/integration, `pip check`, whitespace, secret, benchmark-content, manifest-overlap/integrity, raw-ignore/count, artifact-size, diff, and status suite. Fix only Milestone 1.6 issues, then create one local atomic commit without pushing.

## 2026-07-17 00:57:00 -04:00 — Milestone 1.6 Step 5: verification passed

- **Current step:** Step 5 — complete quality, safety, artifact, and repository review before the local atomic commit.
- **Action performed:** Ran all requested static, test, dependency, whitespace, secret, content-leak, manifest-integrity/overlap, raw-ignore/count, sealed-path, candidate-size, artifact-schema/hash, diff, and status checks; reviewed the exact 22-file candidate set and aggregate evidence.
- **Reason:** The Milestone 1.6 commit must be reproducible, content-safe, internally consistent, and limited to evaluator calibration evidence without raw benchmark output or unauthorized work.
- **Important commands run:** Ruff format/check; Ruff lint; strict `mypy src`; `pytest tests/unit tests/integration`; `python -m pip check`; `git diff --check`; targeted `rg` scans; manifest loader/pair validators; `git check-ignore`; candidate-size enumeration; JSON schema/hash review; `git diff --stat`; and `git status --short --branch`.
- **Result:** Ruff reports 23 files formatted and lint clean. Strict Mypy found no issues in 14 source files. All 67 tests passed (66 unit and 1 integration) in 0.30 seconds. `pip check`, whitespace, secrets, manifest-content, aggregate-content, overlap/coverage, sealed-path, raw-count/ignore, and size checks passed. Each of four calibration/validation raw files has exactly 30 ignored records. Exactly 22 approved files are candidates and none exceeds 1 MiB. Stored summaries/audits consistently use extractor SHA-256 `ffce6538526f9aa21e05ce4d9d6830ec71d3a6334a23fa1e9c7beef3c2053946` and the fresh summary uses prompt SHA-256 `738ea5a3b94e7c75ac0bd50a229bbf04f3fc5d773e14658bc6728bc7a4b18350` plus validation manifest SHA-256 `9582e4b07706a391f00dcfa0d9c68ee86a70cebca6c35f10daa3f3f66c9063f6`.
- **Files changed:** no new change beyond this verification entry; the candidate set remains the 22 reviewed Milestone 1.6 code, tests, identifier manifests, aggregate summaries/audits, and documentation files.
- **Errors or uncertainty:** No final check failed. Git emitted expected LF-to-CRLF conversion warnings on Windows. The admission blocker remains the measured 76.67% fresh extractability, not a repository-quality failure.
- **Next action:** Stage exactly the 22 reviewed files, repeat staged scope/whitespace/secret/content checks, create one local Milestone 1.6 commit, verify a clean branch exactly one commit ahead of `origin/main`, do not push, and stop before Milestone 2.

## 2026-07-17 23:01:49 -04:00 — Milestone 1.7 Steps 1–2: starting state verified and seven rejections diagnosed

- **Current step:** Steps 1–2 — verify the published Milestone 1.6 checkpoint, then classify every rejected output from its ignored 30-example validation artifact before changing the evaluator.
- **Action performed:** Confirmed the repository/remote state and inspected exactly the seven records whose canonical prediction was null. Classified each rejection using response text, the stored truncation flag, and output-token count; recorded only shortened stable-ID prefixes and sanitized terminal-format descriptions here.
- **Reason:** Extractor changes must address demonstrated generic formats without using benchmark labels, guessing from intermediate numbers, or tuning against correctness. Confirmed truncation must be separated from grammar failures.
- **Important commands run:** `git rev-parse --show-toplevel`, `git branch --show-current`, `git rev-parse HEAD`, `git rev-parse origin/main`, `git rev-list --left-right --count main...origin/main`, `git status --short --branch`, and a response-only PowerShell audit of `results/raw/answer_validation/current/predictions.jsonl`.
- **Result:** Starting state passed: `main` was clean at published commit `e1d0576ac7c625de80960edf414d7270ed5cc8e4`, synchronized 0 ahead/0 behind. Rejection diagnosis:

  | ID prefix | Exact category | Sanitized terminal form | Human sees one answer? | Generic support and risk |
  | --- | --- | --- | --- | --- |
  | `acb2cefd0181` | Clearly intended terminal non-integral number | `**Final answer:** 0.74` | Yes | Exact decimal normalization is generic; low risk under an explicit final-answer cue. |
  | `ed4dddc8aa8` | Truncated because `max_new_tokens` was reached | Response stops inside an arithmetic expression | No | Do not extract; a longer deterministic generation is required. |
  | `e49cd952bd1e` | Clearly intended terminal integer in an unsupported wrapper | `Therefore, … must complete **12** [units].` | Yes | A terminal conclusion sentence with one decorated value is generic; low-to-medium risk unless tightly anchored. |
  | `c7a4d6c27eba` | Clearly intended terminal non-integral number | terminal `\boxed{143.50}` | Yes | Exact boxed-decimal normalization is generic; low risk. |
  | `9ddd09e3aa19` | Truncated because `max_new_tokens` was reached | Response stops during a restarted derivation | No | Do not extract; the unfinished response contains conflicting intermediate reasoning. |
  | `bfac9d904152` | Clearly intended terminal integer in an unsupported wrapper | `Therefore, … at 1 AM.` | Yes | A terminal conclusion sentence with exactly one number and a unit is generic; medium risk and requires a strict one-candidate rule. |
  | `7605a69c3f90` | Truncated because `max_new_tokens` was reached | Response stops after an approximate division result | No | Do not infer rounding or flooring; rerun only because the stored limit flag proves truncation. |

  Aggregate categories are 2 unsupported terminal wrappers, 2 explicit non-integral terminal decimals, and 3 confirmed token-limit truncations. All three truncated records have `generation_truncated=true` and exactly 512 output tokens. No remaining record contains a terminal fraction, a complete terminal equation needing special handling, multiple conflicting final answers, or a genuinely ambiguous completed conclusion.
- **Files changed:** `docs/DEVLOG.md` only. Ignored raw predictions were read but not modified; no benchmark question, label, reference answer, full response, dataset, cache, or sealed-final content was committed or changed.
- **Errors or uncertainty:** No shell or repository error occurred. Supporting a bare terminal conclusion sentence carries more false-positive risk than explicit answer markers, so implementation must require a conclusion cue, a terminal sentence, and exactly one syntactically valid numeric candidate. The three truncated responses contain no auditable final intent and remain rejected in the original-output re-score.
- **Next action:** Narrowly version the canonical extractor to support exact rational values and tightly anchored terminal conclusion wrappers, add accepted/rejected regression tests, and keep the strict `Final answer:` parser unchanged. Separately update the sole generation limit from 512 to 768 because truncation is verified.

## 2026-07-17 23:06:52 -04:00 — Milestone 1.7 Step 3: canonical terminal-number extractor implemented

- **Current step:** Step 3 — narrowly extend deterministic canonical extraction while preserving exact-format compliance as a separate unchanged metric.
- **Action performed:** Versioned the canonical extractor as `foundry-terminal-number-v2`; replaced integer-only normalization with exact `Fraction`/`Decimal` normalization; added JSON-safe reduced-fraction serialization; admitted explicit decimals, ASCII/LaTeX fractions, attached percentages, existing currency/unit wrappers, terminal arithmetic equations, decorated conclusion values, and single-number terminal conclusion sentences. Added explicit conflict-list rejection and comprehensive positive/negative tests. Updated raw evaluation record typing so non-integral values are stored exactly as reduced rational strings. The strict parser in `scoring.py` was not edited.
- **Reason:** A clearly expressed wrong non-integral answer must be extractable-but-wrong, while unsupported wrappers need safe generic recognition. Exact rational arithmetic avoids floating-point ambiguity, and tight cue/terminal/conflict rules prevent a generic last-number heuristic.
- **Important commands run:** targeted Ruff format/lint, strict `mypy --strict src`, and targeted extractor/scoring/re-scoring/integration Pytest suites.
- **Result:** Extractor SHA-256 is `32abd7018b0c255997d117daf64d5b50b1fa4bdee8d881c38b87546ca6446bbe`. Ruff and strict Mypy passed; 66 targeted tests passed. Tests prove acceptance of signs, valid commas, integral/non-integral decimals, ASCII and LaTeX fractions, currency, attached percentages, units, boxed/bold/inline-LaTeX/text wrappers, safe terminal conclusion sentences, and terminal equations. Tests also prove rejection of conflicting explicit answers, terminal `and`/`or` candidate lists, arbitrary intermediate numbers, malformed comma grouping, zero-denominator fractions, unfinished expressions, bare trailing numbers, and truncation.
- **Files changed:** `src/foundry/evaluation/answer_extraction.py`, `src/foundry/evaluation/runner.py`, `tests/unit/test_answer_extraction.py`, and `docs/DEVLOG.md`.
- **Errors or uncertainty:** The first targeted test run exposed an over-restrictive decimal-boundary check that also rejected sentence punctuation, plus a test-category mismatch for explicit `or` candidates; both were corrected and the full targeted suite then passed. The broader wrapper rules remain anchored to a terminal conclusion cue and either explicit decoration or a single numeric candidate followed only by units. No benchmark label or question influenced extraction.
- **Next action:** Apply the separately justified 512-to-768 generation-limit change, update exact config/hash tests and dependent manifest bindings as required, and rerun only the three confirmed truncated examples as diagnostics—not as final validation evidence.

## 2026-07-17 23:10:30 -04:00 — Milestone 1.7 Step 4: verified truncation bounded at 768 tokens

- **Current step:** Step 4 — resolve confirmed generation truncation without changing the prompt, model, dataset, or decoding behavior, and keep the diagnostic separate from the final gate.
- **Action performed:** Added `configs/eval/gsm1k_qwen2_5_1_5b_final_evaluator.yaml`, identical to the historical current-prompt config except `max_new_tokens` is 768 rather than 512. Added a typed guard and CLI `--base-config` binding so development-only identifier manifests remain validated against their original configuration while evaluation uses the approved token-limit-only variant. Reran exactly the three confirmed truncations once at 768 tokens, then re-scored those ignored diagnostic outputs with the final extractor. One newly observed answer-leading terminal conclusion (`Therefore, 6 [units] …`) was added through a narrow generic pattern requiring an immediate normalized value, unit words without `and`/`or`, and a conclusion verb; no benchmark answer was consulted.
- **Reason:** Three original responses conclusively hit 512 tokens. A separate config preserves historical evidence and avoids accessing, regenerating, or modifying the sealed-final manifest. The three-record rerun tests the approved bound but is not eligible as final validation evidence.
- **Important commands run:** exact config hashing; targeted Ruff, strict Mypy, and Pytest; a three-ID development-only CUDA diagnostic using the stored truncation flags; response-tail audit; `foundry rescore-answers --max-new-tokens 768`; and `git check-ignore` for all diagnostic artifacts.
- **Result:** Final generation-config SHA-256 is `5f315d5de645f9563b8d1e61bc8e02c3513c453238ad9e1d6f9473489b5a622b`; final extractor SHA-256 is `e099d1c247968fed982cb849022ec3137b1694c15f23a65663a127b8158c06df`. At 768 tokens, one response completed at 602 tokens with an exact final line, one completed at 572 tokens with a clear answer-leading conclusion, and one still hit the 768-token bound during an unfinished derivation. Final-extractor diagnostic re-score: 2/3 extractable, 1/3 exact compliant, 0/3 correct, one truncation rejection, zero false extractions, and zero backend failures. The diagnostic used 1,942 generated tokens in 27.129 seconds (71.59 generated tokens/second), 0.1106 examples/second, and peak RTX 3080 memory of 2,984.10 MiB allocated and 3,176 MiB reserved. These are diagnostic numbers, not the final validation result.
- **Files changed:** `configs/eval/gsm1k_qwen2_5_1_5b_final_evaluator.yaml`, `src/foundry/cli.py`, `src/foundry/evaluation/validation.py`, `src/foundry/evaluation/answer_extraction.py`, `tests/unit/test_config.py`, `tests/unit/test_validation.py`, `tests/unit/test_answer_extraction.py`, and `docs/DEVLOG.md`; ignored diagnostic files under `results/raw/truncation_diagnostic/current_768/` were created and verified ignored.
- **Errors or uncertainty:** Transformers emitted the same non-fatal warnings that model-card sampling defaults are ignored when `do_sample=False`; decoding remained greedy. Raising the bound did not complete one unusually long response, and the approval does not permit a further increase. The diagnostic accuracy was zero but is not a selection criterion and does not alter the configured gate. No CUDA, OOM, backend, manifest, or repository error occurred.
- **Next action:** Re-score all 30 original Milestone 1.6 validation outputs with the final extractor and the historical 512-token truncation metadata, manually audit every newly accepted output, and require zero false extractions before reserving the final fresh set.

## 2026-07-17 23:11:16 -04:00 — Milestone 1.7 Step 5: existing validation re-scored and audited

- **Current step:** Step 5 — measure extractor-only coverage on the unchanged Milestone 1.6 validation outputs and manually verify every newly accepted answer.
- **Action performed:** Re-scored the original ignored 30-record JSONL with `foundry-terminal-number-v2` and the original 512-token truncation rule, without generation or raw-file changes. Audited the four outputs that changed from rejected to extractable by comparing each normalized value with the response's explicit terminal intent; correctness was computed only after extraction.
- **Reason:** The extractor must demonstrate that decimals and terminal wrappers increase answer coverage without selecting intermediate numbers or using benchmark targets to choose candidates.
- **Important commands run:** `foundry rescore-answers --max-new-tokens 512`, a response-only newly-accepted audit, aggregate JSON review, and source-prediction SHA-256 verification.
- **Result:** Exact-format compliance remains 3/30 (10.00%). Extractability rises from 23/30 (76.67%) to 27/30 (90.00%). Accuracy rises from 14/30 (46.67%) to 15/30 (50.00%). Four outputs are newly accepted: two explicit non-integral decimals, one decorated terminal integer, and one terminal integer with a time unit. One was correct and three were clearly expressed but wrong; the two non-integral outputs are counted extractable-but-wrong. Three original outputs remain rejected solely because their stored 512-token metadata proves truncation. Manual audit found zero false extractions; generation failures remain zero.
- **Files changed:** content-free aggregate `results/final_evaluator_calibration/milestone_1_6_validation_rescore.json`, content-free audit `results/final_evaluator_calibration/milestone_1_6_validation_manual_audit.json`, and `docs/DEVLOG.md`. The source raw artifact stayed ignored and unchanged at SHA-256 `e8a8429c07e73e0992cc3b2016548363230950106946f239af2723a2c7448e3e`.
- **Errors or uncertainty:** No re-scoring or audit error occurred. This 90% result is post-calibration evidence on an already observed set and is not the admission decision; only the next untouched 30-example final set can satisfy the gate.
- **Next action:** Deterministically reserve 30 fresh identifiers from the untouched 844-ID pool, leave exactly 814 baseline IDs, add cryptographic integrity/overlap/completeness/tampering tests, and prove the existing calibration/validation partitions remain disjoint without opening or modifying sealed-final contents.

## 2026-07-17 23:13:45 -04:00 — Milestone 1.7 Step 6: final 30/814 development split frozen before use

- **Current step:** Step 6 — reserve one final untouched evaluator-validation set and preserve the remaining main-development baseline before any final generation.
- **Action performed:** Added typed, parent-hashed final-evaluator manifests; deterministic selection; strict loader/saver; tamper detection; pair validation; evaluation adaptation; and bounded build/run CLI commands. Selected 30 IDs from the untouched 844-ID parent with seed `foundry-gsm1k-final-evaluator-validation-v1`, leaving 814 IDs. Rebuilt the split independently in memory and compared the exact stable JSON representation with both saved files.
- **Reason:** The final gate must be evaluated on identifiers not seen while designing the prompt, extractor, or truncation fix, and the future baseline must exclude every calibration/validation ID.
- **Important commands run:** targeted Ruff and Pytest; `foundry build-final-evaluator-validation --size 30`; strict manifest reload; deterministic in-memory rebuild and byte comparison; identifier-only content scan; four-part overlap/completeness test; tamper-rejection test; and sealed-path diff review.
- **Result:** Final validation manifest: 30 IDs, semantic SHA-256 `2234e5ee82cf57e8fb74839a21f7f0ca0d2ff02ddd0fb0e42d93934415b2db93`. Remaining baseline: 814 IDs, semantic SHA-256 `5e810d3ab644bef1d43c598a14a6164ba6464b27fde50e92a2f241816ce87897`. Saved bytes exactly match a deterministic rebuild. The 30 prompt-calibration, 30 prior extraction-validation, 30 final-evaluator-validation, and 814 baseline ID sets are pairwise disjoint and their union is exactly the 904-ID canonical development partition. Seven targeted validation tests pass, including stable hashes and tamper rejection.
- **Files changed:** `src/foundry/evaluation/validation.py`, `src/foundry/cli.py`, `tests/unit/test_validation.py`, `configs/eval/manifests/gsm1k_final_evaluator_validation.json`, `configs/eval/manifests/gsm1k_development_baseline_814.json`, and `docs/DEVLOG.md`.
- **Errors or uncertainty:** No construction, integrity, overlap, or test error occurred. Both new files contain identifiers and provenance only—no questions, answers, or labels. The sealed-final file was neither opened nor modified; separation follows transitively because every selected identity is a member of the already isolated canonical development manifest, and the sealed path has no diff.
- **Next action:** Run exactly one 30-example final evaluator validation with the current prompt, pinned model/dataset revisions, greedy 768-token configuration, unchanged strict parser, and final extractor; then manually audit all 30 accepted/rejected outputs before applying the admission gate.

## 2026-07-17 23:17:03 -04:00 — Milestone 1.7 Step 7: final fresh gate failed after full audit

- **Current step:** Step 7 — execute the single final untouched 30-example evaluator validation, audit every response, and apply the predeclared admission gate without further parser iteration.
- **Action performed:** Ran `final-evaluator-validate` once on exactly the 30 final-evaluator IDs using the current prompt, pinned Qwen/GSM1K revisions, greedy decoding, 768 maximum new tokens, unchanged strict parser, and `foundry-terminal-number-v2`. Reviewed the terminal intent of all 25 accepted outputs and all five rejections. Stored raw output only under ignored `results/raw/`; copied only content-free aggregate and audit records into tracked results.
- **Reason:** Only a previously untouched set can test whether the final evaluator generalizes. Manual review is required to detect false extractions that aggregate scoring cannot reveal.
- **Important command run:** `.venv\\Scripts\\python.exe -m foundry.cli final-evaluator-validate` with base config `gsm1k_qwen2_5_1_5b_smoke.yaml`, final config `gsm1k_qwen2_5_1_5b_final_evaluator.yaml`, canonical development source, 874-ID parent, 844-ID source baseline, and 30-ID final validation manifest; followed by complete response-only audit, derived-metric calculation, raw SHA-256, and ignore checks.
- **Result:** 30/30 attempted; 25/30 extractable (83.33%); 5/30 rejected (16.67%); 5/30 exact-format compliant (16.67%); 13/30 correct (43.33%); 12 extractable-but-wrong; zero backend generation failures; zero confirmed false extractions. Two accepted-but-wrong answers were explicit non-integral decimals. The five rejections were one response still truncated at 768 tokens and four complete, human-clear terminal answers in unsupported prose wrappers. Evaluation took 125.095 seconds at 0.2398 examples/second; total runtime including model load was 126.885 seconds; 4,491 input and 9,451 output tokens; average output 315.03 tokens; generated-token throughput 75.55 tokens/second; peak RTX 3080 memory was 2,985.59 MiB allocated and 3,196 MiB reserved.
- **Files changed:** content-free `results/final_evaluator_validation/current/summary.json`, content-free `results/final_evaluator_validation/current/manual_audit.json`, ignored raw run files under `results/raw/final_evaluator_validation/current/`, and `docs/DEVLOG.md`. Raw prediction SHA-256 is `0d74229947519d8f5d2227b034da8e0f944f9044b47e062d3caa9db10e5f1440` and the file contains exactly 30 ignored records.
- **Errors or uncertainty:** The 83.33% extractable-answer rate is below the required 90% gate. Transformers emitted expected ignored-sampling-default warnings; greedy decoding remained unchanged. No CUDA, OOM, backend, dataset, manifest, or audit error occurred. The four clear false rejections show remaining grammar undercoverage, but the approved protocol forbids another post-hoc parser iteration. Accuracy is diagnostic and did not determine the gate.
- **Next action:** Do not freeze the evaluator or 814-ID pool for Milestone 2, do not begin Milestone 2, and do not propose another calibration milestone. Document the failed gate and exactly two user choices: proceed with unextractable outputs counted incorrect while reporting coverage separately, or reconsider the base model/benchmark.

## 2026-07-17 23:19:25 -04:00 — Milestone 1.7 Step 8: failed gate and stop decision documented

- **Current step:** Step 8 — preserve the complete technical interpretation, safety boundaries, and next decision before final repository verification.
- **Action performed:** Added decision D-010; updated milestone status, evaluator behavior, accepted/rejected formats, rational normalization, false-extraction risks, 30/30/30/814 split lineage, config/extractor/manifest hashes, diagnostics, final metrics, audit result, and the failed admission rule across the plan, decision log, learning notes, experiment registry, and live DEVLOG.
- **Reason:** The repository must distinguish safe extraction precision from insufficient coverage, prevent the 814-ID candidate baseline from being mistaken for authorization, and make the final stop condition understandable without raw benchmark content.
- **Important actions:** compared the old-set 90.00% re-score with the untouched-set 83.33% result; recorded zero false extractions and four false rejections; documented exact `Fraction`/`Decimal` behavior; retained exact compliance as a separate metric; and limited the next decision to the two approved fallback choices.
- **Result:** Documentation consistently states that Milestone 1.7 is complete but its gate failed, no evaluator or baseline is admitted, and no further calibration milestone is proposed. The next user choice is either to approve Milestone 2 with unextractable outputs scored incorrect and coverage reported separately, or to reconsider the base model/benchmark.
- **Files changed:** `docs/PROJECT_PLAN.md`, `docs/DECISIONS.md`, `docs/DEVLOG.md`, `docs/LEARNING_NOTES.md`, and `experiments/EXPERIMENTS.md`.
- **Errors or uncertainty:** No documentation error occurred. The remaining uncertainty is a research-policy choice, not a hidden technical failure: conservative deterministic extraction has zero audited false positives but only 83.33% fresh coverage.
- **Next action:** Run the complete formatting, Ruff lint, strict Mypy, unit/integration, dependency, whitespace, secret, benchmark-content, manifest-integrity/overlap, raw-ignore/count, artifact-consistency, sealed-path, diff, and status suite. Fix only Milestone 1.7 issues, then create one local atomic commit without pushing.

## 2026-07-17 23:20:41 -04:00 — Milestone 1.7 Step 8: complete verification passed

- **Current step:** Step 8 — verify code quality, deterministic integrity, content safety, artifact containment, and exact commit scope before the local atomic commit.
- **Action performed:** Ran every requested formatter, linter, type, unit, integration, dependency, whitespace, secret, content-leak, manifest, ignore, artifact-consistency, size, sealed-path, diff, and status check. Independently loaded and hash-verified the final config/extractor/manifests; proved pairwise disjointness and complete 904-ID development coverage; compared the tracked final summary with the generated ignored summary; and verified old re-scoring source provenance.
- **Reason:** The final milestone commit must be reproducible, content-safe, free of raw benchmark material, and limited to the approved evaluator blocker work even though the scientific gate failed.
- **Important commands run:** `ruff format .`; `ruff check .`; `mypy --strict src`; `pytest tests/unit`; `pytest tests/integration`; `python -m pip check`; `git diff --check`; targeted `rg` secret/content scans; typed manifest loaders/validators; four-way overlap/coverage checks; SHA-256 comparisons; `git check-ignore`; `git ls-files results/raw`; artifact-size checks; sealed-path diff; and repository status/diff review.
- **Result:** Ruff reports 23 files formatted and lint clean. Strict Mypy found no issues in 14 source files. All 89 unit tests and one integration test passed. `pip check`, whitespace, secrets, benchmark-content, manifest integrity/tampering, 30/30/30/814 overlap/completeness, tracked-summary consistency, source hashes, raw ignore/count, raw tracking, sealed-path, and size checks passed. Final raw output contains exactly 30 ignored records with SHA-256 `0d74229947519d8f5d2227b034da8e0f944f9044b47e062d3caa9db10e5f1440`.
- **Files changed:** the reviewed Milestone 1.7 set consists of five documentation files; four source files; three unit-test files; one final evaluator config; two identifier-only manifests; and four content-free aggregate/audit JSON files. Ignored raw/diagnostic artifacts are excluded.
- **Errors or uncertainty:** No final verification failed. Git emitted expected LF-to-CRLF conversion warnings on Windows. The only blocker is the measured 83.33% fresh extractability, not repository quality or hardware execution.
- **Next action:** Stage exactly the reviewed files, repeat staged whitespace, secrets, benchmark-content, sealed-path, and scope checks, create one local `eval:` commit, verify the worktree is clean and exactly one commit ahead of `origin/main`, do not push, and stop before Milestone 2.

## 2026-07-17 23:21:48 -04:00 — Milestone 1.7 Step 8: local atomic commit created

- **Current step:** Step 8 — create the single local Milestone 1.7 commit and verify the final repository state without pushing.
- **Action performed:** Staged exactly the reviewed 19 files and repeated staged scope, whitespace, secrets, benchmark-content, raw-path, and sealed-path checks. The first commit attempt failed because this Windows Git installation had no author identity configured. Read the author/committer identity from published commit `e1d0576` and supplied the same GitHub noreply identity through process-local `GIT_AUTHOR_*` and `GIT_COMMITTER_*` variables only; no global or repository Git configuration changed. Created and amended the single local commit so this incident is included in the live record.
- **Reason:** The milestone requires one atomic local commit, but inventing an identity or permanently configuring Git would exceed what was necessary. Reusing the repository's published identity for one process preserves authorship continuity without persistent configuration.
- **Important commands run:** explicit `git add -- <19 reviewed paths>`; `git diff --cached --check`; staged secrets/content/scope scans; `git commit -m "eval: resolve final evaluator blockers"`; `git log -1` identity inspection; process-local author variables; `git commit --amend --no-edit`; final `git rev-parse`, ahead/behind, log, and status checks.
- **Result:** One local atomic Milestone 1.7 commit exists on `main`; `origin/main` remains at published Milestone 1.6 commit `e1d0576`; the branch is one commit ahead and zero behind; the worktree is clean. The final commit hash is reported in the completion response because amending this entry determines that hash.
- **Files changed:** the same reviewed 19-file Milestone 1.7 scope; ignored raw artifacts remain untracked. This final entry only adds commit-process provenance to `docs/DEVLOG.md` inside the same commit.
- **Errors or uncertainty:** Initial `git commit` error: `Author identity unknown` / `fatal: unable to auto-detect email address`. It was resolved without persistent Git configuration. No push, remote mutation, second commit, code/test change, or scope expansion occurred.
- **Next action:** Stop before Milestone 2 and wait for the user to choose one of the two documented fallback paths.

## 2026-07-17 23:28:32 -04:00 — Milestone 2 Steps 1–2: Milestone 1.7 published and evaluator contract frozen

- **Current step:** Steps 1–2 — publish the completed Milestone 1.7 checkpoint, then record the explicitly approved one-time evaluator-gate exception before any baseline generation.
- **Action performed:** Verified clean `main` at local commit `7786fe7c5d80953255b6a6ee15c4dcccb275e6d6`, exactly one ahead of `origin/main` at `e1d0576`; pushed only that commit; fetched and verified local/remote equality at 0 ahead/0 behind. Loaded and validated the exact Milestone 1.7 base/final configs, development lineage, 814-ID manifest, prompt, extractor, and hashes without opening the sealed-final manifest. Added decision D-011 and updated the project plan with the one-time exception and immutable evaluation contract.
- **Reason:** Milestone 2 must begin from a published, reproducible checkpoint and must make explicit that unextractable answers count wrong while coverage and exact compliance remain separate. Hash verification prevents silent substitution after the evaluator gate exception.
- **Important commands run:** `git rev-parse`, `git branch --show-current`, `git rev-list --left-right --count`, `git status --porcelain`, `git push origin 7786fe7...:refs/heads/main`, `git fetch origin main`, typed config/manifest loaders, prompt/extractor hash functions, 814-to-development subset validation, and sealed-path status review.
- **Result:** Local `main` and `origin/main` both point to `7786fe7`; the worktree was clean after push. Frozen identities match exactly: prompt `738ea5a3...b18350`, extractor `e099d1c2...c06df`, config `5f315d5d...a622b`, and 814-ID manifest `5e810d3a...7897`. Model/dataset revisions and greedy 768-token decoding match D-011. The 814 IDs are wholly contained in the canonical development partition.
- **Files changed:** after the push confirmation, `docs/DECISIONS.md`, `docs/PROJECT_PLAN.md`, and `docs/DEVLOG.md` only.
- **Errors or uncertainty:** No push, hash, manifest, or repository error occurred. No sealed-final content or file was opened, hashed, or modified. Non-overlap with sealed-final follows from the previously established canonical development partition and the verified 814-ID subset relation rather than renewed sealed access.
- **Next action:** Add only the bounded baseline-run adapter/metrics needed to evaluate the frozen 814-ID manifest, validate hardware/disk/CUDA and all disjoint development partitions, then execute one uninterrupted deterministic baseline with no per-example retries.

## 2026-07-17 23:30:44 -04:00 — Milestone 2 Step 3: baseline adapter and preflight validated

- **Current step:** Step 3 — add the minimum bounded execution/summary path for the frozen 814-ID baseline and prove the machine and development-only inputs are ready before generation.
- **Action performed:** Added a typed adapter that accepts only the `main_development_baseline` purpose with exactly 814 IDs, a `development-baseline` CLI command, progress reports every 25 examples, and aggregate schema v3 fields for accuracy among extractable answers, extractable-wrong/unextractable/truncated counts, average output tokens, and generated-token throughput. Added/updated unit and fake integration tests. Revalidated all four development partitions and frozen hashes, checked CUDA/PyTorch, disk space, and live GPU memory, and confirmed the output target did not already exist.
- **Reason:** Existing commands intentionally permitted only bounded 30-ID validation. Milestone 2 needs one auditable 814-ID path without relaxing any model/evaluator setting, plus visible progress during a long local GPU run.
- **Important commands run:** targeted Ruff format/lint; strict Mypy; 55 targeted unit/integration tests; `nvidia-smi`; `torch.cuda.is_available()` and `torch.cuda.mem_get_info()`; `Get-PSDrive C`; typed config/manifest loaders; frozen hash comparisons; 30/30/30/814 overlap/completeness validation; and sealed-path status review.
- **Result:** Targeted Ruff and strict Mypy pass; 55 targeted tests pass. CUDA is available through PyTorch 2.5.1+cu121 on the RTX 3080; driver 610.47; 10,240 MiB total GPU memory with 9,091 MiB reported free; 208.10 GiB disk free. Frozen prompt, extractor, config, model/dataset revisions, and 814-ID manifest match D-011. The development partitions are pairwise disjoint and cover exactly 904 IDs. The ignored baseline output target is new.
- **Files changed:** `src/foundry/cli.py`, `src/foundry/evaluation/runner.py`, `src/foundry/evaluation/validation.py`, `tests/unit/test_validation.py`, `tests/integration/test_evaluation_smoke.py`, and `docs/DEVLOG.md`, in addition to the Step 2 documentation files.
- **Errors or uncertainty:** The first targeted integration run expected token-rate fields from the fake backend, which deliberately supplies no token counts; the test—not runtime code—was corrected to require `null`, then all targeted checks passed. GPU free-memory readings include normal desktop use and may fluctuate. No frozen artifact or sealed-final path changed or was opened.
- **Next action:** Start exactly one deterministic 814-example development-baseline run under the frozen contract, write complete records only beneath ignored `results/raw/`, and do not retry any example. Monitor progress and stop if the single run itself fails.

## 2026-07-18 00:24:07 -04:00 — Milestone 2 Step 3: frozen 814-example baseline completed

- **Current step:** Step 3 — execute the one approved base-model development evaluation and preserve complete per-example output only in the ignored raw-results area.
- **Action performed:** Ran `development-baseline` once on exactly the frozen 814-identifier manifest using the pinned Qwen/GSM1K revisions, current prompt, `foundry-terminal-number-v2`, greedy decoding, and the unchanged 768-token configuration. Progress was reported every 25 examples. No individual example was retried. Verified that the 814-record prediction file is ignored by Git.
- **Reason:** This run establishes the untouched base model's frozen development baseline under the documented one-time exception, with unextractable outputs counted incorrect while extractability and exact-format compliance remain separate.
- **Important command run:** `.venv\\Scripts\\python.exe -m foundry.cli development-baseline --base-config configs/eval/gsm1k_qwen2_5_1_5b_smoke.yaml --config configs/eval/gsm1k_qwen2_5_1_5b_final_evaluator.yaml --development-manifest configs/eval/manifests/gsm1k_development.json --source-pool-manifest configs/eval/manifests/gsm1k_development_baseline.json --source-baseline-manifest configs/eval/manifests/gsm1k_development_baseline_844.json --baseline-manifest configs/eval/manifests/gsm1k_development_baseline_814.json --output-dir results/raw/development_baseline/qwen2_5_1_5b` with `HF_HOME=data/huggingface`.
- **Result:** 814/814 attempted; 521 correct; 64.00% end-to-end accuracy; 752 extractable (92.38%); 69.28% accuracy among extractable answers; 231 extractable-but-wrong; 62 unextractable; 130 exact-format compliant (15.97%); three truncated; zero backend failures. Failure counts were 42 ambiguous terminal answers, eight conflicting answers, seven with no terminal answer, two malformed terminal answers, and three truncated generations. Evaluation took 3,160.074 seconds at 0.2576 examples/second; total runtime including 1.847 seconds model loading was 3,161.921 seconds. The run used 113,720 input tokens and 234,106 output tokens (287.60 output tokens/example) at 74.08 generated tokens/second. Peak RTX 3080 memory was 3,133,062,144 bytes allocated and 3,353,346,048 bytes reserved.
- **Files changed:** ignored `results/raw/development_baseline/qwen2_5_1_5b/summary.json` and `results/raw/development_baseline/qwen2_5_1_5b/raw/predictions.jsonl`, plus this DEVLOG checkpoint. The raw prediction file is approximately 1.20 MB and remains ignored.
- **Errors or uncertainty:** Transformers emitted only the previously observed warnings that sampling defaults are ignored under greedy decoding. No CUDA, OOM, model-loading, dataset, backend, or manifest failure occurred. This is a development baseline, not a final sealed benchmark score. Unextractable outputs are conservatively scored wrong by the approved exception.
- **Next action:** Create a deterministic sample of no more than 100 incorrect development records, manually audit it into a provisional and explicitly non-exhaustive failure taxonomy, and commit only identifier references and aggregate/content-free interpretations—not benchmark questions, labels, or raw outputs.

## 2026-07-18 00:31:46 -04:00 — Milestone 2 Step 4: deterministic failure inventory audited

- **Current step:** Step 4 — measure the complete failure population, select at most 100 mathematical failures without using content, and manually classify recurring development weaknesses.
- **Action performed:** Added a typed content-free prediction loader, aggregate counter, and deterministic selector ranked by SHA-256 of a fixed seed plus stable identifier and row index. Selected exactly 100 of the 231 extractable-but-wrong records; sample SHA-256 is `3ae1e307e3576a07d563f872e24c6345b8bcbbf2cee65415f226bc1c6fc9981d`. Manually reviewed all 100 questions and responses locally and assigned one provisional primary category per record. Stored only aggregate counts, unique 12-character identifier prefixes, hashes, and sanitized limitations in the tracked inventory.
- **Reason:** A content-independent sample makes the analysis reproducible without choosing failures by apparent severity. Manual review provides an initial human-auditable weakness inventory while keeping full benchmark questions, labels, and raw responses out of Git.
- **Important commands run:** targeted Ruff, strict Mypy, and five new unit tests; deterministic sampler/count invocations; cached pinned development-dataset loading for local audit; SHA-256 checks for the raw predictions and ordered sample; and `git check-ignore` for raw containment.
- **Result:** Complete measured failure population: 293 incorrect total, comprising 231 extractable-but-wrong and 62 unextractable; 684 exact-format-noncompliant; three truncated; zero backend failures. Sample taxonomy: multi-step bookkeeping/omission 28; target/language interpretation 18; constraint/distribution/discrete reasoning 15; time/unit/sequence reasoning 14; arithmetic execution 12; rate/ratio/percentage/average 12; benchmark ambiguity or annotation risk 1. Primary counts sum to 100; secondary causes may overlap.
- **Files changed:** `src/foundry/evaluation/failure_inventory.py`, `tests/unit/test_failure_inventory.py`, tracked aggregate-only `results/development_baseline/qwen2_5_1_5b/{summary.json,failure_inventory.json}`, and this DEVLOG entry. Ignored raw output remains unchanged.
- **Errors or uncertainty:** Manual audit found two false extractor acceptances in the 100-record sample: a terminal currency conclusion where an earlier percentage was selected, and a stated loss whose sign was not preserved. Both remained mathematically wrong, so neither changes the 521-correct count, but this contradicts the premise that all remaining extraction errors are conservative. The 521 records scored correct were not audited, so accidental label matches remain an unmeasured risk. Three read-only inspection attempts needed correction: a quoting syntax error, an incorrect dataset configuration (`main` instead of pinned `default`/`test`), and a Windows console encoding error after 19 records. None wrote artifacts or reran generation.
- **Next action:** Document the measured baseline, sampled taxonomy, and newly discovered precision limitation across the plan, decisions, learning notes, and experiment registry; then run the complete repository verification suite without altering the frozen evaluator or rerunning the model.

## 2026-07-18 00:33:05 -04:00 — Milestone 2 Step 5: baseline interpretation documented

- **Current step:** Step 5 — make the frozen contract, complete aggregate result, provisional taxonomy, limitations, and next decision independently reviewable without raw benchmark content.
- **Action performed:** Added decision D-012, marked Milestone 2 complete in the plan, recorded the larger-run extraction lesson, and registered experiment EXP-20260718-005 with exact hashes, hardware, commands, measurements, artifact paths, and scope exclusions. Kept D-011's original one-time exception rationale as historical provenance while explicitly recording that the larger sample weakened its precision premise.
- **Reason:** A reproducible score must remain distinguishable from claims about semantic extractor correctness. The documentation must preserve the exact run while preventing the 92.38% measured extractor rate from being mistaken for fully audited intent coverage.
- **Important actions:** copied every aggregate field from the ignored summary into a semantically identical content-free tracked location; recorded raw prediction SHA-256 `73d52dace0f27577b1177bdfa81dfbb4c88252107c9b04e2ff49dbbd93da6cc0`; separated complete measured counts from sampled interpretations; and stated that no further milestone is approved. The tracked/raw summary byte hashes differ only because the files use different line endings; parsed JSON equality is the required artifact check.
- **Result:** Documentation now consistently reports 64.00% end-to-end accuracy, 92.38% extractability, 69.28% accuracy among extracted answers, 15.97% exact-format compliance, 62 unextractable outputs, three truncations, zero backend failures, and the bounded taxonomy. It also states that two sampled false extractions did not alter the correct count but block a claim of established full-baseline extractor precision.
- **Files changed:** `docs/PROJECT_PLAN.md`, `docs/DECISIONS.md`, `docs/DEVLOG.md`, `docs/LEARNING_NOTES.md`, `experiments/EXPERIMENTS.md`, and the two aggregate/content-free tracked result files.
- **Errors or uncertainty:** No documentation or artifact-copy error occurred. The baseline is reliable as the exact frozen evaluator's output, but not yet proven reliable as a semantic model-intent measurement for comparing future trained candidates. The provisional taxonomy covers only a deterministic sample of extractable-but-wrong records.
- **Next action:** Run all formatting, linting, strict typing, unit/integration, dependency, whitespace, secret, benchmark-content, manifest, raw-ignore, tracked-size, artifact-consistency, sealed-path, and repository-status checks. Fix only Milestone 2 issues, create one local atomic commit, do not push, and stop.

## 2026-07-18 00:34:55 -04:00 — Milestone 2 Step 5: complete verification passed

- **Current step:** Step 5 — verify code quality, deterministic integrity, artifact containment, content safety, and exact commit scope after the baseline and audit.
- **Action performed:** Ran the complete formatter, lint, strict type, unit, integration, dependency, whitespace, credential, benchmark-question, manifest-chain, overlap/completeness, frozen-hash, raw-ignore/count/hash, aggregate-equality, inventory-consistency, content-free-schema, tracked-size, sealed-path-status, diff, and repository-scope suite. Reviewed the full source/test diff and both new aggregate artifacts. Tightened the new raw-record loader to reject booleans masquerading as integers and added a regression test.
- **Reason:** The Milestone 2 commit must be reproducible, development-only, content-safe, and unable to silently include raw generations, benchmark text, large artifacts, or a malformed deterministic sample.
- **Important commands run:** `ruff format .`; `ruff format --check .`; `ruff check .`; `mypy --strict src`; separate unit and integration Pytest suites; `python -m pip check`; `git diff --check`; typed config/manifest loaders and all three development split-pair validators; prompt/extractor/config/manifest hash assertions; 30/30/30/814 overlap/completeness proof; parsed tracked/raw summary equality; failure-sample/hash/category checks; exact 904-development-question leak scan; credential-pattern scan; `git check-ignore`; `git ls-files results/raw`; tracked-size review; sealed-path status; and complete status/diff review.
- **Result:** Ruff reports 25 files formatted and lint clean. Strict Mypy found no issues in 15 source files. All 95 unit tests and one integration test passed. `pip check`, whitespace, secrets, exact development-question leakage, frozen hashes, 30/30/30/814 overlap/completeness, manifest integrity/tampering, raw count/hash/ignore/tracking, aggregate equality, content-free schemas, 100-ID sample reproduction, category totals, tracked-size limit, and sealed-path status all passed. The ignored raw file has exactly 814 records and SHA-256 `73d52dace0f27577b1177bdfa81dfbb4c88252107c9b04e2ff49dbbd93da6cc0`.
- **Files changed:** the reviewed scope is five documentation files, three evaluation source files plus the CLI, three unit/integration test files, and two content-free aggregate result files. No config, prompt, parser, extractor, lock, model, dataset, manifest, sealed-final, or raw-result artifact changed.
- **Errors or uncertainty:** No final verification failed. Git emitted expected Windows LF-to-CRLF warnings. The tracked and ignored summary byte hashes differ because of line endings, but parsed JSON equality passes. The known scientific limitation remains the two false extractions found in the sampled wrong-output audit; verification cannot infer extractor precision in the unaudited scored-correct population.
- **Next action:** Stage exactly the 14 reviewed Milestone 2 paths, repeat staged whitespace, secret, benchmark-content, raw-path, sealed-path, size, and scope checks, create one local `eval:` commit with process-local repository identity if needed, verify a clean branch exactly one commit ahead of `origin/main`, do not push, and stop.

## 2026-07-18 00:35:41 -04:00 — Milestone 2 Step 6: reviewed scope staged for local commit

- **Current step:** Step 6 — stage the complete reviewed milestone as one atomic local commit without pushing.
- **Action performed:** Staged exactly 14 reviewed paths: five documentation/experiment files, four evaluation/CLI source files, three tests, and two content-free aggregate result files. Repeated staged whitespace, credential, prohibited-path, raw-path, sealed-path, size, and scope checks; confirmed no unstaged tracked changes remained before this live-log entry.
- **Reason:** Explicit staging prevents ignored raw predictions, model/data caches, benchmark content, frozen configurations, or unrelated workspace files from entering the milestone commit.
- **Important commands run:** explicit `git add -- <14 reviewed paths>`; `git diff --cached --check`; staged name/count/prohibited-path assertions; staged credential scan; index-object size review; unstaged-diff check; staged stat/name review; and repository status.
- **Result:** Exactly 14 expected files are staged; no candidate exceeds 1 MiB; no raw, config, manifest, sealed-final, cache, model, dataset, secret, or unrelated path is staged. The staged change contains 786 insertions and 16 deletions before this final live-log addition.
- **Files changed:** this final DEVLOG entry only; it will be restaged into the same 14-file atomic scope.
- **Errors or uncertainty:** No staged check failed. Expected Windows line-ending warnings remain non-blocking. The scientific extraction limitation is unchanged and fully disclosed.
- **Next action:** Restage this DEVLOG entry, repeat final staged whitespace/scope/safety checks, create exactly one local commit, verify a clean branch one ahead and zero behind `origin/main`, do not push, and stop.

## 2026-07-18 16:12:31 -04:00 — Milestone 2.1 Step 1: synchronized frozen audit source verified

- **Current step:** Step 1 — verify the published Milestone 2 checkpoint and exact ignored raw source before beginning the label-blind correct-response audit.
- **Action performed:** Temporarily prepended `C:\\Program Files\\Git\\cmd` to the process PATH; verified repository root, branch, local/remote refs, divergence, and cleanliness; checked the ignored raw prediction file, record/correct counts, raw hash, frozen model/dataset/prompt/extractor/config/manifest identities, development-only manifest chain, raw tracking, and sealed-path status without opening the sealed-final manifest.
- **Reason:** The audit must use the exact existing 814-record Milestone 2 output and must stop rather than regenerate if it is missing or altered. Starting from synchronized published history also prevents an audit commit from accidentally absorbing unpublished work.
- **Important commands run:** `git rev-parse`, `git rev-list --left-right --count`, `git status`, `git check-ignore`, `git ls-files results/raw`, SHA-256 calculation, typed config/manifest loaders and split validators, and content-free JSON record counts.
- **Result:** Local `main` and `origin/main` both point to `f4ac19f4e89917d3fee1e2831daad4757cee5f84`; 0 ahead/0 behind; worktree initially clean. Raw predictions exist with exactly 814 development records, 521 currently correct, 90 strict-parser correct and 431 canonical-only correct. Raw SHA-256 is `73d52dace0f27577b1177bdfa81dfbb4c88252107c9b04e2ff49dbbd93da6cc0`. Frozen prompt `738ea5a3...b18350`, extractor `e099d1c2...c06df`, config `5f315d5d...a622b`, and 814-ID manifest `5e810d3a...7897` match the published summary.
- **Files changed:** `docs/DEVLOG.md` only, after all clean-state and integrity checks passed.
- **Errors or uncertainty:** No verification error occurred. The raw records contain no sealed-final marker; the sealed-final manifest was not opened, hashed, or modified. No raw artifact is tracked. An initial memory-registry search returned no relevant project-memory entry and did not affect the repository.
- **Next action:** Build a deterministic label-blind audit view for only the 521 currently correct-scored records, exposing response/extraction evidence but withholding reference answers; add tests and keep detailed audit records under ignored `results/raw/`.

## 2026-07-18 16:15:39 -04:00 — Milestone 2.1 Step 2: label-blind audit view created

- **Current step:** Step 2 — construct a reproducible audit view that exposes extraction evidence for the 521 correct-scored responses while withholding benchmark scoring information.
- **Action performed:** Added a typed audit-view module with immutable raw-source expectations, exact extraction-rule/source-span tracing, completion/token metadata, terminal-context value enumeration, generic suspicion flags, and stable configuration/view hashes. Added six focused unit tests. Built and saved the complete detailed view plus content-free view summary under ignored `results/raw/correct_response_audit/current/`.
- **Reason:** Intent must be classified from the model completion and extraction evidence alone. A label-free, hash-bound view prevents benchmark agreement from influencing that judgment and makes the freeze point independently auditable.
- **Important commands run:** targeted Ruff formatting/lint, strict Mypy, Pytest, audit-configuration hashing, `build_label_blind_views`, `save_label_blind_views`, forbidden-field assertions, and `git check-ignore` for both generated audit files.
- **Result:** Exactly 521 label-blind views were created: 90 strict-parser accepted and 431 canonical-only. Extraction rules: 107 literal final-answer lines, 49 standalone decorated values, 28 explicit answer cues, 139 direct conclusion statements, 158 conclusion-verb prose, 16 decorated conclusions, and 24 terminal-prose conclusions. The raw-source hash remains `73d52dac...a6cc0`; audit configuration hash is `e50df38364b88d4900dfecc948cd56d0d552e050971ca47a7a21264699ee4122`; view hash is `b0fa85ffa26413137e05992a2b368ba4e059d04a649b47605fb70cbbe7e63dee`. Detailed JSONL is ignored and contains no `reference_answer` or `correct` field.
- **Files changed:** `src/foundry/evaluation/correct_audit.py`, `tests/unit/test_correct_audit.py`, `docs/DEVLOG.md`, and ignored `results/raw/correct_response_audit/current/{label_blind_views.jsonl,view_summary.json}`.
- **Errors or uncertainty:** No build, trace, hash, type, test, or ignore failure occurred. Generic flags identify 41 terminal contexts with multiple values and four with negative-intent language; 337 canonical-only records use broad conclusion rules. Flags prioritize review but are not classifications.
- **Next action:** Audit every view sequentially without labels, recording confirmed intent, false acceptance, or ambiguity. Report progress after approximately 100, 200, 300, 400, and all 521 records before freezing classifications.

## 2026-07-18 16:16:23 -04:00 — Milestone 2.1 Step 3 progress: 100/521 audited

- **Current step:** Step 3 — label-blind review of correct-scored responses 1–100.
- **Action performed:** Reviewed the extraction rule, exact source span, terminal context, response tail where flagged, strict/canonical status, completion metadata, and competing-value indicators for the first 100 views. Wrote 100 working classifications under ignored raw results without benchmark labels.
- **Reason:** Sequential checkpoints make the full 521-record audit observable and prevent a partial review from being presented as complete.
- **Result:** 100 confirmed intended answers; zero confirmed false acceptances; zero ambiguous cases. Seven multiple-value terminal contexts were benign: the extracted value was explicitly identified by the final conclusion while other values were supporting calculations or constraints. Common patterns were boxed/bold values, explicit answer cues, terminal equations, and direct prose conclusions.
- **Files changed:** ignored `results/raw/correct_response_audit/current/working_classifications.jsonl` and `docs/DEVLOG.md`.
- **Errors or uncertainty:** No classification remained unresolved in this batch. Reference answers and questions were not consulted or present in the audit view.
- **Next action:** Audit views 101–200 using the same frozen label-blind evidence and update the ignored working classification file.

## 2026-07-18 16:17:18 -04:00 — Milestone 2.1 Step 3 progress: 200/521 audited

- **Current step:** Step 3 — label-blind review of correct-scored responses 101–200.
- **Action performed:** Reviewed the next 100 extraction traces and extended the ignored working classification file to 200 records.
- **Result:** 200 cumulative confirmed intended answers; zero false acceptances; zero ambiguous cases. Five additional multi-value terminal contexts clearly separated goals, operands, or intermediate calculations from an explicit final extracted result. No percentage/currency miscapture, lost-sign pattern, truncation, or contradictory conclusion was found.
- **Files changed:** ignored working classifications and `docs/DEVLOG.md`.
- **Errors or uncertainty:** No unresolved record. Benchmark labels/questions remained hidden.
- **Next action:** Audit views 201–300 and preserve the same label-blind classification boundary.

## 2026-07-18 16:18:05 -04:00 — Milestone 2.1 Step 3 progress: 300/521 audited

- **Current step:** Step 3 — label-blind review of correct-scored responses 201–300.
- **Action performed:** Reviewed the next 100 traces and extended ignored working classifications to 300.
- **Result:** 300 cumulative confirmed intended answers; zero false acceptances; zero ambiguous cases. Eight multi-value contexts clearly named the extracted terminal result. One generic negative-intent flag described a positive magnitude (“short by” an amount); the unsigned extracted magnitude matched the model's explicit intent.
- **Files changed:** ignored working classifications and `docs/DEVLOG.md`.
- **Errors or uncertainty:** No unresolved case and no occurrence of either previously observed false-acceptance mechanism among this batch's correct-scored records.
- **Next action:** Audit views 301–400 label-blind.

## 2026-07-18 16:18:28 -04:00 — Milestone 2.1 Step 3 progress: 400/521 audited

- **Current step:** Step 3 — label-blind review of correct-scored responses 301–400.
- **Action performed:** Reviewed the next 100 extraction traces and extended the ignored working classifications to 400 records.
- **Result:** 400 cumulative confirmed intended answers; zero confirmed false acceptances; zero ambiguous cases. Seven additional multiple-value contexts clearly separated intermediate calculations or stated goals from the terminal result. Two generic negative-intent flags described positive magnitudes (an amount lost and a weight lost), so the unsigned extracted values faithfully represented the model's explicit intent.
- **Files changed:** ignored `results/raw/correct_response_audit/current/working_classifications.jsonl` and `docs/DEVLOG.md`.
- **Errors or uncertainty:** No unresolved record. No percentage/currency miscapture, truncation, contradictory terminal answer, or generic last-number acceptance was found. Benchmark labels/questions remained hidden.
- **Next action:** Audit the final views 401–521 using the same label-blind evidence, then freeze all classifications before any score join.

## 2026-07-18 16:19:26 -04:00 — Milestone 2.1 Step 3 complete: 521/521 audited label-blind

- **Current step:** Step 3 — complete label-blind classification of every correct-scored response.
- **Action performed:** Reviewed views 401–521 sequentially and completed the ignored working classification file for all 521 responses. Each decision used only the completion, extraction rule, exact source span, terminal context, completion/token metadata, and generic suspicion flags.
- **Result:** 521 confirmed intended answers; zero confirmed false acceptances; zero ambiguous cases. Across the full population, all 41 multi-value terminal contexts explicitly distinguished the extracted conclusion from calculations or constraints. All four negative-intent flags expressed an unsigned positive magnitude (for example, an amount or weight lost), consistent with the model's written intent. No correct-scored response exhibited the prior percentage/currency false-match pattern.
- **Files changed:** ignored `results/raw/correct_response_audit/current/working_classifications.jsonl` and `docs/DEVLOG.md`.
- **Errors or uncertainty:** No record remained unresolved; no output was incomplete. This is an intent audit, not an independent re-solution of benchmark questions, and no question, reference label, or `correct` flag was visible during classification.
- **Next action:** Validate completeness and freeze the 521 classifications with a stable hash. Only after that immutable freeze will benchmark scoring metadata be joined.

## 2026-07-18 16:23:01 -04:00 — Milestone 2.1 Step 4: label-blind classifications frozen

- **Current step:** Step 4 — validate completeness, internal consistency, evidence fidelity, and immutable classification provenance before score joining.
- **Action performed:** Added typed freeze/load validation and tests, normalized all 521 working decisions into a stable ignored JSONL schema, verified every identifier and extraction-evidence field against its label-blind view, saved a content-free freeze summary, and independently recomputed the classification hash.
- **Reason:** Freezing the intent decisions before benchmark labels are joined prevents score agreement from influencing or silently changing the audit classifications.
- **Important commands run:** targeted Ruff, strict Mypy, and eight audit unit tests; `freeze_label_blind_classifications`; SHA-256 recomputation; line-count and `git check-ignore` verification.
- **Result:** Freeze status is `frozen_before_score_join`; 521/521 records are present and unique; counts are 521 confirmed intended, zero false acceptance, zero ambiguous. Frozen classification SHA-256 is `669a866e984c35908bdb9e5443cb989733fd762d11bf62456387a25a5c12e14c`; label-blind view SHA-256 remains `b0fa85ffa26413137e05992a2b368ba4e059d04a649b47605fb70cbbe7e63dee`; audit configuration SHA-256 remains `e50df38364b88d4900dfecc948cd56d0d552e050971ca47a7a21264699ee4122`.
- **Files changed:** `src/foundry/evaluation/correct_audit.py`, `tests/unit/test_correct_audit.py`, `docs/DEVLOG.md`, and ignored `results/raw/correct_response_audit/current/{classifications.jsonl,freeze_summary.json}`.
- **Errors or uncertainty:** Initial targeted verification exposed three Ruff line-length errors and one strict-Mypy construction issue; both were fixed before freezing. All repeated targeted checks passed, and detailed audit artifacts remain ignored.
- **Next action:** Load only the hash-verified frozen classification file, join it with the unchanged existing score records, and write a content-free aggregate audit summary.

## 2026-07-18 16:23:28 -04:00 — Milestone 2.1 Steps 5–9: score join and trust decision complete

- **Current step:** Join frozen intent classifications to existing scores, review known failure patterns, reassess the provisional taxonomy, and apply the predeclared decision rule.
- **Action performed:** Loaded the detailed classification file only after verifying its frozen SHA-256, matched all 521 identifiers to the unchanged 814-record scoring artifact, calculated the audited accuracy bounds, compared the two prior generic false-acceptance mechanisms, and wrote a 3.2 KB content-free aggregate summary.
- **Reason:** The join determines whether any correct score was a coincidental extractor match while preserving the original evaluator result and keeping label information out of the intent judgments.
- **Important commands run:** `load_frozen_classifications`, `build_content_free_audit_summary`, raw-source hash and population checks, identifier-set equality, pattern/classification intersections, and aggregate JSON serialization.
- **Result:** Frozen evaluator score remains 521/814 = 64.0049%. Audited counts are 521 intended correct, zero false-positive correct, and zero ambiguous; lower bound, upper bound, and adjusted exact accuracy therefore all equal 521/814 = 64.0049%, with a 0/521 = 0% confirmed false-positive rate. The strict/canonical-only correct split is 90/431. The percentage-plus-currency terminal-collision pattern occurs zero times. Four closely related unsigned negative-language cases occur, but every one explicitly states a positive magnitude and none is a false acceptance. Decision-rule outcome: `BASELINE TRUSTED`.
- **Files changed:** `results/development_baseline/qwen2_5_1_5b/correct_response_audit.json` and `docs/DEVLOG.md`; detailed joins/classifications remain ignored.
- **Errors or uncertainty:** No join, hash, count, or consistency error occurred. The result establishes extractor intent precision for the 521 correct-scored development outputs; it does not independently re-solve questions, expand the provisional 100-record failure taxonomy, or make a sealed-final claim.
- **Next action:** Document the audit method, trust decision, taxonomy implications, limitations, and next proposed milestone across the five required project records, then run the full safety and verification suite.

## 2026-07-18 16:25:27 -04:00 — Milestone 2.1 Step 8: aggregate evidence and project records updated

- **Current step:** Make the bounded audit independently reviewable without committing benchmark content or detailed completions.
- **Action performed:** Added decision D-013, registered experiment EXP-20260718-006, updated the current phase and next proposed milestone, and documented label blindness, freeze ordering, suspicious-pattern findings, taxonomy implications, limitations, and the `BASELINE TRUSTED` decision across all five required project records.
- **Reason:** Future work must be able to distinguish the immutable frozen evaluator score from the exhaustive intent audit, understand why the score is now trusted, and retain the limits of the provisional development taxonomy.
- **Result:** Documentation consistently reports 521/521 intended answers, zero false-positive correct answers, zero ambiguity, a 90/431 strict/canonical-only split, and identical 64.0049% frozen/lower/upper/adjusted accuracy. It preserves the two known wrong-output false extractions, states that the 231 extractable-wrong and 62 unextractable populations remain useful with caveats, and proposes only a separately approved Milestone 3 targeted synthetic-data design.
- **Files changed:** `docs/PROJECT_PLAN.md`, `docs/DECISIONS.md`, `docs/DEVLOG.md`, `docs/LEARNING_NOTES.md`, `experiments/EXPERIMENTS.md`, and aggregate-only `results/development_baseline/qwen2_5_1_5b/correct_response_audit.json`.
- **Errors or uncertainty:** No documentation or aggregate-generation error occurred. Windows Git reports expected future LF-to-CRLF normalization warnings. No raw output, question, reference answer, dataset row, model artifact, or identifier list is included in the tracked summary.
- **Next action:** Run the complete format, lint, type, unit/integration, dependency, whitespace, secret, benchmark-leak, raw-ignore, aggregate-consistency, size, sealed-path, diff, and repository-status verification suite; fix only Milestone 2.1 issues.

## 2026-07-18 16:27:14 -04:00 — Milestone 2.1 Step 10: full verification passed

- **Current step:** Verify code quality, audit reproducibility, aggregate correctness, content containment, and final commit scope.
- **Action performed:** Ran Ruff format/check, Ruff lint, strict Mypy, all unit and integration tests, `pip check`, `git diff --check`, exact cached development-question leakage and credential-pattern scans, independent raw/classification hash and count checks, aggregate arithmetic/provenance/schema checks, detailed-raw ignore and tracking checks, frozen evaluator/config/manifest diff checks, candidate-size review, and repository status/diff review.
- **Reason:** The pushed milestone must be reproducible from unchanged raw evidence, contain no benchmark content or secrets, leave the evaluator frozen, and add only the bounded audit implementation, tests, aggregate result, and documentation.
- **Important commands run:** `ruff format .`; `ruff format --check .`; `ruff check .`; `mypy --strict src`; separate `pytest tests/unit` and `pytest tests/integration`; `python -m pip check`; `git diff --check`; offline development-only dataset leak scan; SHA-256/count scripts; `git check-ignore`; `git ls-files results/raw`; frozen-path diff and file-size checks.
- **Result:** Ruff reports 27 files formatted and lint clean. Strict Mypy found no issues in 16 source files. All 103 unit tests and one integration test passed. `pip check`, whitespace, exact 904-development-question leak, secret, raw hash/count, 521-classification hash/count, aggregate arithmetic/provenance, forbidden-field, raw ignore/tracking, frozen-stack diff, and tracked-size checks passed. No candidate exceeds 1 MB; no detailed raw artifact is tracked.
- **Files changed:** No additional project artifact changed during verification; this DEVLOG checkpoint records the completed checks.
- **Errors or uncertainty:** The first targeted implementation pass found and corrected three Ruff line-length errors and one Mypy type-construction error before the freeze. The full suite had no failure. The offline dataset loader emitted an expected cache-use notice, and Git emitted expected LF-to-CRLF warnings.
- **Next action:** Stage exactly the eight reviewed Milestone 2.1 paths, repeat staged scope/content/whitespace/size checks, create one atomic `eval:` commit with process-local published identity if required, verify `origin/main` has not moved, push without force, and confirm a clean synchronized branch.

## 2026-07-18 16:28:02 -04:00 — Milestone 2.1 Step 10: remote and staged scope verified

- **Current step:** Confirm published history is unchanged and stage only the reviewed milestone artifacts.
- **Action performed:** Fetched `origin/main` without merging, confirmed local and remote both remain at `f4ac19f4e89917d3fee1e2831daad4757cee5f84` with zero ahead/behind, staged the exact five documentation files plus audit aggregate, module, and unit test, and repeated staged whitespace, raw/sealed-path, scope, and size checks.
- **Result:** Exactly eight approved files are staged; no raw or sealed path is staged; no staged file exceeds 1 MB; `git diff --cached --check` passes. The complete candidate is 1,069 inserted and 12 removed lines, primarily the typed audit/freeze/aggregate implementation and live DEVLOG checkpoints.
- **Errors or uncertainty:** The first read-only divergence assertion compared Git's tab-delimited `0 0` output to a space-delimited string and raised after already printing the correct zero/zero state. It did not change repository or remote state; the corrected staged verification passed. Expected LF-to-CRLF warnings remain non-fatal.
- **Next action:** Restage this final DEVLOG entry, repeat staged content and scope checks, create one atomic Milestone 2.1 commit, then push only if the remote tip still equals `f4ac19f`.

## 2026-07-18 16:37:27 -04:00 — Milestone 3 Step 1: synchronized starting state verified

- **Current step:** Verify the trusted Milestone 2.1 commit and unchanged development-only evidence before designing synthetic-data contracts.
- **Action performed:** Temporarily exposed Git in the current shell, fetched `origin/main` without merging, verified repository root/branch/tips/divergence/worktree, recomputed the ignored raw prediction hash and population counts, checked frozen prompt/extractor/config/manifest/classification provenance, and confirmed raw containment.
- **Reason:** Milestone 3 must derive only content-free failure patterns from the exact trusted baseline and must stop rather than silently use regenerated, changed, or sealed evidence.
- **Important commands run:** `git fetch origin main`; `git rev-parse`; `git rev-list --left-right --count`; `git status --porcelain`; `Get-FileHash`; local JSON count/hash inspection; `git check-ignore`; `git ls-files results/raw`.
- **Result:** Repository is `C:/Users/Admin/Projects/Foundry`, branch `main`; local and `origin/main` both equal `0d838318b22404eac1b497ad859a0cfb8b6c7343`; divergence is 0/0 and worktree was clean. Raw SHA-256 remains `73d52dace0f27577b1177bdfa81dfbb4c88252107c9b04e2ff49dbbd93da6cc0`; 814 records comprise 521 correct, 231 extractable-wrong, and 62 unextractable. Frozen manifest/config/extractor/prompt/classification hashes match the trusted aggregate. Raw predictions are ignored and no `results/raw` path is tracked.
- **Files changed:** `docs/DEVLOG.md` only.
- **Errors or uncertainty:** No verification error or state drift. The sealed-final manifest and content were not opened, hashed, or loaded.
- **Next action:** Build an ignored diagnostic classification view for all 293 development failures and review them sequentially, reporting progress near 100, 200, and 293.

## 2026-07-18 16:41:59 -04:00 — Milestone 3 Step 2 progress: 100/293 failures reviewed

- **Current step:** Complete the development failure taxonomy from existing development questions, labels, and responses while keeping all content-bearing records ignored.
- **Action performed:** Materialized only the 904 development examples from the cached pinned dataset, joined the 814 baseline IDs locally, created 293 ignored review views, reused the 100-record Milestone 2 manual taxonomy as prior evidence, used a deterministic local bag-of-words suggestion only to order the remaining review, and directly reviewed/classified failures 1–100. Each record received one primary category, secondary tags, reasoning/output kind, confidence, targetability, generator representability, and deterministic-verification feasibility.
- **Reason:** Exhaustive review is needed to rank synthetic targets from measured weaknesses rather than extrapolating the provisional 100-record sample. The ignored view prevents questions, answers, and completions from entering tracked artifacts or any future generator input.
- **Result:** 100 reviewed: 76 reasoning and 24 output/extraction failures. Primary counts: 28 multi-step bookkeeping/omission; 18 target/language; 12 rate/ratio/percentage/average; 8 constraint/distribution/discrete; 5 time/unit/sequence; 4 arithmetic execution; 1 benchmark ambiguity/annotation risk; 24 output-format/extraction. Confidence is 98 high, 1 medium, 1 low; 99 are automatically targetable and one benchmark-risk record is not. Review found one additional wrong-output false extraction: the completion clearly concluded with the intended duration but the extractor selected an earlier trip count. It remained scored wrong and does not affect the trusted correct numerator.
- **Files changed:** ignored `results/raw/failure_taxonomy/current/{review_views.jsonl,working_classifications.jsonl}` and `docs/DEVLOG.md`.
- **Errors or uncertainty:** The local suggestion layer overemphasized surface keywords in several records; direct review overrode those suggestions. It is diagnostic assistance only, not an LLM judge or committed classifier. Dataset loading was offline/cached and accessed development examples only; sealed-final content was not loaded.
- **Next action:** Directly review failures 101–200, record the second progress checkpoint, and continue to all 293 before freezing aggregate counts or generator priorities.

## 2026-07-18 16:43:07 -04:00 — Milestone 3 Step 2 progress: 200/293 failures reviewed

- **Current step:** Continue exhaustive development-failure classification using question, reference, completion, extraction evidence, and terminal intent.
- **Action performed:** Directly reviewed failures 101–200, overrode surface-keyword suggestions where the actual failure was a target, sequence, rounding, bookkeeping, or extractor-intent error, and updated the ignored working taxonomy with primary/secondary categories, confidence, kind, and generator/verifier feasibility.
- **Result:** 200 cumulative reviewed: 150 reasoning and 50 output/extraction failures. Primary counts: 49 bookkeeping/omission; 34 target/language; 19 constraint/distribution/discrete; 17 rate/ratio/percentage/average; 17 time/unit/sequence; 12 arithmetic; 2 benchmark ambiguity/annotation risk; 50 output-format/extraction. Confidence: 188 high, 10 medium, 2 low. This batch confirmed three additional wrong-output false extractions in which a clear terminal answer differed from an earlier selected number; combined with the two already known and the first-batch discovery, six wrong-output false extractions are now documented. All remained outside the correct numerator.
- **Files changed:** ignored working classifications and `docs/DEVLOG.md`.
- **Errors or uncertainty:** Category boundaries can overlap, especially bookkeeping versus domain skill and target interpretation versus constraint errors. One malformed “two times less” benchmark wording was conservatively assigned benchmark-ambiguity risk with low confidence and excluded from automatic targeting.
- **Next action:** Review failures 201–293, audit every extractor-intent mismatch systematically, freeze all 293 content-free classifications, and compute complete aggregate distributions.

## 2026-07-18 16:45:06 -04:00 — Milestone 3 Step 2 complete: 293/293 failures reviewed and frozen

- **Current step:** Finish the complete taxonomy, systematically audit terminal-intent mismatches, validate coverage, and freeze detailed ignored classifications before selecting synthetic targets.
- **Action performed:** Directly reviewed failures 201–293, corrected domain-versus-root-cause assignments, scanned all 231 extractable-wrong completions for disagreement between extracted and terminal values, validated sequential coverage/unique IDs, attached the false-extraction tag to every confirmed case, and wrote canonical ignored classification/freeze files.
- **Result:** Final primary counts: output-format/extraction 69; multi-step bookkeeping/omission 68; target/language interpretation 53; rate/ratio/percentage/average 28; constraint/distribution/discrete 27; time/unit/sequence 24; arithmetic execution 22; benchmark ambiguity/annotation risk 2. Failure kind: 224 reasoning and 69 output. Confidence: 274 high, 17 medium, 2 low. 291/293 are automatically targetable/generatable/verifiable in principle; the two benchmark-risk records are excluded. Seven false extractions are confirmed among wrong outputs, including the two previously reported; they remain wrong and do not alter the exhaustive zero-false-positive audit of the 521 correct-scored responses. Frozen taxonomy SHA-256 is `964d0c18b60d4f0262f0ec711b2f13b396ca4fa1c921f0dc2e91205d393cb692`.
- **Files changed:** ignored `results/raw/failure_taxonomy/current/{working_classifications.jsonl,classifications.jsonl,freeze_summary.json}` and `docs/DEVLOG.md`.
- **Errors or uncertainty:** One intermediate freeze was regenerated because the final 93 direct-review overrides had not first been copied back to the working file; the corrected canonical freeze re-applied all overrides, verified counts, and replaced only ignored provisional artifacts. Category boundaries remain interpretive and secondary tags overlap by design.
- **Next action:** Score all reasoning categories for prevalence, independent generation/verification, ambiguity, diversity, contamination risk, RTX feasibility, and measurable impact; select exactly three reasoning targets plus a separate output-contract track.

## 2026-07-18 16:52:00 -04:00 — Milestone 3 Steps 3–6 and 8–10: design contracts implemented

- **Current step:** Rank the complete failure taxonomy, choose the bounded pilot curriculum, and turn the design into typed, testable contracts without implementing a generator.
- **Action performed:** Selected multi-step bookkeeping/omission, rate/ratio/percentage/average, and constraint/distribution/discrete reasoning as the three pilot targets; separated `terminal-final-answer-contract-v1` as a shared output track; compared procedural templates, local-model paraphrasing, and frontier-model generation; selected fully procedural latent programs with controlled templates; defined exact rational schema, dual-verifier contracts, contamination stages and thresholds, matched 4,000-example targeted/generic budgets, future RTX estimates, and generation/training gates. Added 19 original unit tests covering exact arithmetic, provenance, verifier disagreement, answer-line enforcement, obvious paraphrases, number swaps, structural copies, semantic escalation, curriculum selection, and matched budgets.
- **Reason:** The first pilot needs labels derived from executable structure rather than model trust, while its targeted-versus-generic comparison must differ only in curriculum selection.
- **Important commands run:** targeted `ruff format`, `ruff check`, strict `mypy src`, and `pytest tests/unit/synthesis -q`.
- **Result:** Architecture A is selected because it is reproducible, dependency-light, locally executable, and lowest-risk for labels and contamination. The pilot contract is 120 smoke candidates followed, only after approval and gates, by matched 4,000-example targeted and generic datasets (3,600 train/400 synthetic validation each, including 800 output-contract examples). Focused Ruff and Mypy passed; all 19 synthesis-contract tests passed.
- **Files changed:** `src/foundry/synthesis/`, `configs/synthesis/gsm1k_phase1.yaml`, `tests/unit/synthesis/`, and `docs/DEVLOG.md`.
- **Errors or uncertainty:** Initial focused lint found one mutable-construction default warning; it was replaced with one frozen module-level policy and all repeated checks passed. Runtime, disk, token, and future QLoRA figures are engineering estimates rather than measurements. A local semantic-similarity backend has deliberately not been downloaded; its exact artifact must be pinned before pilot generation, and a missing semantic check forces manual review rather than acceptance.
- **Next action:** Write the content-free aggregate taxonomy and comprehensive frozen design record, then update the project plan, decisions, learning notes, and experiment register before the full repository verification.

## 2026-07-18 16:57:00 -04:00 — Milestone 3 Step 11: frozen design documented

- **Current step:** Make the exhaustive taxonomy and all pre-generation decisions independently reviewable without tracking benchmark content.
- **Action performed:** Wrote the complete design record, added D-014, registered EXP-20260718-007, updated the project phase and learning notes, and created a 9.2 KB aggregate taxonomy containing only counts, content-free metadata, hashes, and stable 12-character identifier prefixes. Documented category ranking, the three selected tracks, shared output contract, typed schema, architecture comparison, dual-verifier methods, contamination thresholds, matched control, size/compute estimates, staged plan, success gates, risks, and exact proposed Milestone 4.
- **Reason:** Generator and training decisions must be frozen before synthetic examples exist, and later reviewers need an auditable link from measured failure counts to the proposed controlled experiment.
- **Important commands run:** independent JSON validation; synthesis/taxonomy contract hash computation; SHA-256 checks for the design config and contracts; repository status review.
- **Result:** Detailed taxonomy SHA-256 remains `964d0c18b60d4f0262f0ec711b2f13b396ca4fa1c921f0dc2e91205d393cb692`; content-free taxonomy contract SHA-256 is `021837a1f1a3bb5a189b1f39c808bb907e415e28d8fa722a8a03c3114717cf28`; synthesis design contract SHA-256 is `910bf21dba7cef833fd9f7bd83842034e9e7261cf93979d7cdddc0479094d347`; synthesis configuration SHA-256 is `7c087ac45c9027ab872cfecbc0dbf6123b60ec088b35e6d6ddc4dfd9094a99d5`.
- **Files changed:** `docs/{PROJECT_PLAN,DECISIONS,DEVLOG,LEARNING_NOTES,SYNTHETIC_DATA_DESIGN}.md`, `experiments/EXPERIMENTS.md`, and `results/development_baseline/qwen2_5_1_5b/complete_failure_taxonomy.json`.
- **Errors or uncertainty:** No documentation-generation error occurred. Runtime, disk, token, and future QLoRA numbers are planning estimates. Semantic thresholds are frozen, but the exact local encoder is an explicit Milestone 4 decision and no embedding model or scan was used here.
- **Next action:** Run the complete repository verification and safety suite, correct only Milestone 3 issues, then review the exact staged commit scope before publishing one atomic commit.

## 2026-07-18 16:58:22 -04:00 — Milestone 3 Step 12: full verification passed

- **Current step:** Verify design consistency, code quality, content containment, and exact repository scope before the atomic commit and push.
- **Action performed:** Ran full Ruff format/check and lint, strict Mypy, all unit and integration tests, `pip check`, `git diff --check`, secret-pattern and tracked-size scans, an exact and 12-token-window scan against every locally reviewed development failure question/response, raw hash/count/identifier and aggregate consistency checks, synthesis-config assertions, raw ignore/tracking checks, frozen-evaluator diff checks, and sealed-path status review without reading sealed contents. Added exact hash regression assertions for both content-free design contracts and repeated the complete code/test suite.
- **Reason:** The published design must be testable, must preserve the evaluator and sealed partition, and must contain no benchmark content, raw model output, secret, dataset, model artifact, or generated training example.
- **Important commands run:** `ruff format .`; `ruff format --check .`; `ruff check .`; `mypy --strict src`; separate unit/integration Pytest runs; `python -m pip check`; `git diff --check`; structured JSON/YAML/hash checks; exact and 12-token development-content scan; `git check-ignore`; `git ls-files results/raw`; frozen-path and tracked-size review.
- **Result:** Ruff reports 37 files formatted and lint clean. Strict Mypy found no issues in 22 source files. All 123 unit tests and one integration test passed. Dependency, whitespace, secret, development-content, 293-ID completeness/uniqueness, raw/classification hash, aggregate schema, matched 4,000/4,000 contract, contamination-threshold, raw ignore/tracking, sealed-path status, frozen-evaluator diff, and 1 MB candidate-size checks passed. The detailed taxonomy and raw prediction hashes remain unchanged.
- **Files changed:** One additional unit-test assertion freezes the published taxonomy and synthesis contract hashes; this DEVLOG entry records verification. Formatting made no content change.
- **Errors or uncertainty:** The first structured safety command used configuration key names that differed from the actual frozen YAML and stopped at an assertion before its remaining shell checks. The corrected check used the exact YAML schema and passed completely. Expected Windows LF-to-CRLF Git warnings are non-fatal. Sealed contents were not opened or hashed.
- **Next action:** Stage exactly the 18 reviewed Milestone 3 files, repeat staged whitespace/content/secret/size/scope checks, confirm `origin/main` has not moved, create one atomic commit, push without force, and verify a clean synchronized branch.

## 2026-07-18 16:59:03 -04:00 — Milestone 3 Step 12: remote and staged scope verified

- **Current step:** Confirm published history is unchanged and prepare the exact reviewed snapshot for one atomic commit.
- **Action performed:** Fetched `origin/main` without merging, confirmed local and remote still equal trusted commit `0d838318b22404eac1b497ad859a0cfb8b6c7343` with zero ahead/behind, staged exactly the 18 reviewed Milestone 3 paths, and inspected staged names/statistics plus unstaged state.
- **Result:** The staged scope contains five project records, the new design document, one synthesis config, one content-free aggregate, six design-contract modules, and four unit-test files. No raw, evaluator, manifest, sealed-final, model, cache, dataset, dependency-lock, or training artifact is staged. There were no unstaged tracked changes before this live checkpoint.
- **Errors or uncertainty:** The first staged whitespace check correctly flagged two Markdown hard-break spaces in the new design header. They were removed; no substantive design content changed. Expected LF-to-CRLF warnings remain non-fatal.
- **Next action:** Restage the corrected design header and this checkpoint, repeat all staged scope, whitespace, secret, content, hash, and size checks, then commit and push only if `origin/main` remains unchanged.

## 2026-07-18 17:09:44 -04:00 — Milestone 4 Step 1: frozen starting state verified

- **Current step:** Verify the published Milestone 3 state before selecting a semantic artifact or implementing generation.
- **Action performed:** Temporarily exposed Git in the current shell, fetched `origin/main` without merging, verified repository root/branch/tips/divergence/worktree, recomputed the four frozen evidence hashes, confirmed ignored raw development artifacts remain local, and checked frozen-evaluator and sealed-path status without opening sealed contents.
- **Reason:** The bounded generator smoke must be derived from the exact published design and trusted development-only evidence, not an altered evaluator, taxonomy, configuration, or remote history.
- **Important commands run:** `git fetch origin main`; `git rev-parse`; `git rev-list --left-right --count`; `git status --porcelain`; SHA-256 recomputation; content-free contract hash functions; `git check-ignore`; `git ls-files results/raw`; frozen-path diff/status checks.
- **Result:** Repository is `C:/Users/Admin/Projects/Foundry` on `main`; local and `origin/main` both equal `e99be66304eeeede98d1787cf9b1edb049a4a057`; divergence is 0/0 and the worktree began clean. Detailed taxonomy, content-free taxonomy, synthesis contract, and synthesis-config hashes exactly match the approved values. Raw baseline/taxonomy records exist and are ignored; no `results/raw` path is tracked; frozen evaluator paths are unchanged.
- **Files changed:** `docs/DEVLOG.md` only.
- **Errors or uncertainty:** No repository, hash, or containment error. The sealed-final file was not opened, read, or hashed; only its Git status was checked.
- **Next action:** Compare official metadata for at most three qualifying local embedding artifacts, select exactly one, document its immutable configuration before download, and validate it only on original hand-authored fixtures.

## 2026-07-18 17:14:00 -04:00 — Milestone 4 Step 2: semantic artifact selected before download

- **Current step:** Compare no more than three qualifying local sentence encoders and pin one operational configuration before retrieving weights.
- **Action performed:** Reviewed official Hugging Face model cards and repository metadata for `sentence-transformers/all-MiniLM-L6-v2`, `intfloat/e5-small-v2`, and `BAAI/bge-small-en-v1.5`; resolved each immutable revision and required-file size; compared licenses, dimensions, pooling, normalization, prefix behavior, dependencies, CPU/Windows compatibility, and offline use; selected and configured MiniLM without downloading weights yet.
- **Reason:** The frozen semantic thresholds need one reproducible local encoder, but model choice must not be tuned on GSM1K or silently introduce a large framework/dependency change.
- **Result:** Selected `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, Apache-2.0, 384 dimensions, 91,577,897 expected bytes for the eight approved files, CPU float32, attention-mask mean pooling, L2 normalization, cosine by normalized dot product, maximum length 256, batch size 32, `trust_remote_code=False`, and local-only loading after download. Existing versions are PyTorch `2.5.1+cu121`, Transformers `4.46.3`, Hugging Face Hub `0.36.2`, and safetensors `0.8.0`; no new package is required.
- **Files changed:** `configs/synthesis/semantic_all_minilm_l6_v2.yaml`, `docs/SYNTHETIC_DATA_DESIGN.md`, and `docs/DEVLOG.md`.
- **Errors or uncertainty:** All three candidates satisfy the size/license/local constraints. MiniLM is smallest and avoids E5's mandatory similarity prefix and BGE's different CLS-pooling geometry. Suitability under the immutable 0.75/0.82 thresholds is not assumed; it must now pass original fixtures or the milestone stops before generation.
- **Next action:** Download only the eight pinned MiniLM files into the ignored synthesis cache, validate local-only deterministic embeddings on original fixtures, and stop if the frozen thresholds cannot separate the fixture classes acceptably.

## 2026-07-18 17:17:48 -04:00 — Milestone 4 Step 2 complete: semantic artifact validated

- **Current step:** Retrieve only the selected encoder files and test deterministic local-only behavior against original fixtures before procedural generation.
- **Action performed:** Confirmed the synthesis cache is ignored, downloaded exactly the eight pinned MiniLM files, verified every file and total byte count, implemented a strict local artifact loader and CPU float32 attention-mask mean-pooling encoder, ran six hand-authored fixture pairs twice in one process and once in a fresh process, and saved full fixture text/results only under ignored raw output.
- **Reason:** Generator execution is permitted only if the selected encoder behaves acceptably with the already frozen 0.75/0.82 thresholds and remains deterministic without network or remote code.
- **Important commands run:** pinned `snapshot_download` with an eight-file allowlist; local file/hash verification; `AutoTokenizer`/`AutoModel.from_pretrained` with `local_files_only=True`, `trust_remote_code=False`, and safetensors; focused Ruff, strict Mypy, four unit tests; three deterministic fixture passes.
- **Result:** Required artifact size is exactly 91,577,897 bytes; local cache disk size is 91,578,751 bytes; artifact-manifest SHA-256 is `2754d14e196cdd11da43a01af8bfd43b479344265102daeae2f7dffab61eda5c`. Similarities/outcomes: exact duplicate 0.99999988/reject; number swap 0.95753330/reject; close paraphrase 0.93767148/reject; related-but-different 0.50201571/pass; unrelated -0.01185362/pass. The structurally equivalent rewrite scores 0.57151735/pass semantically, as expected, and is required to be rejected by the earlier latent-structure gate. Deterministic output SHA-256 is `4998fa509da71f7e1f681059d8fd68ea91deae0e3e6a3b38a912c6341cd73ba0` in repeated and fresh processes.
- **Files changed:** `src/foundry/synthesis/semantic.py`, `tests/unit/synthesis/test_semantic.py`, original fixture JSON, content-free `results/synthesis_smoke/semantic_artifact_validation.json`, semantic config/design docs, and this DEVLOG; model files and detailed fixture results remain ignored.
- **Errors or uncertainty:** The first focused check found one Ruff import-location rule and one untyped-Transformers Mypy import; both received narrow source fixes and repeated checks passed. The encoder cannot by itself detect every structural copy, which is why structural hashing is a mandatory earlier gate. No threshold or fixture was changed after observing the results.
- **Next action:** Implement the multi-step bookkeeping generator and its exact DAG/state-ledger verifier pair, with deterministic seeds, difficulty controls, original templates, and no benchmark-facing generator interface.

## 2026-07-18 17:19:56 -04:00 — Milestone 4 Step 3A: bookkeeping generator implemented

- **Current step:** Implement the first approved procedural family with two genuinely different exact verification paths.
- **Action performed:** Added a benchmark-blind generator interface and deterministic bookkeeping generator covering sequential additions/removals, inbound/outbound transfers, and exact grouping. Easy/medium/hard produce two/three/four dependent updates; grouping variants append exact division. Added primary topological DAG execution, independent signed-ledger replay with conservation/inverse checks, value-neutral structural signatures, stable IDs, output-track rendering, constraint checks, and four original unit tests.
- **Reason:** Bookkeeping/omission is the largest reasoning-failure category and must be representable by exact state transitions rather than model-authored prose or labels.
- **Result:** Same seed/difficulty/variant produces byte-equivalent drafts and structure hashes. Both verifier methods agree with the canonical exact answer, grouping is integral, difficulty increases dependency depth, and public generator parameters are limited to seed, difficulty, variant, and output-track flag. Focused Ruff and strict Mypy passed; all four bookkeeping tests passed.
- **Files changed:** `src/foundry/synthesis/generators/{__init__,bookkeeping}.py`, `tests/unit/synthesis/test_bookkeeping_generator.py`, and `docs/DEVLOG.md`.
- **Errors or uncertainty:** Initial focused checks found one unnecessary encoding argument and two unvalidated object-to-integer conversions; narrow fixes added explicit payload type validation and repeated checks passed. Controlled prose diversity and contamination behavior remain to be measured in the bounded smoke.
- **Next action:** Implement the rate/ratio/percentage/average family with exact rational equations and a cross-multiplication/inverse/unit verifier.

## 2026-07-18 17:21:41 -04:00 — Milestone 4 Step 3B: rational-relation generator implemented

- **Current step:** Implement the second approved procedural family with exact rational labels and independent equation checks.
- **Action performed:** Added deterministic modes for per-interval rates, scaled ratios, percentages, weighted averages, and combined rates. All modes use integers and `Fraction`; hard weighted averages add a third weighted group. Added exact equation evaluation, a separate cross-product/inverse-substitution/unit-consistency verifier, controlled original templates, structural signatures, constraint checks, and four unit tests spanning every mode.
- **Reason:** Rate/ratio/percentage/average errors are common, independently generatable, and especially suitable for exact cross-checks that expose denominator or weighting mistakes.
- **Result:** All five modes reproduce exactly from seed and variant. Primary and independent methods agree with the canonical rational answer and have distinct method families. Weighted-average difficulty increases relationship count; percentage labels are exact fractions. Focused Ruff and strict Mypy passed; all four tests passed.
- **Files changed:** `src/foundry/synthesis/generators/rates.py`, `tests/unit/synthesis/test_rates_generator.py`, and `docs/DEVLOG.md`.
- **Errors or uncertainty:** One initial test incorrectly rejected the explanatory phrase “without floating point” rather than checking numeric representation; it was replaced with a direct exact-`Fraction` assertion and all checks passed. Surface diversity and benchmark similarity remain unmeasured until the bounded smoke.
- **Next action:** Implement the bounded discrete-allocation generator with constructive exact solutions, uniqueness checks, and an independent finite-domain enumerator.

## 2026-07-18 17:23:12 -04:00 — Milestone 4 Step 3C: discrete-constraint generator implemented

- **Current step:** Implement the third and final approved procedural family with guaranteed finite-domain targets and independent uniqueness proof.
- **Action performed:** Added deterministic two-type allocation, complete-package, equal-distribution, and dual-resource-capacity modes. Every mode constructs an exact nonnegative integer target, records explicit finite-domain constraints, and renders original controlled prose. Added a constructive exact solver, a separate brute-force enumerator that requires exactly one requested result, ambiguity/integrality validation, difficulty-controlled domain growth, and four unit tests.
- **Reason:** Constraint/distribution failures require more than arithmetic correctness: the generator must prove integrality, feasibility, bounds, and uniqueness before a label can be trusted.
- **Result:** All four modes reproduce from seed and variant; constructive and bounded-enumeration methods agree with distinct evidence families; every tested target is unique and integral; hard difficulty expands the bounded domain; and output-track completions contain exactly one canonical terminal line. Focused Ruff, strict Mypy, and all four tests passed.
- **Files changed:** `src/foundry/synthesis/generators/discrete.py`, `tests/unit/synthesis/test_discrete_generator.py`, and `docs/DEVLOG.md`.
- **Errors or uncertainty:** No implementation or focused-check failure occurred. Brute-force domains are intentionally small for the smoke and must retain explicit bounds if future generation is approved. Template diversity and contamination acceptance remain empirical smoke-test questions.
- **Next action:** Implement the shared output-contract scheduling, exact 60/60 attempt plan, ordered acceptance pipeline, development-question-only contamination loader, stable raw records, and deterministic replay.

## 2026-07-18 17:32:07 -04:00 — Milestone 4 Steps 4–7: bounded acceptance pipeline implemented

- **Current step:** Connect the three generators to the shared output-contract schedule, ordered quality gates, development-only contamination scanner, and reproducible 120-attempt plan.
- **Action performed:** Added the exact 60-targeted/60-generic curriculum with 12 output-contract attempts per group; implemented schema checks, family-specific dual verification, ambiguity checks, exact/numeric-template/latent-structure/five-token overlap gates, local MiniLM screening, unresolved-review rejection, stable raw records, content-free summaries, resource timing, and deterministic hashes. Added a loader that validates the identifier-only 904-row development manifest and exposes only the selected question field to the contamination scanner. Added CLI, unit, and integration coverage.
- **Reason:** The counted smoke needs one immutable path from seed to candidate decision, while benchmark content must remain isolated from every generator interface and tracked artifact.
- **Important commands run:** offline 904-question loader preflight with `HF_HOME=data/huggingface`; exact plan-count inspection; focused Ruff, strict Mypy, and synthesis Pytest runs.
- **Result:** Offline preflight loaded exactly 904 development questions; the plan contains 120 attempts, 60 per group and 24 output-contract attempts. Focused Ruff and strict Mypy passed, and all 44 synthesis tests passed before the replay-control addition. Smoke contract hash is `2fd1c4dfde7b9b404187e97225138c86d1f09cfd7e817bca7bd7940b2f0aad1a`.
- **Files changed:** `configs/synthesis/gsm1k_phase1_smoke.yaml`; `src/foundry/synthesis/{contamination,deduplication,pipeline,cli}.py`; generator and pipeline tests; `docs/DEVLOG.md`.
- **Errors or uncertainty:** Three preflight snippets initially used stale helper names or the default Hugging Face cache; all failed before generation. The corrected call used the implemented interfaces and approved local cache and passed. No candidate has yet been counted. Actual acceptance, semantic-review, and contamination rates remain empirical.
- **Next action:** Verify the replay controls and full synthesis test set, then start the single counted process and pause after exactly 60 attempts for the required progress checkpoint.

## 2026-07-18 17:34:43 -04:00 — Milestone 4 Steps 8–9: 120 attempts completed and replay matched

- **Current step:** Run the single bounded candidate smoke and prove that construction and decisions reproduce without counting another candidate set.
- **Action performed:** Started one CLI process with the frozen 120-attempt plan, paused inside that process at attempt 60, resumed to 120 without replacements, wrote full attempts only to ignored raw output, then ran a separately marked dry replay with the same master seed and no manual overrides.
- **Reason:** The smoke must measure fixed-attempt yield and prove deterministic IDs, program/text hashes, verifier evidence, contamination results, decisions, and aggregates before any readiness recommendation.
- **Important commands run:** `.venv/Scripts/python.exe -m foundry.synthesis.cli --config configs/synthesis/gsm1k_phase1_smoke.yaml --pause-at 60`; the same command with `--replay`; offline dataset and model-cache environment variables.
- **Result:** At 60 attempts, 18 were accepted and 42 rejected. At 120, 24 were accepted and 96 rejected. Verification produced zero primary failures, independent failures, disagreements, ambiguous targets, or generator exceptions. Rejections were 25 numeric-template copies, 50 five-token overlaps, 7 semantic rejections, and 14 unresolved semantic-review cases. Dry replay matched decision SHA-256 `968fd548510251dcb375e69d95ac7bb7d10a2800b0306d01ff835463cf9b45eb` and deterministic aggregate SHA-256 `6465afffdeb7c4165fb76b92b05b95b6e4af26e2e354454365b6693eb5c4e595` exactly.
- **Files changed:** ignored counted/replay raw records, content-free `results/synthesis_smoke/procedural_v1_summary.json`, and `docs/DEVLOG.md`.
- **Errors or uncertainty:** The 20% acceptance rate is far below the frozen 75% gate. The measured bottleneck is limited controlled-template diversity under intentionally strict duplicate screening, not arithmetic verification. Fourteen review-band decisions remain unresolved until the required human audit; no threshold will be changed and no replacements will be generated.
- **Next action:** Manually inspect all 120 attempts, resolve the 14 review-band cases conservatively, record false-label/invalid-acceptance/rejection-quality findings, and then compute the unchanged readiness gate.

## 2026-07-18 17:41:26 -04:00 — Milestone 4 Steps 10–13: full audit completed and readiness blocked

- **Current step:** Audit every fixed attempt, resolve semantic review cases, apply the unchanged readiness gate, and record the evidence without tracking candidate or benchmark text.
- **Action performed:** Reviewed all 120 rendered questions against latent programs, traces, exact answers, both verifier records, output-contract lines, decision reasons, and closest semantic matches. Compared every accepted question's closest development match locally. Conservatively rejected all 14 review-band cases because each was a generated-to-generated near-template match, finalized those decisions without resampling, replayed the finalized state, wrote per-ID audit decisions under ignored raw output, and updated the content-free summary and project records.
- **Reason:** Correct numeric labels alone do not prove that rendered training questions are coherent, diverse, uncontaminated, and appropriate for the claimed difficulty/category.
- **Important commands run:** read-only 15-record audit batches; local-only MiniLM closest-development comparisons; manual-decision finalization and dry replay; audit-record/gate finalizer; focused Ruff, strict Mypy, and audit unit tests.
- **Result:** 120/120 reviewed; zero false labels, verifier disagreements, incorrect rejections, unresolved contamination cases, or overlooked benchmark resemblance. Five invalid accepted examples were found: four bookkeeping questions combined heterogeneous object counts without defining a common inventory unit, and one discrete-capacity example had grammar and tied-constraint difficulty defects. Finalized replay matched decision SHA-256 `661410933e90680d34a06c1836c7aca6fecfd5bba507c2dfaf3d8ecd5340c8b9` and aggregate SHA-256 `eb85cf9efe130d34164bca20badb9b3dce8f050493abf0e014614332b68f8771`. The readiness gate failed for five invalid acceptances, 20% overall acceptance, only four accepted bookkeeping and four accepted discrete examples, and systematic generator weaknesses.
- **Files changed:** ignored manual semantic decisions/audit/replay records; content-free `results/synthesis_smoke/procedural_v1_summary.json`; `src/foundry/synthesis/audit.py`; audit tests; all required project/design/experiment records; and this DEVLOG.
- **Errors or uncertainty:** Human validity judgments are documented and conservative. The semantic encoder produced no review-band benchmark comparison; all 14 review cases were self-similarity within the generated pool. The exact acceptance bottleneck is measured, but the acceptance gain from future template repairs is unknown. No threshold was changed and no candidate was replaced.
- **Next action:** Run the full repository quality/safety suite, verify no raw candidate, benchmark, cache, model, secret, or sealed content is tracked, then create and push one atomic Milestone 4 commit only if every check passes and `origin/main` has not moved.

## 2026-07-18 17:45:03 -04:00 — Milestone 4 Step 14: full verification passed

- **Current step:** Verify implementation quality, aggregate consistency, content containment, and exact commit scope before publishing the completed-but-blocked smoke result.
- **Action performed:** Ran full Ruff formatting/checks, strict Mypy, every unit/integration test, `pip check`, whitespace checks, secret patterns, exact development-question and changed-file 12-token leak scans, raw/model/cache tracking checks, candidate-plan and output-contract counts, verifier agreement, contamination-decision consistency, finalized replay proof, all four incoming frozen hashes, frozen-evaluator diffs, sealed-path status, and tracked-file size review.
- **Reason:** A failed readiness result is still a valid milestone outcome only if its code, evidence, safety boundaries, and negative conclusion are reproducible and safe to publish.
- **Result:** Ruff is clean; strict Mypy reports no issues in 31 source files; all 149 unit and two integration tests pass; dependencies and whitespace are clean. The 904-question scan found zero exact tracked-question matches and zero 12-token matches in Milestone 4 files. One unchanged Milestone 1.7 sanitized terminal-response fragment produced a conservative 12-token warning; it is not an exact question, predates this scope, and was manually resolved without changing the frozen evaluator. Secrets, raw/model/cache tracking, sealed-path changes, evaluator changes, candidate/replay/audit inconsistencies, and files at or above 1 MB are all zero. Largest review file is `docs/DEVLOG.md` at 174,528 bytes before this entry.
- **Files changed:** `docs/DEVLOG.md` only for this checkpoint; verification itself was read-only and Ruff made no changes.
- **Errors or uncertainty:** The first safety script omitted a required normalizer argument and stopped; the corrected call then surfaced the pre-existing 12-token warning described above. The final refined scan distinguishes zero exact matches across the whole tree from zero conservative-window matches in this milestone and passes both required boundaries. Expected Windows LF-to-CRLF warnings are non-fatal.
- **Next action:** Fetch `origin/main` without merging, confirm the published tip remains `e99be66304eeeede98d1787cf9b1edb049a4a057`, stage only the reviewed Milestone 4 files, repeat staged scope/whitespace/content/secret/size checks, then commit and push without force if remote history is unchanged.

## 2026-07-18 17:46:02 -04:00 — Milestone 4 Step 14: remote and staged scope verified

- **Current step:** Confirm published history is stable and freeze the exact reviewed snapshot for one atomic commit.
- **Action performed:** Fetched `origin/main` without merging, verified local/remote tips and divergence, staged the 29 approved Milestone 4 paths explicitly, reviewed staged names/statistics and unstaged state, and scanned every staged blob for whitespace, secrets, benchmark windows, disallowed paths, model extensions, and size.
- **Result:** Local and `origin/main` both remain `e99be66304eeeede98d1787cf9b1edb049a4a057` with 0 ahead/0 behind. The staged snapshot contains 29 files: two synthesis configs, two content-free summaries, synthesis implementation/tests, and required documentation. It contains zero raw/model/cache/sealed/evaluator paths, zero secrets, zero exact or 12-token development-question matches, and no blob above 176,728 bytes. There were no unstaged project changes before adding this checkpoint.
- **Errors or uncertainty:** No remote-history drift, authentication issue, staged-scope defect, or content leak occurred. Expected LF-to-CRLF warnings are non-fatal.
- **Next action:** Restage this final DEVLOG checkpoint, repeat staged whitespace/scope/status checks, create one atomic Milestone 4 commit, push it without force, and verify synchronized clean local/remote state.

## 2026-07-18 17:46:48 -04:00 — Milestone 4 Step 14: local atomic commit created

- **Current step:** Create the single local milestone commit without changing persistent Git configuration.
- **Action performed:** Attempted the reviewed commit once, observed that this PC has no configured Git author identity, read the prior published commit's author/committer identity, and supplied that same identity through environment variables only to the commit process. Added this checkpoint before publication and amended the still-unpublished local commit so the milestone remains one atomic commit.
- **Result:** The 29-file Milestone 4 snapshot committed successfully. Persistent local/global Git identity configuration remains absent and unchanged; no extra commit, raw artifact, or project-file change was introduced outside the recorded milestone scope.
- **Errors or uncertainty:** The first commit command failed with `Author identity unknown` and created no commit. The session-local retry succeeded. The pre-publication amend changes only the final DEVLOG checkpoint, so the final commit hash must be read after the amend rather than using the temporary pre-amend hash.
- **Next action:** Re-run post-commit status/scope/tests, fetch `origin/main` again without merging, push the one final commit only if the remote is unchanged, and verify synchronized clean state.

## 2026-07-18 18:10:49 -04:00 — Milestone 4.1 Step 1: frozen starting state verified

- **Current step:** Verify the exact published Milestone 4 state and local evidence before changing generator rendering or validation.
- **Action performed:** Temporarily exposed Git in the current shell, fetched `origin/main` without merging, verified root/branch/tips/divergence/worktree, checked the ignored 120-record smoke artifact, recomputed the frozen taxonomy/synthesis/config hashes, loaded the pinned semantic configuration, and reviewed evaluator/sealed-path status without reading sealed contents.
- **Reason:** The blocker-resolution smoke must preserve the exact evaluator, categories, contamination thresholds, semantic artifact, and published history while changing only the approved generator rendering, typing, quality, and difficulty defects.
- **Result:** Repository is `C:/Users/Admin/Projects/Foundry` on clean `main`; local and `origin/main` both equal `c023ee068fe88400fbf378ade8c8e8e9fdcebfee` with 0/0 divergence. All 120 Milestone 4 attempts remain ignored and readable. Semantic model/revision remain MiniLM `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, `trust_remote_code=False`, thresholds 0.75/0.82. Frozen taxonomy, synthesis, and config hashes remain `021837…`, `910bf…`, and `7c087…` respectively.
- **Files changed:** `docs/DEVLOG.md` only.
- **Errors or uncertainty:** No state drift, missing evidence, threshold change, evaluator change, or containment error. Sealed-final contents were not opened, hashed, inspected, or compared.
- **Next action:** Analyze the five invalid acceptances and all 96 rejection records by template, vocabulary, structure, category, and semantic-match scope, then freeze content-free root causes before implementation.

## 2026-07-18 18:12:30 -04:00 — Milestone 4.1 Step 2: previous smoke root causes frozen

- **Current step:** Diagnose every Milestone 4 invalid acceptance and rejection cluster using only ignored generated records.
- **Action performed:** Reviewed the five audited invalid acceptances; grouped all 25 numeric-template, 50 five-token, seven automatic-semantic, and 14 manual-semantic rejections by category, mode, renderer variant, match scope, scenario prefix, question ending, and similarity; and recomputed value-neutral structure-signature uniqueness.
- **Reason:** Repairs must target measured generator defects rather than weakening the frozen scanner or adding cosmetic name/number substitutions.
- **Result:** All 96 rejections matched earlier generated candidates; none matched development content. Discrete produced all 25 exact number-neutral template copies because each of four modes had one sentence skeleton. Five-token rejects were 34 bookkeeping, 12 rate/ratio, and four discrete, averaging 0.594 Jaccard. Semantic automatic rejects were seven bookkeeping examples (mean 0.924); manual rejects were eight bookkeeping and six rate examples (mean 0.782). Bookkeeping reused ten scenario prefixes and six question endings across 53 attempts; rate reused eight scenarios across 34; discrete reused eight scenarios across 33. Latent hashes appeared 120/120 unique only because the monotonically increasing template variant was included, not because mathematical/rendering structures were genuinely distinct. The five invalid acceptances were four heterogeneous-object/unclear-ledger bookkeeping renderings and one discrete rendering with singular/plural disagreement and tied capacity constraints.
- **Files changed:** `docs/DEVLOG.md` only; all detailed generated text remained ignored.
- **Errors or uncertainty:** One read-only diagnostic script stopped after reporting most aggregates because it attempted to read a computed property absent from serialized raw records; the corrected hash calculation used the recorded structure signature. No evidence changed. Semantic similarity remains an imperfect proxy, but every review/reject match was confirmed to be generated-to-generated repetition.
- **Next action:** Add typed object/unit/location/operation contracts and deterministic renderer-quality metadata, then prove through tests that all five sanitized prior defects fail explicit rules before expanding template pools.

## 2026-07-18 18:23:18 -04:00 — Milestone 4.1 Steps 3–8: typed rendering and diversity repairs implemented

- **Current step:** Implement the approved object/unit safety, controlled rendering diversity, bookkeeping invariants, discrete grammar/difficulty rules, and deterministic quality validation.
- **Action performed:** Added typed object, unit, location, quantity, and ledger-operation contracts; rebuilt bookkeeping around a single compatible ledger; expanded bookkeeping to two mathematical families with eight renderers and 24 scenarios; expanded the unchanged rate mathematics to five families with six renderers and 20 scenarios; expanded discrete generation to four families with six renderers and 20 scenarios; made dual-capacity constraints untied; defined easy/medium/hard search-space ranges of 9–35, 36–80, and 81–200; and inserted a rule-based quality stage before deduplication and semantic screening.
- **Reason:** Milestone 4 proved that exact arithmetic alone did not guarantee valid language or useful diversity. Training candidates need compatible objects and units, explicit targets, grammatical renderings, meaningful difficulty, and varied sentence/scenario structure without relaxing any contamination gate.
- **Important commands run:** focused `pytest` over object/unit, quality, template-diversity, generator, and integration tests; direct deterministic generation/quality sweeps across 53 bookkeeping, 34 rate, and 33 discrete variants; numeric-template and structural-hash coverage checks.
- **Result:** All 39 focused tests pass. At the intended per-category smoke scale, every family produced distinct number-neutral rendered hashes (53/53 bookkeeping, 34/34 rate, 33/33 discrete) and all generated renderings passed the new quality rules. Existing primary/independent verifier agreement remains intact. The four sanitized bookkeeping regressions are rejected for incompatible object/unit combinations; the sanitized discrete regression is rejected for singular/plural mismatch and tied constraints.
- **Files changed:** synthesis object/unit and quality modules; all three approved generators; pipeline quality integration; generator/object/unit/quality/diversity tests; one existing grouping test updated to the new deterministic family mapping; and this DEVLOG.
- **Errors or uncertainty:** An initial focused test run exposed one stale grouping-variant assertion and eager evaluation in a discrete search-space expression; both were corrected without changing any frozen threshold or verifier. Ruff currently reports line-length cleanup still required in hand-authored template literals; functional tests pass, and formatting/lint cleanup will occur before the bounded smoke.
- **Next action:** Freeze a new master seed and raw/summary paths in the fresh 120-attempt configuration, verify the exact 60/60 curriculum and 12/12 output-contract allocation, then exercise the unchanged contamination and semantic pipeline.

## 2026-07-18 18:27:29 -04:00 — Milestone 4.1 Steps 9–11: fresh bounded smoke completed

- **Current step:** Process the one fresh 120-attempt smoke under the frozen lexical and semantic contamination policy.
- **Action performed:** Froze master seed `foundry-phase1-procedural-smoke-master-v2-rendering-diversity`, preserved the 60 targeted/60 generic and 12/12 output-track allocation, manually pre-resolved nine 0.75–0.82 generated-to-generated review-band matches as conservative rejections, and ran the counted smoke once with a required pause after 60 attempts.
- **Reason:** The repaired implementation must be judged on fresh fixed candidates without replacements, post-result threshold changes, benchmark-derived generation, or another model.
- **Important command run:** `.venv\Scripts\python.exe -m foundry.synthesis.cli --config configs/synthesis/gsm1k_phase1_smoke_v2.yaml --manual-decisions results/raw/synthesis_smoke/procedural_v2/manual_semantic_decisions.json --pause-at 60`.
- **Result:** At 60 attempts, 52 were accepted and eight rejected. Final count was exactly 120 attempts, 86 accepted and 34 rejected. Targeted accepted 52/60; generic accepted 34/60. By family: bookkeeping 30/53, rate/ratio 29/34, discrete 27/33. Rejections were 19 automatic semantic, nine manual semantic, and six five-token overlaps; exact, number-neutral, and latent-structure rejections were zero. All verifiers agreed, all 24 output contracts validated, and there were zero type/quality-stage rejections, ambiguity failures, generator exceptions, unresolved contamination cases, or GPU use.
- **Files changed:** fresh ignored raw attempts and manual semantic decisions; tracked content-free `results/synthesis_smoke/procedural_v2_summary.json`; fresh smoke config; this DEVLOG.
- **Errors or uncertainty:** The scanner emitted the existing non-fatal Windows Hugging Face symlink-cache warning. Only the 904 manifest-selected development questions were returned to the scanner; benchmark answers and sealed-final examples were not returned, compared, logged, or used as generator inputs. The 71.67% yield is four candidates below the immutable 90/120 gate.
- **Next action:** Replay the exact 120 decisions, then manually inspect every candidate and finalize the unchanged readiness decision.

## 2026-07-18 18:31:02 -04:00 — Milestone 4.1 Steps 12–14: replay and complete audit finalized

- **Current step:** Prove deterministic replay and manually audit every fresh candidate before applying the readiness gate.
- **Action performed:** Replayed the same plan/seed/semantic decisions without counting new attempts; reviewed all 120 questions, latent programs, traces, canonical labels, verifier evidence, output-contract status, contamination decisions, and accepted/rejected outcomes; then wrote detailed audit records only to the ignored raw directory and finalized the tracked aggregate.
- **Reason:** Exact labels do not guarantee faithful or grammatical training text, and the gate requires deterministic reconstruction plus zero invalid acceptances.
- **Result:** Replay matched decision SHA-256 `84bd6c622b30034a5932a4098c166b8710e39bbf4756e74b1c7c51cf54ce84a3` and aggregate SHA-256 `0e2e20a3516beacb651dfafea96be9b3e95760fbede8804ae6bea76eb6657ed6`. Audit found zero false labels, incorrect rejections, verifier disagreements, unresolved contamination cases, or overlooked benchmark resemblance. It found 11 invalid accepted renderings: residual attributive-plural/grouping grammar, a duplicated weighted-average group, a weighted-average conclusion that asked for a count instead of the computed mean, a rate rendering without an explicit interval denominator, a discrete sentence with an elided object noun, an awkward plural capacity phrase, and the irregular plural `shelfs`. The gate failed because 86 is below 90, invalid acceptances are nonzero, and systematic renderer weaknesses remain.
- **Files changed:** ignored replay proof/records and 120-row manual audit; tracked aggregate summary; this DEVLOG.
- **Errors or uncertainty:** Arithmetic and rejection decisions are fully audited, but the quality validator did not catch the 11 language/target defects, so the committed implementation is evidence of a failed blocker-resolution attempt rather than a generator approved for full production. No post-result rule or template change was made, and no second smoke will be run automatically.
- **Next action:** Document the negative result and exact remaining blockers, run the complete verification and containment suite, then create and push one atomic Milestone 4.1 commit without beginning full generation or training.

## 2026-07-18 18:34:03 -04:00 — Milestone 4.1 Step 16: full verification and safety review passed

- **Current step:** Verify the tested negative result, repository containment, and exact publication scope before creating the atomic commit.
- **Action performed:** Ran Ruff format/check, strict Mypy, all unit/integration tests, `pip check`, `git diff --check`, secret scanning, exact and 12-token development-question leak scans, frozen evaluator/semantic-config and sealed-path diffs, raw/model/cache tracking checks, count/output-contract/agreement/contamination consistency checks, template-diversity statistics, the five-defect regression suite, deterministic replay proof validation, changed/tracked file-size review, and repository status review.
- **Reason:** A failed readiness gate is publishable only when the implementation, negative evidence, containment boundary, and stop decision are reproducible and safe.
- **Result:** Ruff passes; strict Mypy reports no issues in 33 source files; 176 unit and two integration tests pass; dependencies and whitespace are clean. The 904-development-question scan found zero exact and zero 12-token matches in added/changed material. Secrets, frozen evaluator/config changes, sealed-path changes, raw/model/cache tracking, files at or above 1 MiB, candidate count inconsistencies, verifier disagreements, accepted duplicates, unresolved contamination, and replay mismatches are all zero. The dedicated 30-test type/quality/diversity/discrete regression suite passes. Expected Windows LF-to-CRLF notices and the existing Hugging Face symlink-cache warning are non-fatal.
- **Files changed:** `docs/DEVLOG.md` only for this checkpoint; verification was otherwise read-only and Ruff left all files formatted.
- **Errors or uncertainty:** The readiness gate remains failed by design: 86/120 acceptance, 11 invalid accepted renderings, and four documented systematic quality weaknesses. Passing repository checks does not convert the generator into a production-ready dataset source.
- **Next action:** Fetch `origin/main` without merging, confirm it still equals `c023ee068fe88400fbf378ade8c8e8e9fdcebfee`, stage only the reviewed Milestone 4.1 files, repeat staged containment checks, commit once, push without force, and verify clean 0/0 synchronization.

## 2026-07-18 18:34:35 -04:00 — Milestone 4.1 Step 16: remote and staged publication scope verified

- **Current step:** Confirm stable remote history and freeze the exact reviewed snapshot for one atomic commit.
- **Action performed:** Fetched `origin/main` without merging, verified local/remote tips and divergence, staged the 21 approved Milestone 4.1 paths explicitly, and reviewed staged names, statistics, whitespace, ignored-path boundaries, secrets, content leaks, model/cache patterns, and file sizes.
- **Result:** Local and `origin/main` both remain `c023ee068fe88400fbf378ade8c8e8e9fdcebfee` with 0 behind/0 ahead. The staged snapshot contains typed synthesis code, three generator revisions, deterministic quality integration, tests, fresh config, one content-free aggregate summary, and required documentation. It contains no raw candidates, benchmark text/answers, sealed/evaluator changes, model/cache files, embedding weights, secrets, or file at or above 1 MiB. No unstaged project change remains apart from this final DEVLOG checkpoint.
- **Errors or uncertainty:** No remote drift, authentication problem, merge/rebase requirement, or staged containment defect occurred. Expected Windows line-ending notices remain non-fatal.
- **Next action:** Restage this checkpoint, repeat staged whitespace/status review, create one atomic Milestone 4.1 commit using the existing published identity only for the commit process if necessary, push without force, and verify synchronized clean state.

## 2026-07-18 18:42:13 -04:00 — Milestone 4.2 Step 1: synchronized frozen starting state verified

- **Current step:** Verify published history, ignored Milestone 4.1 evidence, and every frozen evaluator/contamination boundary before changing the renderer architecture.
- **Action performed:** Temporarily exposed Git in the current shell; checked repository root, branch, local and stored remote references, live GitHub `main`, divergence, worktree, previous raw-attempt availability, frozen evaluator/config paths, and the pinned local semantic artifact without opening sealed-final content.
- **Reason:** The final procedural-lineage attempt may replace only natural-language realization; generator/verifier mathematics, curriculum, evaluator, semantic model, thresholds, and remote history must remain fixed.
- **Result:** Repository is `C:/Users/Admin/Projects/Foundry` on clean `main`; local, `origin/main`, and live remote all equal `39ea07cdb3d09e2811f8aa871ec3378bebc743f8` with 0/0 divergence. The ignored Milestone 4.1 attempt file contains 120 rows. MiniLM remains pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, `trust_remote_code=False`, CPU-local snapshot available, with immutable manual/reject thresholds 0.75/0.82. No frozen evaluator or sealed path is modified.
- **Files changed:** `docs/DEVLOG.md` only.
- **Errors or uncertainty:** The initial PowerShell probe attempted unavailable `ConvertFrom-Yaml`; a read-only Python config load immediately verified the same fields. The requested starting hash is exact, but its published subject is `synth: test generator rendering blocker fixes`, rather than the descriptive subject quoted in the prompt. Neither issue changes repository state.
- **Next action:** Review the eleven ignored invalid accepted records, map each failure from semantic information through rendering loss to an architectural invariant, and add content-free taxonomy plus sanitized regression fixtures.

## 2026-07-18 19:01:00 -04:00 — Milestone 4.2 Step 2: eleven rendering defects classified

- **Current step:** Freeze a content-free taxonomy for every invalid acceptance from the Milestone 4.1 audit.
- **Action performed:** Re-read only the eleven ignored generated records and traced each failure from available semantic metadata through its rendered surface and the prior validator.
- **Reason:** The final procedural repair must address lost semantic and grammatical information architecturally rather than matching the observed strings.
- **Result:** Four records involved attributive-plural or grouping-noun morphology; one repeated a weighted-average semantic group; one paired a weighted-mean computation with a count question; one omitted the interval denominator from a rate statement; one elided an object noun without a licensed antecedent; one used an incompatible plural-capacity construction; and one guessed the irregular plural `shelfs`. The remaining grouping construction belongs to the same explicit-role morphology class. Existing metadata retained most entity/value information but lacked typed grammatical roles, target kinds, semantic-node accounting, explicit denominator evidence, and an irregular lexicon, so its boolean `grammar_complete` flag could not detect the defects.
- **Files changed:** `docs/DEVLOG.md`; content-bearing diagnostics remained ignored.
- **Errors or uncertainty:** The taxonomy is complete for all eleven audited records. It does not claim to enumerate every possible English defect; the new compiler therefore enforces general invariants for the affected information classes.
- **Next action:** Implement semantic/sentence IR, explicit morphology, typed targets, semantic-node coverage, and a centralized compiler, then encode all eleven defect classes as sanitized regressions.

## 2026-07-18 19:19:00 -04:00 — Milestone 4.2 Steps 3–7: typed realization compiler implemented

- **Current step:** Replace generator-owned final prose with typed semantic IR and a centralized deterministic English realization layer.
- **Action performed:** Added typed problem, entity, lexeme, quantity, unit, rate, group, target, clause-plan, render-signature, coverage, morphology-evidence, and compiled-realization contracts; added explicit irregular noun and verb morphology; implemented one compiler for bookkeeping, rate/ratio, weighted-mean, and discrete frames; added target-compatible question selection, explicit rate denominators, unique weighted-group validation, and exact semantic-node coverage; and wired all three mathematical generators to emit IR and consume compiled prose.
- **Reason:** Grammar, target, and coverage correctness must be consequences of typed contracts rather than post-hoc regexes. Mathematics and verifier implementations remain separate and unchanged.
- **Result:** Deterministic generation sweeps over 100 variants per family produced 300 compiled candidates with zero typed-realization failures or generator exceptions. Render hashes now include semantic-IR and meaningful render-signature hashes. All complete questions used by the pipeline come from the compiler.
- **Files changed:** `src/foundry/synthesis/realization/`, generator contracts and the three approved generators, pipeline integration, and this DEVLOG.
- **Errors or uncertainty:** The first bookkeeping integration passed the obsolete pre-compiler string into the draft; the draft invariant rejected it immediately and the call was corrected to use compiler output. Some legacy rendering helpers remain temporarily present but are no longer on the candidate path; they will be removed or proven unreachable before final verification.
- **Next action:** Add regression and contract tests, remove obsolete generator prose paths, then run the bounded renderer stress test and its 60-render manual sample.

## 2026-07-18 19:31:00 -04:00 — Milestone 4.2 Step 8: renderer stress gate failed

- **Current step:** Exercise typed rendering, morphology, target consistency, semantic coverage, internal diversity, and generated-to-generated semantic collision behavior before any counted full-pipeline smoke.
- **Action performed:** Constructed exactly 300 deterministic in-memory renders per approved family (900 total), validated every typed realization, encoded only generated questions with the pinned local MiniLM artifact, computed content-free collision statistics, discarded the in-memory corpus, and audited a deterministic 20-render sample per family by reconstructing the same seeds.
- **Reason:** The stress gate must prove both correctness and plausible scale capacity before another 120-candidate contamination run is allowed.
- **Important command run:** `run_renderer_stress(Path.cwd())` using master seed `foundry-m4.2-typed-renderer-stress-v1`; deterministic sample selection over the same candidate IDs for the required 60-render audit.
- **Result:** All 900 renders passed typed validation: zero morphology, target-type, missing-node, duplicated-node, or grammar-metadata failures; zero exact or latent-structure duplicates; 900 distinct render signatures; and all eleven prior sanitized defect classes were rejected by tests. However, there were 99 number-neutral template collisions. Generated-to-generated nearest-neighbor similarity was at least 0.82 for 899/900 renders (median 0.936379; maximum 0.996438). The 60-render audit found zero false mathematical labels but 13 unnatural request clauses caused by imperatives such as `Determine`, `Find`, `Calculate`, `Report`, or `Identify` being punctuated as direct questions; two also used awkward hyphenated count compounds. This is a systematic realization defect, not an isolated observed-string issue.
- **Files changed:** tracked content-free `results/synthesis_smoke/renderer_v3_stress_summary.json`, realization compiler/contracts/stress code, sanitized tests, and this DEVLOG. No question corpus was persisted.
- **Errors or uncertainty:** MiniLM similarity is sensitive to unavoidable mathematical language, but 899 automatic-rejection-band neighbors plus 99 number-neutral collisions do not demonstrate the required material scale improvement. The manual sample independently confirms a systematic surface defect. Therefore the stress gate is failed regardless of semantic-metric interpretation.
- **Next action:** Do not run the fresh 120-candidate smoke and do not create a Milestone 4.3. Freeze the procedural-lineage stop decision, document the constrained local-model realization pivot, run repository verification, then commit and push this negative milestone result.

## 2026-07-18 19:47:00 -04:00 — Milestone 4.2 Steps 14–15: documentation and verification passed

- **Current step:** Verify the typed-compiler implementation, negative stress evidence, containment boundaries, and procedural-lineage stop decision before publication.
- **Action performed:** Updated the project plan, decisions, learning notes, synthesis design, experiment registry, and live log; ran Ruff format/lint, strict Mypy, all tests, dependency integrity, whitespace, secret, development-content leak, raw/cache/model tracking, frozen-evaluator/sealed-path status, regression, stress-summary consistency, size, and repository-scope checks.
- **Reason:** A failed readiness gate must still leave reproducible code/evidence, accurate stop documentation, and zero sensitive or benchmark-derived tracked material.
- **Important commands run:** `ruff format src tests`; `ruff check src tests`; `mypy --strict src`; `pytest -q`; focused 20-test realization/regression suite; `python -m pip check`; `git diff --check`; offline 904-development-question exact and 12-token leak scan; secret-pattern scan; `git ls-files` raw/model/cache checks; frozen evaluator and sealed-path status; content-free stress-summary assertions; tracked/candidate file-size review.
- **Result:** Ruff passes; strict Mypy reports no issues in 40 source files; all 198 tests pass; the focused 20 realization tests pass; dependencies and whitespace are clean. The development leak scan found zero exact and zero 12-token matches across all 904 approved development questions. Secret hits, tracked raw files, tracked model/cache artifacts, frozen evaluator changes, sealed-path changes, and candidate files at or above 1 MiB are all zero. The content-free stress summary is internally consistent and confirms that no fresh 120-candidate smoke ran.
- **Files changed:** typed realization source and generator/pipeline integration; realization and updated compatibility tests; one 2,159-byte content-free summary; the six required project records. New realization source totals 52,856 bytes and new tests total 7,381 bytes. No dependency file changed; the existing ignored semantic artifact remains 91,578,751 bytes.
- **Errors or uncertainty:** The first leak-scan script omitted the required `replace_numbers` argument and stopped with a `TypeError`; the corrected read-only scan returned the zero-match result. Peak RAM was not instrumented during the four-second stress run, so no current peak-RSS measurement is claimed; GPU use was zero and no dependency changed. This measurement omission does not alter the failed gate, but it is a reporting limitation.
- **Next action:** Fetch without merging, verify `origin/main` is still `39ea07c…`, stage only reviewed Milestone 4.2 files, repeat staged containment checks, create one atomic commit, push without force, then verify clean 0/0 synchronization.

## 2026-07-18 19:50:00 -04:00 — Milestone 4.2 Step 15: remote and publication scope verified

- **Current step:** Freeze the exact reviewed negative-result snapshot and confirm publishing cannot overwrite remote work or leak prohibited artifacts.
- **Action performed:** Fetched `origin/main` without merging; confirmed local/remote tips and 0/0 divergence; explicitly staged the 23 approved source, test, content-free result, and documentation paths; reviewed staged names, statistics, whitespace, raw/model/cache/evaluator/sealed boundaries, and unstaged state.
- **Result:** Local and `origin/main` both remain `39ea07cdb3d09e2811f8aa871ec3378bebc743f8`. The staged snapshot contains 23 files with 2,162 insertions and 115 deletions before this final checkpoint. It contains zero raw, model, cache, embedding-weight, frozen-evaluator, or sealed-final paths, and no unstaged tracked project change remains.
- **Errors or uncertainty:** No remote drift, authentication issue, merge/rebase requirement, staged whitespace failure, or scope defect occurred. Expected Windows LF-to-CRLF notices are non-fatal.
- **Next action:** Restage this checkpoint, repeat staged whitespace/status review, create one atomic Milestone 4.2 commit with the published identity supplied process-locally if required, push without force, and verify clean synchronized local/remote state.

## 2026-07-18 20:12:00 -04:00 — Milestone 5A Step 1: synchronized design starting state verified

- **Current step:** Verify the exact published typed-realizer result and every frozen boundary before designing the local-model realization pivot.
- **Action performed:** Temporarily exposed Git in the current shell; checked repository root, branch, local/stored/live remote tips, divergence, worktree, frozen evaluator/synthesis path status, tracked raw/model/cache paths, and selected non-sealed configuration hashes without opening sealed-final content.
- **Reason:** This milestone may design interfaces and policy only. It must begin from the published negative result and cannot alter the benchmark evaluator, synthesis mathematics, model caches, or sealed partition.
- **Result:** Repository is `C:/Users/Admin/Projects/Foundry` on clean `main`; local, stored `origin/main`, and live GitHub `main` all equal `ca835716c6cf456699a99f82cc9e09be800de828` with 0/0 divergence. Frozen path status, sealed-path status, tracked raw artifacts, and tracked model/cache artifacts are all zero. The published subject is `synth: test typed realization compiler`, not the descriptive subject quoted in the approval, but the required full hash is exact.
- **Files changed:** `docs/DEVLOG.md` only after verification completed.
- **Errors or uncertainty:** No state drift, remote change, benchmark fixture, sealed access, or containment error occurred. Git status verification does not read sealed-final contents.
- **Next action:** Inspect official metadata/model cards for exactly the three approved candidate realization models, distinguish documented facts from local compute estimates, and recommend one primary plus one fallback without downloading weights.

## 2026-07-18 20:27:00 -04:00 — Milestone 5A Step 2: three-model comparison completed

- **Current step:** Compare the three approved local surface-realization candidates and freeze one primary plus one fallback without downloading model weights.
- **Action performed:** Inspected official Hugging Face model-card and repository metadata for the exact approved candidates; resolved immutable revisions; measured published repository and weight-byte metadata; inspected only lightweight configuration, tokenizer, chat-template, and generation-configuration files in memory; and compared license, parameter count, standard-Transformers support, non-thinking controls, dependency requirements, expected RTX 3080 fit, and reproducibility risk.
- **Reason:** The pivot needs a model narrow enough for local, deterministic wording generation while preserving structured placeholders. Choosing it before implementation prevents later model-driven changes to the interface or quality gates.
- **Important commands run:** read-only Hugging Face Hub metadata calls for `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen3-1.7B`, and `HuggingFaceTB/SmolLM3-3B`; in-memory HTTP reads of their pinned `config.json`, tokenizer/chat-template metadata, and `generation_config.json`.
- **Result:** Primary recommendation is `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (Apache-2.0): documented hard non-thinking mode, standard Qwen3 Transformers support, and an estimated 4.5–5.5 GiB FP16/BF16 inference footprint make it the strongest balance of instruction following, naturalness, structure preservation, and 10 GiB fit. Fallback is the already cached and locally proven `Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (Apache-2.0), which works with the existing Transformers 4.46.3 stack. Qwen3 would require a separately approved pinned Transformers upgrade to at least 4.51; SmolLM3 requires at least 4.53, has the largest published weight payload, and offers no project-specific advantage large enough to offset that added footprint. No candidate documents a strict JSON guarantee, so JSON and placeholder fidelity remain deterministic admission checks rather than assumed model behavior.
- **Files changed:** `docs/DEVLOG.md` only. No dependency, model, cache, benchmark, evaluator, or synthesis-mathematics file changed.
- **Errors or uncertainty:** Official metadata calls report Qwen3 as 1.7B/1.4B non-embedding in the card while repository tensor metadata totals about 2.03B parameters; both will be reported rather than conflated. Windows execution is proven for Qwen2.5 in this project but only inferred from standard PyTorch/Transformers compatibility for Qwen3 and SmolLM3. VRAM and speed values are planning estimates, not measurements. SmolLM3 stores its chat template in a separate pinned Jinja artifact rather than the tokenizer JSON. No weights were downloaded.
- **Next action:** Define the model’s wording-only responsibility, strict input/output schemas, placeholder and semantic-node invariants, deterministic filling compiler, round-trip validation layers, and separate benchmark-contamination versus internal-diversity policies.

## 2026-07-18 20:46:00 -04:00 — Milestone 5A Steps 3–4: constrained realization interface implemented

- **Current step:** Formalize the local model’s wording-only responsibility and make its slot-preserving interface executable without adding a model runtime.
- **Action performed:** Added frozen dataclasses for immutable placeholder types, ordered semantic events, bounded style controls, value-blind realization requests, strict clause-to-node maps, model responses with no answer field, and compiler-filled realizations. Added a versioned system prompt and exact hash. Implemented strict JSON parsing with exact field sets, placeholder inventory and occurrence equality, no raw numeric literals, target/style equality, per-clause semantic-node accounting, required rate-interval preservation, deterministic post-validation filling, and stable response/template/replacement hashes.
- **Reason:** The model may improve English only if it cannot see or control values or labels and if deterministic code can prove that the proposed wording still represents the procedural IR before values are inserted.
- **Important commands run:** focused Ruff format/lint; 20 unit tests under `tests/unit/synthesis/realization`; strict Mypy over the realization package.
- **Result:** The strict parser and contract tests pass. Sanitized fixtures prove rejection of unknown fields, missing/extra/duplicated placeholders, invented numeric literals, invented/omitted/duplicated semantic nodes, clause-to-slot mismatches, target/style changes, missing rate intervals, malformed replacement inventories, and answer-bearing output. The deterministic compiler fills the exact slot set only after validation. The initial focused Mypy run found two redundant boolean casts; they were removed without changing behavior.
- **Files changed:** new `model_contracts.py`, `validation.py`, and `policy.py`; realization package exports; `configs/synthesis/local_realization_design.yaml`; original unit fixtures in `tests/unit/synthesis/realization/`; this DEVLOG.
- **Errors or uncertainty:** Structural validation cannot by itself prove that arbitrary English is natural, so the contract deliberately keeps deterministic morphology/grammar checks and mandatory human audit downstream. The configuration remains `design_only: true`; it contains no loader, pipeline command, model inference, retry loop, or dataset-writing path.
- **Next action:** Freeze all round-trip layers and the bounded human-audit protocol, then separate generated-to-benchmark contamination screening from a future independently calibrated generated-to-generated diversity policy.

## 2026-07-18 21:02:00 -04:00 — Milestone 5A Steps 5–6: round-trip and naturalness validation designed

- **Current step:** Define how a fluent model template is proved semantically faithful and how naturalness is judged without trusting an automatic grammar score.
- **Action performed:** Froze a 13-layer admission path: strict schema, exact slot equality, semantic-node coverage, target and intent equality, unit/denominator preservation, entity integrity, clause mapping/order, filled-question consistency, primary exact execution, independent verifier, output contract, deterministic language checks, and contamination/diversity screening. Defined a complete human audit in which the first semantic-preservation and naturalness decisions hide the canonical answer and verifier outcomes; uncertain cases reject. An optional reverse-model parse is limited to an additional rejection signal and can never accept an example or break a verifier tie.
- **Reason:** Natural wording is useful only when every mathematical fact remains under procedural control. Separate answer-blind semantic review reduces confirmation bias, while dual execution preserves trustworthy labels.
- **Result:** The design explicitly maps each required validation layer to a deterministic rejection boundary and retains the existing morphology, reference, coverage, verifier, and output-contract checks. Every future beam candidate—not only the selected candidate—must be audited, with wording-bearing records ignored and only content-free categories/hashes committed.
- **Files changed:** `docs/LOCAL_MODEL_REALIZATION_DESIGN.md`, design-only realization contracts/tests, and this DEVLOG.
- **Errors or uncertainty:** No deterministic grammar system proves naturalness, and one human audit is still subjective. The design responds conservatively: uncertainty cannot pass, and no LLM judge is permitted. No implementation or model output was used to validate the proposed audit.
- **Next action:** Analyze the scientific mismatch between benchmark-contamination similarity and internal curriculum diversity, select exactly one future policy, and freeze its pre-generation calibration boundary.

## 2026-07-18 21:08:00 -04:00 — Milestone 5A Step 7: semantic-similarity roles separated

- **Current step:** Decide whether one MiniLM threshold can validly serve both benchmark-contamination safety and internal curriculum diversity.
- **Action performed:** Compared the two scientific roles against Milestone 4.2's 899/900 generated-neighbor result and froze policy option 2: preserve the pinned MiniLM artifact and 0.75/0.82 thresholds for generated-to-development contamination, but use exact text, number-neutral template, latent-structure, five-token Jaccard 0.35, and a separately calibrated semantic policy for generated-to-generated diversity.
- **Reason:** Sentence embeddings measure broad semantic/topical proximity, not latent-program equivalence. Arithmetic problems in one family can be meaningfully different while remaining close in embedding space, whereas weakening the benchmark policy would create a real contamination risk.
- **Result:** `local_realization_design.yaml` leaves the internal semantic threshold deliberately unset and makes independent pre-smoke calibration mandatory. A future approved implementation must use original fixtures covering copies, number swaps, structural copies, paraphrases, same-skill/different-program pairs, related-but-distinct questions, and unrelated questions; it must freeze the resulting behavior and hash before viewing model-generated smoke outputs. If the fixtures do not support a safe separation, generation stops.
- **Files changed:** realization policy/config/tests, `docs/LOCAL_MODEL_REALIZATION_DESIGN.md`, and this DEVLOG. Existing semantic source/configuration and thresholds were not edited.
- **Errors or uncertainty:** Milestone 5A does not choose a new internal numeric threshold because doing so without an independent fixture experiment would be unsupported. This is an explicit unresolved implementation gate, not a silent weakening of duplicate controls.
- **Next action:** Freeze deterministic beam behavior, the 120-IR/360-candidate maximum smoke, unchanged readiness gates, compute/storage estimates, and the exact dependency/download decision required for implementation.

## 2026-07-18 21:24:00 -04:00 — Milestone 5A Steps 8–12: deterministic smoke and pivot design frozen

- **Current step:** Complete the future generation protocol, bounded smoke, compute estimates, typed design artifacts, and architectural records without executing the design.
- **Action performed:** Compared greedy, fixed beam, and seeded-sampling strategies; selected fixed three-beam deterministic search with thinking and sampling disabled, 256 maximum new tokens, stable beam ordering, a 90-second timeout, and no retries. Froze a 120-IR/360-candidate maximum future smoke with the existing targeted/generic/output-contract allocations and unchanged readiness gates. Added compute/storage estimates, one primary/fallback model decision, dependency boundary, strict interfaces, original tests, and the complete design document; updated project plan, decisions, learning notes, synthesis design, and experiment registry.
- **Reason:** Generation behavior, budgets, gates, and model/dependency choices must be fixed before model output exists, otherwise the implementation smoke could adapt to its own failures.
- **Result:** Qwen3 FP16 under ordinary Transformers is recommended; quantization and vLLM add complexity without solving the 10 GiB smoke constraint. Estimated model repository download is 4.08 GB decimal, inference peak 4.5–5.5 GiB VRAM, generation 15–45 minutes for 120 IRs, and complete audit 2–4 hours. Focused Ruff passes; strict Mypy passes all 43 source files; all 22 new realization design tests pass. The design configuration remains non-executable and pins zero-defect/90-clean/15-per-family gates.
- **Files changed:** `src/foundry/synthesis/realization/{model_contracts,validation,policy}.py`, package exports, `configs/synthesis/local_realization_design.yaml`, 22 new unit cases, `docs/LOCAL_MODEL_REALIZATION_DESIGN.md`, and the six required project records.
- **Errors or uncertainty:** Runtime, VRAM, throughput, placeholder fidelity, naturalness, yield, and deterministic replay are unmeasured estimates until a separately approved implementation smoke. Qwen3 requires a dedicated Transformers >=4.51 dependency lock and one weight download; neither occurred. The future internal semantic threshold remains intentionally unset pending pre-generation original-fixture calibration.
- **Next action:** Run the complete repository verification and safety suite, review the diff for scope and factual consistency, fetch without merging, then create and push one atomic design-only commit if every check passes.

## 2026-07-18 21:39:00 -04:00 — Milestone 5A Step 13: full verification and safety review passed

- **Current step:** Verify code quality, design consistency, benchmark containment, frozen artifacts, and publication scope before committing the architectural design.
- **Action performed:** Ran Ruff format check/lint, strict Mypy, all tests, `pip check`, `git diff --check`, high-confidence secret patterns, exact and 12-token scans against all 904 development questions, frozen evaluator/math/semantic/dependency diffs, sealed-path status, raw/model/cache tracking, Qwen3/SmolLM3 cache-existence checks, candidate/tracked size review, and repository status/diff review.
- **Reason:** A design commit must be executable enough to validate its contracts while proving that it contains no benchmark fixture, answer, generated data, model artifact, cache, secret, dependency change, evaluator change, or sealed content.
- **Result:** Ruff passes; strict Mypy reports no issues in 43 source files; all 221 tests pass; dependencies and whitespace are clean. Fourteen candidate files contain zero high-confidence secrets, zero exact development-question matches, and zero 12-token development-question matches. Frozen evaluator, generator mathematics, existing semantic artifact/configuration, and requirements diffs are zero; sealed-path status is zero; tracked raw/model/cache files are zero; no Qwen3 or SmolLM3 cache exists. The largest candidate is 222,718 bytes, below 1 MB. Config SHA-256 is `d6e6ca82681b702e07c71a9732a8c81159ea7a9bca78c73193228f72ca4ec3a5`.
- **Files changed:** this DEVLOG only for the checkpoint; verification was otherwise read-only and format check made no change.
- **Errors or uncertainty:** The first safety helper derived paths directly from porcelain status and therefore treated the new test directory as one directory, scanning 12 files rather than every file. The corrected helper combined `git diff --name-only` with `git ls-files --others --exclude-standard`, scanned all 14 individual files, and returned the same zero findings. The leak scanner read only the 904 manifest-selected development questions and emitted no content; it never opened sealed-final rows.
- **Next action:** Fetch `origin/main` without merging, confirm it still equals `ca835716…`, explicitly stage only the 14 reviewed files, repeat staged whitespace/secret/content/model/cache/size/scope checks, then create and push one atomic commit without force if remote history is unchanged.

## 2026-07-18 21:44:00 -04:00 — Milestone 5A Step 13: remote and staged scope verified

- **Current step:** Freeze the exact reviewed design snapshot and prove that publishing cannot overwrite remote work or leak prohibited artifacts.
- **Action performed:** Fetched `origin/main` without merging; verified local/remote tips and 0/0 divergence; explicitly staged the 14 approved files; ran staged whitespace, exact path scope, high-confidence secret, exact/12-token development-question, file-size, and design-config hash checks using index blobs.
- **Result:** Local and `origin/main` both remain `ca835716c6cf456699a99f82cc9e09be800de828`. The staged snapshot has 14 files, zero forbidden raw/model/cache/evaluator/sealed/dependency paths, zero secret hits, zero exact or 12-token development-question matches, no blob at or above 1 MB, and config SHA-256 `d6e6ca82681b702e07c71a9732a8c81159ea7a9bca78c73193228f72ca4ec3a5`. No unstaged project change existed before this checkpoint.
- **Errors or uncertainty:** No remote drift, authentication issue, merge/rebase requirement, staged whitespace error, or containment defect occurred. Expected Windows LF-to-CRLF notices are non-fatal and do not alter the staged bytes.
- **Next action:** Restage this checkpoint, repeat staged whitespace/status review, create one atomic `synth:` commit with the existing published identity supplied process-locally if necessary, push without force, and verify synchronized clean state.

## 2026-07-18 19:56:36 -04:00 — Milestone 5B Step 1: synchronized frozen starting state verified

- **Current step:** Verify published history, frozen evaluator/mathematics, design hashes, Python, ignore boundaries, and remote synchronization before policy calibration or model work.
- **Action performed:** Temporarily exposed Git in the current shell; checked repository root, branch, local/stored/live remote tips, divergence, worktree, latest commit scope, frozen evaluator and generator-math status, sealed-path status by name only, design and system-prompt hashes, CPython version/executable, and ignore rules for model/raw artifacts.
- **Reason:** The implementation smoke may change only the separately approved realization path. It must begin from the published design and cannot silently alter benchmark evaluation, procedural labels, or remote history.
- **Result:** Repository is `C:/Users/Admin/Projects/Foundry` on clean `main`; local, stored `origin/main`, and live GitHub `main` all equal `85179d7b8d3f9b48f0a36c92119c4f5049708ecd` with 0/0 divergence. Design SHA-256 is `d6e6ca82681b702e07c71a9732a8c81159ea7a9bca78c73193228f72ca4ec3a5`; prompt SHA-256 is `9d3e808d5c887d974919728d5afb51df9daa5760d467d3c08648799aeddcc393`. Evaluator, procedural generators/verifiers, and sealed-path status are unchanged. CPython is 3.12.10 at the approved per-user installation. `data/` and `results/raw/` are ignored.
- **Files changed:** `docs/DEVLOG.md` only after verification completed.
- **Errors or uncertainty:** The published subject is `synth: design constrained local realization`, not the descriptive `synthesis:` subject in the approval, but the required full hash is exact. `.venv-realization` is not currently covered by `.gitignore`; a narrow ignore entry is required before environment creation. No sealed-final file was opened.
- **Next action:** Create original content-free diversity fixtures, evaluate no more than three policies with the already pinned MiniLM artifact, freeze one policy and hashes before downloading or running Qwen3, and stop if no policy credibly separates duplicates from distinct same-skill questions.

## 2026-07-18 20:02:45 -04:00 — Milestone 5B Step 2: internal-diversity policy calibrated and frozen

- **Current step:** Separate true generated duplicates from legitimate same-skill variation before any Qwen3 output can influence the policy.
- **Action performed:** Authored 24 original benchmark-independent fixture pairs covering exact duplicates, number swaps, structural rewrites, close paraphrases, same-skill/distinct scenarios, related-but-distinct questions, unrelated questions, and two explicitly ambiguous parallel scenarios. Predeclared and evaluated exactly three policies using the existing local-only MiniLM revision on CPU, then froze `evidence-gated-balanced-v1` with exact/template/latent/five-token hard controls plus evidence-gated semantic review/rejection.
- **Reason:** The 0.75/0.82 benchmark firewall measures contamination risk, while internal curriculum diversity needs supporting structural evidence so topical arithmetic similarity does not become automatic rejection.
- **Important commands run:** focused Ruff/Mypy/tests; local `PinnedSentenceEncoder` load with `local_files_only=True`; one 48-text embedding pass shared across all three predeclared policies; typed policy/hash verification.
- **Result:** Fixture semantic SHA-256 is `e5ba09dc45c6afd58c2c6f9435a33756cb0bb5c20ab73d41313bc49e09c17b89`; calibration SHA-256 is `e855e29a953cbb6b0563e73def3ee4bceb3bbbf20a6f0c5dc7f18573d43063ab`; selected policy SHA-256 is `26c030e8497c4727e286ff3e89d4720cee1c2681a224b8a93b8c515ef521cc90`; frozen policy-file SHA-256 is `e734e1ac0f71e21365381e3f1b92391eae54c36ace0c05efec019368232ef355`. The selected policy rejected all 12 duplicate/paraphrase fixtures and passed all ten clearly distinct fixtures: zero duplicate escapes and zero distinct automatic rejections. The legacy policy auto-rejected two distinct related questions; the conservative policy failed to auto-reject one close paraphrase.
- **Files changed:** `diversity.py`; candidate/frozen policy YAML; 24 original fixtures; five unit tests; content-free aggregate calibration summary; this DEVLOG.
- **Errors or uncertainty:** Two deliberately ambiguous parallel scenarios were labelled manual review but passed automatically under all evidence-gated candidates because MiniLM scores were only 0.5930/0.6278 and lexical overlap was low. They were documented and excluded from threshold tuning rather than used to add a post-hoc signal. The future smoke's mandatory audit of every returned beam remains the conservative safeguard. Generated-to-development MiniLM revision, trust setting, and 0.75/0.82 thresholds are unchanged. No Qwen model was downloaded or run.
- **Next action:** Add the narrow `.venv-realization/` ignore rule, produce an exact dedicated dependency lock from the approved Python/Torch/Transformers constraints, create the environment, install only that lock, and verify CUDA and package integrity before model download.

## 2026-07-18 20:10:01 -04:00 — Milestone 5B Step 3: dedicated realization environment verified

- **Current step:** Isolate the approved Qwen3 runtime from the frozen evaluation environment and prove the pinned Windows/CUDA stack is compatible.
- **Action performed:** Added only `.venv-realization/` to ignore rules; generated an exact pip-compile lock from minimal pinned inputs; created the environment with CPython 3.12.10; installed PyTorch 2.5.1 CUDA 12.1, Transformers 4.51.3, tokenizers 0.21.4, PyYAML 6.0.2, and their exact transitive dependencies; installed this project editable without dependency resolution; ran `pip check` and CUDA/GPU probes.
- **Reason:** Model-runtime dependencies must not destabilize the existing evaluation `.venv`, and Qwen3 requires a newer Transformers/tokenizers pair than the frozen benchmark stack.
- **Important commands run:** `pip-compile --allow-unsafe ... requirements-realization.in`; `py -3.12 -m venv .venv-realization`; realization-environment `pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements-realization.lock.txt`; `pip install --no-deps -e .`; `pip check`; PyTorch CUDA probe; `nvidia-smi`.
- **Result:** Environment creation took 4.651 seconds, dependency installation 93.341 seconds, and editable project installation 4.455 seconds. Lock SHA-256 is `c89dbc0195ef21168e3cd9f73ddb8de7db80e0d79166f3d6fe130294f5bd56f5`. Runtime is Python 3.12.10, PyTorch `2.5.1+cu121`, CUDA runtime 12.1, Transformers 4.51.3, tokenizers 0.21.4, and PyYAML 6.0.2. CUDA is available; GPU is NVIDIA GeForce RTX 3080; `nvidia-smi` reports 10,240 MiB with 9,581 MiB free and driver 610.47. `pip check` passes. The ignored environment occupies 4,868,609,494 bytes.
- **Files changed:** `.gitignore`, `requirements-realization.in`, generated `requirements-realization.lock.txt`, ignored `.venv-realization/`, and this DEVLOG. The existing `.venv` and system packages were unchanged.
- **Errors or uncertainty:** The first verbose pip-compile invocation had to download the 2,449.3 MB CUDA wheel to resolve the exact local-version pin; it completed successfully and the install reused the local cache. Pip printed only the normal notice that a newer unapproved pip exists; no upgrade was performed. Torch reports 10,239 MiB by integer flooring its byte count while `nvidia-smi` reports the expected 10,240 MiB.
- **Next action:** Download only `Qwen/Qwen3-1.7B` at the full frozen revision into the ignored Hugging Face cache, verify every artifact and hash, measure download/load resources, then disable network loading and prove a fresh `local_files_only=True` reload.

## 2026-07-18 20:14:51 -04:00 — Milestone 5B Step 4: pinned Qwen3 download and offline reload verified

- **Current step:** Acquire only the approved local realization model, verify its immutable contents, and prove the runtime can reload it without network access.
- **Action performed:** Downloaded `Qwen/Qwen3-1.7B` at full revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` into the ignored Hugging Face cache; hashed/inventoried every snapshot file; then launched a fresh process with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `local_files_only=True`, `trust_remote_code=False`, safetensors, and FP16 to load tokenizer and model onto the RTX 3080.
- **Reason:** Revision drift, incomplete cache state, custom code, or an implicit network dependency would invalidate replay and the bounded local-compute claim.
- **Result:** Download took 146.704 seconds. The exact-revision snapshot contains 12 files totaling 4,079,450,110 bytes, including two safetensors shards totaling 4,063,515,592 bytes; the complete model cache also occupies 4,079,450,110 logical bytes. Config SHA-256 is `1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197`; tokenizer-config SHA-256 `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101`; tokenizer JSON `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`; generation config `2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2`; chat template `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`. Offline tokenizer load took 0.203 seconds and model-to-GPU load 2.133 seconds. The model has 1,720,574,976 parameters, FP16 on CUDA. Peak process working set was 7,403,778,048 bytes; peak allocated/reserved VRAM 3,441,689,088/3,451,912,192 bytes.
- **Files changed:** ignored `data/huggingface/hub/models--Qwen--Qwen3-1.7B/` and this DEVLOG. No fallback, second encoder, model server, or unrelated checkpoint was downloaded.
- **Errors or uncertainty:** Hugging Face emitted the known Windows no-symlink warning; this cache nevertheless occupies only the measured snapshot size. The first offline load successfully reached GPU but its post-load peak-RAM call failed because the Windows API wrapper lacked explicit handle/argument types. A tiny diagnostic fixed the telemetry wrapper, and the second fresh offline load produced the measurements above. Qwen3's `enable_thinking=False` template includes an empty `<think>…</think>` prefix before the generation boundary; generated-token slicing excludes that prompt prefix, and any thinking marker in decoded model output remains a hard rejection.
- **Next action:** Implement the local Qwen runtime, strict response schema/prompt construction, 27-layer validation, frozen diversity/benchmark screening integration, exact 120-IR plan, raw/hash-only recorders, telemetry, progress checkpoints, and tests before generating any counted realization output.

## 2026-07-18 20:36:20 -04:00 — Milestone 5B Steps 5–8: constrained runtime and counted IR plan implemented

- **Current step:** Turn the frozen design into an offline runtime and lock the exact 120 fresh semantic IRs before any counted Qwen output is generated.
- **Action performed:** Added a value-blind IR-to-placeholder request compiler, a strict JSON prompt protocol, an offline-only FP16 three-beam Qwen3 runtime, deterministic response/fill/language validation, separate generated-to-development and generated-to-generated screening indexes, a question-only ignored development export boundary, a fully pinned smoke configuration, and the exact targeted/generic IR plan. Added tests for all 120 request compilations, curriculum/output-track counts, the benchmark-content firewall at the model API, and hard rejection of thinking, Markdown, numeric calculation, and ambiguous-pronoun output.
- **Reason:** The language model must vary wording without seeing labels or benchmark data and without being able to override deterministic semantic, mathematical, diversity, or contamination decisions.
- **Important commands run:** Focused Ruff format/lint; strict Mypy over realization sources; focused realization unit tests; source review of the frozen generator/verifier dispatch and semantic policies.
- **Result:** The plan contains exactly 120 unique IR seeds: targeted 33/14/13 and generic 20/20/20 across bookkeeping/rates/discrete, with exactly 12 output-contract IRs per group. Every planned procedural IR compiles to a unique value-blind request without exposing the canonical answer or benchmark content. The selected internal policy and unchanged 0.75/0.82 development firewall remain separate. Focused strict Mypy passes; the first focused test pass found one over-specific capitalization assertion in a deliberately malformed fixture, which is being corrected before the counted run.
- **Files changed:** realization request/prompt/runtime/screening/smoke modules; the pinned smoke YAML; new unit tests; the question-export boundary; and this DEVLOG. Ignored model and raw directories remain untracked.
- **Errors or uncertainty:** The full runtime has not generated model output yet. Manual-review semantic outcomes are deliberately not auto-approved: a pending review may be selected by the automatic layers, then the mandatory all-beam human audit can reject that IR without substituting a later beam. Beam alternatives are compared only with selections from earlier IRs so a beam is not automatically rejected merely for sharing its own IR's latent program.
- **Next action:** Finish focused checks, export only the 904 development questions to the ignored scanner boundary using the frozen dataset environment, validate the complete counted configuration, then run exactly 120 IRs and 360 returned beams with progress checkpoints at 30/60/90/120.

## 2026-07-18 20:55:01 -04:00 — Milestone 5B Steps 9–10: counted smoke and all-beam audit complete

- **Current step:** Execute the one approved Qwen3 smoke without post-hoc changes, then manually classify every returned beam before exposing labels or verifier outcomes.
- **Action performed:** Exported only 904 question strings and stable IDs through the ignored contamination-scanner boundary; ran exactly 120 new IRs with three beams each; retained all 360 outputs under ignored raw results; inspected all 360 beams through 37 exact-template groups; classified naturalness, semantic preservation, and pipeline correctness; then checked the frozen dual-verifier evidence for label integrity.
- **Reason:** The fixed candidate budget and hidden-label first pass prevent retries, cherry-picking, and label-informed judgments of whether the model preserved the supplied problem.
- **Important commands run:** Question-only development export; offline `python -m foundry.synthesis.realization.smoke` with the frozen YAML; content-free raw-result aggregation; ignored template-group preparation; ignored all-beam manual-audit expansion.
- **Result:** The run accounted for exactly 120 IRs and 360 beams in 812.465 seconds. JSON parsing succeeded for 181/360 beams; 179 were malformed, predominantly because the 256-token output ended before the JSON object closed. No beam passed the complete automatic stack: parsed outputs systematically omitted semantic events, produced invalid clause maps/discourse order, or both. Manual audit found 63 natural-but-drifted beams, 59 semantics-preserving-but-unnatural beams, 297 unnatural beams total, and 301 semantic-drift beams. All 360 automatic rejections were correct; invalid acceptances, incorrect rejections, false labels, verifier disagreements, and backend failures were all zero. The readiness gate therefore failed at 0/120 clean IRs versus the required 90.
- **Files changed:** Ignored question export, beam records, template groups/decisions, and manual audit; tracked content-free summary; realization runtime/tests/config; this DEVLOG.
- **Errors or uncertainty:** Transformers warned that sampling-only defaults are irrelevant under `do_sample=False`, and warned about a missing attention mask because Qwen shares pad/EOS. Each call was a one-item unpadded prompt, so the effective mask was all ones; the warning was recorded and the frozen run was not modified. Peak model/runtime measurements remain subject to process-level rather than machine-wide RAM accounting. No beam reached benchmark or internal semantic screening because earlier required layers failed.
- **Next action:** Replay all 120 IRs with the same snapshot, environment, prompt, ordering, seed, and generation settings; compare byte/hash-identical beam text, ordering, decisions, and aggregate evidence. No fallback or prompt adjustment will be attempted.

## 2026-07-18 21:12:15 -04:00 — Milestone 5B Steps 11–14: exact replay passed and stop decision documented

- **Current step:** Prove the negative realization result is exactly reproducible, freeze the failed gate, and document the narrow next decision without changing the tested policy.
- **Action performed:** Replayed all 120 IRs/360 beams in a fresh offline process using the same snapshot, environment, request ordering, seed, prompt, chat template, and generation settings; compared exact beam text and all deterministic decisions; updated the aggregate summary, project plan, decision record, learning notes, synthesis design, realization design, and experiment ledger.
- **Reason:** A deterministic failure is actionable evidence only if it can be reproduced without redefining equivalence, and the failed readiness gate must block full generation before verification/commit.
- **Important commands run:** Offline realization smoke with `--replay`; expected-versus-actual deterministic SHA-256 comparison; content-free summary/resource aggregation; documentation updates.
- **Result:** Replay processed 120 IRs and 360 beams in 821.751 seconds and exactly reproduced SHA-256 `a2e6fb565da817ec5e2e6e3c87ba8a54643b2b5ec294dd8f5d24204083d06dcf`. Input/output tokens and peak GPU allocation/reservation also matched the original. D-019 records the stop: `foundry-slot-preserving-json-v1` is not production-ready, full 4,000 + 4,000 generation remains blocked, and no fallback or post-hoc protocol change was run.
- **Files changed:** Content-free result summary and all required project/design/learning/experiment documentation; ignored replay evidence; this DEVLOG. No evaluator, procedural math, benchmark manifest, model artifact, or raw tracked content changed.
- **Errors or uncertainty:** The model runtime is demonstrably deterministic on this machine, but current clean yield is zero, so there is no finite projection for 8,000 accepted examples. The approximate 15-hour projection applies only to 8,000 IR attempts at the measured rate, not usable data. The proposed compact protocol is a future design option, not an approved or measured fix.
- **Next action:** Run the complete repository verification and safety suite, fix only Milestone 5B implementation/documentation issues, inspect the exact staged snapshot, create one atomic commit, push normally to `origin/main`, and verify 0/0 divergence with a clean worktree.

## 2026-07-18 21:20:12 -04:00 — Milestone 5B Step 15: final verification passed

- **Current step:** Verify the complete implementation, reproducibility evidence, containment boundaries, and publication scope before staging the Milestone 5B commit.
- **Action performed:** Ran Ruff formatting and linting, strict Mypy, all unit and integration tests, realization-environment dependency integrity, `git diff --check`, high-confidence secret scanning, exact and 12-token development-question leak scans, frozen evaluator and procedural-mathematics diffs, sealed-path status, raw/model/environment tracking checks, dependency-lock and pinned-model hashes, prompt/chat-template/config/policy hashes, exact IR/beam/output-track counts, placeholder/node/target metrics, verifier and screening consistency, all-beam audit counts, replay identity, ignored-artifact checks, candidate-size review, and repository status review.
- **Reason:** The failed readiness result must still be published as a complete, reproducible, content-free scientific record without exposing raw generations, benchmark content, weights, caches, secrets, or sealed-final material.
- **Important commands run:** `ruff format src tests`; `ruff check src tests`; strict `mypy`; `pytest -q`; realization-environment `pip check`; `git diff --check`; typed smoke-config/plan, policy, model-snapshot, tokenizer/chat-template, raw-count, manual-audit, and replay checks; `git check-ignore`; `git ls-files`; content-free exact/12-token leak and credential scans.
- **Result:** Ruff and strict Mypy pass; all 232 tests pass; dependency and whitespace checks are clean. The 29 candidate files contain zero high-confidence secrets, zero exact development-question matches, and zero 12-token matches. Tracked raw/model/environment paths, frozen-evaluator changes, frozen generator-math changes, sealed-path changes, verifier disagreements, and unexpected downstream screening decisions are all zero. Exactly 120 ignored IR records contain 360 beams; 181 parsed, 41 parsed beams preserved placeholders, zero parsed beams preserved complete semantic-node coverage, and 171 parsed beams preserved target/intent. All 360 audit rows remain correct rejections. Replay remains exact at SHA-256 `a2e6fb565da817ec5e2e6e3c87ba8a54643b2b5ec294dd8f5d24204083d06dcf`. The largest candidate is this DEVLOG at 245,524 bytes, below 1 MiB.
- **Files changed:** This DEVLOG checkpoint only; Ruff left 85 source/test files unchanged and every other verification action was read-only.
- **Errors or uncertainty:** The first read-only leak checker recomputed tokenization inside its inner loop and was stopped after it proved unnecessarily slow; the corrected cached-token implementation completed in three seconds and produced the zero-finding results above. No inference or candidate generation was repeated. Windows LF-to-CRLF notices remain non-fatal.
- **Next action:** Fetch `origin/main` without merging, require the approved starting commit and 0/0 divergence, explicitly stage only the 29 reviewed Milestone 5B files, repeat staged containment and size checks, create one atomic commit, push without force, and verify synchronized clean state. If remote history changed, stop before merging or rebasing.

## 2026-07-18 21:22:00 -04:00 — Milestone 5B Step 15: publication precondition verified

- **Current step:** Confirm the live remote has not changed before freezing the reviewed snapshot.
- **Action performed:** Fetched `origin/main` without merging and checked repository root, branch, local/remote tips, divergence, and worktree scope.
- **Reason:** Publishing is allowed only as a fast-forward of the approved history; unexpected remote work would require stopping rather than merging, rebasing, or overwriting it.
- **Result:** Repository is `C:/Users/Admin/Projects/Foundry` on `main`; local `HEAD` and `origin/main` both remain `85179d7b8d3f9b48f0a36c92119c4f5049708ecd` with 0 ahead and 0 behind. The worktree contains only the 29 reviewed Milestone 5B candidate files; no remote drift occurred.
- **Files changed:** Git remote-tracking metadata and this DEVLOG entry only.
- **Errors or uncertainty:** No authentication failure, remote-history change, merge/rebase requirement, or unapproved file appeared.
- **Next action:** Stage exactly the reviewed milestone files, repeat staged whitespace, secret, development-content, raw/model/cache, frozen-path, size, hash, count, and replay checks, then create and push one atomic commit without force.

## 2026-07-18 21:22:30 -04:00 — Milestone 5B Step 15: staged publication snapshot verified

- **Current step:** Prove the exact Git index is the same safe, reviewed Milestone 5B scope that passed working-tree verification.
- **Action performed:** Explicitly staged 29 named files and scanned the index blobs for whitespace defects, prohibited paths, credentials, exact and 12-token development-question matches, oversized files, lock drift, aggregate-count drift, audit/replay inconsistency, and unstaged tracked changes.
- **Result:** The staged snapshot contains exactly 29 files and 4,288 insertions/15 deletions before this checkpoint. Forbidden raw/model/environment/evaluator/generator-math/sealed paths, secret hits, exact development matches, 12-token matches, unstaged tracked edits, and blobs at or above 1 MiB are all zero. The staged summary records 120 IRs, 360 beams, 360 audits, zero false labels/invalid acceptances, and a passed exact replay. The largest staged blob is this DEVLOG at 249,746 bytes.
- **Files changed:** This DEVLOG checkpoint only after the staged scan; it will be restaged and the final index whitespace/status check repeated.
- **Errors or uncertainty:** Expected LF-to-CRLF working-copy notices are non-fatal; staged bytes and `git diff --cached --check` are clean.
- **Next action:** Restage this checkpoint, repeat final index/status checks, create the one atomic Milestone 5B commit using the existing published identity process-locally if needed, push without force, and verify local/remote synchronization and a clean worktree.

## 2026-07-18 22:04:19 -04:00 — Milestone 5C Step 1: synchronized starting state verified

- **Current step:** Verify the published Milestone 5B state and every local prerequisite before implementing the compact tagged protocol.
- **Action performed:** Temporarily exposed Git in the current shell; fetched `origin/main` without merging; checked repository root, branch, local/remote tips, divergence, worktree, pinned Qwen3 snapshot, realization environment, dependency integrity, CUDA/GPU access, frozen evaluator and procedural generator/verifier paths, sealed-path status by name only, and raw/model/environment tracking.
- **Reason:** The micro-smoke must reuse the exact published mathematical and evaluation foundations and may change only the realization protocol lineage.
- **Result:** Repository is `C:/Users/Admin/Projects/Foundry` on clean `main`; local and `origin/main` both equal `90a1806e6929e1297d0936b18e1f6fcfae4b0701` with 0 ahead/0 behind. The exact Qwen3 revision snapshot exists offline. `.venv-realization` remains Python 3.12.10 with PyTorch 2.5.1+cu121, CUDA 12.1, Transformers 4.51.3, tokenizers 0.21.4, and a passing `pip check`; CUDA detects the RTX 3080. Frozen evaluator and procedural generator/verifier diff counts, sealed-path status, and tracked raw/model/environment counts are all zero.
- **Files changed:** This DEVLOG checkpoint only after read-only verification; Git remote-tracking metadata was refreshed.
- **Errors or uncertainty:** None. No sealed-final file was opened, no dependency or model was installed, and no inference ran.
- **Next action:** Implement a separate compact tagged request, prompt, parser, deterministic fill/validation path, runtime stopping criterion, exact 30-IR configuration, and tests while leaving the published verbose protocol and frozen mathematics intact.

## 2026-07-18 22:15:27 -04:00 — Milestone 5C Steps 2–9: compact protocol and fake-model gate passed

- **Current step:** Remove redundant Qwen responsibilities, freeze the compact protocol and 30-IR plan, and prove all deterministic boundaries before counted inference.
- **Action performed:** Added compact typed contracts, value-blind semantic-anchor compilation, concise tagged prompting, strict tag/placeholder/anchor/target validation, deterministic post-validation filling, per-beam `</Q>` stopping, the exact 30-IR configuration and plan, bounded runtime/smoke accounting, original hand-authored unit fixtures, and a three-beam fake-model integration test. Reused the existing procedural generators, dual verifiers, development question export, semantic artifact, and internal-diversity policy without changing them.
- **Reason:** Qwen should produce only natural connective wording around immutable event and target anchors; deterministic code already owns mappings, operations, values, units, targets, labels, verification, and screening.
- **Important commands run:** Ruff format/lint; strict Mypy; focused compact unit/integration tests; all-30 request construction; configuration/hash and allocation checks; realization-environment `pip check`.
- **Result:** Compact system/user/combined prompt hashes are `d5ea32af1d6df2c6bc06f7a315cab68084c26c0c719d5d08d1cb7c4628630222`, `6ed50a9c49434ec2e61d4d6065a9854aca414523b7b14eb5e33c60eee3285454`, and `6aec762647b03708268da0ae85c5c584821bdc2c82064a6534251c695a48fcf8`; normalized micro-smoke config SHA-256 is `78d1f3e37dad7e97c7e857a0f60efa5459af55b246a34dc653efceb01b47438b`. The plan has exactly 30 unique fresh seeds: 15 targeted and 15 generic, totals 13 bookkeeping/9 rate/8 discrete, and exactly three output-contract IRs per group. Every request compiles without exposing values, answers, target metadata, or benchmark content. Ruff and strict Mypy pass; 20 focused tests pass; `pip check` passes.
- **Files changed:** New compact realization/config modules and tests, plus this DEVLOG. No environment, dependency, model, frozen evaluator, procedural mathematics, benchmark artifact, or raw result changed.
- **Errors or uncertainty:** The first focused test run found three incorrect test assumptions—an outdated attribute name, a missing-tag fixture that correctly became outside text, and a self-canceling swap expression. Only the original fixtures were corrected; protocol behavior and hashes were unchanged. No real model inference has run yet.
- **Next action:** Run exactly 30 IRs and three Qwen3 beams per IR once with the frozen compact prompt/config, offline cache, no retries or replacements, and progress checkpoints after 15 and 30 IRs. Store full text only under ignored raw results.

## 2026-07-18 22:16:36 -04:00 — Milestone 5C Step 10: pre-count runtime compatibility defect resolved

- **Current step:** Start the counted compact run and enforce deterministic stopping after each complete `</Q>` tag.
- **Action performed:** Launched the frozen offline command. Transformers loaded the exact model, then aborted the first beam-search call before completing IR 1 because the new stopping criterion returned shape `(active_beams, 1)` while Transformers 4.51.3 requires `(active_beams,)`. Confirmed that no raw beam file, summary, or completed IR was written; corrected only the Boolean return shape and strengthened the existing unit test to assert it; reran Ruff, strict Mypy, all 20 compact tests, and the frozen config hash.
- **Reason:** This was an implementation compatibility defect in the required `</Q>` stopping condition, not evidence from a counted realization output. The approved 30-IR/90-beam budget remained unconsumed.
- **Result:** The corrected criterion matches the installed official `StoppingCriteriaList` contract. Ruff, Mypy, and all 20 tests pass; the prompt hashes and normalized config SHA-256 remain unchanged at their frozen values. No beam output or counted candidate exists from the aborted call.
- **Files changed:** Compact runtime, its stopping regression assertion, and this DEVLOG. No prompt, generation setting, validator, model, dependency, plan, raw result, evaluator, generator, or verifier changed.
- **Errors or uncertainty:** The normal inactive-sampling and pad/EOS attention-mask warnings appeared before the shape error; they are unchanged from Milestone 5B. Each request remains an unpadded single prompt, so the effective attention mask is all ones.
- **Next action:** Restart the still-unconsumed fixed micro-smoke once, require exactly 15/30 progress checkpoints and 90 returned beams, and make no further protocol or policy change after output is observed.

## 2026-07-18 22:21:17 -04:00 — Milestone 5C Steps 10–11: counted micro-smoke and all-beam audit complete

- **Current step:** Run the fixed compact protocol once on 30 fresh IRs, then classify every returned beam without seeing answers or verifier evidence.
- **Action performed:** Processed exactly 30 IRs/90 beams from the new master seed with no retries or replacements; stored complete text only in ignored raw results; inspected every deterministically filled surface in three answer-blind batches covering all 90 beam references; recorded the explicit per-beam classifications in ignored audit records and updated the tracked content-free summary.
- **Reason:** The compact protocol must demonstrate natural, meaning-preserving yield before any 120-IR follow-up; complete audit prevents structurally valid token echo from being mistaken for usable language.
- **Result:** All 90 beams parsed the required tag sequence. Eighty-seven preserved the complete placeholder set, semantic anchors, and target tokens; three altered the terminal question anchor. Automatic clean selections were 0/30. Every output remained unnatural and semantically drifted: Qwen copied the supplied tokens in list order, placed core relation anchors after their arguments, and frequently omitted connecting prepositions, question marks, or event punctuation. All 90 automatic rejections were correct; invalid acceptances, incorrect rejections, false labels, verifier disagreements, timeouts, and backend failures were zero. Counted generation took 83.474 seconds; end-to-end runtime 92.869 seconds; deterministic run SHA-256 is `b9b1a7bc8214c2656b6cd45cb089252f63fbe572c52f910e1148a34cd6a4358a`.
- **Files changed:** Ignored 30-record beam file and 90-record manual audit; tracked content-free compact summary; compact audit recorder; this DEVLOG. No prompt, validator, policy, threshold, model, generator, verifier, evaluator, or benchmark artifact changed after output was observed.
- **Errors or uncertainty:** All beams failed language quality; 60 also failed pre-fill consistency and three failed placeholder/anchor/target preservation. No beam reached benchmark or internal semantic screening because earlier required layers failed, so the zero contamination counts mean no candidate approached acceptance—not that live semantic comparison was exercised.
- **Next action:** Replay the exact 30 IRs using the same snapshot, environment, ordering, prompt, seed, generation settings, and stopping criterion; require exact beam and aggregate hash identity, then apply the unchanged gate and final Qwen3 stop rule.

## 2026-07-18 22:24:09 -04:00 — Milestone 5C Steps 12–13: exact replay passed and final stop rule applied

- **Current step:** Reproduce the complete negative result, apply every unchanged micro-gate criterion, and select exactly one permitted pivot after failure.
- **Action performed:** Replayed all 30 IRs/90 beams in a fresh offline process with the same model snapshot, environment, prompt, ordering, seed, generation settings, and stopping criterion; compared exact beam text and all deterministic decisions; finalized the content-free summary and documentation.
- **Reason:** A failed micro-smoke can block further spending only if its output and decisions are exactly reproducible and its gate is applied without post-hoc relaxation.
- **Result:** Replay processed 30/90 in 92.849 seconds and exactly reproduced SHA-256 `b9b1a7bc8214c2656b6cd45cb089252f63fbe572c52f910e1148a34cd6a4358a`. Count, false-label, accepted-drift, invalid-acceptance, verifier, unresolved-contamination, and replay gates pass. Clean acceptance fails at 0/30 versus 22; bookkeeping/rate/discrete minima fail at 0/8, 0/6, and 0/5; the systematic wording-defect gate fails. The final Qwen3 stop rule is active.
- **Files changed:** Ignored replay evidence; tracked content-free finalizer/summary and required project, decision, learning, synthesis, realization-design, experiment, and DEVLOG documentation.
- **Errors or uncertainty:** No replay mismatch occurred. Contamination counts are zero because no beam reached those late layers, so they do not demonstrate a live semantic pass rate. The stronger-model pivot remains a recommendation only; no model candidate, dependency, download, or inference is approved.
- **Next action:** Run the complete formatting, lint, typing, test, dependency, whitespace, secret, development-content, sealed-path, raw/model/cache, hash, count, verifier, replay, file-size, and repository-scope verification suite; fix only Milestone 5C defects, then create and push one atomic commit if the remote is unchanged.

## 2026-07-18 22:29:15 -04:00 - Milestone 5C Step 14: full verification passed

- **Current step:** Verify the complete compact-protocol implementation, measured negative result, ignored evidence, and publication scope before creating the atomic milestone commit.
- **Action performed:** Ran Ruff formatting and linting, strict Mypy, all unit and integration tests, realization-environment dependency integrity, `git diff --check`, high-confidence secret scanning, exact and 12-token development-question leak scans, frozen evaluator/procedural-mathematics and sealed-path status checks, raw/model/environment tracking and ignore checks, prompt/config/model/plan hash validation, exact IR/beam/group/output-track/audit counts, timeout and verifier-agreement checks, replay identity, candidate-size review, and repository status review.
- **Reason:** A failed readiness result must be demonstrably reproducible, internally consistent, and safe to publish without exposing raw generations, benchmark material, model files, caches, environments, credentials, or sealed-final content.
- **Result:** Ruff passes after wrapping one overlength audit-description string; strict Mypy reports no issues in 61 source files; all 252 tests pass; `.venv-realization` has no broken requirements; whitespace checks pass. Exactly 21 intended candidate files remain. Secret hits, exact development-question matches, 12-token development matches, forbidden paths, frozen evaluator/generator changes, sealed-path changes, and tracked raw/model/environment artifacts are all zero. The largest candidate is 262,779 bytes. The ignored evidence accounts for 30 IRs, 90 beams, six output-contract IRs, 90 audits, zero timeouts, zero verifier disagreements, and exact replay; the frozen protocol, model revision, generation contract, 30 unique seeds, and failed gate all validate.
- **Files changed:** One compact audit source string was mechanically wrapped; this DEVLOG verification checkpoint was added. No measured data, prompt, policy, threshold, model, environment, evaluator, generator, verifier, benchmark, or sealed-final artifact changed.
- **Errors or uncertainty:** The first combined count helper referenced the plan field as `candidate_seed`; the actual typed field is `random_seed`. The corrected read-only assertion passed and did not change code or evidence. Contamination screens were not reached because all beams failed earlier mandatory language/consistency layers, so zero contamination outcomes must not be interpreted as a measured semantic pass rate.
- **Next action:** Fetch `origin/main` without integrating changes, require it to remain at the approved starting commit, stage exactly the 21 verified files, repeat final index checks, create and push one atomic Milestone 5C commit without force, then verify synchronized clean local and remote state.

## 2026-07-18 22:30:16 -04:00 - Milestone 5C Step 14: staged publication snapshot verified

- **Current step:** Prove the exact Git index is the same safe Milestone 5C scope that passed working-tree verification.
- **Action performed:** Fetched `origin/main` without integrating changes; confirmed local and remote still equal the approved `90a1806e6929e1297d0936b18e1f6fcfae4b0701` at 0 ahead/0 behind; explicitly staged the 21 reviewed files; scanned index blobs for whitespace defects, prohibited paths, credentials, exact and 12-token development-question matches, oversized files, unstaged tracked changes, and aggregate/replay drift.
- **Reason:** Publication must be a safe fast-forward containing only the reproducible compact-protocol implementation and content-free evidence.
- **Result:** The index contains exactly 21 files and 2,643 insertions/15 deletions before this checkpoint. Forbidden paths, secret hits, exact development matches, 12-token matches, and unstaged tracked changes are all zero. The largest staged blob is 265,347 bytes. The staged summary records 30 IRs, 90 beams, 90 audits, a passed replay, and the failed fixed readiness gate.
- **Files changed:** This DEVLOG checkpoint only after the staged scan; it will be restaged and the final index whitespace/status check repeated.
- **Errors or uncertainty:** An initial read-only remote-state assertion compared Git's tab-delimited divergence output as a space-delimited string and raised despite displaying 0/0. The corrected numeric assertion confirmed the expected state; no local or remote history changed.
- **Next action:** Restage this checkpoint, repeat final index/status checks, create the one atomic Milestone 5C commit, push without force, and verify equal local/remote tips, 0 ahead/0 behind, a clean worktree, and no forbidden published artifacts.

## 2026-07-18 22:46:51 -04:00 - Milestone 5D Step 1: synchronized controlled-comparison state verified

- **Current step:** Verify the published Milestone 5C state and every local prerequisite before substituting the approved stronger realization model.
- **Action performed:** Temporarily exposed Git; fetched `origin/main` without integrating changes; checked root, branch, local/remote refs, divergence, worktree, `.venv-realization`, CUDA/GPU access, dependency integrity, the ignored Milestone 5C raw artifact, all 30 semantic-IR and latent hashes, compact prompt/config hashes, frozen evaluator/generator/verifier/validator paths, sealed-path status by name only, and raw/model/environment tracking.
- **Reason:** Milestone 5D is a controlled one-variable model comparison; every IR, prompt, mathematical, validation, and safety input must match Milestone 5C before a different model is introduced.
- **Result:** Repository is `C:/Users/Admin/Projects/Foundry` on clean `main`; local and `origin/main` both equal `1110d987cd3ea5a980e2c80c79ccf2fe38678195` with 0 ahead/0 behind. The environment remains Python 3.12.10, PyTorch 2.5.1+cu121, CUDA 12.1, Transformers 4.51.3, and tokenizers 0.21.4 with a passing `pip check`; CUDA detects the RTX 3080. The ignored M5C record contains exactly 30 IRs/90 beams, 30 unique semantic-IR hashes, and 30 unique latent hashes. Compact system/user/combined/config hashes remain `d5ea32af...0222`, `6ed50a9c...5454`, `6aec7626...fcf8`, and `78d1f3e3...438b`. Frozen-path, sealed-path, and tracked raw/model/environment counts are zero.
- **Files changed:** This DEVLOG checkpoint only after read-only verification; Git remote-tracking metadata was refreshed.
- **Errors or uncertainty:** None. No dependency or model was installed or downloaded, no inference ran, and no sealed-final file was opened.
- **Next action:** Query the official Qwen repository metadata for `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`, verify the immutable revision, Apache-2.0 license, architecture/config/tokenizer/safetensors inventory, expected bytes, and Transformers 4.51.3 support before downloading weights.

## 2026-07-18 22:48:17 -04:00 - Milestone 5D Step 2: official stronger-model metadata verified

- **Current step:** Verify the exact official model artifact and local framework compatibility before downloading any weight file.
- **Action performed:** Queried the official Hugging Face model API at the approved immutable revision with file metadata; checked the model card license/library, model and tokenizer configuration, every repository filename, blob identity, large-file SHA-256, and size; confirmed the installed Transformers mapping exposes built-in `Qwen3Config` and `Qwen3ForCausalLM`; checked local free disk space.
- **Reason:** The experiment permits exactly one official model substitution and forbids dependency changes, remote code, quantizations, and fallback checkpoints.
- **Result:** `Qwen/Qwen3-4B-Instruct-2507` resolves exactly to `cdbee75f17c01a7cc42f958dc650907174af0554`, Apache-2.0, Transformers, `model_type=qwen3`, and `Qwen3ForCausalLM`. The official repository has 13 files totaling 8,060,917,568 bytes; three BF16 safetensor shards total 8,044,982,000 bytes, with tokenizer/config/index/support files making up the remainder. Transformers 4.51.3 has native Qwen3 support, so `trust_remote_code=False` is compatible and no dependency change is required. Drive C has 202,540,650,496 free bytes.
- **Files changed:** This DEVLOG entry only. No weight, tokenizer, dependency, model cache, project source, prompt, IR, or raw-result file changed yet.
- **Errors or uncertainty:** The first metadata formatter attempted to coerce Hugging Face card metadata with `dict(...)`, which that object does not support; a corrected read-only `to_dict()` query returned the full verified metadata. The repository weights are stored as BF16; Milestone 5D will load them into FP16 as approved.
- **Next action:** Download only the exact 13-file approved snapshot into the ignored cache, verify all expected large-file hashes and total bytes, then perform a fresh offline tokenizer/model reload with `local_files_only=True` and record load time, RAM, and GPU memory.

## 2026-07-18 22:55:11 -04:00 - Milestone 5D Step 3: pinned model downloaded and offline reload verified

- **Current step:** Materialize and verify only the approved official stronger-model snapshot, then prove it loads offline on the RTX 3080 without dependency or device-map changes.
- **Action performed:** Downloaded the exact 13-file snapshot through `snapshot_download` at the immutable revision into `data/huggingface`; computed SHA-256 for every local snapshot file and compared the large-file hashes and byte totals with official metadata; set Hugging Face and Transformers offline modes; loaded tokenizer and model from the local snapshot with `local_files_only=True`, `trust_remote_code=False`, safetensors, and FP16; asserted that every parameter resides on `cuda:0`.
- **Reason:** The counted comparison may begin only after artifact identity, offline reproducibility, precision, and no-offload behavior are proven.
- **Result:** Download completed in 288.703 seconds. Cache growth was exactly 8,060,917,568 bytes, matching the complete official inventory. The three weight hashes are `75311d91...8ed6`, `0b48adbb...fba1`, and `7dd39ccc...d5d`; tokenizer JSON is `aeb13307...ae4`; config is `5beea1a4...e8ba`; chat template is `64f85b19...c326`. Offline reload resolved exactly `cdbee75f17c01a7cc42f958dc650907174af0554`, loaded 4,022,468,096 FP16 parameters solely on CUDA, and took 0.221 seconds for the tokenizer plus 4.149 seconds for the model. Peak/load-state reserved VRAM was 8,103,395,328 bytes with 1,390,559,232 bytes free; peak process working set was 12,466,991,104 bytes.
- **Files changed:** Ignored Hugging Face cache for only the approved Qwen snapshot, plus this DEVLOG entry. No dependency, prompt, IR, validator, mathematical generator/verifier, benchmark artifact, prior raw result, or sealed-final file changed.
- **Errors or uncertainty:** The optional `hf_xet` accelerator is absent, so the downloader used regular HTTPS as required rather than installing anything. Windows emitted the expected non-fatal no-symlink cache warning; physical cache growth is therefore the full repository size. The official BF16 shards were cast to approved FP16 at load.
- **Next action:** Add a narrow model-substitution contract/harness that reads the unchanged M5C plan and verifies every regenerated IR/latent/event/placeholder/target hash against the preserved M5C artifact, then run one original non-benchmark three-beam probe under the unchanged prompt and generation settings with the fixed VRAM/headroom gates.

## 2026-07-18 23:01:19 -04:00 - Milestone 5D Steps 4-6: controlled IR proof and memory probe passed

- **Current step:** Freeze the one-variable comparison, prove every M5C semantic input is identical, and test three-beam FP16 compatibility before spending the counted 90-beam budget.
- **Action performed:** Added a model-substitution configuration and typed comparison contract; regenerated the M5C plan without creating replacement IRs; compared all plan fields, candidate IDs, semantic-IR hashes, latent hashes, and compact-request hashes against the preserved M5C artifact; added and tested an original non-benchmark procedural memory probe; loaded the approved model offline and generated exactly three beams under the unchanged compact prompt and decoding settings.
- **Reason:** The counted run is valid only if model identity is the sole changed experimental variable and the 10 GB GPU can sustain three-beam decoding without offload or altered settings.
- **Result:** All 30 plan, semantic-IR, latent, and request hashes match; the content-free controlled manifest is `794ade78...ad66`. Ruff, strict Mypy, 21 focused tests, and `pip check` pass. The probe returned and tag-parsed all three beams in 3.575 seconds with no timeout/backend failure; all 4,022,468,096 FP16 parameters remained on `cuda:0`, with no device map or CPU offload. Peak allocated/reserved VRAM was 8,426,049,024/8,489,271,296 bytes, below the 9,932,111,872-byte limit. Free memory after generation was 946,094,080 bytes, 409,223,168 bytes above the 512 MiB minimum. Peak process working set was 12,480,253,952 bytes.
- **Files changed:** New stronger-model configuration, typed comparison/probe modules, original contract test, one ignored probe record, and this DEVLOG. The downloaded snapshot is unchanged. No prompt, compact validator, semantic anchor, IR, label, mathematical generator/verifier, dependency, benchmark artifact, or sealed-final file changed.
- **Errors or uncertainty:** Initial static checks found one unused import and three references to the existing beam field as `text` instead of `raw_text`; both implementation-only defects were fixed before probe inference. Transformers emitted non-fatal warnings that official sampling defaults are inactive under `do_sample=False` and that pad equals EOS without an explicit mask; these warnings also existed in M5C and the controlled generation contract remains unchanged.
- **Next action:** Finish the comparison runner, per-checkpoint preservation telemetry, all-beam audit ingestion, exact replay, and fixed final-gate code; run formatting, linting, typing, focused tests, configuration/hash proofs, then execute exactly the preserved 30 IRs/90 beams once if all pre-count checks pass.

## 2026-07-18 23:04:35 -04:00 - Milestone 5D Steps 5-7: counted comparison harness frozen

- **Current step:** Complete and verify the comparison runner, audit/replay gate, exact progress metrics, and all pre-count safeguards before the single 90-beam run.
- **Action performed:** Reused the compact execution and validation stack through a stronger-model wrapper; added content-free 15/30 progress telemetry, blinded per-beam audit ingestion, exact replay finalization, fixed gate logic, and direct M5C/M5D comparison fields; ran Ruff, strict Mypy, the complete test suite, realization `pip check`, controlled-IR verification, prompt/config/experiment hash checks, protected-path diffs, probe assertions, and counted-artifact absence checks.
- **Reason:** The comparison must be auditable and unable to alter the prompt, IRs, validators, labels, beam budget, or gate after seeing stronger-model output.
- **Result:** Ruff passes; strict Mypy reports no issues in 66 source files; all 253 tests pass; dependencies are intact. Comparison config SHA-256 is `362a9fae...54bc`, combined experiment SHA-256 is `e09e1aca...b377`, and the compact system/user/combined hashes remain unchanged. All 30 plans/semantic IRs/latent programs/requests match M5C. Protected prompt, validator, generator, verifier, evaluator, and eval-config diff count is zero. The probe is passed at 3/3 parsed beams, and no counted M5D beam or summary artifact exists yet.
- **Files changed:** Comparison configuration and implementation modules, compact progress reporting only, stronger-model tests, and this DEVLOG. The ignored probe remains local. No dependency, prompt, validator, IR, mathematical, benchmark, model-cache, or sealed-final change occurred.
- **Errors or uncertainty:** Pre-count lint found one loop-closure reporting pattern and Mypy found one audit counter name collision; both telemetry/aggregation defects were fixed, followed by a clean full verification. These fixes did not affect generation, validation, prompt, or model behavior.
- **Next action:** Run exactly the preserved 30 M5C IRs with three deterministic Qwen3-4B beams each, report the required 15/30 checkpoints, store full generations only under ignored raw results, and freeze outputs without prompt, validator, policy, model, or generation-setting changes.

## 2026-07-18 23:08:23 -04:00 - Milestone 5D Steps 7-8: counted run and blinded all-beam audit complete

- **Current step:** Execute the one-variable stronger-model comparison once, then manually classify every returned beam before viewing labels or verifier evidence.
- **Action performed:** Ran exactly the preserved 30 M5C IRs with three Qwen3-4B beams per IR, no retries or replacements; recorded the 15/30 progress checkpoints; stored 90 full outputs only in ignored raw results; reviewed all outputs in three ten-IR batches without displaying canonical answers or verifier evidence; recorded explicit hash-bound per-beam decisions and then confirmed label/verifier status.
- **Reason:** The stronger model must demonstrate natural, semantically preserved wording under the exact frozen protocol, and complete blinded review prevents token echo or placeholder invention from being mistaken for useful output.
- **Result:** Counted runtime was 121.937 seconds, including 108.983 seconds of generation. Final automatic counts were 71/90 tag-parsed, 47/90 placeholder-preserving, 50/90 semantic-anchor-preserving, 47/90 target-preserving, 19 language-quality passes, 10 filled-consistency passes, and 0 automatically accepted IRs. Input/output totals were 11,410/8,838 tokens. Peak allocated/reserved VRAM was 8,553,572,352/8,703,180,800 bytes; peak process working set was 12,593,078,272 bytes. Deterministic run SHA-256 is `7043fb5f...9f01`. Blinded audit classified all 90 as unnatural and semantically drifted: malformed tags, invented/replaced placeholders or values, and postfixed token-list predicates were systematic. All 90 automatic rejections were correct; clean IRs, false labels, invalid acceptances, incorrect rejections, verifier disagreements, timeouts, and backend failures are all zero.
- **Files changed:** Ignored 30-record/90-beam raw output and ignored 90-row decision/audit files; tracked content-free summary and audit helper; this DEVLOG. No prompt, validator, policy, threshold, IR, model, dependency, mathematical generator/verifier, benchmark artifact, or sealed-final file changed after output was observed.
- **Errors or uncertainty:** No counted execution error occurred. The stronger model improved neither clean yield nor the systematic realization failure. Zero contamination outcomes do not establish a live screening pass rate because every beam failed earlier mandatory layers.
- **Next action:** Replay the exact same 30 IRs/90 beams with the same snapshot, environment, prompt, seed, order, and generation settings; require exact beam/decision/hash identity, then apply the unchanged gate and final local-model stop rule.

## 2026-07-18 23:11:30 -04:00 - Milestone 5D Steps 9-11: exact replay passed and template-bank pivot recorded

- **Current step:** Reproduce the complete stronger-model result, apply every unchanged readiness criterion, and record the mandatory final architecture decision.
- **Action performed:** Replayed all 30 IRs/90 beams offline with the same model snapshot, environment, prompt, seed, ordering, and decoding; compared beam text/order and deterministic decisions; applied the predeclared gate; wrote a content-free 13-file model artifact manifest; updated the project plan, decisions, learning notes, synthesis/realization design, experiment registry, summary, and DEVLOG.
- **Reason:** A final negative model comparison must be exactly reproducible and its stop rule must be applied without post-hoc prompt, model, validator, or threshold changes.
- **Result:** Replay exactly reproduced `7043fb5f94cbd95fe76391fa167ba766acf5080f77b0fede7197c00b8b9a9f01`, all 30 IRs, all 90 beams, and all preservation/rejection counts. Replay generation/total time was 109.120/121.947 seconds with identical tokens and GPU peaks. Count, beam, false-label, accepted-drift, invalid-acceptance, verifier, unresolved-contamination, replay, and resource-practicality criteria pass. Clean acceptance fails at 0/30 versus 22; bookkeeping/rate/discrete fail at 0/8, 0/6, and 0/5; the systematic wording-defect gate fails. The final local-model stop rule is active, a 120-IR run is not justified, and the only recommendation is a separately approved manually vetted offline natural-language template bank over the retained exact IR/label/verifier/screening stack.
- **Files changed:** Required documentation, content-free model artifact and comparison summaries, comparison implementation/config/tests, compact progress reporting, and this DEVLOG. Ignored beams, audit decisions, audit rows, probe, replay, and model cache remain local only.
- **Errors or uncertainty:** No replay or gate error occurred. Benchmark/internal semantic screens were not reached because every beam failed earlier required layers, so empty screening outcomes do not measure semantic pass rates. No template-bank design or implementation is approved yet.
- **Next action:** Run the complete formatting, linting, strict typing, tests, dependency, whitespace, secret, development-content, sealed-path, raw/model/cache, revision/license/hash, IR/beam/audit, verifier, replay, size, and repository-scope verification suite; fix only M5D defects, then create and push one atomic commit if the remote is unchanged.

## 2026-07-18 23:14:30 -04:00 - Milestone 5D Step 11: full verification passed

- **Current step:** Verify the complete stronger-model implementation, exact negative result, artifact identity, ignored evidence, and publication scope before creating the atomic milestone commit.
- **Action performed:** Ran Ruff formatting/linting, strict Mypy, all tests, realization-environment dependency integrity, `git diff --check`, high-confidence secret scanning, exact and 12-token development-question leak scans, protected prompt/validator/math/evaluator and sealed-path status checks, raw/model/environment tracking and ignore checks, official model revision/license/file-manifest hashes, config/protocol/experiment/control hashes, exact IR/beam/decision/audit counts, preservation/verifier/gate consistency, exact replay identity, tracked-file size review, and repository status review.
- **Reason:** The final local-model failure must be published as a complete, reproducible, content-free record without raw generations, model/cache files, benchmark material, secrets, or sealed-final content.
- **Result:** Ruff passes; strict Mypy reports no issues in 66 source files; all 253 tests pass; `.venv-realization` has no broken requirements; whitespace checks pass. Exactly 17 intended candidate files remain. Secret hits, exact development matches, 12-token matches, forbidden paths, protected-path changes, sealed-path changes, and tracked raw/model/environment artifacts are all zero. The largest candidate is 284,549 bytes. All 13 approved snapshot files hash to manifest `d1a81539...a7a`; 30 IRs, 90 beams, 90 decisions, 90 audits, zero verifier disagreements/false labels, passed replay, failed gate, and the template-bank recommendation validate.
- **Files changed:** This DEVLOG verification entry only after the reviewed M5D code, config, summaries, tests, and documentation. No measured output, prompt, validator, policy, threshold, model, dependency, IR, label, mathematical code, benchmark artifact, or sealed-final file changed.
- **Errors or uncertainty:** No final verification error occurred. Empty benchmark/internal screening outcomes reflect early mandatory-layer failures and are not evidence of a semantic-screen pass rate.
- **Next action:** Fetch `origin/main` without integrating changes, require it to remain at the approved starting commit, stage exactly the 17 verified files, repeat index whitespace/scope/leak/hash/count checks, create and push one atomic M5D commit without force, then verify synchronized clean local and remote state.

## 2026-07-18 23:15:24 -04:00 - Milestone 5D Step 11: staged publication snapshot verified

- **Current step:** Prove the exact Git index is the same safe, content-free M5D scope that passed working-tree verification.
- **Action performed:** Fetched `origin/main` without integrating changes and confirmed it remains synchronized at the approved starting commit; explicitly staged 17 reviewed files; scanned index blobs for whitespace defects, prohibited paths, credentials, exact and 12-token development-question matches, oversized files, unstaged changes, and aggregate count/audit/replay drift.
- **Reason:** Publication must be a fast-forward containing only the controlled model-comparison implementation, metadata, content-free evidence, tests, and documentation.
- **Result:** The index contains exactly 17 files and 1,540 insertions before this checkpoint. Forbidden paths, secret hits, exact development matches, 12-token matches, and unstaged tracked edits are all zero. The largest staged blob is 286,824 bytes. The staged summary records 30 IRs, 90 beams, 90 audits, a passed replay, and the failed fixed gate.
- **Files changed:** This DEVLOG checkpoint only after the staged scan; it will be restaged and the final index whitespace/status check repeated.
- **Errors or uncertainty:** None. Expected LF-to-CRLF working-copy notices are non-fatal; staged bytes and `git diff --cached --check` are clean.
- **Next action:** Restage this checkpoint, repeat final index/status checks, create the one atomic M5D commit, push without force, and verify equal local/remote tips, 0 ahead/0 behind, a clean worktree, and no forbidden published artifact.
## 2026-07-18 23:58:48 -04:00 - Milestone 6A Step 1: synchronized starting state verified

- **Current step:** Verify the published Milestone 5D state and the protected evaluation, mathematics, verification, and contamination surfaces before implementing the offline template bank.
- **Action performed:** Temporarily added `C:\Program Files\Git\cmd` to this PowerShell process; fetched `origin/main` without integrating changes; checked repository root, branch, local/remote tips, ahead/behind, worktree state, the existing Python environment, tracked raw/model/environment paths, and diffs across the frozen evaluator, synthesis generators, dual verifiers, semantic artifact configuration, contamination controls, and benchmark manifests.
- **Reason:** Milestone 6A must begin from the exact published negative live-realization result and must reuse the trusted mathematical and safety stack without modifying it.
- **Result:** Repository root is `C:/Users/Admin/Projects/Foundry`; `main` and `origin/main` both point to `057130d5da79751ab9b05508dd83209e5a6b65b5`; ahead/behind is `0/0`; the worktree is clean. CPython 3.12.10 and the main `.venv` remain healthy (`pip check` passed). Protected-path diff count, sealed-final-path status count, and tracked raw/model/environment artifact count are all zero.
- **Files changed:** This DEVLOG entry only. No template, generator, verifier, evaluator, manifest, result, model, dependency, or cache file changed during verification.
- **Errors or uncertainty:** None. The sealed-final partition was neither opened nor hashed.
- **Next action:** Map the existing typed semantic IR and compiler evidence, then implement a fail-closed versioned template-bank contract that consumes the trusted IR without duplicating mathematical logic.
## 2026-07-19 00:08:15 -04:00 - Milestone 6A Steps 2-7: template-bank contracts, families, and static validation complete

- **Current step:** Replace the closed live-model path with a versioned, fail-closed bank that renders only trusted typed semantic IR.
- **Action performed:** Added `TemplateSpec` and `SentencePlanSpec` contracts; recorded original provenance and human-review-pending status; implemented 18 bookkeeping, 20 rate/ratio, and 20 discrete semantic frames with four independently structured plans per frame; added deterministic slot filling, target/relation/difficulty compatibility, typed morphology and coverage evidence, stable template/render hashes, and a separate pre-smoke internal-diversity policy with six content-free fixture classes. Added a fresh 120-attempt configuration and focused unit/integration tests. Swept every planned attempt through the new renderer and both deterministic language validators.
- **Reason:** The bank must express exact procedural facts in natural English without asking a model to invent wording, labels, mappings, or mathematics, and it must fail closed on incompatible or untyped inputs.
- **Result:** The source-controlled bank contains 58 semantic frames and 232 sentence plans/render signatures: 18/72 bookkeeping, 20/80 rate/ratio, and 20/80 discrete. The scheduled 120 attempts select 120 unique signatures and 120 unique procedural candidates. All 120 renderings pass semantic-node, morphology, unit, target, punctuation, duplicate-clause, and output-contract validation. The existing eleven sanitized renderer regressions remain covered and pass. Ruff, strict Mypy, and 37 focused tests pass after correcting one capitalization edge case and one repeated-setting phrase found by the new sweep.
- **Files changed:** New `src/foundry/synthesis/template_bank/` contracts, bank, renderer, policy, and smoke implementation; three synthesis configuration/fixture files; focused tests; this DEVLOG. Existing generators, verifiers, evaluator, benchmark manifests, semantic thresholds/model, and dependencies are unchanged.
- **Errors or uncertainty:** The initial static sweep exposed two deterministic wording defects before any counted smoke: a lower-case sentence after a fronted condition and a duplicated setting name. Both were fixed architecturally in centralized sentence normalization or the affected original plan, then the complete 120-render sweep passed. Human naturalness review remains pending and will use the ignored packet produced by the counted run.
- **Next action:** Run the single fresh 120-attempt bank smoke through exact verification plus unchanged development contamination screening, report the required 60/120 checkpoints, then replay it exactly and create the ignored user-review packet.
## 2026-07-19 00:11:02 -04:00 - Milestone 6A Steps 8-13: counted smoke, replay, packet, and Codex inspection complete

- **Current step:** Execute the single bounded template-bank smoke under frozen mathematical, language-quality, diversity, and development-contamination controls; replay it; and prepare the user review evidence.
- **Action performed:** Ran exactly 120 fresh attempts from the new master seed with no replacements; used 60 targeted and 60 generic allocations and 12 output-contract attempts per group; screened 904 development questions only inside the read-only contamination scanner; ran the pinned local MiniLM encoder on CPU; repeated the complete 120-attempt construction/decision path; wrote ignored counted/replay records and the 120-question user packet; then Codex inspected all 120 rendered questions without calling that inspection a human audit.
- **Reason:** The milestone must measure whether the offline bank is mathematically safe, deterministic, contamination-safe, and linguistically credible before any full generation can be considered.
- **Result:** At 60 attempts, 60 were accepted; at 120, 118 were accepted and two rejected. Targeted accepted 60/60; generic 58/60. Bookkeeping/rate/discrete accepted 53/53, 33/34, and 32/33. Easy/medium/hard accepted 39/40, 39/40, and 40/40. All 24 output-contract attempts passed. Rejections were one duplicate latent program and one number-neutral template copy. Both verifiers agreed on all 120; false labels, deterministic-rule language defects, target mismatches, benchmark lexical/semantic rejections, unresolved contamination cases, exact text duplicates, and reused render signatures were zero. Counted and replay SHA-256 both equal `bf87e7af166f5dca107c9777337216e6da7a656b4eec3efb372dc98d1bfa5487`. Counted/replay runtime was 1.579/1.585 seconds; peak process RAM 914,022,400 bytes; ignored raw evidence 440,657 bytes; GPU use zero.
- **Files changed:** Ignored `results/raw/template_bank_smoke/attempts.jsonl`, `replay.jsonl`, and `human_review.md`; tracked content-free summary and Codex-inspection evidence; required documentation and this DEVLOG. No benchmark content, answer, generated question, model/cache artifact, dependency, generator, verifier, evaluator, or sealed-final file is tracked.
- **Errors or uncertainty:** The initial generated summary incorrectly treated a correctly rejected latent duplicate as an automatic gate failure even though the fixed gate applies duplicate prohibitions to accepted examples; that reporting-only condition was corrected from frozen raw/replay evidence without regeneration. More importantly, Codex inspection found 13 clearly invalid or unnatural questions: duplicated frame/grouping nouns, invalid ordinal inflection, malformed compound/grouping nouns, and awkward literal frame-label realization. These are systematic language defects missed by deterministic validation. Human review remains pending and may find additional defects.
- **Next action:** Record the failed technical gate, review instructions, and no-full-generation decision in all project documentation; run the complete repository verification suite; then commit and push the content-free milestone evidence if every safety check passes.
## 2026-07-19 00:13:21 -04:00 - Milestone 6A Step 14: full verification passed

- **Current step:** Verify code quality, exact evidence, scope, ignore boundaries, benchmark firewall, and publication safety before creating the atomic Milestone 6A commit.
- **Action performed:** Ran Ruff formatting/linting, strict Mypy, all unit and integration tests, main-environment `pip check`, `git diff --check`, high-confidence secret scanning, exact and 12-token development-question leak scans over every changed/untracked file, protected evaluator/generator/verifier/contamination/dependency diffs, sealed-path status, raw/model/environment tracking and ignore checks, bank capacity/schema checks, eleven historical regressions, exact 120-plan rendering checks, candidate/group/difficulty/output-track counts, verifier/replay/summary consistency, Codex-inspection finalization, tracked-file size review, and repository scope review.
- **Reason:** A failed technical gate is still scientific evidence and must be complete, reproducible, content-free, and safe before publication.
- **Result:** Ruff passes; strict Mypy reports no issues in 73 source files; all 261 tests pass; dependencies and whitespace are clean. Secret hits, exact development-question matches, 12-token development matches, protected-path changes, sealed-path changes, and tracked raw/model/environment artifacts are all zero. All three raw files are ignored. The summary proves 120 attempts, 118 automatic accepts, exact replay, zero false labels/disagreements/unresolved contamination, 13 content-free Codex findings, and a failed final gate. No candidate file reaches 1 MiB; the largest is this DEVLOG at approximately 297 KiB.
- **Files changed:** Twenty-two intended paths: seven existing documentation files plus three configs/fixtures, one dedicated design document, two content-free results, seven template-bank source files, and two tests. Ignored local raw evidence remains outside Git.
- **Errors or uncertainty:** A PowerShell array-count display reported `2` because it counted two nested command-result arrays; the flattened path listing and Git status contain the expected 22 individual candidate paths. This did not affect any file scan; the leak and secret loops processed the flattened path values. Expected LF-to-CRLF working-copy notices are non-fatal.
- **Next action:** Fetch `origin/main` without integrating changes, require it to remain at the approved starting commit, stage exactly the 22 reviewed files, repeat index whitespace/scope/secret/content/size/evidence checks, create one atomic commit, push without force, and verify synchronized clean local/remote state.
## 2026-07-19 00:14:14 -04:00 - Milestone 6A Step 14: staged publication snapshot verified

- **Current step:** Prove that the exact Git index matches the verified, content-free Milestone 6A scope before publication.
- **Action performed:** Fetched `origin/main` without integration and confirmed it remains at the approved starting commit; explicitly staged 22 reviewed paths; ran index whitespace, path-scope, forbidden-extension, high-confidence secret, exact/12-token development-content, summary/count/replay/gate, raw-tracking, size, and unstaged-change checks.
- **Reason:** Publication must be a fast-forward containing only the offline-bank implementation, content-free negative evidence, tests, configurations, and documentation.
- **Result:** Local and remote remain synchronized at `057130d5...65b5`. The index contains exactly 22 intended files, zero forbidden raw/model/cache/evaluator/generator/verifier/sealed paths, zero secrets, zero exact development-question matches, zero 12-token matches, no unstaged tracked changes, no tracked raw files, and no blob at or above 1 MiB. The largest staged blob is 298,941 bytes. Staged evidence records 120 attempts, 118 automatic accepts, two safe rejections, exact replay, 13 Codex surface findings, and a failed final gate.
- **Files changed:** This DEVLOG checkpoint only after staging; it will be restaged and all final index/status checks repeated.
- **Errors or uncertainty:** The first staged 12-token helper used a walrus-bound variable inside a nested comprehension and raised `NameError`; the corrected explicit loop completed and returned zero exact and zero 12-token matches. No staged bytes changed because of that read-only helper failure.
- **Next action:** Restage this checkpoint, repeat final index whitespace/count/status checks, create the atomic milestone commit with process-local author identity, push without force, and verify clean synchronized local/remote state.

## 2026-07-19 00:41:43 -04:00 - Milestone 6B Steps 1-2: starting state and 13-defect taxonomy verified

- **Current step:** Verify the exact published Milestone 6A state and turn every prior surface defect into a content-free architectural regression before changing the composition compiler.
- **Action performed:** Temporarily added Git to this PowerShell process; fetched `origin/main` without integrating changes; verified repository root, branch, local/remote tips, divergence, worktree, CPython environment, ignored Milestone 6A evidence, and protected evaluator/generator/verifier/contamination paths. Inspected only the 13 locally ignored synthetic surfaces identified in Milestone 6A and assigned one primary defect class to each.
- **Reason:** Milestone 6B must start from the published evidence and prevent whole classes of composition failures, rather than patching individual output strings.
- **Result:** `C:/Users/Admin/Projects/Foundry` is on clean `main`; local `main`, tracking `origin/main`, and fetched `origin/main` all equal `822596a6f93435285e86c8fc63714341c525a833`; divergence is `0/0`. The three ignored Milestone 6A evidence files remain local. The 13 defects divide into five adjacent duplicate head-noun cases (`dispatch record record`, `receiving record record`, `materials register register`, `equipment register register`, and `dual recipe plan plan`); two duplicated/malformed grouping-head cases (`paired collections collections` and `matched batches collections`); two invalid ordinal cases (`1th`/`2th` in two weighted-mean surfaces); and four internal-frame/unsupported-compound leaks (`selected share inventory`, `two resource capacity inventory`, `parallel channels process`, and `paired supply limit inventory`). Existing validation missed them because internal semantic identifiers were normalized directly into prose and noun/ordinal composition had no typed provenance or head-role invariant.
- **Files changed:** This DEVLOG entry only. No source, template, generator, verifier, evaluator, policy, threshold, dependency, benchmark, raw result, or sealed-final artifact changed during verification and diagnosis.
- **Errors or uncertainty:** None. The defect classes are content-free and each observed failure has a deterministic architectural invariant: approved surface lexemes only, correct ordinal morphology, one typed noun head per phrase, and complete token provenance. The sealed-final partition was neither opened nor hashed.
- **Next action:** Implement approved surface lexemes, centralized ordinals, typed noun-phrase composition, and token-level provenance; add all 13 sanitized regression fixtures; then expand the complete bank before any fresh counted smoke.

## 2026-07-19 00:51:59 -04:00 - Milestone 6B Steps 3-7: composition compiler and full-bank static gate passed

- **Current step:** Replace identifier-derived prose with typed lexical composition, validate the complete template bank, and inspect a deterministic cross-family sample before authorizing the counted smoke.
- **Action performed:** Added explicit approved surface lexemes separate from internal frame IDs; routed noun heads through a typed `NounPhraseSpec`; implemented numeric and bounded word-form ordinals; added token-level source provenance and semantic-node accounting; blocked raw identifiers, repeated heads, invalid ordinals, unlicensed nodes, and unsupported composition. Added all 13 sanitized regressions. Expanded every one of 232 sentence plans across 10 deterministic fixtures covering all difficulty levels and both output-track states, then Codex inspected a stratified 90-render sample (30 per family) without treating that inspection as human review.
- **Reason:** The architecture must prove broad composition correctness statically instead of relying on a small smoke to rediscover individual surface failures.
- **Result:** The first expansion correctly stopped with 20 internal-label detections and 29 repeated-clause detections. The causes were an over-broad identifier substring check and event sentences that lacked sequence identity when equal operations recurred. Word-boundary matching and ordinal event provenance fixed those shared causes. A second inspection identified omitted repeated units in one combined-rate form and a punctuation portability concern; both were corrected centrally. The final expansion hash is `78802a61...6d8e99`: 2,320/2,320 valid renders, 232 distinct render signatures, zero noun, identifier, ordinal, morphology, target, or semantic-coverage failures. All 13 prior defects are blocked, and the final 90-render Codex sample has zero invalid or unnatural findings. The measured 15 exact and 1,192 number-neutral duplicate expansions are expected repeated fixture realizations of the same plans, not accepted-dataset duplicates; the counted pipeline retains unique-signature and duplicate screening.
- **Files changed:** Template contracts, bank definitions, renderer, new composition and expansion modules, focused unit tests, fresh smoke configuration, content-free expansion/inspection summaries, ignored synthetic inspection packets, and this DEVLOG. Mathematical generators, dual verifiers, benchmark evaluator, contamination model/thresholds, dependencies, and sealed-final artifacts are unchanged.
- **Errors or uncertainty:** The static expansion is automated plus Codex inspection, not genuine human review. The high number-neutral count measures deliberate multi-fixture reuse within this non-dataset expansion and does not relax any accepted-candidate gate. No training dataset was persisted.
- **Next action:** Wire provenance into the unchanged acceptance pipeline, create ignored HTML/Markdown user-review packets, then run exactly one fresh 120-attempt smoke and its exact deterministic replay.

## 2026-07-19 00:56:06 -04:00 - Milestone 6B Steps 8-12: fresh smoke, replay, and user packet complete

- **Current step:** Run the one authorized fresh candidate smoke under the unchanged mathematical, contamination, diversity, and quality controls; reproduce it exactly; and prepare genuine user review.
- **Action performed:** Processed exactly 120 fresh deterministic attempts with no replacement: 60 targeted and 60 generic-control candidates, using the required 53 bookkeeping, 34 rate/ratio, and 33 discrete allocation; 40 attempts at each difficulty; and 12 output-contract attempts in each group. Added token provenance before contamination screening. Replayed all 120 attempts with the same seed and wrote ignored JSONL evidence plus ignored Markdown and interactive HTML packets. Codex separately inspected the 120 surfaces without representing that as human review.
- **Reason:** The final bounded smoke must establish technical trust and reproducibility while reserving the decisive natural-language judgment for the user.
- **Result:** The 60-attempt checkpoint was 58 accepted/2 rejected; the final result is 116 accepted/4 rejected (96.6667%). Targeted and generic each accepted 58/60. Bookkeeping accepted 53/53, rates 31/34, and discrete 32/33. Easy/medium/hard accepted 38/40, 39/40, and 39/40. All 24 output-contract attempts passed. The four safe rejections are three latent-program copies and one number-neutral template copy. Primary and independent verifiers succeeded and agreed on all 120; false labels, deterministic language defects, target mismatches, development lexical/semantic rejections, unresolved contamination, exact text duplicates, reused render signatures, and GPU use are zero. Counted and replay decision hashes both equal `f5caa7e8...a254`. Counted/replay runtime was 1.670/1.612 seconds; peak process RSS was 912,838,656 bytes; ignored artifacts occupy 539,386 bytes. The fixed technical gate passes: `TECHNICALLY READY - HUMAN REVIEW PENDING`.
- **Files changed:** Ignored `results/raw/template_bank_smoke_v2/attempts.jsonl`, `replay.jsonl`, `human_review.md`, and `human_review.html`; tracked content-free summary, smoke/provenance/review-packet implementation, and this DEVLOG. No generated question, raw record, benchmark content, model/cache file, dependency, generator, verifier, evaluator, or sealed-final artifact is tracked.
- **Errors or uncertainty:** No execution, verifier, replay, or contamination error occurred. Fifty generated-to-generated similarity cases were recorded for review under the frozen policy but did not override the structural duplicate controls. Codex inspection is not genuine human vetting; user review is still required and may reject wording that deterministic checks accept.
- **Next action:** Run every required formatting, linting, strict typing, test, dependency, whitespace, secret, benchmark-leak, sealed-path, raw/cache, regression, expansion, count, signature, provenance, verifier, contamination, replay, HTML-ignore, file-size, and repository-scope check before committing.

## 2026-07-19 01:00:48 -04:00 - Milestone 6B Step 14: full verification and safety audit passed

- **Current step:** Prove code quality, evidence integrity, benchmark isolation, ignore boundaries, and publication safety before constructing the atomic Git index.
- **Action performed:** Ran Ruff format/lint, strict Mypy, all unit/integration tests, main-environment `pip check`, and `git diff --check`; scanned changed/untracked files for high-confidence secrets; compared them against all 904 development questions using exact and 12-token checks; reviewed protected evaluator/generator/verifier/contamination/dependency paths and sealed-path references; checked raw/cache/environment tracking and ignore rules; validated the 13 regressions, 2,320 static records, 120 counted and replay records, provenance hashes, verifier/target/contamination decisions, signature uniqueness, review packet controls/count/privacy, evidence hashes, and tracked-file sizes.
- **Reason:** Technical readiness and a positive smoke must not publish raw questions, benchmark material, review state, secrets, caches, or evidence that differs from the ignored source records.
- **Result:** Ruff passes; strict Mypy reports no issues in 75 source files; all 291 tests pass; dependencies and whitespace are clean. Secret hits, exact development-question leaks, 12-token development leaks, protected changes, sealed-source references, and tracked raw/cache/environment artifacts are all zero. Static evidence is 2,320/2,320 with zero failure kinds. Counted/replay evidence is 120/120 with 116 accepted, 120 valid provenance hashes, exact decision/provenance replay, zero false labels/disagreements/language defects/target mismatches/unresolved contamination/exact duplicates/signature reuse. The HTML parses to 120 cards, has all three decision controls, local storage, JSON export, and no answer-bearing fields. Twenty intended changed/untracked paths remain; the largest is this DEVLOG at about 310 KiB.
- **Files changed:** This DEVLOG checkpoint only after the reviewed compiler, tests, configuration, content-free results, and documentation. Ignored expansion/smoke/review artifacts remain local and untracked.
- **Errors or uncertainty:** No verification error occurred. Expected LF-to-CRLF working-copy notices are non-fatal. The technical gate does not replace the still-pending genuine user review.
- **Next action:** Repeat final format/lint/type/test/whitespace checks after this checkpoint, fetch `origin/main` without integrating changes, require the approved remote tip, stage exactly the 20 reviewed paths, scan the Git index, then commit and push without force.

## 2026-07-19 01:01:55 -04:00 - Milestone 6B Step 14: staged publication snapshot verified

- **Current step:** Prove that the exact Git index matches the verified, content-free Milestone 6B scope before publication.
- **Action performed:** Fetched `origin/main` without integration and confirmed the approved local/remote tip and `0/0` divergence; explicitly staged the 20 reviewed paths; checked index whitespace, path scope, forbidden raw/cache/model/environment paths, high-confidence secrets, unstaged/untracked files, and staged blob sizes.
- **Reason:** The pushed commit must be a fast-forward containing only the compiler, tests, frozen configuration, content-free summaries, and documentation—not local review packets or raw synthetic questions.
- **Result:** Local and remote remain at `822596a6...a833`. The index has exactly 20 intended paths, zero unstaged files, zero untracked files, zero forbidden paths, zero secret hits, and no blob at or above 1 MiB. The largest staged blob is this DEVLOG at about 312 KiB. The previously completed development-leak, sealed-path, evidence, replay, provenance, packet-privacy, and protected-scope checks apply unchanged because there are no unstaged bytes.
- **Files changed:** This DEVLOG checkpoint only after staging; it will be restaged and final index whitespace/status checks repeated.
- **Errors or uncertainty:** None. LF-to-CRLF notices describe Git's configured working-copy conversion and do not indicate staged-content corruption.
- **Next action:** Restage this checkpoint, repeat final index/status and remote-tip checks, create one atomic Milestone 6B commit, push without force, then verify equal local/remote tips, `0/0` divergence, clean worktree, and ignored local packets.

## 2026-07-19 17:35:17 -04:00 - Milestone 6C-R Steps 1-3: genuine review verified and rejection patterns aggregated

- **Current step:** Import the user's completed Milestone 6B language review, prove packet identity, and classify the rejected wording before changing the bank.
- **Action performed:** Verified clean synchronized `main` at `2650f6adf63a8815403f472b925863890cade4e2`; checked the review export only at `C:\Users\Admin\Downloads\foundry-template-bank-smoke-v2-review.json`; matched its SHA-256, schema, review kind, candidate IDs, and counts against the ignored Milestone 6B packet and attempt records; aggregated decisions by group, family, difficulty, output-contract state, template, and sentence plan; inspected the 60 rejected surfaces with their template metadata while preserving the user's decisions exactly.
- **Reason:** The previous automatic gate cannot authorize full generation until the genuine review is imported, and replacements must address recurring human-rejected language rather than override or reinterpret individual decisions.
- **Result:** Review SHA-256 is `564a8ca584984ee7a0b997eec4a6a6f377308c869b62cf65ebeef5375cef0791`; all 120 IDs are unique and match the prior packet; decisions are 60 approve, 60 reject, and 0 unsure. Targeted approved 26/60 and generic 34/60. Bookkeeping approved 15/53, rate/ratio 26/34, and discrete 19/33. Easy/medium/hard approved 22/40, 17/40, and 21/40. Output-contract attempts approved 13/24; other attempts approved 47/96. `direct_relation` and `constraint_sequence` were approval-only; `chronological_active` and `condition_fronted` were rejection-only. The other eight plan IDs were mixed and require compatibility-specific repair. The previous full-generation language gate is formally failed.
- **Files changed:** This DEVLOG entry only. The genuine review JSON, prior questions, prior packets, and raw attempt records remain ignored and unchanged. No evaluator, mathematical generator, canonical label, dual verifier, contamination control, dependency, benchmark manifest, or sealed-final artifact changed.
- **Errors or uncertainty:** The export contains no user defect labels. Root causes are therefore explicitly analysis inferred from recurring rejected wording and metadata, not user-supplied explanations. No sealed-final path was accessed.
- **Next action:** Freeze a content-free approved/quarantine manifest tied to the review hash, reauthor the defective sentence-plan paths in worksheet-quality English, and add regression tests for every recurring rejected pattern.

## 2026-07-19 17:46:04 -04:00 - Milestone 6C-R Steps 3-5: review manifest, language repair, and static gate passed

- **Current step:** Freeze the genuine-review evidence, replace every quarantined plan family, and prove the full revised bank is mechanically and linguistically sound before a counted smoke.
- **Action performed:** Imported the ignored genuine review through a strict hash- and candidate-identity validator; wrote content-free review summary and quarantine manifests; versioned the bank and renderer as v3; replaced all 12 affected sentence-plan families with direct, worksheet-quality constructions while retaining the two consistently accepted plan families; added review-import, quarantine, wording, and smoke-contract tests; expanded all 232 sentence plans across 10 deterministic fixtures; and Codex inspected a stratified 90-render sample (30 per category). The initial inspection caught one shared duplicated-quantifier construction (`In total, a total of ...`), so its complete-package composition rule was corrected and covered by a family-wide regression before the expansion and inspection were repeated.
- **Reason:** The 60 rejected questions must remain immutable historical evidence, while only reviewed sentence-plan paths can be replaced. The static gate prevents a fresh smoke from spending candidate budget on a systematic wording defect.
- **Result:** The tracked review manifest is tied to review SHA-256 `564a8ca...791`; it records exactly 60 approved and 60 quarantined candidates without question text. Focused tests pass (42/42). The final expansion hash is `fc3c6a...cc6c`: 2,320/2,320 valid renders, 232 distinct render signatures, zero deterministic validation failures, zero exact sentence-plan duplicates, and zero number-neutral sentence-plan duplicates. After the shared quantifier repair, the final 90-render Codex inspection found zero invalid or unnatural questions and no systematic composition defect. This is Codex inspection, not genuine human review.
- **Files changed:** Review-import module; template contracts, bank, renderer, expansion and smoke orchestration; v3 smoke configuration; focused unit/integration tests; content-free review/static evidence; and this DEVLOG. Ignored review input, prior raw artifacts, generators, canonical labels, verifiers, evaluator, benchmark manifests, dependencies, contamination thresholds, and sealed-final artifacts are unchanged.
- **Errors or uncertainty:** The first static inspection failed as designed on one duplicated quantifier; the architectural fix and rerun resolved it. Expansion-level duplicate counts still include deliberate reuse of each sentence plan across ten fixture variants and are not accepted-candidate duplicates; sentence-plan identity metrics are zero. Human review of the fresh v3 packet remains pending.
- **Next action:** Run exactly one fresh 120-attempt v3 smoke with no replacements, require exact deterministic replay, and stop the pipeline if the unchanged automatic gate fails.

## 2026-07-19 17:49:50 -04:00 - Milestone 6C-R Steps 8-10: fresh smoke replayed exactly; packet gate failed

- **Current step:** Execute the single authorized 120-attempt v3 smoke, reproduce every decision, and apply the fixed 110/120 admission gate before creating review material.
- **Action performed:** Ran exactly 120 fresh attempts with seed `foundry-template-bank-smoke-master-20260719-v3`, no replacements, the required targeted/generic and category allocations, 40 attempts at each difficulty, and 24 output-contract attempts. Replayed the same attempts exactly. After confirming the milestone contract's 110 threshold, removed the prematurely emitted HTML/Markdown packet from the failed run and changed the workflow to create user-review packets only after a passing technical gate; added a focused fail-closed test.
- **Reason:** The milestone explicitly prohibits packet creation below 110 automatic passes. A failed result must remain a negative experiment rather than becoming a review packet or being improved through a second seed, replacements, relaxed screens, or post-result tuning.
- **Result:** At 60 attempts, 56 were accepted and 4 rejected. Final automatic acceptance is 104/120 (86.6667%): targeted 56/60 and generic 48/60; bookkeeping 39/53, rate/ratio 33/34, and discrete 32/33; easy/medium/hard 32/40, 34/40, and 38/40; output-contract 21/24. Rejections are 15 number-neutral template copies and one latent-program copy. All 120 primary and independent verifications passed and agreed; false labels, deterministic language defects, target mismatches, benchmark lexical/semantic rejections, unresolved contamination, exact text duplicates, and reused render signatures are zero. Counted and replay decision hashes both equal `44cd52653ec1e45a3d603f3858c9051e6d48a53d6fdc3417ba9989563c171e0f`. Counted/replay runtime is 1.612/1.530 seconds; peak RSS is 923,930,624 bytes; ignored attempts/replay total 380,785 bytes; GPU use is zero. The technical gate fails solely because 104 is below 110; no v3 human-review, Codex-audit, or assisted-review packet exists.
- **Files changed:** Content-free v3 smoke summary; smoke packet gating and its focused test; ignored attempts/replay evidence; and this DEVLOG. The two packet files emitted before the contract check were removed. No raw result is tracked.
- **Errors or uncertainty:** The smoke implementation originally wrote review packets before computing the gate; that fail-open sequencing defect is now corrected. The static sentence-plan diversity measure did not predict 15 runtime number-neutral collisions, chiefly among repeated bookkeeping surface structures. Fixing selection/composition diversity would require separate approval and a new smoke; no policy, threshold, seed, template, or result was changed after observing this run.
- **Next action:** Document the failed technical gate and diversity blocker, run the full quality and safety suite, then publish the atomic negative-result milestone without creating a second-review packet.

## 2026-07-19 17:54:42 -04:00 - Milestone 6C-R Step 11: verification and safety suite passed

- **Current step:** Verify code quality, evidence reproducibility, review identity, benchmark isolation, fail-closed packet behavior, and publication scope before creating the atomic commit.
- **Action performed:** Ran Ruff formatting/check and lint, strict Mypy, all tests, main-environment `pip check`, and `git diff --check`; reran the genuine-review importer and 2,320-render expansion; checked review hash/schema/identity/counts, approved/quarantine manifest counts and content-free keys, 120 counted/replay record identities, allocations, verifier agreement, rejection categories, decision hashes, signature/accepted-text/accepted-latent uniqueness, and failed-gate packet suppression. Scanned all 21 changed/untracked files for high-confidence secrets and exact/12-token matches to the 904 development questions; reviewed protected evaluator/generator/verifier/contamination/config/dependency paths, sealed-path changes, tracked raw/cache/environment paths, ignored evidence, and candidate sizes.
- **Reason:** A failed technical gate still needs a complete, reproducible, content-free publication record, and it must not leak the genuine review, rendered questions, benchmark content, raw evidence, or a misleading review packet.
- **Result:** Ruff reports 115 files formatted and lint-clean; strict Mypy reports no issues in 76 source files; all 296 tests pass; dependencies and whitespace are clean. Review SHA/count/identity checks pass; the rerun preserves manifest SHA `fa4cf33a...5798`, review-summary SHA `69feacf5...a6d`, and expansion SHA `fc3c6a16...cc6c`. Static expansion is 2,320/2,320 with zero failure kinds and zero exact/number-neutral duplicate sentence plans. Counted/replay evidence is 120/120 with exact decision hash `44cd5265...1e0f`, 120/120 verifier agreement, and the expected 104 accepted plus 15 numeric-template and one latent-copy rejection. Secret hits, exact development-question hits, 12-token development hits, protected changes, sealed-path changes, and tracked raw/cache/environment artifacts are zero. Both raw v3 evidence files are ignored; all four forbidden v3 review-packet/audit files are absent. The largest candidate is this DEVLOG at 323,041 bytes, below 1 MiB.
- **Files changed:** This DEVLOG checkpoint only after the reviewed code/config/tests/content-free summaries and documentation. Ignored review input and raw attempts/replay remain outside the publication set.
- **Errors or uncertainty:** One initial review-import rerun used obsolete option names and exited before reading/writing; the corrected declared interface passed. One initial ignore command incorrectly passed an external Downloads path to Git; the corrected check proves the review is outside the repository and both in-repository evidence files are ignored. Neither command affected artifacts or results. Expected LF-to-CRLF notices are non-fatal.
- **Next action:** Repeat final format/lint/test-independent whitespace checks after this entry, fetch `origin/main` without integration, require the approved remote tip, stage exactly the 21 reviewed paths, inspect the index, then commit and push without force.

## 2026-07-19 17:55:49 -04:00 - Milestone 6C-R Step 11: final gate hardening and re-verification passed

- **Current step:** Close the last fail-closed admission gap before staging the verified publication snapshot.
- **Action performed:** Wired the already reported primary-verifier, independent-verifier, target-mismatch, benchmark-lexical, benchmark-semantic-reject, and benchmark-semantic-review counts directly into the technical-gate Boolean instead of relying only on their downstream summary fields. Reran Ruff format/lint, strict Mypy, all tests, dependency integrity, and whitespace checks after the change. Fetched `origin/main` without integration.
- **Reason:** A future passing count must not create a review packet if any separately stated correctness or benchmark-isolation condition fails, even though the current run has zero in every such field.
- **Result:** The measured v3 result and failed gate are unchanged. Ruff passes; strict Mypy reports no issues in 76 source files; all 296 tests pass; dependencies and whitespace are clean. Local and remote remain at `2650f6adf63a8815403f472b925863890cade4e2`, divergence is `0/0`, and the worktree contains exactly the 21 intended milestone paths before staging.
- **Files changed:** Smoke gate orchestration and this DEVLOG; Ruff mechanically reformatted the source. No config, policy, template, generator, verifier, evaluator, benchmark, raw evidence, or result changed.
- **Errors or uncertainty:** None. The hardening adds only stricter future rejection conditions and cannot convert the present failure into a pass.
- **Next action:** Stage exactly the 21 reviewed paths, run index scope/whitespace/secret/content/size/evidence checks, commit with the approved message, and push only if the remote tip is still unchanged.

## 2026-07-19 17:56:34 -04:00 - Milestone 6C-R Step 11: staged publication snapshot verified

- **Current step:** Prove the exact Git index contains only the verified, content-free Milestone 6C-R scope before committing.
- **Action performed:** Staged the 21 explicit reviewed paths; ran cached whitespace, path-scope, forbidden raw/review/model/cache/environment, unstaged/untracked, high-confidence secret, exact development-question, 12-token development-question, summary/manifest, and blob-size checks against the Git index.
- **Reason:** Publication must not include the genuine review export, any rendered question, raw candidate/replay evidence, failed-run review page, secret, benchmark content, or local cache.
- **Result:** The index has exactly 21 intended paths, zero forbidden paths, zero unstaged files, zero untracked files, zero secret hits, zero exact development-question hits, and zero 12-token development hits. Staged evidence records 60 approvals/60 quarantines, 104/120 automatic accepts, a failed fixed gate, and no packet. The largest staged blob is this DEVLOG at 327,781 bytes, below 1 MiB.
- **Files changed:** This DEVLOG checkpoint only after staging; it will be restaged and the final index/remote checks repeated.
- **Errors or uncertainty:** None. Expected Windows line-ending notices do not indicate content corruption.
- **Next action:** Restage this checkpoint, verify the remote still equals `2650f6a...e2`, create the atomic commit with the exact approved message, push without force, and verify synchronized clean state.

## 2026-07-19 18:21:26 -04:00 - Milestone 6D Step 1: synchronized frozen starting state verified

- **Current step:** Verify repository identity, published state, frozen evidence, and protected implementation boundaries before diagnosing or changing runtime allocation.
- **Action performed:** Temporarily prepended `C:\Program Files\Git\cmd` to this PowerShell process; fetched `origin/main` without integration; checked repository root, branch, local/remote tips, divergence, worktree state, the genuine review SHA-256, presence and ignore status of both Milestone 6C-R raw evidence files, and diffs across the repaired template bank, mathematical generators, dual verifiers, benchmark evaluator/configuration, semantic and contamination configuration.
- **Reason:** Milestone 6D may change only deterministic runtime allocation and scheduling. Any drift in the language bank, mathematics, evaluation, review evidence, or safety controls would invalidate the controlled experiment.
- **Result:** Repository root is `C:/Users/Admin/Projects/Foundry`; branch is `main`; local and `origin/main` both equal `457dc5e65e6828380d91c2b3c930ccf1c1ad98db`; divergence is `0/0`; the worktree is clean. Review SHA-256 remains `564a8ca584984ee7a0b997eec4a6a6f377308c869b62cf65ebeef5375cef0791`. Both `template_bank_smoke_v3` attempts/replay files exist and are ignored. Protected-path diff count is zero. No sealed-final path was opened, hashed, inspected, or compared.
- **Files changed:** This DEVLOG entry only. No source, configuration, raw evidence, model, dataset, template, generator, verifier, evaluator, contamination control, or Git configuration changed.
- **Errors or uncertainty:** None.
- **Next action:** Diagnose all 15 number-neutral collisions and the one latent-program collision, recording only content-free identifiers, hashes, collision partners, stages, and causes.

## 2026-07-19 18:26:41 -04:00 - Milestone 6D Step 2: all 16 runtime collisions diagnosed

- **Current step:** Resolve every Milestone 6C-R duplicate rejection to its earlier candidate and distinguish allocation defects from capacity, label, language, and contamination failures.
- **Action performed:** Read the ignored 120-record v3 smoke in attempt order; recomputed each number-neutral SHA-256; mapped template IDs to content-free semantic frames; and joined each rejected record to the first earlier record with the same number-neutral or latent-program hash. Recorded candidate prefixes, group/category/difficulty/output state, frame/template/plan IDs, full signatures/hashes, earlier candidate prefixes, and collision stages in memory only.
- **Reason:** Collision-free scheduling must address the actual identity reused by the old allocator and must not misdiagnose conservative duplicate rejection as a language or mathematical defect.
- **Result:** All 15 number-neutral rejections selected the same sentence-plan ID as an earlier candidate but under a different semantic-frame/template ID; their generated scenario lexemes and clause structure then collapsed to the same number-neutral question. Four occurred in targeted bookkeeping, ten in generic bookkeeping, and one in generic weighted averages. The pattern is allocator imbalance plus repeated plan/scenario realization, not different plans accidentally paraphrasing each other and not insufficient compatibility for the 120-slot smoke. The lone latent rejection was discrete attempt 117, whose latent-program hash matched attempt 105 despite a different plan; it is a seed-schedule collision. Every collision has one deterministic earlier partner and exact stage (`numeric_template_copy` or `latent_program_copy`).
- **Files changed:** This DEVLOG entry only. Complete questions and raw records remain ignored and unchanged.
- **Errors or uncertainty:** One supplemental read-only formatting command had a quoting syntax error and exited before reading or writing evidence; its delimiter-based replacement succeeded. The review export supplied no collision labels; classifications derive from exact signatures and ordered partner joins.
- **Next action:** Formalize the in-memory combination-space enumeration, measure active plan-level, domain-aware, and number-neutral capacities across all required strata, and apply the 8,000-example plus 125% cross-dataset capacity gate before writing allocator code.

## 2026-07-19 18:29:43 -04:00 - Milestone 6D Step 3: full-generation capacity gate failed

- **Current step:** Enumerate the repaired bank's finite runtime combination space and compare every active uniqueness layer with the combined targeted/generic 8,000-example quotas and fixed 125% attempt pools.
- **Action performed:** Added a typed, content-free capacity auditor. It enumerates exact generator structural periods (bookkeeping 240, rates 25, discrete 80), all three difficulties, compatible semantic frames/templates/plans/domains, both reachable bookkeeping positive-state-guard branches, and existing plan-level, domain-aware, semantic-frame, and number-neutral signatures. It treats output-contract enabled/disabled questions as one shared uniqueness pool, diagnoses all 16 source collisions, computes group/category/difficulty/output quotas, and writes no question corpus. Added focused deterministic count, stop-boundary, collision-inventory, and content-free tests; generated the tracked aggregate audit.
- **Reason:** The milestone forbids implementing an allocator or spending another 120-attempt smoke unless every future cross-dataset stratum can satisfy both its accepted quota and its predeclared 125% pool under unchanged duplicate controls.
- **Result:** Capacity audit SHA-256 is `8b921822bf10da964cf357cf3851084a2e0bd15ffc5dc549a85e04f84c9ccd7b`. The future accepted quotas total 8,000 and require 10,003 fixed attempts. Bookkeeping requires 4,418 attempts but has 72 active plan signatures, 1,728 domain-aware signatures, and 768 unique number-neutral surfaces. Rates require 2,834 but have 80 active, 400 domain-aware, and 88 number-neutral signatures. Discrete requires 2,751 but has 80 active, 1,600 domain-aware, and 320 number-neutral signatures. Under all current controls, combined limiting capacity is 232/10,003 (2.3193%). Every category, difficulty, and output-contract stratum fails. Even replacing the active plan-level identity with the already-existing domain-aware signature would not pass because the number-neutral pools remain only 768/88/320.
- **Files changed:** New capacity-audit module, focused unit tests, content-free `template_bank_v3_capacity_audit.json`, and this DEVLOG. No rendered-question dataset was persisted; source raw attempts remain ignored.
- **Errors or uncertainty:** The audit's first style pass found one overlength content-free string, then strict Mypy found one redundant cast; both implementation-only issues were fixed before focused tests. Ruff, strict Mypy, and both focused tests pass. Capacity counts are exact for the existing finite structural cycles; numeric values do not enlarge a number-neutral surface pool.
- **Next action:** Apply the mandatory stop rule: do not implement allocation or latent scheduling, do not freeze a 120 schedule, do not run a smoke, and do not create a packet. Document the required future expansion in independently reviewed sentence plans, scenario/lexical domains, and structural surfaces before any allocator milestone can be reconsidered.

## 2026-07-19 18:32:28 -04:00 - Milestone 6D Step 4: failed capacity boundary documented

- **Current step:** Freeze the content-free collision diagnosis, exact capacity evidence, stop-rule decision, and next authorization boundary across project records.
- **Action performed:** Updated the project plan, decision ledger, learning notes, synthesis design, template-bank design, experiment ledger, and this DEVLOG. Recorded future accepted/attempt quotas, active/domain-aware/number-neutral capacity by family, audit/source hashes, the absence of an allocator/schedule/smoke/packet, and minimum structural expansion needs.
- **Reason:** The failed preflight is the milestone result. Publishing only a vague blocker would make it easy to repeat a small-smoke scheduling change that cannot satisfy the actual 8,000-example cross-dataset contract.
- **Result:** Documentation now consistently states that 8,000 accepted examples require 10,003 attempts while the existing number-neutral pools are only 768 bookkeeping, 88 rate/ratio, and 320 discrete. Required minimum number-neutral additions are 3,650, 2,746, and 2,431 respectively. Duplicate and contamination policies remain unchanged.
- **Files changed:** Seven required documentation/experiment files, the capacity auditor, its focused tests, and one aggregate content-free result. No bank wording, template, generator, verifier, evaluator, policy, threshold, manifest, raw evidence, schedule, candidate, packet, dataset, or model artifact changed.
- **Errors or uncertainty:** Capacity counts are exact for the current finite bank and generator cycles. They do not estimate the authoring effort or guarantee that future language additions will survive human review and contamination screening.
- **Next action:** Run the complete formatting, typing, test, dependency, content-leak, sealed-final, raw-artifact, protected-path, capacity-consistency, size, and Git-state verification suite. Commit and push only if every check passes and the remote tip remains unchanged.

## 2026-07-19 18:36:30 -04:00 - Milestone 6D Step 5: full verification passed

- **Current step:** Verify the failed capacity result is exact, deterministic, content-free, correctly scoped, and safe to publish.
- **Action performed:** Ran Ruff format/check and lint, strict Mypy, all unit/integration tests, main-environment `pip check`, and `git diff --check`; reran the capacity auditor byte-for-byte; validated source/audit hashes, collision partners, category/difficulty/output strata, failed gate, and all false implementation flags; scanned the ten candidate files for high-confidence secrets and exact/12-token matches to all 904 development questions; reviewed protected bank/renderer/generator/verifier/evaluator/policy/config paths, sealed-path status, raw/cache/environment tracking, raw-evidence ignore rules, forbidden schedule/packet artifacts, and candidate sizes.
- **Reason:** A negative preflight must be as reproducible and safe as a positive run, and the stop boundary must be independently proven before creating the publication commit.
- **Result:** Ruff passes; strict Mypy reports no issues in 77 source files; all 298 tests pass in 12.67 seconds; dependencies and whitespace are clean. Capacity rerun is byte-identical with file SHA-256 `bf107a9fb98c5c0a4182d5755ecf73d97a62b328a3e9d1df1c4caa37a5166553` and internal audit SHA-256 `8b921822...ccd7b`. The source-attempt hash matches; 16 collision diagnoses and every failed stratum validate. Secret hits, exact development-question hits, 12-token development hits, protected-path changes, sealed-path changes, forbidden new schedule/packet/raw files, and tracked raw/cache/environment artifacts are all zero. Both source evidence files remain ignored. The largest candidate is this DEVLOG at 339,060 bytes, below 1 MiB.
- **Files changed:** This verification checkpoint only after the reviewed auditor, test, aggregate, and documentation changes. No measured evidence, template bank, renderer, generator, verifier, evaluator, policy, threshold, manifest, dependency, raw record, packet, dataset, model artifact, or sealed-final file changed.
- **Errors or uncertainty:** One ad hoc assertion initially expected different content-free root-cause labels and was corrected after reading the audit schema; the formal tests and evidence were unaffected. One combined shell scan was rejected before execution by command policy, so its read-only checks were rerun as smaller commands and passed. Offline development-data loading used the exact pinned local cache only.
- **Next action:** Fetch `origin/main` without integration, require the approved starting tip and `0/0` divergence, stage exactly the ten verified paths, repeat index scope/whitespace/leak/size/evidence checks, create the exact atomic commit, push without force, and verify synchronized clean state.

## 2026-07-19 18:37:08 -04:00 - Milestone 6D Step 6: publication index verified

- **Current step:** Prove the exact Git index contains only the verified Milestone 6D negative-result scope before committing.
- **Action performed:** Fetched `origin/main` without integration and verified the published tip was unchanged; staged the ten explicit audit, test, aggregate, and documentation paths; ran cached whitespace, expected/extra/forbidden path, unstaged/untracked, high-confidence secret, blob-size, collision-inventory, failed-gate, and stop-flag checks against the index. The completed working-tree development-content and protected-path scans remain applicable because there were no unstaged or untracked bytes.
- **Reason:** The requested commit must publish the reproducible capacity blocker without accidentally including raw questions, source attempts, schedules, packets, benchmark content, caches, secrets, or unauthorized implementation.
- **Result:** Local and remote remain at `457dc5e...98db` with divergence `0/0`. The index has exactly ten intended paths, zero missing/extra/forbidden paths, zero unstaged or untracked files, zero secret hits, clean whitespace, and no blob at or above 1 MiB. Staged evidence records 15 number-neutral plus one latent collision, failed capacity, and five false allocator/schedule/smoke/packet flags. The largest staged blob is this DEVLOG at 341,612 bytes.
- **Files changed:** This DEVLOG checkpoint only after staging; it will be restaged and the final index and remote-tip checks repeated.
- **Errors or uncertainty:** None. Expected Windows LF-to-CRLF working-copy notices do not change staged semantic content.
- **Next action:** Restage this checkpoint, verify the final index and remote tip, create the exact commit `synthesis: enforce collision-free runtime allocation`, push without force, and confirm synchronized clean local/remote state.

## 2026-07-19 18:37:38 -04:00 - Milestone 6D Step 7: commit identity blocker resolved without configuration

- **Current step:** Create the verified atomic commit without changing local or global Git configuration.
- **Action performed:** Rechecked the ten-path index, clean unstaged/untracked state, unchanged local/remote tip, and `0/0` divergence; attempted the exact commit. Git rejected it before creating a commit because this shell had no author identity. Read the published parent commit's author and committer name/email for process-local reuse only.
- **Reason:** The milestone requires a commit and push but explicitly forbids Git configuration changes. Git's supported environment variables provide the same published project identity for one process without persisting configuration.
- **Result:** No commit or file mutation resulted from the failed Git attempt. The index remains intact at the approved starting commit. Parent identity is `KavinSivaharan <140466612+KavinSivaharan@users.noreply.github.com>` for both author and committer.
- **Files changed:** This DEVLOG error record only; no Git config, source, evidence, raw artifact, or remote state changed.
- **Errors or uncertainty:** Exact Git error: `Author identity unknown` followed by `fatal: unable to auto-detect email address (got 'Admin@DESKTOP-384F1AQ.(none)')`.
- **Next action:** Restage this record, set `GIT_AUTHOR_*` and `GIT_COMMITTER_*` only in the commit process, retry the same exact message, then push and verify synchronization.

## 2026-07-19 18:40:22 -04:00 - Milestone 6D Step 8: fresh-clone test dependency removed

- **Current step:** Correct a post-push reproducibility defect without amending or rewriting the published Milestone 6D commit.
- **Action performed:** Audited the new unit test's inputs after publication and found that both tests directly opened ignored local v3 attempts. Replaced that dependency with an original, hand-authored temporary collision fixture containing 15 deterministic number-neutral copies and one latent copy. The production auditor still reads the genuine ignored evidence when explicitly run; its tracked aggregate and measured hashes remain unchanged.
- **Reason:** A unit test must pass in a fresh clone where ignored raw evidence is intentionally absent. Local success alone was insufficient proof because this desktop retained the source file.
- **Result:** Focused tests pass with the hermetic fixture and no benchmark content. Ruff format passes; Ruff lint initially reported only import ordering, which was corrected directly. Strict Mypy remains clean in 77 source files. No production source, result, capacity count, decision, raw evidence, or published commit was changed or rewritten.
- **Files changed:** The capacity unit test and this DEVLOG follow-up only.
- **Errors or uncertainty:** The already-pushed `fa696d6` commit remains immutable. A small follow-up commit is required so `origin/main` is genuinely fresh-clone safe; this is disclosed rather than hidden through amend or force-push.
- **Next action:** Run the full suite and safety checks again, commit the two-file hermetic-test correction, push normally, and report both the required Milestone 6D commit and the final synchronized follow-up tip.

## 2026-07-19 18:41:08 -04:00 - Milestone 6D Step 9: hermetic follow-up verified

- **Current step:** Verify and publish the fresh-clone-safe test correction without altering Milestone 6D evidence or production behavior.
- **Action performed:** Ran Ruff format/check and lint, strict Mypy, all tests, `pip check`, `git diff --check`, high-confidence secret scanning, exact and 12-token development-content scans, production/sealed/raw/cache scope checks, and audit-file hash verification over the two-file follow-up.
- **Reason:** The follow-up must prove both that the test no longer depends on ignored evidence and that the correction does not change the measured capacity result.
- **Result:** Ruff passes; strict Mypy remains clean in 77 source files; all 298 tests pass in 12.59 seconds; dependencies and whitespace are clean. Secret, exact-development, 12-token-development, production-source, sealed-path, and raw/cache hits are zero. The aggregate audit file remains byte-identical at SHA-256 `bf107a9f...6553`.
- **Files changed:** Only the hermetic capacity unit test and this DEVLOG. No production source, audit result, documentation decision, raw evidence, benchmark artifact, schedule, packet, or model/data artifact changed.
- **Errors or uncertainty:** None. The fixture is original, temporary, and content-free; it exists only while the unit test runs.
- **Next action:** Commit this disclosed two-file follow-up without amending history, push normally, and verify local `main` and `origin/main` are synchronized and clean.

## 2026-07-19 18:51:01 -04:00 - Milestone 6E Step 1: synchronized frozen starting state verified

- **Current step:** Verify repository identity, published state, capacity/review evidence, and all protected implementation boundaries before correcting generated-to-generated reuse policy.
- **Action performed:** Temporarily prepended `C:\Program Files\Git\cmd` to this PowerShell process; fetched `origin/main` without integration; checked root, branch, local/remote tips, divergence, and worktree; verified the Milestone 6D capacity file/internal hashes and ignored v3 source evidence; located the genuine v2 review export in Downloads and matched its SHA-256 to the tracked review contract; reviewed diffs across the repaired bank/renderer, mathematical generators, dual verifiers, benchmark evaluator/configuration, and contamination model/threshold files.
- **Reason:** Milestone 6E may change only generated-to-generated identity/reuse policy and balanced runtime allocation. Drift in human review, wording, mathematics, labels, evaluation, or benchmark contamination would invalidate the controlled correction.
- **Result:** Repository is `C:/Users/Admin/Projects/Foundry` on `main`; local and `origin/main` both equal `00a419b61e43594736abe1c808aab33ff9156cf2`; divergence is `0/0`; worktree was clean. Milestone 6D file SHA-256 remains `bf107a9f...6553` and internal audit SHA-256 remains `8b921822...ccd7b`. Genuine review SHA-256 remains `564a8ca5...0791`. Both v3 source records remain ignored. Protected-path and sealed-path change counts are zero; no sealed-final file was opened, hashed, inspected, or compared.
- **Files changed:** This DEVLOG entry only. No policy, template, renderer, generator, verifier, evaluator, config, raw evidence, packet, dataset, model artifact, dependency, or Git configuration changed.
- **Errors or uncertainty:** The first review-path probe used an incorrect in-repository filename and returned no hash; the export was then located read-only at its existing Downloads path and matched the tracked contract exactly. No state changed during the failed probe.
- **Next action:** Formally separate exact-question, latent-program, structural-problem, and surface-template identities; derive bounded caps from frozen quotas; and calibrate no more than three policies on original fixtures before inspecting new smoke outputs.

## 2026-07-19 19:00:47 -04:00 - Milestone 6E Step 2: bounded-reuse policy calibrated and frozen

- **Current step:** Separate exact-question, latent-program, structural-problem, and surface-template identity, then select a quota-derived generated-to-generated reuse policy before any new smoke output exists.
- **Action performed:** Added a typed policy/calibration contract, three predeclared policy candidates, and 14 original pairwise fixtures spanning exact copies, number swaps, distinct programs under one reviewed plan, cross-dataset shared wording, latent copies, close paraphrases, and usage-cap violations. Derived per-dataset/family attempt and acceptance caps mechanically with `ceil(1.25 * quota / active identities)` and retained unchanged development-contamination metadata for MiniLM revision `1110a243...b4d41` at 0.75 review and 0.82 rejection.
- **Reason:** A reviewed sentence plan is language machinery, not an example identity. Harmless controlled reuse must be distinguished from exact-question or latent-program duplication without weakening benchmark-contamination screening or choosing policy after observing a new smoke.
- **Result:** `bounded-balanced-template-reuse-v1` matches all 14/14 fixtures; its policy SHA-256 is `66443bc8...25f0`. The legacy one-use alternative matches 12/14 because it rejects harmless number-neutral reuse. The permissive exact/latent-only alternative matches 11/14 because it misses close-paraphrase review and cap violations. Fixture-set SHA-256 is `2a829eea...419b`; calibration SHA-256 is `fd731501...2693`. The selected policy allows a reviewed plan or number-neutral signature to recur only while exact question and latent hashes remain globally unique and every predeclared usage cap remains unexceeded.
- **Files changed:** New bounded-reuse fixture/config contracts, typed audit module, focused hermetic unit tests, ignored content-free calibration/audit outputs, and this DEVLOG. No template wording, generator, verifier, evaluator, benchmark-contamination policy, raw question, dataset, or model artifact changed.
- **Errors or uncertainty:** Initial Ruff found only formatting/line-length and one unused loop-name issue; strict Mypy found one `Any` return. Those implementation issues were corrected before focused tests. Original fixtures contain no benchmark-derived content.
- **Next action:** Apply the selected policy to the frozen 8,000-example quotas, enumerate verified unique latent-program supply, and require the full 10,003 fixed-attempt pool to pass before implementing an allocator.

## 2026-07-19 19:00:47 -04:00 - Milestone 6E Step 3: revised full-generation capacity gate failed

- **Current step:** Recalculate full-generation capacity under bounded plan reuse, balanced frame/target/difficulty/output caps, globally unique latent programs, and targeted/generic plus train/validation isolation.
- **Action performed:** Derived exact caps for every dataset/family, enumerated finite generator modes where possible, and ran a fixed 20,000-candidate-per-family constructive probe using SHA-256-derived seeds. Every counted available latent program was generated by the unchanged procedural generator and checked through the unchanged primary and independent verifiers; no program corpus was persisted. Compared bounded identity and latent supply with all accepted quotas and their fixed 125% attempt pools.
- **Reason:** Removing the erroneous one-use surface restriction does not prove that the mathematical generators can supply enough independently instantiated programs. The approved gate requires both bounded language reuse and 10,003 globally unique, deterministic attempts.
- **Result:** Surface identity capacity passes, and bookkeeping passes with 5,524 bounded programs for 4,418 attempts. Rate/ratio fails with 1,632 balanced unique programs for 2,834 attempts, a 1,202 shortfall and 57.5865% ratio. Its finite modes provide only 96 rate-total, 336 ratio-scale, 104 percentage, and 384 combined-rate programs; weighted average contributes 712 under its balanced frame cap. Discrete fails with 2,073 for 2,751, a 678 shortfall and 75.3544% ratio: two-type and complete-package modes each contribute 865 under balanced frame caps, while equal-distribution and dual-capacity supply only 253 and 90. The overall capacity gate is false; `full_generation_feasible`, allocator, schedule, smoke, replay, and packet flags are all false. Internal audit SHA-256 is `1a40db7b...1129`; the content-free audit file SHA-256 is `26cc9c37...402d`.
- **Files changed:** Content-free audit evidence and this DEVLOG in addition to the policy/config/test files above. No allocator, schedule, candidate smoke, replay, v3 packet, raw generated question, benchmark record, or training artifact was created.
- **Errors or uncertainty:** The 20,000-seed probe is constructive evidence rather than an infinite upper bound for weighted-average, two-type, and bookkeeping modes; those modes already exceed their applicable balance caps. The smaller rate and discrete modes use exact finite bounds. The failure is therefore not caused by insufficient probing and cannot be repaired by number substitution alone while the current parameter ranges and frame-balance caps remain frozen.
- **Next action:** Apply the mandatory stop boundary. Document the selected reuse policy and the narrower remaining latent-program-capacity blocker, run all verification and safety checks, then commit and push the negative result. Do not implement an allocator or issue a second review packet.

## 2026-07-19 19:03:59 -04:00 - Milestone 6E Step 4: negative gate result documented

- **Current step:** Freeze the corrected identity model, selected policy, exact caps, capacity evidence, stop boundary, and next authorization decision across project records.
- **Action performed:** Updated the project plan, decision ledger, learning notes, synthesis design, template-bank design, experiment ledger, and this DEVLOG. Recorded all four identity layers; principal per-dataset/family plan, number-neutral, and frame caps; the unchanged benchmark screen; constructive mode limits; category shortfalls; and the absence of downstream artifacts.
- **Reason:** Milestone 6E corrects the prior conclusion that language-bank expansion alone was necessary. The published record must distinguish the resolved over-conservative surface policy from the remaining mathematical program-space blocker so a future proposal changes only the deficient layer.
- **Result:** Documentation consistently states that bounded language reuse passes, bookkeeping passes, rates are short by 1,202 attempts, and discrete reasoning is short by 678. The next proposed decision is a narrow expansion of verified rate and discrete latent-program ranges or modes—not more prompt work, threshold weakening, or thousands of new sentence plans. Focused Ruff, strict Mypy, and four bounded-reuse tests pass.
- **Files changed:** Seven required documentation/experiment files plus the content-free policy/config/audit source, two aggregate JSON results, and focused unit tests. No bank wording, renderer, mathematical generator, verifier, evaluator, contamination rule, dependency, raw question, packet, dataset, model artifact, or sealed-final content changed.
- **Errors or uncertainty:** The fixed probe is constructive for unbounded modes, but those modes already exceed their balance caps; every limiting small mode has an exact finite bound. The capacity failure therefore does not depend on extrapolating unseen probe results.
- **Next action:** Run the complete formatting, linting, strict typing, test, dependency, whitespace, secrets, benchmark-leak, sealed-path, raw/cache, protected-path, policy, cap, capacity, deterministic-audit, tracked-size, and repository-state checks. Commit and push only if all pass and `origin/main` remains unchanged.

## 2026-07-19 19:06:57 -04:00 - Milestone 6E Step 5: full verification passed

- **Current step:** Prove that the policy correction and negative capacity result are deterministic, content-free, correctly scoped, and safe to publish.
- **Action performed:** Ran Ruff formatting and linting, strict Mypy, all unit/integration tests, main-environment `pip check`, and `git diff --check`; reran both aggregate audits byte-for-byte; validated three-policy fixture outcomes, every derived cap, 20,000-seed constructive probe counts, dual-verifier evidence, exact finite mode bounds, category capacity/shortfall, failed gate, and false downstream flags. Scanned all 13 candidate files for high-confidence secrets and exact/12-token matches to all 904 cached development questions in offline mode. Reviewed protected bank/renderer/generator/verifier/evaluator/contamination/dependency paths, sealed-path status, raw/cache tracking, ignored v3 evidence, absent packet artifacts, prior M6D/review hashes, candidate sizes, and repository scope.
- **Reason:** A capacity failure changes the project's next decision. It must be as reproducible and publication-safe as a successful smoke, without accidentally shipping fixture-derived questions as datasets, benchmark content, raw artifacts, or unauthorized implementation.
- **Result:** Ruff reports 119 files formatted and lint clean; strict Mypy reports no issues in 78 source files; all 302 tests pass in 18.74 seconds; `pip check` and whitespace pass. Calibration and capacity file SHA-256 values replay identically at `38f4c9c...8827` and `26cc9c37...402d`; internal hashes remain `fd731501...2693` and `1a40db7b...1129`. Secret, exact-development, 12-token-development, protected-path, sealed-path, tracked raw/cache/environment, second-packet, and oversized-file counts are zero. The old M6D evidence and genuine review hashes remain exact. All 13 expected candidate paths and no extras are present; the largest is this DEVLOG at 358,833 bytes.
- **Files changed:** This verification checkpoint only after the reviewed policy/config/source/tests/results/documentation set. No template wording, renderer, generator, verifier, evaluator, contamination threshold, dependency, benchmark manifest, raw record, packet, schedule, dataset, model artifact, or sealed-final content changed.
- **Errors or uncertainty:** The first offline leak scan named a nonexistent stale evaluation config and failed before loading rows; rerunning with the frozen final-evaluator config loaded exactly 904 cached development questions and found zero hits. One path-scope wrapper initially counted two nested PowerShell arrays rather than 13 flattened paths; its corrected read-only rerun confirmed 13/13 expected and zero extras. Neither diagnostic failure wrote project evidence.
- **Next action:** Fetch `origin/main` without integration, require the approved tip and `0/0` divergence, stage exactly these 13 paths, repeat cached scope/whitespace/secret/content/size/evidence checks, create the exact atomic commit, push without force, and confirm synchronized clean state.

## 2026-07-19 19:07:29 -04:00 - Milestone 6E Step 6: publication preflight passed

- **Current step:** Confirm remote history is unchanged before forming the exact Milestone 6E publication index.
- **Action performed:** Fetched `origin/main` without integration and checked repository root, branch, local tip, remote tip, and numeric ahead/behind counts.
- **Reason:** The verified negative result may be pushed only as a fast-forward from the approved synchronized starting state; unexpected remote work would require stopping before merge or rebase.
- **Result:** Root is `C:/Users/Admin/Projects/Foundry`; branch is `main`; local and remote both remain `00a419b61e43594736abe1c808aab33ff9156cf2`; ahead/behind is `0/0`. The 13 reviewed working-tree paths remain unstaged.
- **Files changed:** This DEVLOG entry only after a read-only fetch. No source, evidence, config, Git setting, or remote state changed.
- **Errors or uncertainty:** The first divergence assertion compared Git's tab-delimited counts with a literal space-delimited string and falsely stopped after reporting `0/0`; the numeric rerun passed. No staging, commit, integration, or file mutation occurred during the failed assertion.
- **Next action:** Stage exactly the 13 verified paths, repeat index-only scope, whitespace, secret, development-content, evidence, and size checks, then create the requested atomic commit if the index is exact.

## 2026-07-19 19:08:17 -04:00 - Milestone 6E Step 7: publication index verified

- **Current step:** Prove the exact Git index contains only the verified bounded-reuse correction and negative capacity evidence before committing.
- **Action performed:** Staged the 13 explicit paths; checked cached whitespace, expected/extra/forbidden paths, unstaged and untracked state, high-confidence secrets, exact and 12-token matches against all 904 cached development questions, content-free result keys, calibration/capacity decisions, downstream stop flags, and staged blob sizes.
- **Reason:** The publication must not include raw questions, a packet, a schedule, benchmark material, caches, secrets, or protected language/math/evaluation changes, and it must reproduce the failed gate rather than imply a smoke occurred.
- **Result:** The index contains exactly 13 intended paths, 2,314 insertions before this checkpoint, zero missing/extra/forbidden paths, zero unstaged/untracked paths, clean whitespace, zero secret hits, zero exact or 12-token development hits, and no blob at or above 1 MiB. Staged evidence has 14 fixtures, 14 selected-policy matches, 10,003 required attempts, a false capacity gate, content-free aggregate keys, and all allocator/schedule/smoke/replay/packet flags false.
- **Files changed:** This DEVLOG checkpoint only after staging; it will be restaged and the final index, remote tip, and commit subject checked again.
- **Errors or uncertainty:** None. The offline staged leak scan loaded development questions only from the pinned local cache; no benchmark answer or sealed-final content was loaded.
- **Next action:** Restage this checkpoint, recheck the exact 13-path index and unchanged remote tip, create `synthesis: permit bounded balanced template reuse` with process-local published identity, push normally, and verify clean `0/0` synchronization.

## 2026-07-19 19:21:36 -04:00 - Milestone 7A Step 1: synchronized protected starting state verified

- **Current step:** Verify the published Milestone 6E state, genuine review evidence, and every protected language, mathematics, verifier, evaluator, and contamination boundary before freezing the signal-first pilot.
- **Action performed:** Temporarily prepended Git to this PowerShell process; fetched `origin/main` without integration; checked root, branch, local/remote tips, divergence, and worktree; hash-reviewed the repaired template bank/compiler, generators, verifiers, evaluator tree, contamination module, bounded-reuse contracts, and final-evaluator config; rechecked the genuine user-review hash, old capacity evidence, ignored v3 raw evidence, CPython, dependencies, and sealed-path status.
- **Reason:** Milestone 7A may reduce quotas and add allocation/smoke machinery, but it may not change reviewed wording, mathematical generation, labels, dual verification, evaluation, benchmark contamination, or sealed-final state.
- **Result:** Repository is `C:/Users/Admin/Projects/Foundry` on `main`; local and `origin/main` both equal `0e5f6d221f2eb10a4bf93fcb264c629878829e34`; ahead/behind is `0/0`; worktree is clean. Genuine review SHA-256 remains `564a8ca...0791`; policy remains `bounded-balanced-template-reuse-v1`; old measured capacities remain 5,524 bookkeeping, 1,632 rates, and 2,073 discrete. CPython is 3.12.10 and `pip check` passes. Protected and sealed path changes are zero; no sealed-final file was opened, hashed, inspected, or compared.
- **Files changed:** This DEVLOG entry only. No quota, policy, template, generator, verifier, evaluator, contamination rule, raw artifact, schedule, dataset, packet, dependency, or Git configuration changed during verification.
- **Errors or uncertainty:** Two read-only protected-file probes used stale filenames (`render.py`, then `evaluation/evaluator.py`) and stopped without mutation. The corrected probe used `renderer.py` and the tracked evaluator tree hash. All actual protected paths passed.
- **Next action:** Freeze exactly 1,000 accepted examples per dataset, the user-specified 2,504 fixed attempts, 900/100 splits, 20% output track, and even difficulty allocations; rerun the capacity audit before implementing an allocator.

## 2026-07-19 19:33:12 -04:00 - Milestone 7A Step 2: reduced signal-first capacity gate passed

- **Current step:** Freeze the 1,000-targeted plus 1,000-generic signal-first quotas and prove that every dataset/family/difficulty/output/split stratum fits the unchanged bounded-reuse and finite latent-program limits.
- **Action performed:** Added a strict signal-pilot configuration and typed audit contract for exactly 2,000 accepted examples, 1,800 training examples, 200 synthetic-validation examples, 400 output-contract examples, and the user-specified 2,504 fixed attempts. Reserved mode allocations within exact finite generator bounds, derived quota-based surface caps through `bounded-balanced-template-reuse-v1`, and checked all shared cross-dataset latent limits. Added focused tests for exact totals, every nested stratum gate, finite combined capacity, and unchanged policy/safety boundaries.
- **Reason:** The earlier 10,003-attempt plan exceeded rate and discrete latent supply. The smaller scientific pilot may proceed only if its complete fixed attempt pool fits without generator expansion, quota adjustment, cross-dataset overlap, or post-result cap tuning.
- **Result:** The capacity gate passes. Combined required/available attempt capacity is 1,106/1,384 for bookkeeping (125.1356%), 709/752 for rates (106.0649%), and 689/750 for discrete constraints (108.8534%). Every targeted and generic difficulty, output-contract, and train/validation stratum passes; the rate family remains the tightest but retains 43 predeclared slots of headroom. Configuration SHA-256 is `5231a73d...6c67`; internal audit SHA-256 is `b87bc992...1f6`; serialized audit file SHA-256 is `6722734b...fe74`. All four focused tests pass.
- **Files changed:** New signal-pilot YAML, typed capacity/audit module, content-free aggregate audit JSON, focused unit tests, and this continuous DEVLOG. No reviewed template wording, generator mathematics, verifier, evaluator, contamination threshold, raw question, dataset, packet, dependency, or sealed-final content changed.
- **Errors or uncertainty:** The first audit exposed one infeasible generic combined-rate acceptance allocation (85 requested versus a balanced cap of 84). Before any allocation or smoke output existed, one accepted slot was moved to the generic percentage mode, preserving all user-specified dataset/family/difficulty/output/split totals and staying within its exact finite bound. Initial focused Ruff/Mypy findings were formatting and type annotations only and were corrected. Capacity proves schedulability under declared identities; the allocator must still construct the complete unique schedule deterministically.
- **Next action:** Implement the global deterministic allocator, prove exact quotas, stable ordering, cap compliance, latent uniqueness, cross-dataset/split isolation, tamper detection, and fail-closed behavior, then construct the full content-free 2,504-slot schedule.

## 2026-07-19 19:44:08 -04:00 - Milestone 7A Step 2 correction: target-type compatibility invalidated the preliminary pass

- **Current step:** Validate the preliminary aggregate capacity result against every frozen reuse identity before implementing any allocator.
- **Action performed:** Added an exact deterministic maximum-flow compatibility check joining each dataset's per-target-type cap, semantic-frame-derived per-mode cap, and the shared finite mode supply across targeted and generic datasets. Recomputed both accepted and attempt capacity for each family and dataset, then proportionally reserved failed shared pools across difficulty, output-contract, and train/validation strata so the audit reports rather than hides every shortfall.
- **Reason:** The first aggregate calculation summed target-type capacity without enforcing which modes share one target type. That was insufficient: rate-total and combined-rate both consume `total_quantity`; two-type allocation and equal distribution both consume `count`. An allocator cannot satisfy the frozen policy merely because the total number of target-type identities multiplied by a cap is large enough.
- **Result:** The corrected capacity gate is false. Bookkeeping still passes at 1,384 compatible slots for 1,106 attempts. Rates provide only 695 compatible cross-dataset slots for 709 attempts, a shortfall of 14, after the 223 combined `total_quantity`, 184 ratio, 104 percentage, and 184 weighted-mean ceilings are enforced. Discrete provides only 598 for 689, a shortfall of 91: combined count capacity is 288, complete-package/group-count capacity is 220 under frame balance, and exact dual-capacity supply is 90. Generic discrete also fails independently at 399/417; each of its difficulty, output, and split attempt strata is proportionally short. Internal audit SHA-256 is `522b5b4e...7aaf`; serialized evidence SHA-256 is `b18b7e44...6995`. Four focused tests pass and now assert the negative gate and all downstream false flags.
- **Files changed:** The signal-pilot config/audit/tests, aggregate content-free evidence, and this DEVLOG only. No allocator, full schedule, smoke, replay, review packet, generated dataset, template wording, generator, verifier, evaluator, contamination rule, dependency, or sealed-final content was created or changed.
- **Errors or uncertainty:** The earlier Step 2 entry's claimed pass was preliminary and is superseded by this correction; it remains in the append-only log to preserve the diagnostic sequence. The corrected computation is exact for the declared identity/mode graph and does not depend on stochastic probing. The rates and discrete shortfalls cannot be fixed by a different ordering of the same slots.
- **Gate status:** **FAILED.** The explicit reduced-pilot stop rule is active.
- **Next action:** Do not implement allocation, create the 2,504-slot schedule, run a smoke or replay, or issue a packet. Record the negative evidence in required project documents, run bounded verification of the capacity-only change, and report the exact blocker for a new user decision.

## 2026-07-19 19:51:26 -04:00 - Milestone 7A Step 3: negative capacity evidence verified; downstream work remains stopped

- **Current step:** Verify that the corrected capacity-only result is deterministic, content-free, correctly scoped, and safe to report without creating downstream artifacts.
- **Action performed:** Ran full Ruff formatting/linting, strict Mypy, all unit/integration tests, main-environment dependency integrity, and whitespace checks. Rebuilt the audit byte-for-byte; validated exact quota totals, every compatible capacity and failed stratum, all false downstream flags, and the absence of content-bearing keys. Scanned every changed/untracked path for high-confidence secrets and exact/12-token matches to all 904 cached development questions in offline mode; checked protected source/config paths, sealed-path changes, raw/cache tracking, packet absence, ignore rules, and file sizes.
- **Reason:** The stop decision rests on a newly corrected compatibility model. It must be reproducible and must not accidentally introduce the allocator, questions, benchmark content, or any protected change that the failed gate forbids.
- **Result:** Ruff reports 121 files unchanged and lint clean; strict Mypy reports no issues in 79 source files; all 306 tests pass in 18.29 seconds; `pip check` and `git diff --check` pass. Audit replay is byte-identical at file SHA-256 `b18b7e44...6995` and internal SHA-256 `522b5b4e...7aaf`. The 11 candidate paths have zero secret hits, zero exact or 12-token development hits, zero protected/sealed-path changes, zero tracked raw files, zero packet files, and no file at or above 1 MiB. The configured raw path remains ignored.
- **Files changed:** Seven required documentation/experiment records, one signal-pilot config, one typed capacity module, one content-free aggregate audit, and one focused unit-test module. No template, generator, verifier, evaluator, contamination rule, dependency, allocator, schedule, smoke, replay, packet, dataset, model artifact, or sealed-final content changed.
- **Errors or uncertainty:** The first path-scope wrapper counted two nested PowerShell arrays rather than 11 flattened paths; a corrected read-only rerun verified all 11 candidate paths and zero violations. Expected LF-to-CRLF notices are non-fatal. The failed capacity proof is exact over the frozen graph; the only scientific uncertainty is whether the uniform target/frame cap policy itself should remain the intended policy for an intentionally nonuniform curriculum.
- **Gate status:** **FAILED.** No commit or push is created because the user-directed capacity stop occurred before allocator/schedule/smoke completion and the requested commit subject would falsely claim scheduled pilot data.
- **Next action:** Report the blocker and wait for explicit approval of a separate target-type/semantic-frame cap-policy decision. Keep full dataset generation, training, benchmark evaluation, and sealed-final access blocked.

## 2026-07-19 20:10:42 -04:00 - Milestone 7B Stage A: stopped 7A evidence reverified for publication

- **Current step:** Preserve the accurate Milestone 7A stop result as its own publication before changing the balancing policy.
- **Action performed:** Fetched `origin/main` without integration; verified root, branch, tips, `0/0` divergence, all 11 intentional candidate paths, and zero staged paths; inspected the complete capacity-only scope; replayed the audit byte-for-byte; asserted all six dataset/family and three combined compatibility results; reviewed protected template/math/verifier/evaluator/contamination paths, raw tracking, ignored pilot output, and forbidden model/environment/cache status.
- **Reason:** Milestone 7B is authorized to change exactly one balancing policy after the prior negative result is independently preserved. Publishing the stop evidence first prevents the corrected scientific record from being overwritten by the subsequent solution.
- **Result:** Local and `origin/main` both remain `0e5f6d221f2eb10a4bf93fcb264c629878829e34`, ahead/behind `0/0`. The audit remains byte-identical with internal SHA-256 `522b5b4e...7aaf` and file SHA-256 `b18b7e44...6995`; all required/capacity pairs reproduce exactly. The 11-path scope contains only the signal-pilot config/auditor/test, content-free evidence, and seven required records. Protected, raw, sealed-path, model, environment, cache, dataset, allocator, schedule, smoke, replay, and packet changes are zero.
- **Errors or uncertainty:** None. The working tree is intentionally dirty until this accurate stop result is committed and pushed as a separate atomic publication.
- **Gate status:** **PASSED.** The stopped audit is ready for publication.
- **Next action:** Repeat formatting/type/test/dependency/whitespace/content/safety checks, stage exactly the 11 paths, commit `analysis: record signal-first capacity blocker`, push normally, and require a clean synchronized tree before Stage B begins.

## 2026-07-19 20:13:18 -04:00 - Milestone 7B Stage A: publication verification passed

- **Current step:** Verify and publish only the stopped Milestone 7A capacity evidence.
- **Action performed:** Repeated Ruff formatting/linting, strict Mypy, all tests, `pip check`, and whitespace validation; scanned the exact 11-path publication candidate for high-confidence secrets and exact/12-token matches against all 904 cached development questions in offline mode.
- **Reason:** The first commit must be independently safe and reproducible before the authorized balancing correction begins.
- **Result:** Ruff reports 121 files unchanged and lint clean; strict Mypy reports no issues in 79 source files; all 306 tests pass in 19.06 seconds; dependencies and whitespace pass. Secret, exact-development, and 12-token-development hits are zero. Audit evidence and protected-scope results remain unchanged.
- **Errors or uncertainty:** Expected Windows LF-to-CRLF notices are non-fatal. No test or safety failure occurred.
- **Gate status:** **PASSED.** Publication may proceed.
- **Next action:** Stage the exact 11 paths, repeat index scope and safety checks, create the accurate stop-result commit, push without force, and confirm clean `0/0` synchronization.

## 2026-07-19 20:18:37 -04:00 - Milestone 7B Stage B: maximally balanced feasible submode policy selected

- **Current step:** Replace only the infeasible semantic-frame/target-type equal-style caps with a deterministic allocation across the unchanged generator modes.
- **Action performed:** Added three predeclared content-free policy candidates and nine original fixtures covering ample modes, one/two low-capacity modes, impossible total and subordinate capacity, stable remainders, targeted/generic largest-remainder splitting, and exact difficulty/output margins. Implemented iterative water-filling, stable largest-remainder dataset splitting, and standard-library integer min-cost flow minimizing squared deviation for subordinate margins. Applied the algorithm independently to accepted and attempt quotas; no benchmark or generated result informed weights.
- **Reason:** The family totals are feasible, but forcing every target/frame identity toward one uniform cap contradicts finite submode compatibility. Water-filling gives each low-capacity mode its verified maximum and redistributes only the unavoidable remainder, producing the most even feasible mixture without hand weights.
- **Result:** `maximally-balanced-feasible-submodes-v1` matches 9/9 fixtures; equal-style caps match 5/9 and fail necessary redistribution; proportional-to-capacity matches 6/9 but overconcentrates high-capacity modes. Attempt water-filling produces bookkeeping 553/553; rates 96/170/104/170/169 in stable mode order; and discrete 200/200/199/90. Largest-remainder splitting gives targeted rates 39/70/43/70/70 and targeted discrete 79/79/79/35, with complementary generic allocations. Fixture SHA-256 is `5101e3b2...bd47`; policy SHA-256 `75a47622...3ce9`; config SHA-256 `b9f6f9b2...fbed`; calibration SHA-256 `85ab3cc9...864d`. Four focused tests pass.
- **Files changed:** New submode policy/fixture configuration, typed policy implementation, focused tests, content-free calibration evidence, updated mode-allocation fields in the already published signal-pilot config, and this DEVLOG. No generator mode/range, template wording, label, verifier, evaluator, contamination rule, dependency, raw question, schedule, dataset, model artifact, or sealed-final content changed.
- **Errors or uncertainty:** Initial Ruff/Mypy findings were one unused import and two local type annotations; they were corrected before focused tests. The selected policy proves quota arithmetic and subordinate optimization behavior; the real 2,504-attempt audit must still verify every frozen surface cap and latent capacity before allocator work.
- **Gate status:** **PASSED.** Policy calibration is frozen before real schedule or smoke outputs.
- **Next action:** Run the revised real capacity audit across dataset/family/submode/target/difficulty/output/split strata, including concentration and balance metrics, and stop if any 2,504-attempt requirement remains infeasible.

## 2026-07-19 20:25:06 -04:00 - Milestone 7B Stage C: revised 2,504-attempt capacity gate passed

- **Current step:** Apply the preselected feasible submode policy to every real dataset/family/mode/target/difficulty/output/split stratum and retained surface-reuse control.
- **Action performed:** Recomputed accepted and attempt mode quotas by water-filling; jointly split accepted and attempt pools so every accepted mode fits its candidate pool; used deterministic integer minimum-deviation matrices for difficulty, output-contract, and future train/validation margins; balanced each mode across compatible semantic frames; checked raw unique mode capacity and unchanged sentence-plan, number-neutral, and plan-plus-scenario caps. Recorded target coverage, headroom, maximum concentration, and normalized Shannon entropy.
- **Reason:** Fixture success is not sufficient. The exact 2,504-slot contract must fit every real margin and retained identity layer before any candidate seed, allocator, or question is created.
- **Result:** The capacity gate passes. Bookkeeping assigns 553/553 with 5,524 raw capacity and 4,418 headroom (entropy 1.0000). Rates assign 96/170/104/170/169 with 1,632 capacity and 923 headroom (entropy 0.9810); rate-total and percentage are saturated exactly. Discrete assigns 200/200/199/90 with 2,073 capacity and 1,384 headroom (entropy 0.9685); dual capacity is saturated exactly. Retained surface capacity is 864/400/400 for targeted bookkeeping/rates/discrete against 688/292/272, and 576/528/560 for generic against 418/417/417. All modes and target types have nonzero coverage; all difficulty/output/split margins are exact; all dataset/family gates pass. Internal audit SHA-256 is `ba1f8131...7d68`; serialized evidence SHA-256 `9980a473...a863`. Five focused tests pass.
- **Files changed:** New content-free revised capacity evidence and focused assertions in addition to the Stage B policy/config/source/tests and this DEVLOG. No allocator, schedule, rendered question, template, generator, label, verifier, evaluator, contamination threshold, dependency, dataset, model artifact, or sealed-final content changed.
- **Errors or uncertainty:** Independent water-filling of accepted and attempt totals initially assigned one accepted rate-total and one accepted dual-capacity example beyond the targeted attempt split. Before the audit gate, the split was replaced by a deterministic minimum-squared-deviation split constrained by accepted lower bounds; family/mode totals and proportional intent are unchanged. One Mypy object narrowing annotation was corrected. The dry allocator must still construct 2,504 distinct latent programs and fit actual plan/scenario assignments.
- **Gate status:** **PASSED.** Allocator implementation is authorized.
- **Next action:** Implement the deterministic global allocator, predeclare finite seed pools, construct all 2,504 content-free slots, and prove quotas, caps, unique latent programs, cross-dataset/split isolation, stable ordering, tamper detection, and fail-closed behavior before any 120-question smoke.
### 2026-07-19 — Milestone 7B Stage D: deterministic pilot allocator and dry schedule

- Corrected the preliminary subordinate capacity model to enforce verified per-mode and per-difficulty latent capacities. The discrete `equal_distribution` mode is limited to 33 easy, 60 medium, and 160 hard unique programs; constrained allocation assigns 33/60/106 attempts and shifts the remaining easy and medium quota to compatible modes without changing any family or mode total.
- Fixed the allocator's shared-bookkeeping-frame accounting: bookkeeping's 18 semantic frames serve both `inventory` and `grouping`, so their frozen cap is derived from the family total rather than applied independently to each mode.
- The revised 2,504-attempt capacity gate passed. The global allocator created 2,504 content-free slots, 2,504 unique latent-program hashes, 2,504 unique semantic-IR hashes, and 2,504 dual-verifier agreements with zero targeted/generic or train/synthetic-validation latent overlap.
- Dry schedule SHA-256: `63f3d631aacd38cbd9a365ad0f4333e1fa565c28985b1c38baa142fa6d4e84e1`; aggregate summary SHA-256: `b4044ae7acdeb3684f47b4828714647fcb497939d3ab8f0e4c5eeef1465326c8`.
- An exact independent dry-schedule replay reproduced the complete tracked summary byte-for-byte. No rendered questions or complete datasets were created.

### 2026-07-19 — Milestone 7B Stage E/F: review-smoke technical gate failed

- Froze a content-free 120-slot review schedule with SHA-256 `021956fff2321bd28779390870b5d90030806cb69d8b97d18165b4aea3f67332`. The schedule predicted 120 unique latent programs, semantic IRs, render signatures, and number-neutral identities.
- Processed exactly 120 scheduled candidates without replacements: 115 passed all automatic gates and 5 were conservatively rejected. Targeted accepted 59/60; generic control accepted 56/60. Bookkeeping accepted 52/53, rate/ratio accepted 33/34, and discrete accepted 30/33.
- Mathematical and safety results were clean: zero false labels, verifier disagreements, target mismatches, deterministic language defects, exact rendered duplicates, latent-program duplicates, render-signature duplicates, benchmark-contamination rejections, or unresolved contamination cases.
- The counted decision SHA-256 and replay decision SHA-256 were both `7bc2250d189602f955853797141aa8557ea236a68e9ae01cd8f2a867fb7b1ec1`; deterministic replay passed exactly.
- The technical gate failed solely because runtime number-neutral normalization found five collision pairs across ten otherwise distinct questions. The schedule's predicted number-neutral identity included semantic metadata and therefore was not equivalent to the actual rendered-text normalization used by the runtime duplicate gate.
- Per the stop rule, no second human-review packet was created, no additional smoke was run, and no final Milestone 7B commit or push was made. Complete questions remain ignored locally.

### 2026-07-19 — Milestone 7C Stage A: Milestone 7B preservation verification

- Reproduced every frozen 7B hash and count from ignored and tracked evidence. The complete dry schedule rebuilt exactly at `63f3d631...e84e1` with summary `b4044ae7...26c8`; the counted and replay smoke decisions both reproduced `7bc2250d...1ec1`.
- Ruff formatting and linting passed across 127 files; strict Mypy passed across 82 source files; all 317 unit and integration tests passed; `pip check` and `git diff --check` passed.
- The 22-file publication candidate contains no protected evaluator, generator, verifier, contamination, sealed, raw, cache, environment, model, credential, or secret path. Exact and 12-token scans against all 904 locally cached development questions found zero matches. No raw or environment path is tracked, and the largest candidate is this DEVLOG at 388,778 bytes.
- Content-free validation confirms 2,504 unique latent programs, zero dataset/split overlap, 115/120 smoke acceptances, exactly five runtime number-neutral collisions, and absent review packets. The measured failed gate is preserved without alteration.
- Next action: stage exactly the 22 verified files, repeat index checks, create the accurate Milestone 7B preservation commit, push it, and confirm clean synchronized state before diagnosing the five ignored collision pairs.

### 2026-07-19 20:38:00 -04:00 — Milestone 7C Stage A: Milestone 7B evidence published

- **Action performed:** Created and pushed commit `23c2cdb5ab2a931df6b711b1f3fc571c748035c3` with the verified feasible-submode policy, 2,504-slot metadata schedule, failed 115/120 smoke evidence, tests, and documentation.
- **Result:** Local `main` and `origin/main` matched at the new commit, ahead/behind was 0/0, and the worktree was clean before identity correction began. No raw question, dataset, model, cache, environment, secret, credential, or sealed-final content was published.
- **Gate status:** **PASSED.** Collision diagnosis was authorized.

### 2026-07-19 20:47:00 -04:00 — Milestone 7C Stages B/C: five collisions classified and canonical identity integrated

- **Action performed:** Inspected all five ignored collision pairs without committing rendered text. Added the typed `NumberNeutralIdentity` interface and made both schedule construction and runtime screening call the same unchanged NFKC/lowercase/number-replacement/tokenization implementation. Added an explicit normalizer version, contract hash, runtime equality assertion, and five sanitized collision-class regressions.
- **Result:** Every pair had different scheduler metadata but the same normalized runtime surface. Differences included semantic-frame and template IDs in all five pairs, and varying difficulty, output-track, scenario, or lexical metadata in some pairs. None of those differences reliably survived rendering and normalization. Canonical contract SHA-256 is `e57f12e06673c21bb4c257c651d96b791a8b991260a3e5fd7c5fd11bdc3a72eb`; current normalizer source SHA-256 is `57d77d32b4631aaecacb02d454f39000f7e13fec457b398ff983817df05c3d1f`.
- **Files changed:** contamination identity module, signal allocator/smoke contracts, focused tests, and content-free collision evidence. Normalizer semantics, templates, mathematics, verifiers, and contamination thresholds did not change.
- **Gate status:** **PASSED.** Runtime-exact schedule reconstruction was authorized.

### 2026-07-19 20:55:00 -04:00 — Milestone 7C Stage D: runtime-exact 2,504-slot reconstruction failed

- **Action performed:** Rebuilt fixed latent pools and attempted the complete schedule with actual rendered questions hashed by the canonical runtime identity before allocation. Independently enumerated every weighted-average realization in the fixed 4,096-per-difficulty pool to establish an upper-bound proof rather than treating a greedy error as conclusive.
- **Result:** The allocator exhausted the targeted weighted-average stratum. The fixed pool contains 9,265 unique latent programs and 10,512 difficulty realizations, but its 16 internal render signatures collapse to eight runtime number-neutral identities. With unchanged caps, targeted supports at most 8 x 5 = 40 of 70 required attempts (short 30), and generic supports at most 8 x 6 = 48 of 100 (short 52). Content-free blocker evidence SHA-256 is `8caa804c04844a2bea0fa4f3d5af9a3f7ef63c381d15d148a91ecd02691f4368` before final documentation-only changes.
- **Errors or uncertainty:** This is a mathematical upper-bound failure under the fixed pool and caps, not merely an allocator heuristic failure. The previous schedule used identities containing non-surface metadata and therefore cannot satisfy the runtime contract.
- **Gate status:** **FAILED.** Per the approved stop rule, no fresh review schedule, 120-question smoke, replay, Codex audit, or assisted packet was created.
- **Next action:** Verify and publish the accurate runtime-exact scheduling blocker. Await an explicit user decision on a scientifically valid surface-capacity correction; do not generate datasets or train.

### 2026-07-19 21:07:00 -04:00 — Milestone 7C Stage G: failure result verification passed

- **Action performed:** Ran repository-wide Ruff formatting and lint, strict Mypy, all unit and integration tests, `pip check`, and `git diff --check`; validated the five-pair collision inventory, unchanged normalizer semantics and hashes, fail-closed identity equality, quota/cap proof, and review-packet suppression. Scanned all 16 candidate paths for high-confidence secrets and exact/12-token matches against all 904 ignored development questions; reviewed protected evaluator/generator/verifier/template/dependency paths, sealed-path changes, raw/cache/environment tracking, ignore rules, and tracked sizes.
- **Result:** Ruff reports 128 files formatted and lint-clean; strict Mypy reports no issues in 82 source files; all 327 tests pass in 34.98 seconds; dependency and whitespace checks pass. Collision-evidence SHA-256 is `deaf8e54d4a43085076428aaa0efa84bfcd6fea9211d956f8f99bbcd10b47726`; blocker-evidence SHA-256 is `8caa804c04844a2bea0fa4f3d5af9a3f7ef63c381d15d148a91ecd02691f4368`. Secret, exact-development, 12-token-development, protected-path, sealed-path, and tracked raw/model/environment counts are zero. The largest candidate is this DEVLOG at 393,850 bytes before this entry, below 1 MiB.
- **Gate status:** **VERIFIED FAILURE.** The accurate blocker result is eligible for the required commit and push. A fresh smoke, replay, and review packet remain prohibited by the failed full-schedule gate.
- **Next action:** Fetch without integration, require the published tip to remain `23c2cdb5...35c3`, stage exactly the 16 verified paths, repeat index-only checks, commit `analysis: record runtime-exact scheduling blocker`, push normally, and confirm a clean synchronized worktree.

### 2026-07-19 - Milestone 7D Stage A/B: state verified and surface-policy calibration started

- **Action performed:** Verified clean synchronized `main` at `8653980c...02c4`, ignored raw-review boundaries, the frozen template/math/verifier/evaluator/contamination scope, and the unchanged canonical runtime normalizer. Enumerated runtime identities from the fixed 4,096-per-mode/difficulty pools using every compatible reviewed sentence plan.
- **Result:** The measured inventory is 384 identities for each bookkeeping mode; 20 for rate-total, ratio-scale, percentage, and combined-rate; 8 for weighted-average; and 80 for each discrete mode. Weighted-average exactly reproduces the prior blocker proof. Added a predeclared three-policy, ten-fixture comparison and a submode-local mechanical cap contract; no real capacity or smoke result informed selection.
- **Errors or uncertainty:** One read-only verification probe used an outdated loader name and was immediately rerun with the correct loader; no state changed. Full policy calibration and the real 2,504-slot capacity gate remain pending.
- **Gate status:** Starting-state gate **PASSED**. Policy-selection gate is in progress.
- **Next action:** Run formatting, typing, and focused policy tests; freeze the policy/config/fixture/calibration hashes; then audit all eleven real submodes before allocator reconstruction.

### 2026-07-19 - Milestone 7D Stages B/C: submode-local policy and capacity gate passed

- **Action performed:** Compared exactly three predeclared policies on ten original content-free fixtures, derived attempt and accepted caps for every dataset/family/submode from the measured canonical runtime identities, and audited both dataset-local and combined capacity.
- **Result:** `submode-local-balanced-surface-reuse-v1` matches 10/10 fixtures; the family-level and permissive alternatives each match 6/10. Weighted-average caps are targeted 11 attempts/8 accepted, generic 16/12, and combined 27/20. Every one of the eleven submodes has nonnegative attempt and accepted headroom. Fixture SHA-256 is `7f32574c...15be5`; policy `acb01f6d...4babc`; config `c0c73305...638a3`; calibration `29218773...2554b`; capacity audit `dfad68c4...44ac3`.
- **Errors or uncertainty:** A display-only PowerShell loop had a syntax error and was replaced by an equivalent read-only Python report. The policy proof establishes arithmetic capacity; the allocator must still construct a complete deterministic schedule while satisfying all unchanged plan, frame, scenario, exact, latent, split, and cross-dataset controls.
- **Gate status:** Policy calibration **PASSED**. Full surface-capacity audit **PASSED**. Allocator reconstruction is authorized.
- **Next action:** Make dataset and global runtime-identity counters use the frozen submode-local caps, rebuild all 2,504 slots, and validate exact schedule/runtime identity and all retained controls before authorizing the review smoke.

### 2026-07-19 - Milestone 7D Stage D: first allocator reconstruction exposed submode-balance defect

- **Action performed:** Integrated dataset-local and combined submode runtime-identity caps into both assignment and validation, then ran the complete fixed-pool reconstruction.
- **Result:** All eleven latent pools matched their exact 2,504 slot requirements, including 170 weighted-average slots. Surface assignment later exhausted the targeted weighted-average `clinic` scenario under the unchanged plan-plus-scenario cap.
- **Errors or uncertainty:** Diagnosis found that latent scenario and lexical usage were balanced by dataset/family, not dataset/family/submode. Earlier rate modes could therefore consume balancing priority and leave the scarce weighted-average mode with more than its per-scenario plan capacity. This is an allocator-key defect rather than evidence that the frozen submode caps lack mathematical capacity.
- **Gate status:** Reconstruction is **not yet decided**. The authorized allocator implementation may correct deterministic submode-local balancing, but no quota, cap, wording, identity, threshold, generator, or verifier may change.
- **Next action:** Balance scenario and lexical selection at dataset/family/submode granularity, rerun the same construction, and stop if the exact assignment remains infeasible.

### 2026-07-19 - Milestone 7D Stage D: corrected compatibility gate failed; mandatory stop

- **Action performed:** Added content-free allocator-exhaustion diagnostics, verified the weighted-average latent scenario mix was already balanced, and partitioned all eight fixed runtime identities by their actual difficulty compatibility. Updated the capacity audit so aggregate cap arithmetic cannot authorize scheduling without this compatibility check.
- **Result:** Easy and medium share four identities; hard uses four disjoint identities. Frozen targeted easy/medium attempts are 47 under cap 11, giving capacity 44 and shortfall 3. Generic requires 66 under cap 16, giving 64 and shortfall 2. Combined requires 113 under cap 27, giving 108 and shortfall 5. The earlier aggregate pass is superseded. Corrected config SHA-256 is `fc090135...ac1a5`; calibration `0187a2d...1d337`; audit `7c0f2913...10f5`.
- **Errors or uncertainty:** The first failed assignment was initially attributed to family-level scenario balancing. Correcting that key preserved an even 14-per-scenario targeted distribution but failed again; diagnostics then exposed the binding difficulty-specific identity caps. This is now a mathematical compatibility shortfall, not a greedy allocator claim.
- **Gate status:** **FAILED.** Per the explicit stop rule, no complete 2,504-slot schedule, fresh 120-question smoke, deterministic replay, Codex audit, or assisted review packet will be created.
- **Next action:** Document and repository-wide verify the accurate blocker, commit `analysis: record submode surface-capacity blocker`, push normally, and wait for a user decision. Do not generate datasets or train.

### 2026-07-19 - Milestone 7D final verification passed for accurate blocker publication

- **Action performed:** Ran repository-wide Ruff format/lint, strict Mypy, all unit and integration tests, `pip check`, and `git diff --check`; replayed calibration and capacity evidence byte-for-byte; validated the three-policy fixture comparison, all eleven aggregate cap derivations, weighted-average difficulty partition, exact stop flags, and absence of downstream artifacts. Scanned all 13 candidate paths for high-confidence secrets and exact/12-token matches against exactly 904 cached development questions in offline mode; checked protected generator/verifier/evaluator paths, sealed-path changes, tracked raw/model/environment files, review-packet writes, ignore rules, and tracked sizes.
- **Result:** Ruff reports 130 files unchanged and lint clean; strict Mypy reports no issues in 83 source files; all 331 tests pass in 33.50 seconds; dependencies and whitespace pass. Calibration and capacity evidence replay byte-identically with file SHA-256 values `0e37a440...663e` and `902b969f...9ecd`, and internal hashes `0187a2d4...d337` and `7c0f2913...10f5`. Secret, exact-development, 12-token-development, protected-path, sealed-path, tracked raw/model/environment, and packet-write counts are zero. The configured raw full-schedule path remains ignored; the largest candidate is this DEVLOG at 401,628 bytes before this entry, below 1 MiB.
- **Errors or uncertainty:** The first leak-scan implementation recomputed each candidate document's n-grams for every benchmark question and was stopped after its process IDs were verified; an optimized equivalent precomputed each candidate n-gram set once and completed with identical criteria. No repository state or evidence was altered by the stopped read-only process.
- **Gate status:** **VERIFIED FAILURE.** The accurate 13-file blocker result is eligible for the required commit and push. No schedule, smoke, replay, or packet was created.
- **Next action:** Fetch without integration, require `origin/main` to remain at `8653980c...02c4`, stage exactly the 13 verified paths, repeat index-only checks, commit `analysis: record submode surface-capacity blocker`, push normally, and confirm clean `0/0` synchronization.

### 2026-07-19 - Milestone 7E Stage A: frozen starting state verified

- **Action performed:** Verified the repository root, `main` branch, local and remote commit, ahead/behind count, worktree, ignored raw-review evidence, and every published Milestone 7D policy, fixture, calibration, capacity, runtime-normalizer, and identity-contract hash before editing.
- **Result:** Local `main` and `origin/main` both point to `bfdbd4a89d29f8cdccde5eaca0517af397e8a289`, ahead/behind is `0/0`, and the worktree is clean. The raw Milestone 6 review directory remains ignored. Frozen content-free hashes match: policy `acb01f6d...4babc`, fixture `7f32574c...15be5`, config `fc090135...ac1a5`, calibration `0187a2d...1d337`, capacity audit `7c0f2913...10f5`, normalizer source `57d77d32...c3d1f`, and identity contract `e57f12e0...a72eb`.
- **Errors or uncertainty:** None. No sealed-final path was accessed. Because the repository is clean at the published commit, the tracked template bank, generators, verifiers, evaluator, and contamination controls are unchanged.
- **Gate status:** Starting-state gate **PASSED**.
- **Next action:** Define and calibrate `minimal-compatible-difficulty-reallocation-v1`, changing only rate-family attempt difficulty placement while preserving every frozen total and control.

### 2026-07-19 - Milestone 7E Stages B/C: minimal difficulty policy and capacity gate passed

- **Action performed:** Compared the frozen allocation, `minimal-compatible-difficulty-reallocation-v1`, and a broader redistribution on nine original content-free fixtures. Deterministically enumerated feasible weighted-average and compensating rate-submode shifts while requiring accepted-cell compatibility, exact submode row margins, and exact dataset difficulty columns. Re-audited the frozen submode-local runtime-identity capacities.
- **Result:** The selected policy passes 9/9 fixtures. Targeted moves weighted-average easy `2` and medium `1` to hard, compensated by ratio-scale hard to easy `1`, ratio-scale hard to medium `1`, and combined-rate hard to easy `1`. Generic moves weighted-average easy `1` and medium `1` to hard, compensated by ratio-scale hard to easy `1` and combined-rate hard to medium `1`. Weighted-average easy/medium demand is now targeted `44/44`, generic `64/64`, combined `108/108`; hard is targeted `26/44`, generic `36/64`, combined `62/108`. Fixture SHA-256 is `c8ae7f75...e6231`; policy `edd80f67...93d7`; config `c05883f2...068c`; original allocation `10a99906...cb8d`; corrected allocation `1f9abb51...d332`; calibration `ec00bfe5...6a11`; corrected capacity `4ab358b7...04cb`.
- **Errors or uncertainty:** Initial focused verification exposed two mechanical issues: the allocator needed the corrected subordinate audit rather than the compact capacity proof, and an existing slot-margin snapshot still named the previous allocation. Both were corrected without changing the selected shifts or any frozen quota. Strict Mypy then passed for the three touched source modules; 14 of 15 focused tests passed, with only the expected old snapshot mismatch before it was updated.
- **Gate status:** Policy selection **PASSED**. Corrected compatibility and complete capacity gate **PASSED**.
- **Next action:** Build and validate the complete 2,504-slot content-free schedule using submode-local canonical runtime-identity caps; stop before smoke if deterministic scheduling cannot satisfy every frozen quota and reuse constraint.

### 2026-07-19 - Milestone 7E Stage D: exact complete-schedule gate failed

- **Action performed:** Matched all eleven latent pools, integrated submode-local runtime-identity caps, and attempted deterministic surface allocation. Greedy diagnostics first confirmed the corrected weighted-average stratum; a bounded exact matcher then jointly enforced sentence-plan, plan-plus-scenario, frame, exact-text, and canonical runtime-identity caps. Complete-packages latent selection was additionally balanced by dataset, difficulty, and scenario before the final exact feasibility check.
- **Result:** The weighted-average correction remains feasible. The first exact joint blocker is `generic_control/constraint_distribution_or_discrete_reasoning/complete_packages`: 121 attempts cannot be assigned under caps of seven per plan, one per plan/scenario pair, 25 per frame, and two per runtime identity, despite 1,337 verified unique latent programs, 20 plans, 20 scenario domains, and 80 runtime identities. Content-free blocker SHA-256 is `a387b5ce...41ab`.
- **Errors or uncertainty:** Earlier independent capacity products did not model intersections among all surface resources and therefore overstated schedulability. Multiple deterministic ordering probes were diagnostic only; none relaxed a cap or persisted a schedule. The final result comes from the joint fail-closed matcher after compatibility-aware latent balancing.
- **Gate status:** Complete 2,504-slot schedule gate **FAILED**. Per the mandatory stop rule, no fresh 120-slot schedule, smoke, deterministic smoke replay, Codex audit, or assisted human-review packet was created.
- **Next action:** Document and repository-wide verify the stopped result, commit `analysis: record weighted-average difficulty blocker`, push normally, and wait. The only relevant permitted next choice is reducing the fixed signal-pilot attempt pool; adding weighted-average plans cannot resolve the discrete complete-packages blocker.

### 2026-07-19 - Milestone 7E final verification passed for stopped result

- **Action performed:** Ran repository-wide Ruff formatting/linting, strict Mypy, all unit and integration tests, `pip check`, and `git diff --check`; replayed the policy calibration and corrected arithmetic-capacity evidence; reproduced the exact joint-surface failure; validated all shifts, row/column margins, frozen caps, absent downstream artifacts, and blocker hash. Scanned all 18 changed/untracked paths for high-confidence secrets and exact/12-token matches against exactly 904 cached development questions in offline mode; reviewed protected evaluator/generator/verifier/contamination paths, sealed-path changes, tracked raw/model/cache/environment files, ignored raw schedule path, packet absence, and candidate sizes.
- **Result:** Ruff is clean; strict Mypy reports no issues in 84 source files; all 338 tests pass in 40.89 seconds; dependency and whitespace checks pass. Policy fixtures pass 9/9. Calibration, corrected-capacity, and blocker internal SHA-256 values are `ec00bfe5...6a11`, `4ab358b7...04cb`, and `a387b5ce...41ab`; blocker integrity validates. Secret, exact-development, 12-token-development, protected-path, sealed-path, tracked raw/model/cache/environment, fresh-packet, and oversized-candidate counts are zero. The largest candidate is this DEVLOG at about 409 KiB before this entry, below 1 MiB.
- **Errors or uncertainty:** The first offline leak scan used the final-evaluator config, whose hash does not belong to the development manifest, and stopped before loading rows. The corrected read-only scan used the manifest-matched smoke config, loaded exactly 904 development questions from the pinned offline cache, and returned zero findings. Expected Windows LF-to-CRLF notices are non-fatal.
- **Gate status:** Verified stopped result **PASSED**. The complete schedule and technical gate remain **FAILED**; smoke/replay/packet stages remain unrun by rule.
- **Next action:** Fetch without integration, require `origin/main` to remain at `bfdbd4a8...a289`, stage exactly the reviewed 18 paths, repeat index-only checks, commit `analysis: record weighted-average difficulty blocker`, push normally, and confirm clean `0/0` synchronization.

### 2026-07-19 23:05:05 -04:00 — Milestone 7F Stage A: frozen starting state verified

- **Action performed:** Read the complete Milestone 7F authorization, verified the repository root and `main` branch, compared local `main` with `origin/main`, inspected the worktree, rehashed the three Milestone 7E evidence artifacts, and reran the focused difficulty-policy and exact-allocator tests without accessing any sealed-final path.
- **Result:** Local and remote both point to `6a5554050d7dad5ef4a268b2008d532732f09088`, ahead/behind is `0/0`, and the worktree was clean before this entry. The calibration, corrected-capacity, and blocker evidence file SHA-256 values remain `cf9b1917...351d`, `3379a96f...eba4`, and `d5e9f043...3134`; their recorded internal decision hashes remain unchanged. All 12 focused tests passed in 18.92 seconds.
- **Errors or uncertainty:** None. Exact-commit equality and the empty starting diff confirm that accepted quotas, templates, generators, verifiers, canonical normalizer, reuse/balancing policies, evaluator, and development-contamination controls are unchanged. No sealed-final path was accessed.
- **Gate status:** Starting-state gate **PASSED**.
- **Next action:** Implement a narrowly scoped, deterministic attempt-pool derivation and exact preflight for only `1.15`, `1.125`, and `1.10`, then evaluate them in descending order and stop at the first complete schedule.

### 2026-07-19 23:10:11 -04:00 — Milestone 7F multiplier 1.15 exact preflight failed

- **Action performed:** Derived the `1.15` pool as the ceiling of each frozen accepted family quota, recomputed submode/difficulty/output/split margins through the already-frozen policies, generated every fixed latent pool, and invoked the exact joint surface matcher with unchanged caps.
- **Result:** The candidate contains 2,302 attempts: targeted `633/268/250` and generic `385/383/383` for bookkeeping/rate/discrete. All eleven mathematical modes had enough unique latent candidates. Exact scheduling failed at `generic_control/constraint_distribution_or_discrete_reasoning/complete_packages` because its joint sentence-plan, plan/scenario, frame, exact-surface, and canonical runtime-identity assignment is infeasible.
- **Errors or uncertainty:** No approximate result was treated as capacity proof. The failed candidate produced no complete schedule and no schedule hash. Candidate configuration remains under the ignored raw-results boundary.
- **Gate status:** Multiplier `1.15` **FAILED**.
- **Next action:** Evaluate the next predeclared multiplier, `1.125`, through the identical exact scheduler; do not test `1.10` if `1.125` passes.

### 2026-07-19 23:12:33 -04:00 — Milestone 7F multiplier 1.125 exact preflight failed

- **Action performed:** Derived the 2,253-attempt `1.125` pool and sent all rate and discrete submodes through the bounded exact joint surface matcher. This replaced an inconclusive greedy `dual_capacity` exhaustion with a decision-quality exact preflight while retaining every cap and identity rule.
- **Result:** Targeted attempts are `619/263/245`; generic attempts are `376/375/375`. All unique latent requirements were met. Exact scheduling failed at `generic_control/rate_ratio_percentage_or_average/percentage` under the frozen plan, plan/scenario, frame, exact-surface, and canonical runtime-identity constraints.
- **Errors or uncertainty:** The first run reached the old greedy path for `dual_capacity`; that diagnostic was not used to reject the multiplier. Focused regression checks still prove that the historical 1.25 pool fails closed. No complete 1.125 schedule or schedule hash was produced.
- **Gate status:** Multiplier `1.125` **FAILED**.
- **Next action:** Evaluate the final predeclared multiplier, `1.10`. If it fails, invoke the mandatory stop rule; if it passes, freeze it and proceed without testing any other multiplier.

### 2026-07-19 23:15:29 -04:00 — Milestone 7F exact multiplier selection exhausted; stop rule invoked

- **Action performed:** Evaluated `1.10` through the corrected exact scheduler, then replayed `1.15` with that same final implementation so every recorded candidate result has identical decision authority. Built content-free selection evidence covering all three predeclared pools and no others.
- **Result:** `1.15` (2,302 attempts), `1.125` (2,253), and `1.10` (2,203) all fail exact joint scheduling at `generic_control/rate_ratio_percentage_or_average/percentage`. The initial 1.15 `complete_packages` failure remains a valid earlier subproblem failure, but the final uniform implementation exposes the rate-percentage blocker first. Selection configuration SHA-256 is `c5840a94...707e`; aggregate evidence SHA-256 is `df31ac17...7ad7`.
- **Errors or uncertainty:** No lower or unlisted multiplier was invented. The failure is a surface-compatibility/scheduling result; every candidate had sufficient unique mathematical latent programs. No multiplier was selected, so no complete schedule, fresh review schedule, 120-question smoke, deterministic smoke replay, or review packet exists.
- **Gate status:** Multiplier-selection gate **FAILED** after exhausting exactly the three approved candidates. The mandatory downstream stop rule is active.
- **Next action:** Document the exact stopped result, run repository-wide tests and safety checks, commit `analysis: record reduced-pool scheduling blocker`, push normally, and ask for the next architectural decision: reducing the accepted signal pilot itself.

### 2026-07-19 23:22:17 -04:00 — Milestone 7F final verification passed for stopped result

- **Action performed:** Ran repository-wide Ruff formatting/linting, strict Mypy, all unit and integration tests, `pip check`, and `git diff --check`; replayed all three derived candidate configs and the aggregate selection evidence; validated multiplier order, family ceilings, submode/difficulty/output/split margins, frozen reuse-cap separation, exact failure flags, and downstream artifact suppression. Scanned all 14 candidate paths for high-confidence secrets and exact/12-token matches against exactly 904 cached development questions in offline mode; reviewed protected generator/verifier/evaluator/contamination paths, sealed-path changes, raw/cache/environment/model tracking, ignore rules, and tracked sizes.
- **Result:** Ruff reports 134 files formatted and lint-clean; strict Mypy reports no issues in 85 source files; all 348 tests pass in 89.43 seconds; dependency and whitespace checks pass. Candidate configs and evidence replay exactly. Selection configuration SHA-256 is `c5840a94319351d59dcc1cac9a0b7aae9977928d53518044613c19f7161c707e`; evidence SHA-256 is `df31ac174fc2cac003f87b8a5b1ebd94a84f4aace93d0161975e4b001d027ad7`. Secret, exact-development, 12-token-development, protected-path, sealed-path, tracked raw/cache/environment/model, review-packet, and content-bearing evidence counts are zero. The largest tracked file is this DEVLOG at about 417 KiB, below 1 MiB.
- **Errors or uncertainty:** The first 1.125 diagnostic reached a legacy greedy path and was discarded as inconclusive; the allocator was then routed through the existing bounded exact matcher for every rate and discrete mode, all focused regressions passed, and every multiplier was reevaluated under that final implementation. Expected Windows LF-to-CRLF notices are non-fatal.
- **Gate status:** **VERIFIED FAILURE.** The accurate 14-path stopped result is eligible for `analysis: record reduced-pool scheduling blocker`. Smoke, replay, packet, complete dataset generation, and training remain prohibited.
- **Next action:** Fetch without integration, require `origin/main` to remain at `6a555405...f09088`, stage exactly the 14 reviewed paths, repeat cached scope/whitespace/secret/content/evidence/size checks, commit, push normally, and confirm clean `0/0` synchronization.

### 2026-07-19 - Milestone 7G Stage A: frozen starting state verified

- **Action performed:** Verified the required repository root, `main` branch, synchronized local and remote commit, ahead/behind state, clean worktree, ignored raw-results boundary, frozen attempt-policy configuration, canonical runtime identity tests, and exact-scheduler tests before editing.
- **Result:** Local `main` and `origin/main` both point to `b66290b633fac4434e680be3b9180966950e0ee9`, ahead/behind is `0/0`, and the starting worktree was clean. `results/raw/` remains ignored. The frozen attempt-policy configuration file SHA-256 remains `c5840a94319351d59dcc1cac9a0b7aae9977928d53518044613c19f7161c707e`; 11 focused policy, allocator, and normalizer tests passed.
- **Errors or uncertainty:** None. Exact commit equality and the empty starting diff establish that human-review evidence, repaired templates, generators, verifiers, canonical normalizer, reuse/balancing policies, evaluator, and contamination controls are unchanged. No sealed-final path was accessed.
- **Gate status:** Starting-state gate **PASSED**.
- **Next action:** Implement the predeclared descending accepted-size derivation and selection contract for exactly `900`, `800`, `700`, `600`, and `500`, all at the fixed `1.10` multiplier.

### 2026-07-19 - Milestone 7G size-selection contract implemented

- **Action performed:** Added `largest-feasible-matched-signal-pilot-v1`, stable largest-remainder family derivation, dynamic matched 90/10 split validation, exact 20% output-contract margins, family-level `ceil(accepted * 1.10)` attempts, and deterministic reuse of the frozen submode/difficulty/surface policies. Added focused tests for every approved size, exact quotas, invalid-size rejection, descending selection, and reproducible evidence.
- **Result:** The derived total fixed attempts are `1,981`, `1,762`, `1,544`, `1,320`, and `1,102` for accepted sizes `900`, `800`, `700`, `600`, and `500`, respectively. Ruff format/lint pass and all 22 focused size-selection, prior multiplier, and allocator tests pass.
- **Errors or uncertainty:** Exact surface schedulability has not yet been inferred from these arithmetic totals; each candidate must now pass the runtime-rendered joint matcher. No candidate schedule has yet been selected or persisted.
- **Gate status:** Derivation/implementation gate **PASSED**.
- **Next action:** Evaluate the five accepted sizes in strict descending order, stopping immediately at the first exact complete schedule.

### 2026-07-19 - Milestone 7G accepted size 900 exact preflight failed

- **Action performed:** Derived the matched 900-targeted/900-generic contract at the frozen `1.10` multiplier and ran all 1,981 fixed slots through the actual latent enumerator, deterministic renderer, canonical runtime identity function, and exact joint surface matcher.
- **Result:** Targeted accepted family quotas are `495/210/195`; generic quotas are `300/300/300`. All eleven mathematical modes supplied their required unique latent programs. Exact scheduling failed at `generic_control/rate_ratio_percentage_or_average/ratio_scale`; no complete schedule or schedule hash was produced. Candidate config SHA-256 is `0834b0a3...2311`; allocation SHA-256 is `311f9f4a...87c3`.
- **Errors or uncertainty:** The matcher proves infeasibility under the current joint surface constraints; it does not attribute the blocker to any one cap in isolation. No cap, template, or threshold was changed.
- **Gate status:** Accepted size `900` **FAILED**.
- **Next action:** Evaluate accepted size `800` through the identical exact scheduler.

### 2026-07-19 - Milestone 7G accepted size 800 exact preflight failed

- **Action performed:** Derived the matched 800-targeted/800-generic contract and ran its 1,762 fixed attempts through the unchanged exact scheduler.
- **Result:** Targeted accepted family quotas are `440/186/174`; generic quotas are `267/267/266`. Unique latent supply passed in all modes. Exact surface assignment failed at `generic_control/rate_ratio_percentage_or_average/rate_total`. Candidate config SHA-256 is `8d3a4d0a...799e`; allocation SHA-256 is `09ddab2b...793b`.
- **Errors or uncertainty:** No complete schedule was available to hash. The exact matcher retained all joint caps and runtime identities.
- **Gate status:** Accepted size `800` **FAILED**.
- **Next action:** Evaluate accepted size `700` through the identical exact scheduler.

### 2026-07-19 - Milestone 7G accepted size 700 exact preflight failed

- **Action performed:** Evaluated the 1,544-attempt size-700 pool. The general multidimensional matcher first exhausted its deterministic proof bound on generic weighted average. Because that is neither a feasibility proof nor an infeasibility proof, weighted average was routed through the existing stable constructive allocator, whose priority order directly enforces runtime identity, plan/scenario, frame, plan, and exact-text headroom. The unchanged exact matcher remained authoritative for every other finite rate/discrete mode.
- **Result:** Targeted accepted family quotas are `385/163/152`; generic quotas are `234/233/233`. The constructive weighted-average assignment passed. Exact scheduling then conclusively failed at `generic_control/constraint_distribution_or_discrete_reasoning/two_type_allocation`. Candidate config SHA-256 is `2383ae36...3983`; allocation SHA-256 is `045ec71b...d774`.
- **Errors or uncertainty:** The initial search-bound result was rejected as inconclusive and is not the recorded capacity decision. No cap or identity changed; a focused routing regression prevents recurrence.
- **Gate status:** Accepted size `700` **FAILED** under the final deterministic allocator.
- **Next action:** Evaluate accepted size `600` through the same final allocator.

### 2026-07-19 - Milestone 7G accepted size 600 exact preflight failed

- **Action performed:** Derived and exactly scheduled the matched 600-targeted/600-generic candidate contract with 1,320 total fixed attempts.
- **Result:** Targeted accepted family quotas are `330/140/130`; generic quotas are `200/200/200`. All latent requirements passed. Exact assignment failed at `generic_control/rate_ratio_percentage_or_average/rate_total`. Candidate config SHA-256 is `a781dbbe...35d1`; allocation SHA-256 is `c5bedfbd...21b`.
- **Errors or uncertainty:** None. No complete schedule was produced and no frozen rule changed.
- **Gate status:** Accepted size `600` **FAILED**.
- **Next action:** Evaluate the final approved size, `500`; if it fails, invoke the architecture stop rule.

### 2026-07-19 - Milestone 7G size selection exhausted; architecture stop invoked

- **Action performed:** Evaluated the final matched 500-targeted/500-generic candidate, then froze the descending result set for all five approved sizes at the one approved `1.10` multiplier.
- **Result:** Size 500 contains 1,102 attempts, with targeted accepted family quotas `275/117/108` and generic quotas `167/167/166`. All eleven latent pools passed; exact scheduling failed at `generic_control/rate_ratio_percentage_or_average/ratio_scale`. Candidate config SHA-256 is `d388b40f...38e2`; allocation SHA-256 is `dd822482...7c90`. The complete order is: 900 failed generic ratio-scale; 800 failed generic rate-total; 700 failed generic two-type allocation; 600 failed generic rate-total; 500 failed generic ratio-scale.
- **Errors or uncertainty:** No approved size remains. The scheduler failures concern joint surface compatibility, not mathematical generation or verifier capacity. No result was selected based on desired scale or expected benchmark performance.
- **Gate status:** Size-selection gate **FAILED** after exactly five candidates. The mandatory architecture-stop rule is active.
- **Next action:** Persist content-free aggregate evidence, update every required project record, verify the stopped result repository-wide, commit `analysis: record reduced signal-pilot blocker`, push, and stop. No complete schedule, 120-question smoke, replay, or review packet may be created.

### 2026-07-19 - Milestone 7G final verification passed for stopped result

- **Action performed:** Ran repository-wide Ruff formatting/linting, strict Mypy, all unit and integration tests, `pip check`, and `git diff --check`; replayed all five exact size preflights under the final allocator; reproduced the aggregate evidence and internal hash; validated size order, family largest-remainder quotas, 90/10 splits, 20% output track, family-level 1.10 ceilings, submode allocations, downstream stop flags, and absence of packet/schedule artifacts. Scanned all 14 candidate paths for high-confidence secrets and exact/12-token matches against exactly 904 cached development questions in offline mode; checked protected generator/evaluator paths, sealed-path changes, tracked raw/cache/environment/model files, ignore rules, and tracked sizes.
- **Result:** Ruff is format/lint clean; strict Mypy reports no issues in 86 source files; all 361 tests pass in 88.87 seconds; dependencies and whitespace pass. All five exact blocker decisions replay in order. Selection config SHA-256 is `073f03bbc291502fb92f39c22f4fae70d5496a077d346f82e1a087438c7bbd49`; evidence SHA-256 is `793c0276b42fb7bd26dbdfd95a316ec4fbcb4d6178d5567011d83013b96a407f`. Secret, exact-development, 12-token-development, protected-path, sealed-path, tracked raw/cache/environment/model, and review-packet counts are zero. The largest candidate is this DEVLOG at about 428 KiB before this entry, below 1 MiB.
- **Errors or uncertainty:** The first size-700 diagnostic hit a generic backtracking proof bound and was rejected as inconclusive. The final allocator uses the pre-existing stable constructive weighted-average assignment, passed focused regression, and then produced a conclusive exact two-type-allocation blocker. Expected Windows LF-to-CRLF notices are non-fatal.
- **Gate status:** **VERIFIED FAILURE.** No approved accepted size is schedulable. The accurate stopped result is eligible for commit and push; no smoke, replay, packet, complete dataset, or training is permitted.
- **Next action:** Fetch without integration, require `origin/main` to remain at `b66290b6...ee9`, stage exactly the reviewed 14 paths, repeat index-only scope/whitespace/secret/content/size checks, commit `analysis: record reduced signal-pilot blocker`, push normally, and confirm clean `0/0` synchronization.

### 2026-07-20 - Fast-Track 8A Stage A: starting state verified

- **Action performed:** Read the complete Fast-Track 8A-8C authorization; verified repository root, branch, local/remote tips, divergence, clean worktree, raw ignore boundary, frozen evaluator, generator/verifier tests, repaired bank, genuine review evidence, and contamination tests without opening any sealed-final path.
- **Result:** Local `main` and `origin/main` both point to `a005dda6bf42d09b53a3d50e30cc7fc5a1d90654`; divergence is `0/0`; the worktree was clean. Genuine review SHA-256 remains `564a8ca5...0791`; 72 focused tests pass.
- **Errors or uncertainty:** One exploratory hash command named a nonexistent generic template-bank config and reported only that missing path; the actual typed bank and its focused tests were then verified. No state changed.
- **Gate status:** Starting-state gate **PASSED**.
- **Next action:** Freeze and fixture-calibrate `matched-template-signal-v1` before generating any candidate.

### 2026-07-20 - Fast-Track 8A Stage B: matched-template policy frozen

- **Action performed:** Formalized global exact/normalized/latent/example uniqueness, balanced repeatability for reviewed language identities, `ceil(accepted/inventory)+2` plan/frame/scenario caps, and a 15% number-neutral concentration limit. Added eight original fixtures covering exact/latent/cross-dataset copies, harmless reviewed reuse, concentration, and benchmark contamination.
- **Result:** Calibration passes 8/8 fixtures. Fixture SHA-256 is `63b9eec6...2bbe`; policy `7e56acfa...3518`; canonical configuration `563a52c4...582a`; config file `6f2717d5...f753`; calibration `2e1e7915...aaa1`. Ruff and three focused tests pass.
- **Errors or uncertainty:** None. Benchmark contamination, templates, generators, labels, and verifiers remain unchanged. Number-neutral identity remains measured but is no longer a one-use rejection key.
- **Gate status:** Policy gate **PASSED**.
- **Next action:** Freeze the exact 500+500 quotas and 550+550 fixed attempt schedule, then generate without replacements.

### 2026-07-20 - Fast-Track 8A matched dataset implementation

- **Action performed:** Implemented the strict 500-by-2 dataset contract, exact family/mode/difficulty/output/split quota derivation, deterministic latent selection, balanced reviewed-template assignment, and content-free schedule serialization.
- **Result:** Every scheduled candidate is reconstructible from content-free seeds and assignments; reconstruction checks latent, semantic-IR, exact-text, normalized-text, render-signature, and canonical number-neutral identities. Focused Ruff, strict Mypy, and matched-policy tests pass after resolving only mechanical typing and formatting issues.
- **Errors or uncertainty:** The content-free schedule has not yet been frozen and no candidate has been screened. This prevents contamination outcomes from influencing allocation.
- **Gate status:** Implementation preflight **PASSED**.
- **Next action:** Freeze and hash the complete 1,100-slot schedule, then verify quota and uniqueness invariants before fixed-pool execution.

### 2026-07-20 - Fast-Track 8A fixed schedule frozen

- **Action performed:** Constructed the complete fixed candidate schedule before any benchmark-contamination outcome was inspected, then rebuilt it independently and added focused quota, uniqueness, reconstruction, and tamper-failure tests.
- **Result:** The schedule contains exactly 1,100 slots, 1,100 unique latent-program hashes, 1,100 unique exact-question hashes, and 1,100 unique synthetic IDs. Family attempts are `302/129/119` targeted and `184/184/182` generic. Maximum planned sentence-plan/frame/scenario use is `5/17/15`; 730 number-neutral identities are present. Schedule SHA-256 is `a70cb62ce233724113293de7feedb1b00efe0cd1ba7b607e48831b8cb07dd5eb`; quota SHA-256 is `282647af2bbba68713aa3c401cfb7a77f177a8d560d1c1c4866236dc12ab1a89`.
- **Errors or uncertainty:** Number-neutral repetition is intentionally permitted under the frozen 15% stratum cap. The tracked content-free schedule is approximately 1.61 MB and will receive explicit tracked-size review.
- **Gate status:** Schedule-freeze gate **PASSED**.
- **Next action:** Execute exactly the frozen 1,100 attempts through all deterministic, dual-verifier, lexical, structural, n-gram, and pinned MiniLM development-contamination gates, reporting at each quarter.

### 2026-07-20 - Fast-Track 8A fixed pool generated and replayed

- **Action performed:** Executed all 1,100 pre-scheduled attempts with no replacement or post-result change, using the repaired bank, typed renderer, dual verifiers, deterministic quality controls, development lexical/n-gram screening, and pinned offline MiniLM. Reconstructed and reran the complete pool.
- **Result:** Quarter checkpoints were `275/275`, `500/550`, `775/825`, and `1000/1100` accepted. The final 100 rejections were only `quota_cell_filled`. Exact targeted family acceptances are `275/117/108`; generic are `167/167/166`; difficulty is targeted `167/167/166` and generic `168/167/165` for easy/medium/hard; each dataset has 100 output-track examples and a `450/50` split. Both decision hashes are `4574c969...ea93`. Dataset hashes are targeted `987712f6...2876` and generic `49294282...2e7e`.
- **Errors or uncertainty:** None. Peak process RSS was 945,278,976 bytes; runtime was 28.04 seconds; ignored raw artifacts use 7,747,981 bytes.
- **Gate status:** Dataset-generation and deterministic-reconstruction gates **PASSED**.
- **Next action:** Audit all 1,000 accepted question texts blind to labels and build the ignored stratified human-review packet.

### 2026-07-20 - Fast-Track 8A language audit and review packet complete

- **Action performed:** Applied the frozen six-part language rubric to all 1,000 accepted records using only candidate ID and rendered question. Created ignored Markdown, static HTML, Codex audit JSON, and assisted HTML; selected a deterministic 100-question high-confidence sample split 50/50 between datasets and balanced by family, difficulty, and output track.
- **Result:** Codex advisory recommendations are 1,000 approve, 0 reject, and 0 unsure, all high confidence; no defect category or three-plan systematic issue was found. Audit SHA-256 is `e148e8fd...e99d`; packet SHA-256 is `ca5a3e01...31ab`. Static packet validation confirms 100 unique IDs, matching identity hash, unique DOM IDs, all A/R/U/navigation/export controls, no preselected decision, and Git ignore coverage.
- **Errors or uncertainty:** This is AI-assisted inspection, not human review. The in-app browser security policy blocked direct `file:///` navigation, so no workaround was used; page verification is static/script-contract validation rather than a live-render claim.
- **Gate status:** Automatic data-quality gate **PASSED**; genuine human review remains pending and the one-seed result must remain provisional.
- **Next action:** Update all required records, run repository-wide dataset-stage verification, publish the first atomic commit, then create the isolated training environment.

### 2026-07-20 - Fast-Track 8A dataset-stage verification passed

- **Action performed:** Ran repository-wide Ruff format/lint, strict Mypy, every unit and integration test, `pip check`, and `git diff --check`; validated the policy fixtures, exact quotas, 1,100-slot schedule reconstruction, 1,000 accepted records, manifest/summary/decision hashes, concentration caps, exact/normalized/latent uniqueness, cross-dataset and split isolation, verifier outcomes, contamination outcomes, packet identity and ignore rules. Scanned all 20 candidate paths for high-confidence secrets and exact/12-token matches against 904 locally cached development questions in offline mode; reviewed protected evaluator/generator/verifier/contamination paths, sealed-path changes, raw/environment tracking, and candidate sizes.
- **Result:** Ruff is clean; strict Mypy reports no issues in 89 source files; 363 unit and six integration tests pass; dependencies and whitespace pass. Secret, exact-development, 12-token-development, protected-path, sealed-path, tracked raw/environment, cap, uniqueness, overlap, verifier, contamination, manifest, summary, and replay inconsistency counts are zero. All ten local data/review artifacts are ignored. The content-free schedule (1,606,584 bytes) and ID/hash manifest (1,200,111 bytes) exceed 1 MiB and were explicitly reviewed; neither contains rendered questions, answers, traces, or complete dataset rows.
- **Errors or uncertainty:** Expected Windows LF-to-CRLF notices are non-fatal. Live `file:///` UI testing remains blocked by browser policy; static packet integrity tests pass.
- **Gate status:** Dataset publication gate **PASSED**.
- **Next action:** Fetch without integration, require the unchanged remote tip, stage exactly the 20 reviewed paths, repeat index-only scope/safety checks, commit `data: generate matched 500-example signal datasets`, push normally, and confirm clean `0/0` synchronization.

### 2026-07-20 - Fast-Track 8A dataset commit published

- **Action performed:** Fetched without integration, verified `origin/main` remained unchanged, staged exactly 21 reviewed content-free implementation/evidence/documentation paths, repeated scope and whitespace checks, and committed with a temporary process-only reuse of the latest repository author identity because Git configuration was intentionally left untouched.
- **Result:** Commit `021517438e62210bc37a783269f4b7f367570745` (`data: generate matched 500-example signal datasets`) was pushed. Local and `origin/main` match with `0/0` divergence and a clean worktree. No raw dataset, rendered question, review page, environment, cache, adapter, or prediction was pushed.
- **Errors or uncertainty:** The first commit command failed because this shell had no author identity. No Git configuration was changed; the retry used environment variables only.
- **Gate status:** Dataset publication gate **PASSED**.
- **Next action:** Create and validate the isolated native-Windows QLoRA environment.

### 2026-07-20 - Fast-Track 8B training environment created

- **Action performed:** Created ignored `.venv-training` with CPython 3.12.10 and installed the predeclared minimal stack: PyTorch 2.5.1+cu121, Transformers 4.51.3, tokenizers 0.21.4, PEFT 0.15.2, TRL 0.17.0, bitsandbytes 0.49.2, and Accelerate 1.7.0. Ran dependency, import, CUDA, NF4, and paged-optimizer probes.
- **Result:** Environment creation took 4.59 seconds and dependency installation 123.86 seconds. `pip check` passes; CUDA reports RTX 3080 with 10,736,893,952 bytes. Native NF4 CUDA matrix multiplication and `PagedAdamW8bit` update both pass, reserving 23,068,672 bytes in the probe.
- **Errors or uncertainty:** The first low-level probe called an outdated internal `functional.matmul_4bit` location and failed before work; the unchanged public `bitsandbytes.matmul_4bit` API then passed. No package changed.
- **Gate status:** Training-environment gate **PASSED**.
- **Next action:** Freeze the exact lock, QLoRA recipe, SFT formatting hash, and 32-step smoke implementation before the first optimizer step.

### 2026-07-20 - Fast-Track 8B QLoRA recipe frozen

- **Action performed:** Added an isolated training dependency lock, a strict typed QLoRA recipe, exact chat-format contract, offline data loader, quantized LoRA runtime, final-adapter hashing, reload/inference check, and focused configuration tests. Added `.venv-training/` to the exact Git ignore list after confirming the general `.venv/` entry did not cover it.
- **Reason:** The generic and targeted runs must share one immutable recipe, and no optimizer step may occur until the model revision, tokenizer/chat hashes, dependency lock, SFT formatting, quantization, LoRA modules, optimizer, step count, and checkpoint rule are frozen.
- **Result:** Recipe SHA-256 is `4a9c6043f72d4f5b83dad774ffcd208e17f8c9738c9b34b0ab06919ba2620590`; SFT-format SHA-256 is `34d894187c86e47538dd422b2487de6802627a2435ce216dd69f76a2ed568f14`; dependency-lock SHA-256 is `fc158cd278124af82406a110afb5efcde2346776dce79875fe3cf6aa5ccb4755`. Focused Ruff, strict Mypy, and four unit tests pass.
- **Errors or uncertainty:** The 32-step smoke has not yet run, so full NF4 model loading, backward compatibility, optimizer stability, peak memory, adapter serialization, and reload inference remain gated. No hyperparameter has been tuned.
- **Next action:** Run exactly 32 optimizer steps on at most 128 targeted training records and stop if any compatibility or memory requirement fails.

### 2026-07-20 - Fast-Track 8B deterministic CUDA preflight correction

- **Action performed:** Launched the frozen smoke once, observed a fail-closed cuBLAS determinism error on the first forward pass, confirmed zero optimizer steps completed, and added the required `CUBLAS_WORKSPACE_CONFIG=:4096:8` process setting before the training runtime imports PyTorch.
- **Reason:** `torch.use_deterministic_algorithms(True)` requires a fixed cuBLAS workspace on CUDA 10.2 or newer. The process-level setting makes the frozen run reproducible without changing any model, data, optimizer, or hyperparameter choice.
- **Result:** The first launch performed no training and created no accepted smoke result. Qwen loaded in 4-bit on CUDA and reached the first forward operation before PyTorch stopped it exactly as intended.
- **Errors or uncertainty:** Transformers emitted a Qwen sliding-window/SDPA warning and a PEFT label-name warning; neither caused the stop. Backward, optimizer, memory, save, and reload checks remain untested until the corrected launch completes.
- **Next action:** Rerun the identical 32-step smoke with the deterministic cuBLAS process contract active.

### 2026-07-20 - Fast-Track 8B 32-step QLoRA compatibility gate passed

- **Action performed:** Loaded the pinned Qwen base from the offline cache in NF4 with double quantization, attached rank-16 LoRA modules to all seven approved projection types, trained exactly 32 optimizer steps on the first 128 frozen targeted training records, evaluated at step 25 on all 50 targeted synthetic-validation records, saved one ignored adapter, reloaded it offline, and completed one deterministic non-benchmark inference.
- **Reason:** Native Windows 4-bit model loading alone does not prove that forward, backward, paged 8-bit optimization, gradient accumulation/checkpointing, serialization, and reload all work within the RTX 3080 memory gate.
- **Result:** Gate passed: 32/32 steps, 256 examples processed, 131,072 padded model-input tokens and 50,506 non-padding tokens; logged loss 2.5810 to 0.5413; step-25 validation loss 0.633282; 3,343,800,832 peak allocated and 3,741,319,168 peak reserved VRAM; 2,147,590,144-byte process RSS; 18,464,768 trainable LoRA parameters and no non-LoRA trainable parameters. Runtime was 102.395 seconds after 2.596 seconds setup. The 89,796,953-byte adapter hash is `11159bd5051bf71095952e8ae677ad73d1a7dae545ea7c890634c8155fb77849`; offline reload inference passed.
- **Errors or uncertainty:** A pre-optimizer launcher attempt failed because the isolated environment lacked the repository import path; a subsequent pre-optimizer attempt stopped on the required cuBLAS deterministic workspace setting. Neither completed an optimizer step. The successful counted run emitted non-fatal sliding-window, PEFT label-name, and inherited sampling-parameter warnings. This smoke checks compatibility, not model quality.
- **Next action:** Run the complete training-setup verification and publish the second atomic commit before training either final adapter.

### 2026-07-20 - Fast-Track 8B training-setup verification passed

- **Action performed:** Ran repository-wide Ruff formatting/lint, strict Mypy, every unit and integration test, main and training environment dependency checks, whitespace validation, frozen recipe/lock/tokenizer/chat/data hash checks, smoke-gate assertions, secrets scanning, exact and 12-token development-content scanning against all 904 cached development questions, protected/sealed path review, raw/cache/environment tracking checks, adapter/environment ignore checks, tracked-size review, and complete status/diff review.
- **Reason:** Final adapters may be trained only from a verified published setup that cannot accidentally contain benchmark content, raw data, adapters, caches, environment files, secrets, or a mechanically drifting recipe.
- **Result:** Ruff is clean; strict Mypy reports no issues in 92 source files; 367 unit and six integration tests pass; both environments have no broken requirements; whitespace passes. Recipe and smoke gates pass, and exact/12-token development hits, secret hits, protected/sealed candidate paths, and tracked raw/cache/environment paths are all zero. `.venv-training/` and the smoke adapter are ignored. The two pre-existing content-free files above 1 MiB were reviewed again; no new training file exceeds the threshold.
- **Errors or uncertainty:** The first full unit collection found a duplicate `test_config.py` module name in the nested training test directory; renaming it to `test_qlora_recipe.py` resolved collection and the full suite passed. Expected Git LF-to-CRLF notices are non-fatal. No sealed-final artifact was opened.
- **Next action:** Fetch without integration, require `origin/main` to remain synchronized, stage exactly the 16 approved setup files, repeat index-only safety checks, commit `train: validate RTX 3080 QLoRA compatibility`, push normally, and confirm a clean synchronized branch before final training.

### 2026-07-20 - Fast-Track 8B training setup published

- **Action performed:** Published `train: validate RTX 3080 QLoRA compatibility` and verified the local and remote branch tips after a normal push.
- **Result:** Commit `6d2b96e845870ed5c56cfbb3ba115cbd311d0e71` is on both local `main` and `origin/main` at 0 ahead/0 behind with a clean worktree. No environment, adapter, raw data, cache, benchmark content, or secret was pushed.
- **Next action:** Train the generic-control adapter first under the published recipe.

### 2026-07-20 - Fast-Track 8C generic-control adapter frozen

- **Action performed:** Trained the generic-control adapter from the pinned base and frozen 450/50 generic split for exactly 200 optimizer steps, using the published seed, effective batch, sequence length, optimizer, validation cadence, and final-adapter-only rule. The development benchmark was not loaded.
- **Result:** Training completed 200/200 steps and 1,600 examples in 641.366 seconds. It processed 819,200 padded input tokens and 271,396 non-padding loss tokens; logged loss moved from 3.1699 to 0.1179 and final synthetic-validation loss was 0.153627. Peak allocated/reserved VRAM was 3,343,800,832/3,577,741,312 bytes; process RSS was 1,542,365,184 bytes. The ignored 89,796,953-byte adapter SHA-256 is `36b19165e348fecef09826c8b8807f75c2071be9c2aa622455cf723fb112e3ac`.
- **Errors or uncertainty:** No OOM, NaN, offload, backend, forward, backward, optimizer, or save failure occurred. Development performance is unknown and remains prohibited until both adapters and parity checks pass.
- **Next action:** Train the targeted adapter with the exact same published recipe and seed, then load both adapters and evaluate the predeclared parity gate before any benchmark inference.

### 2026-07-20 - Fast-Track 8C targeted adapter frozen

- **Action performed:** Trained the targeted adapter from the pinned base and frozen 450/50 targeted split for exactly 200 optimizer steps under the same published recipe and seed. The development benchmark was not loaded.
- **Result:** Training completed 200/200 steps and 1,600 examples in 645.737 seconds. It processed 819,200 padded input tokens and 306,766 non-padding loss tokens; logged loss moved from 2.7859 to 0.1199 and final synthetic-validation loss was 0.144995. Peak allocated/reserved VRAM was 3,343,800,832/3,577,741,312 bytes; process RSS was 1,542,606,848 bytes. The ignored 89,796,953-byte adapter SHA-256 is `217a9bcf2a66dcefae1359409dc3962d554de1fad5f1aa880b8a89d38a36406e`.
- **Errors or uncertainty:** No mechanical training failure occurred. Development performance remains unknown.
- **Next action:** Load and hash both frozen adapters, compare every parity field, and apply the predeclared 2% loss-token gate before any benchmark access.

### 2026-07-20 - Fast-Track 8C training parity gate failed; evaluation stopped

- **Action performed:** Recomputed both adapter directory hashes, loaded each adapter offline on its own quantized base, verified all parameters stayed on CUDA, compared the frozen run summaries field by field, and applied the predeclared non-padding token threshold.
- **Result:** Both adapter hashes match and both load with zero offloaded parameters. Base revision, packages, recipe hash, seed-bound configuration, 200 steps, 1,600 examples, 819,200 padded tokens, sequence length, and final-only rule match. Generic processed 271,396 non-padding tokens; targeted processed 306,766. The 35,370-token difference is 11.529961%, above the 2% maximum. Training parity gate failed and `benchmark_evaluation_authorized` is false.
- **Gate status:** **FAILED.** The frozen 814-example development evaluator was not opened or run for either adapter. No generic/targeted development or category-level result exists, and the one-seed signal gate cannot be evaluated.
- **Errors or uncertainty:** Equal padded tensor shapes do not make the loss-bearing token exposure equal because padding labels are masked. The result diagnoses an experimental-control mismatch, not adapter quality.
- **Next action:** Document and publish the accurate stopped result. The narrowest future decision is whether to authorize a new pre-training token-budget-matching design and fresh retraining; do not evaluate these adapters.

### 2026-07-20 - Fast-Track 8C stopped-result verification passed

- **Action performed:** Ran repository-wide Ruff format/lint, strict Mypy, all unit and integration tests, both environment dependency checks, whitespace validation, dataset/split/recipe/adapter/summary hashes, exact step/seed/recipe/padded-token checks, offline adapter loads, fail-closed parity assertions, evaluator/synthesis freeze review, secret scanning, exact and 12-token development-content scans against all 904 cached development questions, protected/sealed path review, raw/cache/environment/adapter ignore and tracking checks, tracked-size review, and complete status review.
- **Result:** Ruff is clean; strict Mypy reports no issues in 93 source files; 369 unit and six integration tests pass; both environments have no broken requirements; whitespace passes. Dataset/split and both adapter hashes match. Parity summary SHA-256 is `31c2cb90faa8d0c0ef94fd40b4334f9743979d04406d78a9b3b1ab6fe3d9762c`; generic/targeted training summary hashes are `f95bda22a4ffb580b5e6e2603e116191f28827a8156eebe4bc75368644ed60ed` and `b54c5d6cfbd7f2b21c05ede27b7e9c078971b4386f24659d307973f9569e74cf`. Exact/12-token development hits, secret hits, evaluator/synthesis changes, protected/sealed candidates, and tracked raw/cache/environment artifacts are all zero. The two pre-existing content-free tracked files above 1 MiB remain reviewed.
- **Errors or uncertainty:** One consistency command initially compared the tracked nested split-hash mapping directly with the recipe's flattened mapping and stopped; a corrected read-only normalization proved all four hashes equal. The gate remains a verified failure, not an implementation error. No development prediction file exists for either adapter.
- **Next action:** Fetch without integration, require `origin/main` to remain at the published training-setup commit, stage only this content-free stopped result, repeat index checks, create an accurate blocker commit, push normally, and report no adapter development scores.

### 2026-07-20 - Milestone 8D starting state verified

- **Action performed:** Read the complete Milestone 8D authorization; verified repository root, `main`, local/remote commit `9ac4202c7e2a0271a4c88d5202d21188250a6b6b`, 0/0 divergence, clean worktree, raw/environment/model-cache ignore boundaries, both frozen dataset hashes, all four split hashes, the base-model revision, original recipe and SFT-format hashes, training-lock integrity, package/CUDA/GPU versions, evaluator source/config/development-manifest hashes, and both preserved parity-failed adapter hashes.
- **Reason:** Token-matched scheduling must begin from the exact published evidence and must not overwrite or benchmark the scientifically invalid equal-example adapters.
- **Result:** Every approved starting hash and version matches. The generic and targeted adapter directory hashes remain `36b19165e348fecef09826c8b8807f75c2071be9c2aa622455cf723fb112e3ac` and `217a9bcf2a66dcefae1359409dc3962d554de1fad5f1aa880b8a89d38a36406e`; both remain ignored and are classified `invalid_for_comparative_evaluation_due_to_token_parity_failure`. `pip check` passes and CUDA sees the 10,240 MiB RTX 3080.
- **Errors or uncertainty:** Two initial read-only verification probes failed before inspecting data because Windows native-argument quoting removed Python string quotes; a stdin-based retry completed successfully. No tracked or ignored artifact changed. No sealed-final file was opened, hashed, inspected, or compared.
- **Next action:** Tokenize all 900 frozen training examples through the existing SFT contract, persist full census rows only under the ignored raw directory, and publish aggregate content-free census evidence.

### 2026-07-20 - Milestone 8D token census passed

- **Action performed:** Added a deterministic census implementation and focused tests, then formatted every frozen training record with the existing three-message SFT contract and tokenized it with the pinned offline tokenizer, 512-token limit, unchanged truncation behavior, and unchanged `input token when attended, otherwise -100` label construction. Replayed the complete census before writing aggregate evidence.
- **Result:** All 900 records have loss-bearing labels and zero records truncate. Generic totals 77,348 loss-bearing tokens across 450 unique records (mean 171.8844, median 144, range 111-323); targeted totals 87,317 (mean 194.0378, median 184, range 111-320). Census SHA-256 values are `eee9b961e28066f7954790d99b94dfd1f05aa28e4da5dd2a326c1124b261e6a3` and `3782412c5b0eeba303c7e4c51a1f963bc35082523a18d980fda590f032eb59f8`; aggregate-summary SHA-256 is `59ffbd01ba613d567eea82d31288b2c006e1d1a0817820ebb3e2d5f0d6e547e7`.
- **Gate status:** **PASSED.** Formatting, tokenizer, chat-template, SFT-format, label masking, sequence length, and reconstruction checks all match across arms.
- **Errors or uncertainty:** None. Complete census rows remain under `results/raw/training/token_matched_v2/` and are ignored. No example text, completion, or label was changed.
- **Next action:** Attempt Method A exactly once with 1,600 occurrences per arm, stratified largest-remainder fourth repeats, and deterministic eight-example step balancing.

### 2026-07-20 - Milestone 8D Method A parity gate failed

- **Action performed:** Allocated exactly 250 fourth repeats in each arm by deterministic largest remainder over family x difficulty x output-contract strata. Within those fixed quotas, selected the longest permissible generic examples and shortest permissible targeted examples, which proves the minimum attainable cross-arm gap because targeted remains larger at that boundary. Ordered each 1,600-occurrence multiset into 200 capacity-eight steps using stable longest-processing-time balancing and replayed both schedules.
- **Result:** Generic totals 278,167 loss-bearing tokens with schedule SHA-256 `649f8013f90ac56478ccdaa71734d989f736553a232d97517c6b461ff29485bb`; targeted totals 307,144 with SHA-256 `c16ecd2ec2a5c335d658cb955332bc3a96286ea65f9586316d3abbb065bb440c`. The exact lower-bound gap is 28,977 tokens, or 9.434337%, versus the frozen 0.5% maximum. Step totals span 1,386-1,394 generic and 1,531-1,538 targeted. Summary SHA-256 is `a7a68503e9f028e74e6559ab354487c08d067567be03a585991fc4adc271fca5`.
- **Gate status:** **FAILED.** Occurrence count, optimizer-step count, 3/4 reuse balance, exact stratum quotas, deterministic reconstruction, content integrity, and safety-envelope checks pass; only token parity fails.
- **Errors or uncertainty:** None. The failure is mathematically unavoidable under Method A's fixed stratum quotas, not a heuristic miss.
- **Next action:** Preserve Method A evidence and implement the preapproved Method B whole-example token-budgeted accumulation fallback, including numerical gradient-equivalence and schedule-accounting tests, before any GPU training.

### 2026-07-20 - Milestone 8D Method B schedule gate and protocol freeze passed

- **Action performed:** Implemented the preapproved whole-example token-budgeted fallback. Deterministic balanced cycles select 1,578 generic and 1,398 targeted occurrences without splitting or packing examples; stable token balancing partitions each arm into exactly 200 optimizer steps. Real four-step windows were moved to the schedule prefixes to support the bounded smoke without a second schedule. Added the frozen v2 recipe contract and custom token-weighted QLoRA runtime, plus focused tests for exact schedule loading, padding exclusion, sequence-boundary preservation, fail-closed token mismatches, and numerical loss/gradient equivalence to a reference combined-token mean.
- **Result:** Generic schedules 271,292 loss-bearing tokens with schedule SHA-256 `38c030d703268e7046a8fa73dd63eceb54f98b52b1d0720a1d0188621992d7f0`; targeted schedules 271,150 with SHA-256 `76f4382505e8985c04d997759f01c508d14784d9e0e1f31e9d2d26eab1decc44`. The 142-token pairwise difference is 0.052342%, both arms are within 0.034% of the 271,200 nominal budget, each example appears three or four times, and every family x difficulty x output stratum differs by at most one repeat. Maximum step totals are 1,373 and 1,360, below the previously exercised 4,096-token padded-step envelope. First-four-step totals are 5,464 and 5,440, a 0.439239% difference. Method B summary SHA-256 is `13d3c75e59ea1e182a6934bffedfcd50ccfa2a34e0300719ffa8fa7c4a6efaf5`.
- **Gate status:** **PASSED.** All Method B scheduling, nominal-budget, pairwise-parity, occurrence-coverage, stratum-balance, safety, smoke-prefix, whole-example, content-integrity, and deterministic-reconstruction checks pass. Seven focused tests pass; strict Mypy and focused Ruff checks are clean.
- **Errors or uncertainty:** The first schedule ordering placed matching four-step windows at the same source index and yielded 0.5490% smoke-prefix parity, just outside the future smoke gate. Before protocol freeze or any model run, the deterministic ordering implementation was corrected to choose the closest real window independently in each arm; no occurrence, total, stratum quota, threshold, or content changed. The frozen result passes at 0.4392%.
- **Next action:** Run one four-optimizer-step fresh-adapter smoke per arm from the identical base, require actual IDs/tokens/steps and scheduler steps to match the frozen prefixes, verify finite losses/gradients, CUDA residency, save, and offline reload.

### 2026-07-20 - Milestone 8D four-step token-parity smoke passed

- **Action performed:** Ran generic then targeted fresh-adapter smokes from the identical pinned base using the first four real steps of each frozen Method B schedule. The custom runtime verified every scheduled occurrence and loss-token count before execution, token-weighted each mean loss, clipped/updated/cleared once per step, saved each ignored adapter, rehashed it, and reloaded it offline on a fresh quantized base.
- **Result:** Generic processed 32 occurrences and exactly 5,464 scheduled/actual loss tokens in 12.260 seconds; targeted processed 28 and exactly 5,440 in 10.778 seconds. Both completed four optimizer and four scheduler steps with finite losses and gradients, zero truncation, no development exposure, and 3,343,712,768/3,577,741,312 bytes peak allocated/reserved VRAM. Adapter hashes are `a86e450af1bf3671c8578d28cad45d6507f919e305dc62c552d27dc5bda7c17e` and `f8af4ad4e38271b986a8e5744f7ed3939232067a648cdbe0ab80b6f2b6d87811`; both reload offline. Actual parity is 24 tokens / 0.439239%. Parity-summary SHA-256 is `6913bda010c407f5e79ce8647ae317771c61858a048a79ea986819577c7ac4d9`.
- **Gate status:** **PASSED.** All metadata, schedule, actual-token, finite-loss, finite-gradient, optimizer/scheduler-step, CUDA, save/hash, offline-reload, and <=0.5% parity requirements pass.
- **Errors or uncertainty:** Both runs emitted the inherited non-fatal sliding-window SDPA and future checkpoint `use_reentrant` warnings already seen in the original compatible recipe. No OOM, offload, backend, optimizer, or serialization failure occurred.
- **Next action:** Run complete repository and protocol verification, publish `train: freeze token-matched QLoRA protocol`, and require a clean synchronized branch before either 200-step retraining run.

### 2026-07-20 - Milestone 8D protocol verification passed

- **Action performed:** Ran repository-wide Ruff formatting/linting, strict Mypy, all unit and integration tests, main/training environment dependency checks, and whitespace validation. Reconstructed both censuses and both 200-step schedules; revalidated Method A failure, every Method B gate, the v2 recipe hash, four-step actual-token parity, dataset/split hashes, old negative-control adapter hashes, content-free evidence, adapter/schedule/raw/cache/environment ignore rules, protected evaluator/generator/verifier/contamination scope, sealed-path status, and tracked sizes. Scanned all candidate files for high-confidence secrets and exact/12-token matches against exactly 904 cached development questions in offline mode.
- **Result:** Ruff is clean; strict Mypy reports no issues in 97 source files; 378 unit and six integration tests pass; both environments have no broken requirements; whitespace passes. Census, schedules, recipe, adapters, and smoke parity reconstruct exactly. Secret, exact-development, 12-token-development, protected-path, sealed-path, tracked raw/cache/environment/adapter, and new oversized-file counts are zero. The only tracked files at or above 1 MiB are the two pre-existing content-free synthesis schedule/manifest artifacts.
- **Gate status:** **PASSED.** The token-matched protocol is eligible for its required publication commit.
- **Errors or uncertainty:** Expected Windows LF-to-CRLF notices are non-fatal. Offline development loading reports that it is using the pinned local cache, as required; no sealed-final artifact was opened.
- **Next action:** Fetch without integration, require `origin/main` unchanged, stage only the approved protocol/config/source/tests/content-free evidence/docs, repeat index-only checks, commit `train: freeze token-matched QLoRA protocol`, push, and confirm a clean synchronized branch before full generic retraining.

### 2026-07-20 - Milestone 8D token-matched protocol published

- **Action performed:** Fetched without integration, verified `origin/main` remained unchanged, staged exactly 23 approved protocol files, repeated cached whitespace/path/raw/adapter/environment safety checks, and published the required first commit.
- **Result:** Commit `02a7a3f16bd7ee4afa75b59b034e7e3c13c26238` (`train: freeze token-matched QLoRA protocol`) is on both local `main` and `origin/main` at 0/0 divergence. The worktree was clean before the full-training log entry. No schedule containing question text, dataset, adapter, checkpoint, prediction, cache, environment, secret, credential, or sealed-final content was pushed.
- **Errors or uncertainty:** The first commit command stopped before creating a commit because no persistent Git author identity is configured; its following push was a no-op. Without modifying local or global Git configuration, the retry supplied the author identity from the prior published commit through command-scoped `git -c` values and succeeded.
- **Next action:** Train the generic-control token-matched adapter first from the untouched pinned base for exactly 200 optimizer steps; do not load development data.

### 2026-07-20 - Milestone 8D generic token-matched adapter frozen

- **Action performed:** Trained a fresh generic-control LoRA adapter from the untouched pinned Qwen base using the published v2 Method B schedule for exactly 200 optimizer/scheduler steps. The runtime validated every occurrence and token count, evaluated only the frozen 50-example synthetic-validation split every 25 steps, saved the final adapter, rehashed it, and reloaded it offline. No development benchmark data was loaded.
- **Result:** The run processed 1,578 occurrences, 807,936 padded tokens, and exactly 271,292 scheduled/actual loss-bearing tokens in 634.798 seconds. Token-weighted step loss moved from 3.262921 to 0.149672; mean step loss was 0.420702 and final synthetic-validation loss was 0.175241. Losses and gradients remained finite. Peak allocated/reserved VRAM was 3,343,712,768/3,577,741,312 bytes; peak process RSS was 1,477,734,400 bytes. The ignored 89,796,953-byte adapter SHA-256 is `c039612d250827525f06269d75da2600c24d2a76c26c11c58a3bfb838e025df1` and offline reload passed.
- **Gate status:** **PASSED.** Base/recipe/schedule, occurrence, actual-token, step/scheduler, finite-loss/gradient, final-save/hash, offline-reload, CUDA-residency, and no-development-exposure checks pass.
- **Errors or uncertainty:** Only the inherited non-fatal sliding-window SDPA and checkpoint `use_reentrant` warnings appeared. No OOM, offload, backend, optimizer, validation, or serialization failure occurred.
- **Next action:** Train the targeted token-matched adapter second from the untouched base and its separately frozen schedule; do not initialize from any existing adapter or load development data.

### 2026-07-20 - Milestone 8D targeted token-matched adapter frozen

- **Action performed:** Trained a fresh targeted LoRA adapter from the untouched pinned base using the published v2 targeted schedule for exactly 200 optimizer/scheduler steps. As in the generic run, only the frozen synthetic-validation split was evaluated during training; the final adapter was saved, hashed, and reloaded offline without development access.
- **Result:** The run processed 1,398 occurrences, 715,776 padded tokens, and exactly 271,150 scheduled/actual loss-bearing tokens in 569.049 seconds. Token-weighted step loss moved from 3.139081 to 0.142186; mean step loss was 0.390246 and final synthetic-validation loss was 0.165387. Losses and gradients remained finite. Peak allocated/reserved VRAM was 3,343,712,768/3,577,741,312 bytes; peak process RSS was 1,478,729,728 bytes. The ignored 89,796,953-byte adapter SHA-256 is `b4a2e55d293ed88ccc2668dd450cc29f683bda7585668126fe75e2253fe3b02e` and offline reload passed.
- **Gate status:** **PASSED.** Base/recipe/schedule, occurrence, actual-token, step/scheduler, finite-loss/gradient, final-save/hash, offline-reload, CUDA-residency, and no-development-exposure checks pass.
- **Errors or uncertainty:** Only the inherited non-fatal warnings appeared. No OOM, offload, backend, optimizer, validation, or serialization failure occurred.
- **Next action:** Apply the final pairwise parity gate to both fresh summaries and adapter hashes before any development-benchmark inference.

### 2026-07-20 - Milestone 8D final training-token parity gate passed

- **Action performed:** Compared all immutable metadata and runtime fields, scheduled versus actual token counts, finite-loss/gradient outcomes, development-exposure flags, and offline-reload outcomes; rehashed both final adapter directories and applied the frozen 0.5% actual-token limit.
- **Result:** Generic actual tokens are 271,292 and targeted actual tokens are 271,150. The absolute difference is 142 and the relative difference is 0.052342%. Both adapter hashes match their summaries; every compared metadata field and all seven parity checks pass. Final parity-summary SHA-256 is `5c4d70c6f4778216b5d3606d83c788af6f73a9c7a13aa3163ad5359de798c63e`; `benchmark_evaluation_authorized` is true.
- **Gate status:** **PASSED.** The frozen development evaluator may now run in the predeclared generic-then-targeted order.
- **Errors or uncertainty:** None. Neither adapter was evaluated before this gate.
- **Next action:** Add only an external PEFT adapter backend and paired-analysis layer, prove the frozen evaluator files/config/hash remain unchanged, then evaluate generic first on the exact 814 development identifiers.

### 2026-07-20 - Milestone 8D generic frozen-development evaluation completed

- **Action performed:** Loaded the fresh token-matched generic-control adapter on the pinned base entirely from local artifacts and evaluated it first, as predeclared, on all 814 frozen development identifiers through the unchanged prompt, extractor, parser, and deterministic greedy generation settings. Raw predictions remain under the ignored token-matched-v2 evaluation directory; only the content-free aggregate summary is tracked.
- **Result:** The run completed 814/814 examples with zero generation/backend failures. It scored 15 correct (1.8428%), extracted 167 answers (20.5160%), exactly complied on 137 (16.8305%), and scored 8.9820% among extractable outputs. There were 152 extractable-but-wrong and 647 unextractable outputs, including 470 with no terminal answer, 173 ambiguous terminal answers, two malformed terminal answers, and two truncations. Runtime was 1,357.107 seconds (0.60094 examples/s); peak allocated/reserved VRAM was 3,248,531,968/3,512,729,600 bytes. The adapter hash remained `c039612d250827525f06269d75da2600c24d2a76c26c11c58a3bfb838e025df1`; all frozen evaluator, model, manifest, prompt, and extractor hashes match.
- **Gate status:** **COMPLETED, RESULT POOR.** Evaluation integrity passed, but accuracy declined by 506 correct answers and 62.1622 percentage points from the frozen base. This is not a declared intermediate stop gate; the authorization requires the targeted evaluation next once final training parity has passed.
- **Errors or uncertainty:** The inherited sliding-window SDPA and ignored sampling-parameter warnings appeared; no OOM, offload, adapter-load, generation-backend, or serialization failure occurred. The severe loss of output-contract compliance is an observed result and will be diagnosed from paired content-free aggregates after the predeclared targeted run, without changing any setting.
- **Next action:** Evaluate the fresh targeted adapter second on the same frozen 814-example development manifest, then perform the paired and category-level analysis and apply the fixed one-seed signal gate.

### 2026-07-20 - Milestone 8D targeted frozen-development evaluation completed

- **Action performed:** Loaded the fresh token-matched targeted adapter second, as predeclared, on the same pinned base and evaluated all 814 frozen development identifiers through the identical unchanged evaluator. Raw predictions remain ignored; the tracked result contains only content-free aggregate metrics and immutable hashes.
- **Result:** The run completed 814/814 examples with zero generation/backend failures. It scored 14 correct (1.7199%), extracted 180 answers (22.1130%), exactly complied on 157 (19.2875%), and scored 7.7778% among extractable outputs. There were 166 extractable-but-wrong and 634 unextractable outputs, including 374 with no terminal answer, 256 ambiguous terminal answers, one conflicting-answer case, and three truncations. Runtime was 1,359.252 seconds (0.59998 examples/s); peak allocated/reserved VRAM was 3,248,531,968/3,512,729,600 bytes. The adapter hash remained `b4a2e55d293ed88ccc2668dd450cc29f683bda7585668126fe75e2253fe3b02e`; all frozen evaluator hashes match.
- **Gate status:** **COMPLETED; ONE-SEED SIGNAL CANNOT PASS.** Targeted is one correct answer below generic and 507 below the frozen base, while its 22.1130% extractability is far below the fixed 91.38% gate. Backend integrity passed, but the predeclared performance requirements did not.
- **Errors or uncertainty:** Only the same inherited non-fatal generation warnings appeared. No OOM, offload, adapter-load, generation-backend, or serialization failure occurred. The magnitude and shared direction of both adapter failures point to a common training/evaluation compatibility problem rather than evidence for the targeted-data hypothesis; paired and category-level analysis remains required before the narrowest diagnosis is frozen.
- **Next action:** Compute the fixed-seed paired bootstrap interval, per-example transitions, and frozen-taxonomy category aggregates; apply every signal-gate clause; document the failure without tuning, retraining, or changing data.

### 2026-07-20 - Milestone 8D paired analysis and one-seed signal gate completed

- **Action performed:** Added a standard-library paired analyzer with strict 814-ID alignment, prediction/summary consistency checks, frozen-taxonomy validation, deterministic transition counts, and a 10,000-replicate paired percentile bootstrap using seed `20260720`. Added focused reproducibility and fail-closed alignment tests, then ran the analyzer on the ignored base, generic, and targeted prediction sets.
- **Result:** Frozen base/generic/targeted correctness is 521/15/14. Targeted wins 11 paired rows and generic wins 12 (net -1); targeted-minus-generic is -0.12285 percentage points with a 95% interval of [-1.22850, +0.98280] points. Generic fixes one base failure and breaks 507 base successes; targeted fixes two and breaks 509. On the frozen 293-row base-failure taxonomy, neither arm fixes an example in the three selected reasoning categories; targeted fixes two of 170 untargeted rows and generic fixes one. Analysis SHA-256 is `dc9e2a767d8a39fef1373b4d3b1c4ca3d922ad2257238e2a2e4cd7c6a240ac9c`.
- **Gate status:** **FAILED.** Targeted misses the >=529 correct, >=generic+4, and >=91.38% extractability clauses. It passes final training parity, zero backend failures, and the frozen failure-taxonomy untargeted-decline clause. The paired interval includes zero but is immaterial to the much larger shared collapse versus base.
- **Errors or uncertainty:** The frozen taxonomy covers the base model's 293 failures only, so its category results cannot classify the 521 base-success rows that both adapters mostly break. This scope is reported explicitly rather than extrapolated. No raw response, question, answer, or stable ID is written to tracked output.
- **Next action:** Complete the required documentation, run repository-wide code, evidence, safety, hash, evaluator-freeze, ignore, and synchronization checks, then create and push the exact final commit without tuning or retraining.

### 2026-07-20 - Milestone 8D final verification passed

- **Action performed:** Ran Ruff format/lint, strict Mypy, all unit and integration tests, both environment dependency checks, whitespace validation, exact dataset/split/census/Method A/Method B/adapter/recipe/final-parity hash reconstruction, paired-analysis replay, evaluator-diff review, secret scanning, offline exact and 12-token development-content scans over every changed text file, sealed-path review, ignore/tracking checks, and tracked-size review.
- **Result:** Ruff is clean; Mypy reports no issues in 99 source files; 382 unit and six integration tests pass; both environments have no broken requirements. All six dataset/split hashes, both census hashes, both schedule hashes, the 0.052342% final parity result, both adapter hashes, the frozen evaluator hashes, and paired-analysis SHA-256 `dc9e2a76...ac9c` match. Exactly 904 cached development questions were scanned: zero exact and zero 12-token changed-file matches. Secret, protected evaluator diff, sealed access, and newly tracked raw/cache/environment/adapter/prediction counts are zero. The only >=1 MiB tracked files are two pre-existing content-free synthesis schedule/manifest artifacts.
- **Gate status:** **PASSED.** The accurate failed-signal result is eligible for its required final commit and push. This verification does not change the signal decision and authorizes no additional experiment.
- **Errors or uncertainty:** A first verification wrapper tried to print a nonexistent Method A convenience key after the underlying replay had already succeeded; reading the real `method_a_gate_passed` field confirmed the unchanged expected failure and exact `a7a68503...fca5` hash. A parallel unit-test handle was not retained after its output window, so the unit suite was rerun directly and passed unambiguously in 124.76 seconds. Neither issue changed source, evidence, or gates.
- **Next action:** Fetch without integration, require `origin/main` to remain at the published protocol commit, stage only the approved Milestone 8D result files, run index-only safety checks, commit `train: compare token-matched targeted and generic adapters`, push normally, and confirm 0/0 divergence with a clean worktree.

### 2026-07-20 - Milestone 8E starting state verified

- **Action performed:** Read the complete Milestone 8E authorization and verified repository root, `main`, local/remote commit `911b48106a80c950e4e0f0f334073c8b224c7b44`, 0/0 divergence, clean worktree, all dataset and split hashes, all four preserved adapter directory hashes, the pinned base/tokenizer/chat-template evidence, training package/CUDA/GPU versions, frozen evaluator commit state, and the existence of all three ignored development-prediction sets. Created only a content-free adapter-status record outside the adapter directories.
- **Result:** Every frozen hash matches. The original generic/targeted adapters remain `36b19165...e3ac` and `217a9bcf...406e` and are marked `invalid_for_comparative_evaluation`; collapsed token-matched adapters remain `c039612d...5df1` and `b4a2e55d...b02e` and are marked `diagnostic_only_shared_sft_collapse`. CPython 3.12.10, PyTorch 2.5.1+cu121, Transformers 4.51.3, tokenizers 0.21.4, PEFT 0.15.2, TRL 0.17.0, bitsandbytes 0.49.2, and Accelerate 1.7.0 remain available on the RTX 3080. No sealed-final file was opened, hashed, inspected, or compared.
- **Gate status:** **PASSED.** Existing failure evidence is preserved, and diagnosis may begin without regenerating benchmark predictions.
- **Errors or uncertainty:** Two read-only verification probes used an incorrect evaluator filename/signature and were rejected before producing evidence. A `pip check` run with `PYTHONPATH=src` exposed repository egg-info requiring general-project PyYAML 6.0.2, while the dedicated training lock intentionally pins installed PyYAML 6.0.3; isolated checks without injected source metadata pass in both environments. No dependency or file changed as a result.
- **Next action:** Classify all existing generic and targeted collapsed outputs using deterministic content-free features and locally retained detailed rows; do not run new GSM1K inference or commit raw text.

### 2026-07-20 - Milestone 8E existing collapse characterized

- **Action performed:** Added and tested a deterministic output taxonomy, then classified the 814 preserved generic and 814 preserved targeted development outputs without running new benchmark inference. The classifier uses response/extractor metadata, EOS/truncation evidence, repetition, safe first-prefix classes, and overlap against development questions solely to detect echo; per-ID hashes and features remain ignored.
- **Result:** Generic averages 48.216 output tokens (median 36) and targeted 49.430 (median 37). Only 143 generic and 167 targeted outputs contain `Final answer:`; 303 and 263 show deterministic repetition. Generic begins with recognized synthetic-trace operation or metadata phrases on 756/814 outputs and targeted on 728/814. Dominant safe prefixes include `multiply the`, `the exact`, and `the typed`. Generic categories include 300 repetitive, 228 reasoning-like without a terminal answer, 112 valid terminal-contract surfaces, 85 ambiguous/malformed answers, and two truncations; targeted includes 260, 193, 120, 161, and three. Question-prefix echo is zero in both arms; only one generic output resembles question/transcript generation. Summary SHA-256 is `c5c54b8472475f59d064fbbadaff7f9316737d636d21bd55e6393f1c63820951`.
- **Gate status:** **CHARACTERIZATION COMPLETE.** The shared failure is short, repetitive, synthetic-trace-style generation with low terminal-contract completion, not benchmark-question echo or token-limit exhaustion.
- **Errors or uncertainty:** The taxonomy labels reasoning-like surface form, not mathematical validity, and therefore does not infer correctness beyond the frozen evaluator. First-prefix fragments are restricted to original generic operation language; no benchmark question or raw response is committed.
- **Next action:** Reconstruct all 900 prior SFT records token by token, identify system/user/assistant/EOS/padding spans, and measure exactly which roles contributed to loss before proposing any correction.

### 2026-07-20 - Milestone 8E role-aware label audit completed

- **Action performed:** Reconstructed the exact official-Qwen chat text, 512-token padding, attention mask, and previous label mask for all 450 generic and 450 targeted training rows. Token boundaries were derived from verified chat-template prefixes and partitioned into system message, user message, assistant header, assistant reasoning, assistant final answer, EOS, and padding. Wrote deterministic 30-row-per-arm token packets only below the ignored raw-training directory; committed evidence is aggregate and content-free.
- **Result:** The previous implementation copied every nonpadding input token into `labels`. All 900 records made system content, user content, and assistant-header tokens loss-bearing. Generic has 77,348 nonpadding tokens: 15,300 system-message, 39,884 user-message, 1,350 assistant-header, 19,335 assistant-reasoning, 579 recognized final-answer, and 900 post-completion/EOS-span tokens. Targeted has 87,317: 15,300, 44,356, 1,350, 24,787, 624, and 900 respectively. Padding is correctly masked in all rows; all have assistant content and an EOS; none is truncated; all use the same system/user/assistant order. Every decoded label span contains its original question, none contains internal metadata, and zero equals the intended assistant completion plus EOS. Summary SHA-256 is `e9aba3ce9dea9b407dfaac98d54a8934ea67ca4dbe00236ab79d49857b84ccb6`.
- **Gate status:** **CONCRETE MASKING DEFECT FOUND.** System and user content contributed to loss in every previous training record, satisfying the declared label-mask defect condition. No correction has yet been applied because completion-format and adapter-load diagnosis remain mandatory.
- **Errors or uncertainty:** The first observational pass incorrectly required every completion to contain one `Final answer:` line and stopped on an actual stored completion without one. The audit was corrected to observe absent/multiple lines rather than reject them; terminal-format prevalence is handled by the next dedicated audit. A token-subsequence question check was also replaced by exact decoded-text containment because BPE boundaries differ when the question is embedded in the user message. Both changes affect diagnosis only, not training data or labels.
- **Next action:** Audit all 1,000 completion strings, compare the prior SFT roles/prompts/response contract against the frozen development evaluator, and AI-inspect 30 deterministic completions per arm without changing any target.

### 2026-07-20 - Milestone 8E completion and evaluation-format alignment audited

- **Action performed:** Audited all 500 accepted synthetic records per arm across frozen training and synthetic-validation splits, compared every stored completion with its deterministic trace and canonical answer, and compared the exact prior SFT prompt/roles/chat behavior with the frozen development evaluator. Wrote 30 deterministic AI-assisted completion samples per arm only to ignored raw storage.
- **Result:** Exactly 100/500 completions per arm—the frozen 20% output-contract track—contain one clear `Final answer: <canonical-number>` line and end with it. The other 400/500 per arm contain no explicit final-answer marker. No completion uses another terminal marker, multiple answer markers, or worksheet/question-generation wording. All 1,000 exactly preserve their deterministic solution traces and exhibit the recognizable procedural trace style; the canonical answer also occurs before the final line in 101 generic and 103 targeted completions. Training and evaluation share system/user role order and the Qwen special-token template, but system text hashes differ, user templates differ by the terminal placeholder (`<canonical-number>` versus `<integer>`), training renders the completed assistant message with `add_generation_prompt=False`, evaluation uses an assistant generation prompt, and the expected response contracts differ. Summary SHA-256 is `989b984baba2c0533a7c59995885e8dfd1a041bc37aa8f79c15aebc4e1b904c7`.
- **Gate status:** **SECOND CONCRETE PROTOCOL DEFECT FOUND.** Eighty percent of each arm systematically lacked the development evaluator's required terminal-answer contract. The data remain mathematically unchanged and trace-consistent; no target was edited in this audit.
- **Errors or uncertainty:** A numeric-boundary fixture initially did not recognize a canonical number immediately before sentence punctuation. The detector was corrected to permit trailing punctuation and the full audit was rerun, changing only this observational count. The `internal program trace` label describes the deterministic procedural surface style and does not allege exposed record metadata; the role audit found zero metadata leakage.
- **Next action:** Run original non-benchmark adapter application probes for base, generic enabled/disabled/re-enabled, and targeted enabled/disabled/re-enabled; inspect adapter state/scaling and prove disabling restores the untouched base exactly.

### 2026-07-20 - Milestone 8E adapter loading and diagnosis gates passed

- **Action performed:** Loaded the untouched pinned base and each collapsed adapter separately, entirely offline and on CUDA, then generated from six original diagnostic prompts with the adapter enabled, disabled in place, and re-enabled. Inspected adapter configuration, every saved tensor key, active adapter names, merge state, scaling, loaded-state count, and per-layer LoRA norms. Full diagnostic responses remain ignored; only hashes and content-free measurements are tracked.
- **Result:** Both arms have exactly one active `default` adapter, 196 LoRA modules, 392 expected A/B tensors, rank 16, alpha 32, dropout 0.05, and scaling 2.0. Missing expected keys, unexpected/saved base keys, and merged modules are all zero. Disabling either adapter exactly restores untouched-base output SHA-256 `d2b230f9...abff`; re-enabling exactly restores each enabled hash. Generic changes three of six original probes and targeted changes one. Adapter hashes remain `c039612d...5df1` and `b4a2e55d...b02e`; base snapshot SHA-256 is `ccc7ba57...1815`; summary SHA-256 is `ebbc367dc2239b344f5d7c92c148fd9cf969842bd916e5eeebbb20e6b83421dc`.
- **Gate status:** **PASSED.** Loading/application is correct: one unmerged adapter, no double application, no offload, no saved base weights, correct scaling, and exact base restoration. The diagnosis gate selects directly evidenced causes **1** (system/user tokens contributed to loss) and **3** (80% of assistant targets lacked the evaluation-required terminal contract). Loading defect 4 is ruled out. The evidence does not establish either defect as the sole cause; corrective retention smokes are required.
- **Errors or uncertainty:** Transformers emitted the inherited sliding-window SDPA warning and ignored sampling-parameter warnings from the pinned generation config while `do_sample=False`; generation completed deterministically. A loading-time figure measures base construction before PEFT wrapping and is retained as setup telemetry rather than an end-to-end load benchmark.
- **Next action:** Freeze 60 original non-benchmark retention prompts, answers, deterministic evaluator, prompt/suite/settings hashes, and untouched-base baseline before implementing or training with assistant-only SFT v3.

### 2026-07-20 - Milestone 8E original retention suite frozen

- **Action performed:** Authored 60 original, non-benchmark, objectively scored prompts: 30 arithmetic items balanced across addition, subtraction, multiplication, division, percentages, ratios, and short multi-step reasoning; 15 strict-format items; and 15 deterministic general-instruction items. Added a frozen local evaluator using the canonical numeric extractor plus exact text/JSON scoring, then evaluated the untouched pinned base offline before any corrective SFT.
- **Result:** Final untouched-base baseline is arithmetic 30/30 (100%), format 14/15 (93.3333%), and instruction 14/15 (93.3333%). Extractability is 59/60 (98.3333%), exact-format compliance 56/60, prompt echo zero, question generation zero, malformed outputs one, and backend failures zero. Runtime was 26.866 seconds with 3,561 input and 1,601 output tokens; peak allocated/reserved VRAM was 3,554,672,640/3,619,684,352 bytes. Suite SHA-256 is `0f0b73d8cd920f20998ba00c81ac6ead4c45f04c972c033e5c30a95bfee63eb9`, prompt SHA-256 `451ed6c7fbaa4f96d3d07fdc7286d491c33969973fb110ba20c494c9517c82fe`, generation SHA-256 `f628fe7faafe94040de3df696a63d98494525cd1a57487b718fd9a86b5292093`, and baseline summary SHA-256 `83858c632dfce15ecd88ac568b88d9145e150330303e1b6b2ab1e2ed6a683e0a`.
- **Gate status:** **PASSED AND FROZEN.** The untouched base itself clears the future absolute 90% format/instruction and extractability gates, so the instrument can validly detect post-training retention loss.
- **Errors or uncertainty:** The first pre-freeze draft used a 128-token ceiling and scored the base at 19/30 arithmetic because verbose but otherwise correct responses were truncated; its strict prompts also yielded only 8/15 format and 9/15 instruction. A second draft with a 384-token ceiling proved arithmetic 30/30 but left brittle format/instruction cases. Before any adapter evaluation or corrected training, those original prompts were simplified to objective base-capable transformations, producing the final frozen result above. These pre-freeze calibration runs used only original prompts and are not downstream recipe selection.
- **Next action:** Implement `foundry-assistant-only-sft-v3`, normalize one terminal line without changing reasoning or canonical answers, mask system/user/header/padding/post-EOS newline, and rebuild assistant-only loss-token censuses and Method B schedules with <=0.5% predicted parity.

### 2026-07-20 - Milestone 8E assistant-only SFT v3 and Recipe 1 smoke completed

- **Action performed:** Implemented `foundry-assistant-only-sft-v3` with the official Qwen chat template, evaluation-aligned system/user wording, unchanged deterministic reasoning, one normalized canonical terminal line, and labels only on assistant completion tokens plus final EOS. Rebuilt both 450-row censuses and deterministic whole-example Method B schedules, then trained fresh generic and targeted 32-step temporary adapters from the untouched base at the predeclared `2e-4` learning rate and evaluated only synthetic validation and the frozen original retention suite.
- **Result:** Static formatting invariants pass on all 900 training rows: zero system/user/header, padding, or post-EOS loss tokens; exactly one terminal line; no metadata/question in decoded labels; no truncation. Generic/targeted unique-source assistant tokens are 22,603/28,126. Complete schedules deliver 90,000/89,995 tokens (0.00556% difference), while both 32-step prefixes deliver exactly 14,404. Census summary SHA-256 is `e6e79e0c...3e11e`; schedule summary `81de989d...e3644`; format hash `3ffba986...35329`; recipe hash `9a968154...e1df7`. Recipe 1 training completed with finite final synthetic-validation losses 0.06349 generic and 0.06779 targeted. Generic retention is arithmetic 25/30, format 10/15, instruction 10/15, extractability 95%; targeted is 22/30, 13/15, 8/15, and 93.3333%. Both have zero echo, question generation, and backend failures.
- **Gate status:** **RECIPE 1 FAILED.** Relative to the 30/30 base arithmetic baseline, generic drops 16.67 points and targeted 26.67, exceeding the ten-point limit; format and instruction accuracy also miss 90% in both arms. Token parity, finite losses, extractability, echo, question-generation, and backend gates pass.
- **Errors or uncertainty:** Training emitted only the inherited sliding-window/checkpoint warnings. The result establishes that assistant-only masking alone does not make `2e-4` retention-safe at 32 steps; it does not authorize changing the suite, labels, optimizer, schedule, rank, or any other setting.
- **Next action:** Run exactly the one predeclared fallback pair at `5e-5`, still 32 steps and 14,404 tokens per arm, and apply the unchanged retention-smoke gate. If it fails, stop before full retraining and benchmark evaluation.

### 2026-07-20 - Milestone 8E fallback retention smoke failed; stop rule applied

- **Action performed:** Trained exactly the one authorized fallback pair from the untouched base at `5e-5`, changing no other recipe field. Each arm ran 32 optimizer/scheduler steps and exactly 14,404 assistant-only loss-bearing tokens, then was evaluated only on its synthetic-validation split and the frozen 60-prompt original retention suite. Added a deterministic paired gate evaluator and applied the unchanged thresholds to both learning-rate recipes.
- **Result:** Fallback training remained finite with final validation loss 0.334604 generic and 0.274047 targeted. Generic retention was arithmetic 28/30, format 15/15, instruction 13/15, and extractability 58/60; targeted was 25/30, 15/15, 14/15, and 57/60. Both had zero prompt echoes, question generation, and backend failures. Generic failed the >=90% instruction gate at 86.67%; targeted failed the <=10-point arithmetic-drop gate at 83.33% versus the 100% base. Gate-summary SHA-256 is `5d1d5f0d31fce710836bd398baeeb469bc4d36c9888476ae38f9721f37671201`.
- **Gate status:** **FAILED — MANDATORY STOP.** Neither predeclared learning rate is retention-safe for both curricula. No recipe is selected; 200-step retraining, retention-checkpoint selection, final parity, and new frozen-development evaluation are not authorized and were not run.
- **Errors or uncertainty:** The diagnosis directly proves prior prompt-label leakage and inconsistent terminal contracts, and v3 removes both. The bounded outcomes show that these corrections alone do not preserve the full retention suite under either allowed learning rate. They do not identify which remaining optimization or data property causes the arm-specific retention failures, and no additional recipe search is authorized.
- **Next action:** Complete repository-wide verification and publish the diagnosis with an accurate analysis commit. The user must decide whether to approve a separately designed training-method investigation; a second seed and benchmark evaluation remain blocked. Human language review is still pending.

### 2026-07-20 - Milestone 8E final verification passed

- **Action performed:** Ran repository-wide Ruff formatting/linting, strict Mypy, every unit and integration test, both environment dependency checks, and whitespace validation. Replayed all assistant-only aggregate hashes and the paired retention gate; revalidated the frozen suite, format, recipe, census, schedules, datasets, splits, four preserved adapters, and four temporary adapters. Scanned every candidate file for high-confidence secrets and exact/12-token overlap with all 904 cached development questions; reviewed protected and sealed paths, raw/model/cache/environment ignore and tracking status, and tracked-file sizes.
- **Result:** Ruff is clean; strict Mypy reports no issues in 109 source files; 408 unit and six integration tests pass; both environments have no broken requirements. Twelve v3 summaries and gate SHA-256 `5d1d5f0d...1201` replay exactly. All two dataset, four split, four preserved-adapter, and four temporary-adapter hashes match. Across 49 candidate paths, secret, exact-development, 12-token-development, protected-path, sealed-path, tracked-raw, and tracked-environment/model counts are zero. All ignored artifact checks pass. The only tracked files at or above 1 MiB are the two pre-existing content-free synthesis schedule/manifest files.
- **Gate status:** **VERIFICATION PASSED; EXPERIMENTAL GATE FAILED AS RECORDED.** The diagnosis and stopped retention-smoke result are eligible for an accurate commit and push. Verification does not authorize full retraining or benchmark evaluation.
- **Errors or uncertainty:** One read-only evidence replay initially referenced a nonexistent convenience key in the role-audit summary; it failed before changing data, was corrected to the actual aggregate schema, and passed. One read-only safety script initially received a nested PowerShell path array and stopped before scanning; the corrected scan passed. Expected Windows LF-to-CRLF notices are non-fatal. No sealed-final artifact was opened.
- **Next action:** Fetch without integration, require `origin/main` unchanged, stage only approved code/tests/config/content-free evidence/docs, repeat index safety checks, create an accurate diagnosis commit, push, and confirm a clean synchronized branch.
### 2026-07-20 - Fast-Track 8F-8H starting state verified

- **Action performed:** Read the complete fast-track authorization; verified repository root, `main`, local/remote commit `b2c02fffefba690c7d49236e4d291ec7e79ef89f`, 0/0 divergence, and a clean worktree. Rehashed both datasets, all four splits, and all four permanently invalid adapters; loaded the frozen v3 recipe and retention suite; checked the schedule summary, pinned training stack, frozen evaluator scope, ignored prior retention outputs, GPU process state, and sealed-path status. Re-ran the original six-prompt adapter sanity audit under an ignored Stage A path.
- **Result:** All frozen identities match. CPython 3.12.10, PyTorch 2.5.1+cu121, Transformers 4.51.3, tokenizers 0.21.4, PEFT 0.15.2, TRL 0.17.0, bitsandbytes 0.49.2, and Accelerate 1.7.0 pass `pip check`; CUDA detects the 10,240 MiB RTX 3080. The untouched-base diagnostic hash remains `d2b230f9...abff`; both collapsed adapters restore that exact hash when disabled and reproduce their enabled hashes when re-enabled. Each has one unmerged `default` adapter, 196 LoRA modules, 392 saved tensors, scaling 2.0, no unexpected/missing keys, and no offload.
- **Gate status:** **PASSED.** Frozen data, adapters, v3 protocol, retention evidence, environment, and evaluator boundary are intact. No GSM1K inference or sealed-final access occurred.
- **Errors or uncertainty:** The first read-only Python hash command lacked a temporary `PYTHONPATH=src` in `.venv-training`; it stopped before changing data. The corrected command passed. NVIDIA process reporting includes ordinary desktop graphics processes but no Python training process.
- **Next action:** Audit all 1,000 current v3 assistant completions, create an ignored deterministic 50-per-arm blind-inspection packet, and commit only content-free aggregate style counts and hashes.
### 2026-07-20 - Fast-Track 8F-8H assistant-target style audit completed

- **Action performed:** Added a deterministic style analyzer, audited the normalized v3 completion for all 1,000 frozen examples, and created ignored blind samples selected by stable hash. Inspected 50 generic and 50 targeted completions using only completion text; recorded the Codex result explicitly as AI-assisted rather than human review.
- **Result:** Generic targets average 49.128 assistant tokens and 3.768 nonempty lines; targeted average 61.246 tokens and 4.324 lines. Deterministic classification marks 376/500 generic and 419/500 targeted as procedural/program-trace style, with internal-operation terminology in 167 and 275. Only 41 generic and 27 targeted targets contain a direct equation pattern. All 1,000 have exactly one final-answer line in final position. The blind Codex sample classified all 100 as procedural or terse trace style and zero as natural assistant style; 20 generic and 25 targeted samples explicitly use internal trace terminology. Summary SHA-256 is `f852c157a568bef5f4c6e61fc40b4d935dbcf79f984dbd3dbcf0fc2afb49d6e3`.
- **Gate status:** **COMPLETED.** The audit is descriptive and does not modify data or select a variant. The predeclared concise-v4 candidate proceeds regardless of this result.
- **Errors or uncertainty:** A focused test initially exposed that an answer-token boundary rejected sentence-final punctuation; the generic numeric boundary was corrected before the final audit. One command continued into the first aggregate generation after the focused test failure because PowerShell does not treat external nonzero exits as exceptions; no training ran, and the result was regenerated after tests passed. The blind sample is Codex inspection, not genuine human review.
- **Next action:** Implement and statically validate concise equation-grounded SFT v4 for every frozen record, with one to four reasoning lines, one final line, <=128 assistant tokens, replay/verifier agreement, and unchanged assistant-only label masking.
### 2026-07-20 - Fast-Track 8F-8H concise assistant SFT v4 frozen

- **Action performed:** Added a separate deterministic concise-v4 compiler for all eleven frozen mathematical modes, refactored the shared tokenizer only to accept an already validated completion, and reconstructed/tokenized every one of the 1,000 frozen records twice with the pinned offline tokenizer. V3 behavior remains unchanged.
- **Result:** All 1,000 v4 targets pass canonical-answer replay, frozen primary/independent verifier evidence, one-to-four equation-grounded-line rules, exactly one terminal line, forbidden-term checks, assistant-only label masking, and stable reconstruction. No record rejects. Targets use one reasoning line on 874 records and two on 126; maximum assistant loss length is 41 versus the 128-token cap. Generic/targeted training sources contain 11,196/11,984 assistant tokens. Format SHA-256 is `0d7b8fbd8ac03060a53e517acfd96afa976dec3f7548a16ebfdc7fcaaa6415a1`; reconstruction SHA-256 is `0ee6bca9c19a99875a3e7e8c482da4c0b61747bca666c2902726e927d6daed31`; summary SHA-256 is `d8630cdad57120a3fdeb1db1fa7b02adff29d3bea0b71fbcf9bab8417f711755`.
- **Gate status:** **PASSED.** Concise-v4 is eligible for the predeclared ladder; it does not replace or delete v3.
- **Errors or uncertainty:** None. The compiler fails closed on unknown modes, inconsistent trace states, inexact group division, verifier failure, format violations, or token overflow. Its terse equations are intentionally an adaptation candidate, not a claim of superior language quality.
- **Next action:** Create two ignored, original, mutually disjoint 90-prompt retention suites; prove no training/development overlap; freeze hashes; and require the untouched base to pass every 90/90/90/95% gate before any adapter sees them.

### 2026-07-20 - Fast-Track 8F-8H disjoint retention suites frozen

- **Action performed:** Authored two ignored, original 90-prompt suites with 45 arithmetic, 20 exact-format, and 25 deterministic instruction items each. Validated them with the generalized frozen retention evaluator, compared all three retention suites against one another, and scanned the 180 new prompts against all 1,000 synthetic training questions and 904 development questions before any adapter evaluation. Evaluated the untouched pinned base on validation and final holdout.
- **Result:** The three suites contain 240/240 unique normalized prompts with zero exact or 12-token training/development overlap. Validation base scores 45/45 arithmetic, 20/20 format, and 23/25 instruction with 100% extractability. Final-holdout base scores 44/45, 20/20, and 23/25 with 98.8889% extractability. Both have zero prompt echo, question generation, and backend failures. Validation suite SHA-256 is `96e88c82...1d10`; final-holdout suite SHA-256 is `3af7b87c...227d`; evaluator source SHA-256 is `884c0d10...69d1`.
- **Gate status:** **PASSED AND FROZEN.** Both suites exceed the required 90% section accuracies and 95% extractability. Prompt text remains ignored and untracked; only content-free hashes and base metrics are eligible for commit.
- **Errors or uncertainty:** Initial drafts exposed reference-span ambiguity and poor base calibration in character reversal, symbol grouping, item selection, and related deterministic prompts. Before any adapter saw either suite, these original prompts were clarified or replaced with simpler objective token operations as authorized. Validation was frozen before final holdout evaluation; neither suite may change after downstream adapter outputs are observed.
- **Next action:** Build four separate 32-step, whole-example, assistant-token-matched ladder schedules for v3/v4 and freeze checkpoint prefixes at steps 8, 16, 24, and 32 before training Variant A.

### 2026-07-20 - Fast-Track 8F-8H adaptation ladder completed

- **Action performed:** Built separate v3/v4 assistant-token censuses and four predeclared Method-B schedules, each with exact whole-example prefix parity. Trained fresh generic and targeted adapters for Variants A-D from the untouched base, saved checkpoints 8/16/24/32, and evaluated all 32 checkpoints only on the original calibration suite.
- **Result:** Every arm received exactly 14,400 loss tokens, with 3,600/7,200/10,800/14,400 cumulative tokens at the four checkpoints and 0% pairwise difference. Variant A (v3, `5e-5`) passes both arms at all checkpoints: generic calibration moves from 30/30, 14/15, 14/15 to 29/30, 15/15, 15/15; targeted moves from 30/30, 14/15, 14/15 to 29/30, 15/15, 15/15. Variants B-D all fail instruction following at every checkpoint with 12/15 or 13/15 in at least one arm. Ladder result SHA-256 is `bb1808486798a606e5be1dc1e2ca66533eca45dfd6790e953369fe8244237739`.
- **Gate status:** **VARIANT A SELECTED AT STEP 32.** It is the only variant with any common passing checkpoint, so later tie-breaks are unnecessary. No GSM1K data or the new validation/final suite influenced selection.
- **Errors or uncertainty:** The first calibration loop stopped before its first evaluation because PowerShell promoted the evaluator's known SDPA warning to a native-command error; no output was scored. Capturing warnings without changing Python generation behavior resolved the orchestration issue. Ladder training used 1,653.039 seconds total, peaked at 3,343,712,768 allocated and 3,577,741,312 reserved VRAM bytes, and created 2.876 GB of ignored adapters/checkpoints/raw evidence.
- **Next action:** Evaluate only Variant A's generic and targeted step-32 checkpoints on the frozen, disjoint 90-prompt validation suite. Both arms must pass before any protocol promotion or 200-step scheduling.

### 2026-07-20 - Fast-Track 8F-8H disjoint retention validation failed

- **Action performed:** Evaluated the calibration-selected Variant A step-32 generic and targeted adapters on `foundry-retention-validation-v1`, without inspecting GSM1K and without evaluating any other variant or checkpoint on validation.
- **Result:** Generic scores 45/45 arithmetic, 20/20 format, 21/25 instruction, and 90/90 extractable. Targeted scores 44/45, 20/20, 21/25, and 89/90. Both have zero prompt echo, question generation, and backend failures. Each instruction accuracy is 84%, below the fixed 90% threshold. Validation gate SHA-256 is `0dc0a92da70a0086bfbd9382b83d0a4bf129b191f16d96e17a4ba27dadf319e4`.
- **Gate status:** **FAILED - MANDATORY STOP.** Full 200-step schedules, full retraining, common full checkpoint selection, final-holdout adapter evaluation, final parity, GSM1K, paired analysis, and a new one-seed decision are not authorized and were not run.
- **Errors or uncertainty:** Calibration and validation disagree specifically on deterministic instruction following; arithmetic, format, extractability, echo, question-generation, backend, finite-loss, and token-parity checks pass. Selecting a different checkpoint after seeing validation would violate the protocol. The final-holdout suite was used only for its pre-adapter untouched-base gate and remains unconsulted for trained adapters.
- **Next action:** Run repository-wide verification, publish the complete negative ladder result with an accurate analysis commit, and request an evidence-based next training-method decision. The pending stratified human language review remains unchanged.

### 2026-07-20 - Fast-Track 8F-8H final verification passed

- **Action performed:** Replayed all content-free schedule and gate summaries; rehashed the frozen datasets, splits, earlier adapters, and all 32 ladder checkpoints; ran repository-wide formatting, linting, type checking, unit/integration tests, dependency checks, whitespace checks, secret and development-content scans, ignore/tracking checks, and tracked-file review. Inspected the complete diff and corrected one mistyped validation-base summary hash in the content-free suite index to match the canonical base result and gate input.
- **Result:** Ruff is format- and lint-clean across 190 files; strict Mypy reports no issues in 115 source files; all 430 tests pass in 94.16 seconds; both environments pass `pip check`; and `git diff --check` passes. Dataset, split, v3/v4 format, suite, census, schedule, checkpoint, adapter, protocol, and gate hashes all match. The changed-file scan found zero exact or 12-token overlaps with 904 development questions, zero high-confidence secret hits, and zero tracked raw/model/environment artifacts. Ladder training consumed 115,200 supervised tokens across eight arms in 1,653.039 seconds; 34 retention evaluations took 985.441 seconds. Peak training allocated/reserved VRAM was 3,343,712,768/3,577,741,312 bytes, peak training RSS was 1,461,723,136 bytes, and ignored ladder evidence occupies 2,876,184,214 bytes.
- **Gate status:** **VERIFICATION PASSED; DISJOINT VALIDATION GATE FAILED AS RECORDED.** The negative result is eligible for one accurate analysis commit and push. No later fast-track stage is authorized.
- **Errors or uncertainty:** The final evidence audit found the validation-suite index typo described above; the underlying base result and validation-gate input already agreed, so no measured score, selection, or decision changed. The first full-test wrapper yielded a live process session after 30 seconds; polling that same process completed normally with 430 passes. Expected Windows LF-to-CRLF notices are non-fatal. No sealed-final path was opened.
- **Next action:** Fetch without integration, require `origin/main` to remain unchanged, stage only approved code/tests/config/content-free evidence/documentation, create an accurate negative-result commit, push, and confirm a clean synchronized branch. The user must decide whether to approve a new training-method diagnosis; validation cannot be reused for tuning.

### 2026-07-20 - Fast-Track 8I-8K starting state verified

- **Action performed:** Read the complete powered-retention authorization; verified repository root, clean synchronized `main` at `31105ba982cc17370d2c2a453bfb442835711b56`, every dataset and split hash, Variant A format/recipe/schedule identities, exact 14,400-token arm parity, pinned training environment, frozen evaluator scope, and ignored prior-retention evidence. Re-ran the six-prompt offline adapter enable/disable/re-enable audit against the selected Variant A step-32 pair.
- **Result:** All frozen evidence matches. Generic adapter SHA-256 is `faa4b72bd0046fbd8d94fc3f364cb7078259403131ea16cd136fdfbbe5408f35`; targeted is `c4e455432b9a3b17ae077dafdaa1c9391ab67b7350628f3aa5f9787677d8bb5b`. Both load fully on CUDA with one unmerged `default` adapter, 196 LoRA modules, 392 expected tensors, rank 16, alpha 32, dropout 0.05, scaling 2.0, zero unexpected/missing keys, and exact untouched-base restoration hash `d2b230f9...abff` when disabled. Sanity summary SHA-256 is `8c8bb04a8f097122be34595817bc7e21580c42b6356b9f60d3a23d993610f94c`. CPython 3.12.10, PyTorch 2.5.1+cu121, Transformers 4.51.3, tokenizers 0.21.4, PEFT 0.15.2, TRL 0.17.0, bitsandbytes 0.49.2, and Accelerate 1.7.0 remain intact; `pip check` passes.
- **Gate status:** **PASSED.** No GSM1K inference or sealed-final access occurred. Existing retention outputs remain ignored and available locally.
- **Errors or uncertainty:** The authorization spells the targeted adapter identity as a 63-character value, `c4e455432b9ab17...d8bb5b`, which cannot be a SHA-256. The synchronized published gate evidence and on-disk selected adapter agree on the 64-character value above; the missing `3` is treated as a transcription defect, not an artifact substitution. A read-only evaluator-hash command referenced nonexistent convenience path `extractor.py`; the actual canonical extractor lives in `answer_extraction.py`, and no evaluator path is modified.
- **Next action:** Audit the existing frozen 25-item validation instruction slice pairwise using only base/generic/targeted raw outputs, expected contracts, and scorer evidence. Preserve the old failed decision unchanged.

### 2026-07-20 - Fast-Track 8I-8K old instruction failure audited

- **Action performed:** Joined the frozen validation suite with untouched-base, generic A/32, and targeted A/32 raw decisions for all 25 instruction prompts. Created an ignored content-bearing transition packet and a tracked content-free aggregate; classified failures without consulting GSM1K or targeted-versus-generic goals.
- **Result:** Scores remain base 23/25, generic 21/25, targeted 21/25. Twenty-one items pass in all three systems; two fail in all three; and two pass in the base but fail identically in both adapters. All four shared adapter failures are genuine instruction noncompliance. The two adapter-only regressions are an uppercase transformation and an ordered-list final-item selection. Generic-only successes, targeted-only successes, arm-unique failures, prompt defects, reference defects, and scorer defects are all zero. Raw packet SHA-256 is `d09f4ed60db2c83c5a94b9ad014ad37ff9e913a456e7c12af1c6775b169599dc`; aggregate SHA-256 is `06434e8d9325e8c6ed4aa8d86a19fdf049d5d43c739621f36b7fce03454c4a6e`.
- **Gate status:** **AUDIT COMPLETE; OLD GATE REMAINS FAILED.** The result supports powered adjudication because the two regressions are genuine but the 25-item estimate remains coarse.
- **Errors or uncertainty:** The first aggregate used the raw suite object's absent convenience hash and emitted `null`; the audit was corrected to obtain the canonical suite identity through the frozen loader, then regenerated. No decision or classification changed.
- **Next action:** Before any new adapter evaluation, create and cross-audit the original 300-prompt adjudication suite, 120-item shared anchor, and disjoint 300-prompt anchor holdout; freeze their prompts, answers, evaluator, generation settings, and hashes.
# 2026-07-20 — Milestones 8I–8K Stage C powered retention artifacts

- **Before:** Build and freeze original, objective retention artifacts before either trained adapter sees them: a 300-item adjudication suite, a 120-item shared training anchor, and an untouched 300-item anchor holdout. Only ignored prompt-bearing files and the tracked content-free evidence record may be created; the retention loader and focused tests may change to recognize the two new suite contracts.
- **Result:** The first offline build correctly failed closed on duplicated newly authored prompts. Before any adapter evaluation, the duplicate constructions were removed by making format inputs unique and by preserving distinct alphabetic context around symbol-count inputs. The final artifacts contain 720 unique IDs and 720 unique normalized prompts: adjudication `100/100/100`, shared anchor `40/40/40`, and holdout `100/100/100` across arithmetic/format/instruction. Exact and 12-token overlap counts are zero against the prior retention suites, the 1,000-example synthetic corpus, and the 904-item development partition; pairwise exact overlap among the three new artifacts is also zero. No benchmark answer or sealed-final content was accessed.
- **Frozen hashes:** adjudication suite `5caf23be79fa01151af6f7db8d45c2b85bfe24b03a29589e482d51731c8358af`, prompt `a94e0f81c845f9af83bc339d411158f628f1eb95103c6e9375326c0a7c214b42`, answer `cd0b39765760fb63fd90ee8c9eb8aaa47902da9c596b2e6134e80e39281e54e1`; anchor `a15df37c7318432576878ff86e567a0f0bac050cc62e2af61081a937f2c1740c`, prompt `aca5eef33cc48aee74080d6408eff40b06df60108933a70c5739214773394fb9`, gold response `9634f8cbf5490ef3ca237ce07ff5f142f118dbf891c9b448a9eda214ebde13a6`; holdout suite `bff18b434a284d848387262dde201601278e5c8b573937b3486bed2bf925696e`, prompt `15c6c993b75f69a90ae422e7facb8ae264768275d2d08364259502dd23e57782`, answer `da30196c0821f8f1bc083ac6fbf5a99a194eae4a68f55bc5bfd75e420ccf2379`; evidence summary `5f19ec7601e101f278c9e1ab6e48446ec3b23cb39c513185f874a395fd529e06`.
- **Uncertainty:** The suites are structurally frozen, but their objective usability still depends on the required untouched-base gate. Any base failure will be classified as a prompt/scorer defect or genuine base weakness; prompts cannot be tuned merely to make the base pass.
- **Next:** Run focused loader tests, then evaluate the untouched base on adjudication and holdout before exposing the adjudication suite to either adapter.

### 2026-07-20 - Fast-Track 8I-8K untouched-base gate failed

- **Before:** Evaluate the untouched pinned base on the new 300-item adjudication instrument first. Require at least `90/100` in each section, `285/300` extractable, and zero backend failures, prompt echo, or ambiguous references before evaluating the holdout or either adapter.
- **Result:** The offline CUDA run completed deterministically in 131.875 seconds at 2.275 items/second, using 3,554,672,640 peak allocated and 3,619,684,352 peak reserved VRAM. Scores were arithmetic `84/100`, format `48/100`, instruction `55/100`, overall `187/300`, and extractable `268/300`. Backend failures, prompt echo, and question generation were zero. Direct inspection of all 113 failed responses found 16 genuine terminal-contract failures, 52 genuine exact-format failures, and 45 genuine deterministic-instruction failures; objectively defective prompts, ambiguous references, bad answers, and scorer defects were zero. Base summary SHA-256 is `977dcf32458159f56c711def4b06c454ef24f747c7cc0b36965c0e7fb4f7589e`; gate evidence SHA-256 is `fa1fec57e87a03f390eb4944427f10c4cf0a716c2938447c922070a451067d48`.
- **Gate status:** **FAILED — STOP BEFORE ADAPTER ADJUDICATION.** The arithmetic, format, instruction, and extractability clauses all failed. Zero-error safety clauses passed.
- **Errors or uncertainty:** No objectively defective original prompt exists to repair under Stage D. The checkpoint emitted non-fatal warnings that sampling parameters are ignored under `do_sample=False`; decoding remained greedy and deterministic. The training environment intentionally lacks Pytest/Ruff, so focused development checks ran in the existing main environment without dependency changes.
- **Next action:** Do not evaluate the untouched holdout, expose either A/32 adapter to adjudication, train the shared-anchor fallback, or run GSM1K. Complete verification, commit and push the accurate stop record, then request an explicit new decision about the retention instrument or project stop.

### 2026-07-20 - Fast-Track 8I-8K stopped-result verification

- **Action performed:** Ran repository-wide Ruff formatting/linting, strict Mypy, all unit and integration tests, main/training environment dependency checks, whitespace validation, exact powered-artifact reconstruction, base-gate reconstruction, old-transition-audit hash checks, selected-adapter directory hashes, offline exact and 12-token scans against all 904 approved development questions, high-confidence secret scans, protected benchmark-evaluator and sealed-path status checks, ignored raw/model/environment tracking checks, tracked content-key checks, candidate-size review, and remote-divergence review.
- **Result:** Ruff is clean; strict Mypy reports no issues in 117 source files; all 432 tests pass in 101.08 seconds; both environments have no broken requirements. The adjudication/anchor/holdout files and both aggregate audit summaries reconstruct exactly. Generic and targeted A/32 adapter hashes remain `faa4b72bd0046fbd8d94fc3f364cb7078259403131ea16cd136fdfbbe5408f35` and `c4e455432b9a3b17ae077dafdaa1c9391ab67b7350628f3aa5f9787677d8bb5b`. Across 18 candidate paths, exact-development hits, 12-token-development hits, high-confidence secrets, protected evaluator changes, sealed-path changes, tracked raw/model/environment artifacts, and content-bearing keys in aggregate results are all zero. The ignored powered-evidence directory uses 520,833 bytes; the largest candidate is this DEVLOG at about 522 KB.
- **Gate status:** **VERIFICATION PASSED; UNTOUCHED-BASE USABILITY GATE REMAINS FAILED.** The accurate stop result is eligible for one atomic analysis commit and push.
- **Errors or uncertainty:** The evaluator source used for the stopped run has SHA-256 `bb7016ecfa6a2a9de326b5dc2ca41f5f7dc406f2023be768e5a487667eecea2f`. Required Ruff formatting later collapsed one parenthesized string literal without changing runtime semantics; the publishable source hash is `28955fb0a25a07008a3187ab5cd9a29cda36a5b3e7d1236c66c6dd718136788c`. Tests and gate reconstruction pass, and no model evaluation was repeated. Expected Windows LF-to-CRLF notices are non-fatal. Peak system RAM was not instrumented during the 131.875-second base run, so no peak-RSS claim is made.
- **Next action:** Explicitly stage only the 18 intended source, test, aggregate-evidence, and documentation paths; validate index scope and safety; commit as an accurate powered-retention base-gate blocker; push; and confirm clean 0/0 synchronization.

### 2026-07-20 - Milestone 8L starting state verified

- **Action performed:** Verified clean synchronized `main` at `d22044883457e194d02a9dd769d9b75010eca0b9`; reconstructed both 500-example dataset hashes and all four split hashes from ignored counted-attempt order; recomputed both selected A/32 adapter directory hashes; confirmed exact 14,400-token/32-step parity; revalidated both frozen retention artifacts; checked the pinned training and main environments; and reran the six-prompt offline adapter enable/disable/re-enable sanity audit without GSM1K access.
- **Result:** Targeted dataset/splits remain `987712f6...2876`, `9f8fef80...e464`, and `1ec20743...0ac`; generic remain `49294282...2e7e`, `52276f04...1dd`, and `42f51218...58c`. Generic/targeted adapter hashes remain `faa4b72bd0046fbd8d94fc3f364cb7078259403131ea16cd136fdfbbe5408f35` and `c4e455432b9a3b17ae077dafdaa1c9391ab67b7350628f3aa5f9787677d8bb5b`. Both adapters load with one unmerged `default` adapter, 196 LoRA modules, 392 saved tensors, rank 16, alpha 32, dropout 0.05, scaling 2.0, and exact disabled-base restoration hash `d2b230f9...abff`; sanity summary SHA-256 is `0464887f9d17c62d09cc2ea06244fc0985e8465104edc9db517611e0427667b7`.
- **Gate status:** **PASSED.** CPython 3.12.10, PyTorch 2.5.1+cu121, Transformers 4.51.3, tokenizers 0.21.4, PEFT 0.15.2, TRL 0.17.0, bitsandbytes 0.49.2, Accelerate 1.7.0, CUDA, and both `pip check` results remain valid. No frozen GSM1K inference or sealed-final content access occurred.
- **Errors or uncertainty:** A first read-only aggregate-dataset check concatenated training and validation files and therefore changed record order; the split hashes still matched. Recomputing from the original counted-attempt order produced both exact expected aggregate hashes. No artifact changed.
- **Next action:** Implement and test `foundry-base-conditioned-retention-v1`, then freeze the adjudication base-correct IDs from the existing untouched-base raw result before evaluating the untouched anchor holdout or reading any new adapter output.

### 2026-07-20 - Milestone 8L adjudication base-correct subset frozen

- **Action performed:** Added a content-free, hash-validated base-conditioned subset contract and subset-aware evaluator path, plus standard-library Wilson-bound and preservation-gate logic with focused regression tests. Selected the frozen adjudication IDs solely from the existing untouched-base raw scorer decisions, retaining suite order and category labels.
- **Result:** The immutable subset contains 187 demonstrated base capabilities: 84 arithmetic, 48 format, and 55 instruction prompts. Its subset SHA-256 is `c76df74b911b96ca43c2663a123e41347fd544bf6644f15522ccaad7b77099e1`; its source base-summary SHA-256 is `977dcf32458159f56c711def4b06c454ef24f747c7cc0b36965c0e7fb4f7589e`. The manifest contains no prompt, reference, or raw-output content.
- **Gate status:** **PASSED AND FROZEN.** Every selected ID belongs to the frozen suite and was objectively base-correct. No adapter result was read during construction, and no reference or scorer changed.
- **Errors or uncertainty:** A focused test initially assumed a repeated instruction-skill label in the older 60-item fixture; those skill labels are unique. The test now creates an original temporary suite variant with a repeated content-free skill label, correctly exercising the maximum-three family-failure rule without changing any frozen artifact.
- **Next action:** Evaluate the untouched pinned base exactly once on the frozen 300-item anchor holdout, verify the sample-size usability gate, and only then freeze the holdout base-correct IDs.

### 2026-07-20 - Milestone 8L anchor holdout base gate and subset passed

- **Action performed:** Evaluated the untouched pinned base exactly once, offline and greedily, on the previously frozen 300-item anchor holdout. Checked every frozen reference through the scorer, confirmed the pre-exposure ambiguity audit, applied the sample-size usability gate, and then froze only the base-correct IDs before reading any adapter result.
- **Result:** The base scores 96/100 arithmetic, 60/100 format, and 54/100 instruction, for 210/300 overall; 283/300 outputs are extractable. Prompt echo and backend failures are zero, question generation is one, and malformed outputs are 17. All 300 references self-score correctly and the frozen artifact audit reports zero ambiguous references. Runtime was 138.868 seconds at 2.160 items/second, with 20,429 input tokens, 8,564 output tokens, and 3,554,672,640/3,617,587,200 peak allocated/reserved VRAM bytes. Base summary SHA-256 is `83afe1ba34c45761d0498764ee72c0c1788762626d0c76431cbc499439c290b9`.
- **Gate status:** **PASSED AND FROZEN.** All section minima (40), the 150-overall minimum, zero-backend requirement, and reference/scorer-integrity requirements pass. The resulting 210-item subset contains 96 arithmetic, 60 format, and 54 instruction IDs; subset SHA-256 is `36be91d08f2ab0e05c491094c53965d1aa4f989a730347768877a2548a62c7a9`.
- **Errors or uncertainty:** The first command invocation did not reach model import because the training environment needed the repository `src` directory on `PYTHONPATH`; setting that process-local variable allowed the one counted evaluation to run. Transformers repeated its known warning that sampling-only defaults are ignored under `do_sample=False`; decoding remained deterministic.
- **Next action:** Evaluate the generic A/32 adapter on the frozen adjudication base-correct subset, then proceed through the remaining three fixed-order adapter/subset cells only while the retention protocol remains unchanged.

### 2026-07-20 - Milestone 8L base-conditioned retention failed

- **Action performed:** Evaluated the existing unmerged Variant A step-32 adapters in the frozen order: generic then targeted on the 187-item adjudication subset, followed by generic then targeted on the independent 210-item holdout subset. Applied the fixed overall/category/Wilson/output-behavior/family-concentration gate to every cell and froze the four-cell pair decision.
- **Result:** Generic adjudication preserves 181/187 (96.7914%; Wilson lower 93.1778%), with sections 84/84, 43/48, and 54/55. Targeted adjudication has the same correctness and Wilson result. Generic holdout preserves 197/210 (93.8095%; Wilson lower 89.6981%), with 90/96, 53/60, and 54/54. Targeted holdout preserves 200/210 (95.2381%; Wilson lower 91.4577%), with 93/96, 53/60, and 54/54. All four have zero prompt echo and backend failures. Adjudication has one question-generation output per arm; generic holdout has one and targeted holdout zero. Maximum instruction-family adapter-only failures are one, one, zero, and zero.
- **Gate status:** **FAILED - SFT LINE STOPPED.** Both arms preserve only 89.5833% of adjudication format behaviors and 88.3333% of holdout format behaviors, below the fixed 90% category minimum. Three cells also fail the zero-question-generation clause. Pair-decision SHA-256 is `433c911d89925b2359c1aeb2bca03bc48eac10842034c1ea9cfb846c1fe0a237`; GSM1K authorization is false.
- **Errors or uncertainty:** The first generic-adjudication launch pointed PEFT at the checkpoint container rather than its nested `adapter` directory and failed before adapter loading or inference. The corrected path was then used consistently. Known sampling-default warnings remained non-fatal under greedy decoding. The common format regression appears in both arms and both independent subsets; this supports stopping the shared SFT method but does not identify a causal mechanism beyond the measured behavior.
- **Next action:** Run complete repository verification, commit as `analysis: stop SFT line after base-conditioned retention failure`, push to synchronized `origin/main`, and stop without GSM1K, further training, second-seed work, or sealed-final access.

### 2026-07-20 - Milestone 8L stopped-result verification passed

- **Action performed:** Reconstructed both base-correct manifests, the holdout usability gate, all four preservation assessments, and the pair decision from ignored raw evidence; rehashed both datasets, all four splits, and both adapters; reconfirmed exact 14,400-token arm parity; and ran repository-wide formatting, linting, strict typing, tests, dependency checks, whitespace validation, secret and development-content scans, protected-evaluator checks, ignored-artifact checks, content-bearing-key checks, tracked-size review, and repository-status review.
- **Result:** Ruff formatting and linting pass across 196 files; strict Mypy passes 118 source files; all 436 tests pass in 106.92 seconds; main and training environments have no broken requirements; and `git diff --check` passes. Dataset/split hashes, adapter hashes, subset hashes, base gate, four failed assessment hashes, and pair decision reconstruct exactly. Across 22 candidate files, exact and 12-token overlap with all 904 approved development questions is zero; high-confidence secret hits, protected GSM1K evaluator changes, sealed-path changes, tracked raw/model/environment artifacts, and content-bearing keys in aggregate evidence are all zero. Candidate files total 971,181 bytes; the largest is this DEVLOG at 532,352 bytes. Ignored base-conditioned raw evidence uses 628,765 bytes, tracked aggregate evidence uses 49,970 bytes, and the existing two adapters use 179,593,906 bytes.
- **Gate status:** **VERIFICATION PASSED; BASE-CONDITIONED RETENTION REMAINS FAILED.** The accurate stopped result is eligible for the mandated negative commit and push. GSM1K remains unauthorized and unrun.
- **Errors or uncertainty:** Peak system RAM was not instrumented during these evaluator runs, so no retrospective peak-RAM claim is made. The first development-overlap wrapper accidentally serialized two PowerShell arrays as `System.Object[]`; it scanned no files and was discarded. The corrected wrapper scanned all 22 candidate paths and found zero hits. Windows LF-to-CRLF notices are non-fatal.
- **Next action:** Fetch without integration, require `origin/main` to remain at the approved starting commit, stage only the 22 verified code/test/aggregate/documentation paths, commit with the exact negative-result subject, push, and confirm a clean synchronized branch.

### 2026-07-21 - Milestone 8M starting state verified

- **Action performed:** Read the complete common-scaling authorization and verified clean synchronized `main` at `9af1840792037e5b7b5f91366536554668acf9bf`; reconstructed both matched datasets and all four split hashes, both base-conditioned subset hashes, the previous failed pair decision, Variant A selection, and exact 14,400-token arm parity. Rehashed and reran the offline enable/disable/re-enable sanity audit for both existing step-32 adapters.
- **Result:** Generic and targeted adapter hashes remain `faa4b72bd0046fbd8d94fc3f364cb7078259403131ea16cd136fdfbbe5408f35` and `c4e455432b9a3b17ae077dafdaa1c9391ab67b7350628f3aa5f9787677d8bb5b`. Each loads as one active unmerged `default` adapter with 196 LoRA modules, 392 saved tensors, rank 16, alpha 32, dropout 0.05, scaling 2.0, zero missing/unexpected keys, and exact disabled-base output hash `d2b230f9...abff`. The sanity summary SHA-256 is `15217f0c60ad2f40ac3f52b0a8f3377c82698025d66fea75c515fa4a9cdb6450`.
- **Gate status:** **PASSED.** CPython 3.12.10, PyTorch 2.5.1+cu121, Transformers 4.51.3, tokenizers 0.21.4, PEFT 0.15.2, TRL 0.17.0, bitsandbytes 0.49.2, Accelerate 1.7.0, CUDA, and both `pip check` results remain valid. No GSM1K or sealed-final content was accessed.
- **Errors or uncertainty:** The approval repeats the known 63-character targeted hash transcription; tracked evidence and disk agree on the 64-character value above. It also states that full scale failed only format, but the immutable 8L evidence records one question-generation output in both adjudication cells and generic holdout. Scale 1.00 remains failed as recorded; every lower scale must satisfy the unchanged zero-question-generation clause.
- **Next action:** Implement typed, exception-safe, one-adapter-only runtime LoRA scaling without modifying tensors or checkpoints; add unit/integration coverage; and prove scale 0.0/base and scale 1.0/unscaled output identity before authoring the new holdout.

### 2026-07-21 - Milestone 8M common LoRA scaling sanity passed

- **Action performed:** Implemented `foundry-common-lora-runtime-scaling-v1` as a typed context manager that requires one active unmerged adapter, snapshots all active-adapter scaling values, applies one finite factor uniformly, rejects nesting and invalid factors, and restores the complete scaling map in `finally`. Integrated optional scaling into the retention evaluator and added unit/integration coverage for factors 0.0/0.25/0.50/0.75/1.0, nested and multi-adapter rejection, invalid inputs, exception restoration, output identity, and state hashes. Ran the two-arm offline CUDA sanity audit.
- **Result:** Both adapters pass. Scale 0.0 produces exact untouched-base diagnostic hash `d2b230f9...abff`; scale 1.0 produces the exact existing unscaled hashes `e2fcda47...fb32` (generic) and `375f57e4...308e` (targeted). Every 196-module scaling dictionary is restored; in-memory adapter hashes, deterministic base-parameter signatures, active adapter names, and on-disk adapter hashes match before/after. Focused tests pass 26/26.
- **Frozen hashes:** Scaling source `1e0506ce89a65ab2699f514730eec0437788fe35d55eeb31e299bdd60fa5ceff`; scale configuration `938ec15a61831208dfc138b7829619f801af1205bc6f165a84083b8a64b0ddfc`; sanity evidence `9f7605fe9f8bac8fc763e67636cd2513dcf932a177d2eae451697c9d87420dea`.
- **Gate status:** **PASSED.** Adapter tensors and base parameters remain unchanged; no merge, checkpoint write, dependency change, GSM1K access, or sealed-final access occurred.
- **Errors or uncertainty:** Initial focused checks caught two style/type issues and a duplicate Pytest module basename before CUDA sanity. They were corrected mechanically; no model result or scale policy changed. The base-parameter signature hashes deterministic identity/version metadata rather than copying all 1.5B base values back to CPU; adapter tensors receive a full bytewise state hash.
- **Next action:** Author exactly 450 original, objectively scored holdout prompts in an ignored artifact; statically validate references and scorers; prove disjointness from prior retention, synthetic training, and all 904 development prompts; then freeze content-free suite evidence before any scaled-adapter inference.

### 2026-07-21 - Milestone 8M independent final holdout frozen

- **Action performed:** Authored and froze an original 450-prompt objective holdout before any scaled-adapter inference. The ignored prompt-bearing suite contains 150 arithmetic, 150 exact-format, and 150 deterministic-instruction items; the tracked artifact contains only hashes and aggregate counts. Ran reference self-scoring plus exact-normalized and 12-token disjointness scans against prior retention prompts, all 1,000 synthetic training questions, and all 904 development questions.
- **Result:** All 450 IDs and normalized prompts are unique; every reference self-scores; ambiguous references are zero; and exact/12-token overlaps are zero in all three comparison groups. Suite SHA-256 is `b856c8ce8e56d98eb7e3fbffdead07ffde7091ab2a20abe5a22ada598136353e`; suite-file SHA-256 is `9f42c6c4ed3fe6b712ea4cb96d666fbea138ae653112a62244568371ddda8939`; prompt SHA-256 is `f9fe2ce26afc8c8b4f53cc91f587c24cab0d292bd84c22b602201275602cdfa6`; answer SHA-256 is `7e3300038720b536728252eeed85bd115213c0d9e26f88092d65916ad4d14da4`; evidence summary SHA-256 is `6a627756f573d3f998f6046b13d8baa2ddee3167054b3a6ffc9b3b4936b87647`.
- **Gate status:** **PASSED AND FROZEN BEFORE SCALED ADAPTER EXPOSURE.** No adapter output, benchmark answer, GSM1K inference, or sealed-final content was accessed.
- **Errors or uncertainty:** The first pre-freeze build correctly failed because the strict alphanumeric prompt normalizer collapsed two pairs of symbol-count fixtures whose punctuation multiplicities repeated. An inert visible alphanumeric prefix made those original fixtures distinct; the complete static audit then passed. This correction occurred before any base or scaled-adapter evaluation.
- **Next action:** Run the untouched base exactly once on all 450 prompts, inspect every base failure, and freeze the base-correct subset only if the predeclared instrument-usability thresholds pass.

### 2026-07-21 - Milestone 8M final-holdout base gate passed

- **Action performed:** Evaluated the untouched pinned base exactly once, offline and greedily, on all 450 frozen final-holdout prompts. Inspected all 132 objective failures without adapter output, validated all references through the frozen scorer, applied the predeclared `60/60/60` section and `250/450` overall usability thresholds, and froze only the 318 base-correct IDs.
- **Result:** Base correctness is arithmetic `112/150`, format `127/150`, instruction `79/150`, and overall `318/450`; extractability is `393/450` (87.3333%). Prompt echo, question generation, backend failures, objectively defective prompts/references/scorers, and reference self-score failures are all zero. The 132 inspected failures comprise 38 genuine arithmetic/terminal-contract failures, 23 genuine format failures, and 71 genuine instruction failures. Runtime was 173.315 seconds at 2.596 examples/second, with 24,561 input tokens, 10,857 output tokens, and 3,554,672,640/3,617,587,200 peak allocated/reserved VRAM bytes.
- **Frozen hashes:** Base summary `dc48cd7656de41ffb0f50103e81a55dd29b8e5c90a620a2a30fd9e8a0897add5`; failure audit `98b96c15934964d47817cb2d9650e21e7290789e6906f224c5cf0c8a3f2239c0`; usability gate `38c87e2f1db61067156b2a07e18be80a066e9373eb6cc1b4f5462644242e9a62`; base-correct subset `0884923ce7ab39f1080282dab0ce51aff7063270d6c97f5c1d70370256012ded` with 112 arithmetic, 127 format, and 79 instruction IDs.
- **Gate status:** **PASSED AND SUBSET FROZEN BEFORE ADAPTER INFERENCE.** The new holdout cannot participate in scale selection.
- **Errors or uncertainty:** Known greedy-decoding warnings remained non-fatal. Console rendering displayed multiplication, division, and em-dash characters with the terminal replacement glyph, but code-point inspection confirmed the frozen UTF-8 prompts contain valid `U+00D7`, `U+00F7`, and `U+2014` characters rather than replacement characters. Peak system RAM was not instrumented for this run.
- **Next action:** Reconstruct the historical scale-1.00 four-cell decision exactly, then evaluate scale 0.75 in the frozen generic/targeted by adjudication/holdout order.

### 2026-07-21 - Milestone 8M common scale 0.75 failed retention

- **Action performed:** Reconstructed all four historical scale-1.00 assessments byte-for-byte, then evaluated scale 0.75 in the frozen order on the adjudication and independent anchor-holdout base-correct subsets. Applied one uniform scale to both arms and all 196 LoRA modules; verified tensor, base-signature, scaling-map, and adapter restoration after every run.
- **Result:** Generic and targeted adjudication each preserve `182/187`: arithmetic `84/84`, format `44/48`, instruction `54/55`, Wilson lower bound 93.8945%, zero question generation, and all gates pass. Generic holdout preserves `196/210`: `87/96`, `55/60`, `54/54`, Wilson lower 89.1222%, but produces 3 question-like outputs. Targeted holdout preserves `197/210`: `88/96`, `55/60`, `54/54`, Wilson lower 89.6981%, but produces 4 question-like outputs. Both holdout cells pass every preservation, category, Wilson, prompt-echo, backend, and family-concentration requirement except the frozen zero-question-generation clause.
- **Gate status:** **FAILED.** Scale 0.75 cannot be selected. Its four gate-summary hashes are `6e228b9b49ea7bf083a4b78d9f16fa9a8c229eba72ead73e8a0e84a30aebbf31`, `2f7d98688658d03d9c4be6866013e7a9e7f323009d2a3410d6792ca92af8420e`, `d2b1c464ab3ef76505f4103ab0850b640b8a8c59593ed4118d6af576dc502a42`, and `d8e27bb6135a52f325c058b7b0233dc8de0081d7c678e3125f546acf5bd817ec` in frozen run order.
- **Errors or uncertainty:** Transformers emitted only the known non-fatal greedy-decoding warnings. The question-generation failures are frozen scorer decisions and are not reinterpreted. The new 450-item final holdout remains unexposed to adapters.
- **Next action:** Evaluate scale 0.50 in the same four-run order. Do not evaluate 0.25 unless 0.50 fails.

### 2026-07-21 - Milestone 8M common scale 0.50 selected

- **Action performed:** Evaluated scale 0.50 in the fixed generic/targeted by adjudication/holdout order and applied the unchanged base-conditioned retention gate to all four cells. Froze the first-passing descending-scale selection using only the two existing subsets; neither the new final holdout nor GSM1K contributed to selection.
- **Result:** Generic adjudication preserves `182/187` (`83/84`, `45/48`, `54/55`); targeted adjudication preserves `183/187` (`84/84`, `45/48`, `54/55`). Generic and targeted holdout each preserve `205/210` (`94/96`, `57/60`, `54/54`). Every cell passes overall/category preservation, Wilson lower bound, prompt echo, zero question generation, zero backend failures, and instruction-family concentration; all model-state restorations pass. Gate hashes are `9cce2ca00c1e13cfc3d32508310f5c28854ddaa8c6df09389b96a28f3fc3b3ee`, `0f8e9e05e3e880b3f0d309c5ce77f969da8942c4435aad6cfeaa9570d757bbd9`, `d9de41c4ed68f451a22009f7688d2329bb1b85b93318899743c4ef52d46786fb`, and `6adcf3b983d35df9ee051840161b6c9a42b079c0c9218d1eea2d5e6b999632bc`.
- **Gate status:** **PASSED; COMMON SCALE 0.50 SELECTED.** Selection summary SHA-256 is `d7455a57001acf68f37369503cce9ef4e3fc30755b2f7f3fd8b7c7055a0b986c`. Scale 0.25 is skipped by the frozen first-passing rule.
- **Errors or uncertainty:** Known greedy-decoding warnings remained non-fatal. Scale 0.50 slightly changes arithmetic behavior on adjudication between arms, but both remain above every frozen threshold. Selection says nothing yet about the newly frozen independent holdout.
- **Next action:** Evaluate generic then targeted exactly once at scale 0.50 on the new 318-item base-correct final-holdout subset. Do not fall back to another scale if either validation cell fails.

### 2026-07-21 - Milestone 8M independent scaled-retention validation passed

- **Action performed:** Evaluated the selected common scale 0.50 exactly once for generic then targeted on the newly frozen 318-item final-holdout base-correct subset. Applied the same retention gate and froze a two-arm independent validation decision; no lower scale, GSM1K prompt, or sealed-final content participated.
- **Result:** Generic preserves `314/318`: arithmetic `110/112`, format `127/127`, instruction `77/79`; targeted preserves `315/318`: `111/112`, `127/127`, `77/79`. Both have 100% extractability, zero malformed outputs, zero prompt echo, zero question generation, zero backend failures, and exact post-run adapter/base/scaling restoration. Generic and targeted gate hashes are `f0441002370b6a396b4c43857c122a6f8b48daad26a99f4648b92d68f6a75772` and `d6eb33f60773d4ff2697e50ce8e6650b5790ebecbb23baacb4e9672e1db059d9`.
- **Gate status:** **PASSED.** Decision is `retention_approved_common_scaled_short_run_adapters`; decision SHA-256 is `6f3e7a29dfbb184f5b6b5eb09fd52060c3c2465c5da2343f85d62d05f8589cc7`. Frozen GSM1K development evaluation is authorized only after repository verification, retention-decision commit, push, and synchronization.
- **Errors or uncertainty:** Known greedy-decoding warnings remained non-fatal. The final holdout demonstrates retention at scale 0.50; it does not estimate GSM1K benefit. Peak system RAM was not instrumented during the retention evaluations.
- **Next action:** Complete documentation and repository-wide verification, commit and push the retention approval, confirm clean `0/0` synchronization, and only then run generic followed by targeted on the frozen 814-item development evaluator.

### 2026-07-21 - Milestone 8M retention-decision verification passed

- **Action performed:** Ran repository-wide Ruff formatting/linting, strict Mypy, all tests, main/training environment dependency checks, whitespace validation, content-free hash validation, exact reconstruction of ten scaled assessments, scale selection, base gate/subset, and final decision, plus dataset/split, adapter, token-parity, state-restoration, development-content, secret, protected-evaluator, sealed-path, ignored-artifact, size, and remote-divergence checks.
- **Result:** Ruff is clean; strict Mypy reports no issues in 121 source files; all 456 tests pass in 96.22 seconds; both environments have no broken requirements; and `git diff --check` passes. All 28 new content-free JSON hashes validate. Dataset and split hashes reconstruct exactly; adapter hashes remain `faa4b72b...8f35` and `c4e45543...bb5b`; Variant A remains exactly 14,400 tokens per arm. Ten scaled gates, the 318-ID subset, selection, and final decision reconstruct exactly from ignored raw evidence. Across 45 candidate files, exact and 12-token development-question hits, high-confidence secrets, content-bearing top-level result keys, forbidden raw/model/environment paths, frozen-evaluator changes, sealed-path changes, and files at or above 1 MB are all zero. The largest candidate is this DEVLOG at 548,535 bytes before this entry.
- **Resources:** The eleven counted base/scaled retention evaluations used 1,658.348 seconds, 169,237 input tokens, and 56,017 output tokens; peak allocated/reserved VRAM was 3,554,672,640/3,751,804,928 bytes. Ignored scaled-retention evidence uses 1,713,791 bytes; the two unchanged selected adapter directories use 179,593,906 bytes. Peak system RAM was not instrumented.
- **Gate status:** **VERIFICATION PASSED; RETENTION APPROVAL IS READY TO PUBLISH.** Local and `origin/main` remain at the approved starting commit with `0/0` divergence before commit.
- **Errors or uncertainty:** The first candidate safety wrapper had an unclosed Python regex string and performed no content scan; the corrected read-only wrapper scanned all 45 files and found zero hits. Expected Windows LF-to-CRLF notices and known greedy-decoding warnings are non-fatal. No sealed-final file was opened.
- **Next action:** Fetch without integration, require the remote tip to remain unchanged, explicitly stage only the verified retention files, repeat staged safety checks, commit as `analysis: approve common-scaled adapters by retention`, push, and confirm a clean synchronized branch before GSM1K.

### 2026-07-21 - Milestone 8M scaled development-evaluation preparation

- **Action performed:** Published the independently validated common-scale retention decision as commit `2a69bca54b8e90b88d51b91e2ea347f221722e5c`, pushed it to `origin/main`, and confirmed a clean synchronized branch. Inspected the existing training-side PEFT development backend and added exception-safe use of the already verified runtime LoRA scaling context at an explicit required scale. The backend continues to call the unchanged frozen evaluator for the 814 development IDs, prompt, greedy decoding, extraction, and scoring.
- **Result:** Focused scaling/backend tests pass `18/18`, strict Mypy passes, and the selected Variant A step-32 adapter directories still hash to `faa4b72bd0046fbd8d94fc3f364cb7078259403131ea16cd136fdfbbe5408f35` and `c4e455432b9a3b17ae077dafdaa1c9391ab67b7350628f3aa5f9787677d8bb5b`. The exact pinned base snapshot is available offline. Runtime summaries will include scale, LoRA-module count, adapter-state hashes, base-parameter signatures, and post-run restoration evidence.
- **Gate status:** **PASSED FOR EVALUATION START.** The frozen evaluator and evaluator configuration are unchanged; generic must run first at scale `0.50`, followed by targeted at the identical scale.
- **Errors or uncertainty:** Focused Ruff initially found one import-order issue, which was fixed mechanically. Peak system RAM is not instrumented by the existing evaluator backend; GPU allocation/reservation and runtime/token throughput remain measured.
- **Next action:** Run the generic adapter on the frozen 814-example development evaluator, preserve raw predictions only under ignored results, then run targeted second with no configuration change.

### 2026-07-21 - Milestone 8M scaled generic development evaluation complete

- **Action performed:** Loaded the selected Variant A step-32 generic adapter from its ignored frozen directory, applied the selected common runtime scale `0.50`, and evaluated it first on exactly the frozen 814-example GSM1K development manifest through the unchanged evaluator. The run was offline and greedy with the frozen 768-token ceiling; raw predictions remain ignored.
- **Result:** Generic scored `387/814` (`47.5430%`), with `768/814` extractable answers (`94.3489%`), 482 exact-format-compliant responses, 46 unextractable responses, 2 truncations, and zero backend failures. It is 134 correct below the frozen base (`521/814`). Runtime was 4,412.066 seconds, generation throughput 31.687 output tokens/second, and peak allocated/reserved VRAM was 3,248,531,968/3,512,729,600 bytes. All 814 prediction rows are present; manifest, prompt, extractor, model, revision, and adapter hashes match; all 196 scaling entries were restored and adapter/base signatures are unchanged. Tracked summary-file SHA-256 is `a6d2f049b77e06f6ce5f8fda6da0b3ebca3fc88e2b168d6c849b62be43c2452b`.
- **Gate status:** **GENERIC ARM COMPLETE.** This intermediate result does not alter the predeclared requirement to run targeted second at the same scale and settings.
- **Errors or uncertainty:** Only the known non-fatal greedy-decoding warnings appeared. Peak system RAM and continuous GPU utilization were not instrumented. The large negative generic delta is measured, not yet a paired comparative conclusion.
- **Next action:** Run the targeted adapter second on the identical frozen 814 IDs at scale `0.50`, then validate exact count/hash/restoration evidence and perform the frozen paired/category analysis.

### 2026-07-21 - Milestone 8M scaled targeted development evaluation complete

- **Action performed:** Loaded the selected Variant A step-32 targeted adapter second, applied the same common runtime scale `0.50`, and evaluated exactly the same frozen 814 development IDs through the unchanged evaluator. Validated row count, model/manifest/prompt/extractor/adapter hashes, zero backend failures, and complete post-run scaling/adapter/base restoration.
- **Result:** Targeted scored `414/814` (`50.8600%`) with `767/814` extractable (`94.2260%`), 479 exact-format-compliant, 353 extractable-but-incorrect, 47 unextractable, 3 truncated, and zero backend failures. It is 107 below the frozen base and 27 above generic. Input/output tokens are 113,720/145,865; peak allocated/reserved VRAM is 3,248,531,968/3,512,729,600 bytes. Summary-file SHA-256 is `bc0710c9a1fc79274204d5a39ab8f7fa241d8a52238d4e90a4f579a2208273a5`.
- **Gate status:** **TARGETED ARM COMPLETE; ABSOLUTE SIGNAL FLOOR CANNOT PASS.** Targeted is below the frozen minimum `529/814`, although paired/category analysis remains required.
- **Errors or uncertainty:** The evaluator reports 33,838.811 seconds of end-to-end wall time and 4.311 generated tokens/second because the process experienced an observed host suspension/long scheduling pause after launch. Those elapsed/throughput values are preserved as measured but are not active GPU compute-time estimates. No model/backend error occurred.
- **Next action:** Run the frozen 10,000-replicate paired bootstrap and failure-taxonomy category analysis, explicitly include current 14,400/14,400 training parity and the all-three-subset retention decision, then apply all seven signal clauses.

### 2026-07-21 - Milestone 8M paired analysis and one-seed signal decision

- **Action performed:** Froze a content-free parity artifact for the actual Variant A step-32 pair (`14,400/14,400` assistant tokens), ran the standard-library paired analysis over aligned base/generic/targeted prediction IDs, performed 10,000 bootstrap replicates with frozen seed `20260720`, summarized the frozen failure taxonomy, and applied all seven signal-gate clauses including independent common-scale retention.
- **Result:** Generic/targeted/base are `387/414/521`. Targeted wins 47 rows generic misses and generic wins 20 targeted misses (net +27); it fixes 58 base failures and breaks 165 base successes, versus generic 54/188. Targeted minus generic is +3.3170 percentage points with 95% interval `[+1.3514, +5.2826]`. Targeted/generic fixes on selected taxonomy categories are bookkeeping `14/12`, rate/ratio `4/5`, and discrete `4/3`; the 170-row untargeted failure-only set is `36/34`. Analysis SHA-256 is `8cd2e7c9556e08850345166b89ed5c1d2c932b96f7ed203e59ef43f50cfcb9ed`; final decision SHA-256 is `2b4f39b542ebe16a4cdfd4835856b9965de9dc04c2384fffaf12a064d736a0ed`.
- **Gate status:** **FAILED.** Six clauses pass: targeted is +27 over generic, extractability is at least 91.38%, backend failures are zero, untargeted-taxonomy decline is within two points, token parity passes, and retention passes all three subsets. The sole failure is targeted `414 < 529`.
- **Errors or uncertainty:** The failure taxonomy covers only 293 frozen base failures, so category results cannot describe the 521 base-success rows where both adapters regress. The result remains provisional pending stratified human language review and second-seed confirmation, but the failed predeclared gate does not justify running a second seed automatically.
- **Next action:** Complete required documentation and full repository/safety verification, create and push `train: evaluate retention-calibrated targeted and generic adapters`, then stop. The user may complete the pending language review and decide whether to stop adaptation or separately approve a materially different retention-preserving architecture.

### 2026-07-21 - Milestone 8M final evaluation verification passed

- **Action performed:** Ran final Ruff formatting/linting, strict Mypy, all unit/integration tests, both dependency checks, whitespace validation, exact deterministic paired-analysis replay, dataset/split and adapter reconstruction, prediction-ID/count alignment, scaling/restoration and seven-clause gate checks, exact and 12-token scans against all 904 development questions, secrets/content-bearing-result scans, protected evaluator/sealed-path diffs, ignored raw/model/environment tracking checks, candidate size review, and remote-divergence checks.
- **Result:** Ruff is clean; strict Mypy reports no issues in 121 source files; all 457 tests pass in 140.44 seconds; both environments have no broken requirements; and `git diff --check` passes. The paired output is byte-identical on replay. All 814 base/generic/targeted IDs align; dataset/split/adapter hashes reconstruct; exact/12-token development hits, secret hits, content-bearing result keys, frozen-evaluator changes, sealed-path changes, tracked raw/model/environment artifacts, and candidate files at or above 1 MiB are all zero. The final candidate set has 15 paths; the largest is this DEVLOG below 1 MiB. Local and `origin/main` remain at retention commit `2a69bca54b8e90b88d51b91e2ea347f221722e5c` with `0/0` divergence before final commit.
- **Resources:** The two GSM1K arms consumed 227,440 input and 285,585 output tokens, with peak allocated/reserved VRAM 3,248,531,968/3,512,729,600 bytes. Their recorded wall interval totals 38,250.878 seconds, including the targeted run's host suspension. Together with eleven base/scaled retention evaluations, Milestone 8M records 39,909.225 seconds, 396,677 input tokens, and 341,602 output tokens; maximum reserved VRAM is 3,751,804,928 bytes. Peak system RAM was not instrumented. Ignored scaled-retention/GSM raw evidence uses 3,400,129 bytes; unchanged adapters use 179,593,906 bytes; the pinned base snapshot uses 3,098,955,668 logical bytes.
- **Gate status:** **FINAL VERIFICATION PASSED; ONE-SEED SIGNAL GATE FAILED.** The verified result is ready for its required final commit and push.
- **Errors or uncertainty:** One read-only development-content scan was initially computationally inefficient and ended without a result; its optimized replacement completed with zero hits. One dataset verification initially concatenated split files rather than preserving the published accepted-attempt order; split hashes were exact and reconstruction using the defined attempt order yielded both exact dataset hashes. Neither issue changed evidence. Expected Windows line-ending notices are non-fatal.
- **Next action:** Stage only the 15 verified content-free/code/documentation paths, repeat index-only safety checks, commit as `train: evaluate retention-calibrated targeted and generic adapters`, push normally, confirm a clean synchronized branch, and stop.

### 2026-07-21 — Milestone 8N starting-state verification

- **Action performed:** Verified the repository root, `main` branch, clean worktree, and exact `0/0` synchronization of local and `origin/main` at `0b6cbe350b6065317fe06aa93b48502ee4e57c56`. Rehashed both ignored Variant A step-32 source adapters, checked the pinned offline base snapshot, confirmed the three frozen base-correct retention-subset identities, reviewed the tracked evaluator/config hashes without opening the sealed-final manifest, and recorded the existing Python/CUDA package stack.
- **Result:** Generic and targeted adapters remain `faa4b72bd0046fbd8d94fc3f364cb7078259403131ea16cd136fdfbbe5408f35` and `c4e455432b9a3b17ae077dafdaa1c9391ab67b7350628f3aa5f9787677d8bb5b`; both declare rank 16, alpha 32, dropout 0.05, no bias, no `modules_to_save`, and the same seven target projections against base revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Training parity remains exactly 14,400/14,400 loss-bearing tokens. CPython 3.12.10, PyTorch 2.5.1+cu121, Transformers 4.51.3, PEFT 0.15.2, and CUDA on the RTX 3080 are available.
- **Gate status:** **PASSED.** Milestone 8N source-adapter compatibility and exact-delta work may begin.
- **Errors or uncertainty:** None. The sealed-final manifest was neither opened nor hashed.
- **Next action:** Add and test one exact targeted-minus-generic task-vector implementation, load both source adapters under distinct names, and verify compatibility before constructing any retained artifact.

### 2026-07-21 — Milestone 8N source compatibility and dense-delta analysis

- **Action performed:** Added a frozen task-vector protocol and streamed all 196 corresponding source LoRA modules in FP32. Validated directory hashes, normalized configurations, tensor keys, shapes, dtypes, ranks, scaling, target projections, and absence of base parameters or `modules_to_save`; calculated dense generic, targeted, and exact targeted-minus-generic statistics without materializing full-model vectors concurrently.
- **Result:** Both sources share the frozen 392-tensor inventory `c3334a15154beae94d6b4bd043d24742cf522b86f48efd044e3a9fe7e1f978b9`. Generic, targeted, and contrastive global Frobenius norms are `1.6918784364`, `1.6980775191`, and `0.5876302228`; generic-targeted cosine similarity is `0.9399098552`. The contrastive norm is `34.6056%` of targeted and `34.7324%` of generic. Every dense value is finite. After final protocol-binding regeneration, analysis SHA-256 is `36ce1b90beee7499aa33e11dacbe163e107a98bda5f1065e3f7841fbd85fbaa2` and its protocol field matches `b4914d5a95bb46a52374b9a390038634f01df99f69a4ef6f79c5bfe4f8d983fa`.
- **Gate status:** **PASSED.** Exact PEFT `cat` composition is allowed.
- **Errors or uncertainty:** The authorization abbreviated the targeted adapter hash with one missing character; the complete prior frozen hash, tracked Milestone 8M evidence, and current directory all agree on `c4e455432b9a3b17ae077dafdaa1c9391ab67b7350628f3aa5f9787677d8bb5b`. Runtime simultaneous named loading and diagnostic reconstruction remain to be verified on CUDA.
- **Next action:** Load both sources under distinct names, construct the unmerged rank-32 adapter, save only that adapter under ignored storage, and require exact dense/logit equivalence plus scale and state-safety gates.
### 2026-07-21 - Milestone 8N exact contrastive construction

- Diagnosed the initial composition error as loss of source precision: PEFT formed the new adapter through FP16 base-dtype factors. The saved FP32 result therefore preserved FP16-rounded values. Construction now reads the two immutable FP32 source safetensors directly; a regression test proves rounded in-memory factors cannot influence the result.
- Froze protocol SHA-256 `b4914d5a95bb46a52374b9a390038634f01df99f69a4ef6f79c5bfe4f8d983fa`. The numerical tolerances were not changed. Functional equivalence is verified using an FP32 verification-only base load so low-rank-versus-dense GEMM association is tested without FP16 residual-stream amplification; production inference remains unchanged.
- Constructed ignored rank-32 `targeted_minus_generic_v1` adapter SHA-256 `84f02df1cbc5ec1015d096164dbfe3833e166a14eda9ffadf62b5d2d2527c961`. Dense equivalence passed across 196 modules (maximum absolute error `1.7462298274040222e-10`; relative Frobenius error `2.9353350495838164e-07`). Functional logit equivalence passed (maximum absolute error `5.626678466796875e-05`; relative Frobenius error `1.959386123746654e-06`).
- Scale-zero and scale-one sanity passed. Source-adapter state, source directories, and base state remained unchanged; no adapter was merged. Hardened construction/equivalence evidence SHA-256 is `3418dedbae18dda5fecf5941cab33b20b201a2e09feabde4d90bc73470df8a7e`; construction summary SHA-256 is `07a99bde03339494cc1ce9cf8428d7ecf7ad35aef58b55038389a3888d2c586c`.
- Hardened construction took `27.863` seconds and reserved at most `9,248,440,320` GPU bytes (8.61 GiB); releasing stale FP16 module references before FP32 verification reduced memory pressure without changing adapter bytes or equivalence metrics. Pre-existing Transformers warnings noted inactive sampling parameters during greedy diagnostics; they did not alter deterministic outputs.

### 2026-07-21 - Milestone 8N contrastive retention scale 1.00

- Adjudication passed: 181/187 preserved (`96.7914%`), arithmetic 80/84, format 46/48, instruction 55/55, Wilson lower bound `93.1778%`, zero prompt echoes, zero question generation, and zero backend failures. Gate-evidence SHA-256: `3e9ee9f0071b6902cd385ce87a7caa7a521921d7ae8b1225346935f5bb4eb4a4`.
- Anchor did not pass despite preserving 204/210 (`97.1429%`): arithmetic 91/96, format 60/60, instruction 53/54, and Wilson lower bound `93.9079%` passed, but one output was classified as question generation against a frozen maximum of zero. Gate-evidence SHA-256: `4a3e7b06df1be2dd0c787eeb8d01d57a1190654584f847c8b70cbbad55493ee1`.
- Adapter/base state restoration passed in both runs. Scale 1.00 was not selected; the predeclared ladder continues to 0.75 without inspecting GSM1K.

### 2026-07-21 - Milestone 8N contrastive retention scale 0.75

- Adjudication passed: 182/187 preserved (`97.3262%`), arithmetic 81/84, format 46/48, instruction 55/55, Wilson lower bound `93.8945%`, and zero prohibited behaviors. Gate-evidence SHA-256: `503300a7ca3706813a255c873f6017666bdb3e88d679ce3582bbb6006f04ea40`.
- Anchor did not pass despite 207/210 preserved (`98.5714%`), arithmetic 94/96, format 59/60, instruction 54/54, and Wilson lower bound `95.8848%`: two outputs were classified as question generation. Gate-evidence SHA-256: `edac709957a61fe72b05ff7c793bea312df08dda343e1c2cbdc9daac01532db2`.
- State restoration and backend checks passed. Scale 0.75 was not selected; the predeclared ladder continues to 0.50 without GSM1K access.

### 2026-07-21 - Milestone 8N contrastive retention scale 0.50

- Adjudication passed: 183/187 preserved (`97.8610%`), arithmetic 81/84, format 47/48, instruction 55/55, Wilson lower bound `94.6300%`, and zero prohibited behaviors. Gate-evidence SHA-256: `8dca6f488f68a1ef41a69006a1b1c0f6cf441d4710c97421d23c23bc47caa70f`.
- Anchor did not pass despite 207/210 preserved (`98.5714%`), arithmetic 94/96, format 60/60, instruction 53/54, and Wilson lower bound `95.8848%`: one output was classified as question generation. Gate-evidence SHA-256: `7b561acc9fa2e9bb48da3779c08a94110386e5e95e51ae7ed0f7f0f634f4311f`.
- State restoration and backend checks passed. Scale 0.50 was not selected; the final authorized selection scale is 0.25.

### 2026-07-21 - Milestone 8N contrastive retention scale 0.25 and stop decision

- Adjudication passed: 184/187 preserved (`98.3957%`), arithmetic 81/84, format 48/48, instruction 55/55, Wilson lower bound `95.3902%`, and zero prohibited behaviors. Gate-evidence SHA-256: `080f44931a4b95be36a3c0e8582243a0d69f6cfadd4189744591efbafc445bf0`.
- Anchor did not pass despite 208/210 preserved (`99.0476%`), arithmetic 94/96, format 60/60, instruction 54/54, and Wilson lower bound `96.5946%`: two outputs were classified as question generation. Gate-evidence SHA-256: `853531884fc2a356f8862d89ee449cd4d73bcf12aab1fd714213af599c5809e6`.
- No scale in the frozen descending ladder passed both selection suites. The hardened selector independently recomputes every metric and gate, binds all eight evaluation/assessment pairs to construction summary `07a99bde03339494cc1ce9cf8428d7ecf7ad35aef58b55038389a3888d2c586c` and the exact adapter hash, and has SHA-256 `b41d975f342820ac34ca693d599677994e3f272243c114c313605beb020ad49a`; `selection_passed=false`, `selected_contrastive_scale=null`, and `gsm1k_authorized=false`.
- Per the approved stop rule, the independent final holdout was not opened for evaluation, GSM1K was not run, and no further scale, merge method, or retraining attempt was started. The exact blocker is persistent question-generation behavior on the anchor suite (1, 2, 1, and 2 cases at scales 1.00, 0.75, 0.50, and 0.25 respectively), despite otherwise strong preservation.

### 2026-07-21 — Milestone 8N final verification passed

- **Action performed:** Rebound the content-free dense analysis to the final frozen protocol, replayed adapter/source/construction/selection hashes, and ran repository-wide Ruff formatting/linting, strict Mypy, focused and full tests, both environment dependency checks, whitespace validation, state-restoration and equivalence checks, high-confidence secret scanning, exact and 12-token scans against all 904 approved development questions, frozen evaluator/generator/verifier review, sealed-path status review, ignored raw/model/cache/environment checks, candidate-size review, and repository status review.
- **Result:** Dense values are unchanged and the regenerated analysis now binds protocol `b4914d5a95bb46a52374b9a390038634f01df99f69a4ef6f79c5bfe4f8d983fa` with analysis SHA-256 `36ce1b90beee7499aa33e11dacbe163e107a98bda5f1065e3f7841fbd85fbaa2`. Ruff is clean across 206 files; strict Mypy reports no issues in 123 source files; 478 unit tests and 7 integration tests pass; both environments have no broken requirements; and `git diff --check` passes. All source/composed hashes, dense and functional equivalence evidence, eight retention evaluations, eight gate summaries, and scale selection replay exactly. Across 30 candidate files, exact-development hits, 12-token-development hits, high-confidence secrets, frozen evaluator/synthesis changes, sealed-path changes, and tracked raw/model/cache/environment artifacts are all zero; no candidate exceeds 1 MiB.
- **Resources:** Exact construction took `27.863` seconds; peak allocated/reserved VRAM was `6,606,550,528`/`9,248,440,320` bytes; process working set/peak working set was `2,393,239,552`/`5,471,981,568` bytes. The eight retention runs took `1,858.707` seconds, used `110,008` input and `57,016` output tokens, and peaked at `3,554,672,640`/`3,730,833,408` allocated/reserved GPU bytes. The exact rank-32 adapter uses `147,771,481` bytes; all contrastive raw evidence, including preserved diagnostic construction attempts, uses `665,985,307` bytes under ignored storage.
- **Gate status:** **VERIFICATION PASSED; CONTRASTIVE RETENTION GATE FAILED.** No contrastive scale was selected, so independent final-holdout and GSM1K inference were not authorized.
- **Errors or uncertainty:** The first development scan incorrectly summed three named 30-item subsets on top of their encompassing 904-item canonical manifest and failed before scanning content; the corrected scan used exactly the canonical 904 unique development rows and found zero hits. Expected Windows line-ending notices are non-fatal. No sealed-final content was opened or hashed.
- **Next action:** Stage only the 30 verified content-free/code/test/documentation paths, recheck the Git index and remote divergence, commit as `analysis: record contrastive adapter retention failure`, push normally, confirm a clean synchronized branch, and stop. The next architectural decision belongs to the user.

### 2026-07-21 — Milestone 9 starting-state verification

- **Action performed:** Verified clean synchronized `main` at `08ae1a2763e5a054972231c0817b6ff8e35fc076`; reconstructed both 500-example dataset hashes and all four 450/50 split hashes; rehashed the pinned untouched base, shared replay anchor, three retention subsets, frozen evaluator contract, and complete contrastive evidence; checked both Python environments, CUDA, ignore/tracking boundaries, and sealed-path status without opening sealed-final content.
- **Result:** Every frozen hash and count matches. The replay anchor contains 120 unique original items, exactly 40 per arithmetic/format/instruction category, with artifact SHA-256 `a15df37c7318432576878ff86e567a0f0bac050cc62e2af61081a937f2c1740c` and gold-response SHA-256 `9634f8cbf5490ef3ca237ce07ff5f142f118dbf891c9b448a9eda214ebde13a6`. CPython 3.12.10, PyTorch 2.5.1+cu121, Transformers 4.51.3, tokenizers 0.21.4, PEFT 0.15.2, TRL 0.17.0, bitsandbytes 0.49.2, Accelerate 1.7.0, CUDA, and the RTX 3080 remain available; both environments pass `pip check`.
- **Gate status:** **PASSED.** Base-anchor evaluation and replay-corpus freezing are authorized.
- **Errors or uncertainty:** One independent combined dependency command briefly reported a PyYAML metadata mismatch, but immediate labelled reruns of both environments passed without any installation or state change. The raw anchor uses its own frozen `anchor_id` schema rather than the existing retention-suite schema, so a strict replay-specific loader/evaluator adapter is required; the anchor itself will not be rewritten.
- **Next action:** Implement the strict anchor contract and deterministic untouched-base evaluation, then freeze only base-correct IDs and actual untouched-base outputs as ignored replay targets. No adapter or benchmark inference may occur.

### 2026-07-21 — Milestone 9 untouched-base replay-anchor evaluation

- **Action performed:** Added the strict `foundry-base-replay-anchor-evaluation-v1` loader/evaluator, bound the frozen 120-item anchor and its prompt/gold/answer identities, then evaluated the pinned untouched Qwen2.5-1.5B base on every prompt with greedy decoding, seed `20260720`, and no adapter loaded. Full prompts and outputs were written only to the ignored replay workspace.
- **Result:** The base scored `40/40` arithmetic, `20/40` format, and `23/40` instruction, for `83/120` correct overall. It produced 114 extractable responses, zero prompt echoes, zero question-generating responses, six scorer-classified malformed outputs, and zero backend failures. Base-result SHA-256 is `d35f07647a57b28231598713860478507bee1556c0c2ad89393a261a127c3295`; content-free summary SHA-256 is `f1f686779ad27dcd26aa9c7005cc7be9dc3a5109883a162931709fadba4931db8`.
- **Resources:** Model load took `3.222` seconds and generation took `30.528` seconds for 120 prompts (`3.931` examples/second), consuming 7,956 input and 1,733 output tokens. Peak allocated/reserved GPU memory was 3,554,672,640/3,617,587,200 bytes on the RTX 3080.
- **Gate status:** **PASSED.** The 83 scorer-correct results meet the minimums of 20 per category and 75 overall with zero backend failures and zero confirmed prompt/scorer defects. Format coverage is exactly at the required minimum and is recorded as a limitation, not changed after evaluation.
- **Errors or uncertainty:** The first process launch failed before loading the model because `.venv-training` does not install the repository package. Relaunching with this repository's `src` directory on the process-local `PYTHONPATH` succeeded without changing dependencies. Transformers emitted inactive sampling-parameter warnings inherited from the model generation configuration; greedy decoding remained in force.
- **Next action:** Freeze the exact 83 scorer-correct IDs and their actual deterministic base outputs as the shared replay corpus, keeping all content-bearing records ignored and tracking only content-free hashes and counts.

### 2026-07-21 — Milestone 9 shared replay corpus frozen

- **Action performed:** Revalidated every base-anchor row, frozen scorer result, response hash, category, base revision, and aggregate count, then selected only scorer-correct items. Persisted complete prompts and actual base responses under ignored raw storage and created a content-free manifest containing only stable IDs, categories, skills, output hashes, and aggregate evidence.
- **Result:** The frozen replay corpus contains 83 records: 40 arithmetic, 20 format, and 23 instruction. Replay-corpus SHA-256 is `b511129f89ce450014b78698e9e439bdaa0947657f301c3e99b2a9955b7ab4d1`; raw-packet file SHA-256 is `a9f25258d23f05a785dfea9f8ae0e05a246b52c9798a0d10e683fdc4e01a87f6`; replay-format SHA-256 is `758dc1f35020e88e04c425b6106e54ea2f577f547afa4762ade9923762af6d66`; manifest SHA-256 is `27ccd1c22bd321d17418ca346e1b3b4022fd696fdde07550e1c56f2864efde18`.
- **Gate status:** **PASSED.** All category/overall minimums, zero-backend, and zero-defect requirements pass. Targets are the actual deterministic untouched-base outputs; predefined gold responses are not replay targets.
- **Errors or uncertainty:** The format category has no headroom above its 20-item minimum, so any later replay-corpus integrity loss must fail closed rather than be repaired or replaced.
- **Next action:** Build and freeze the independent 450-item retention holdout, audit it against every prior retention suite, the 1,000 synthetic questions, and all 904 development questions, then evaluate the untouched base once if the artifact audit passes.

### 2026-07-21 — Milestone 9 independent replay final holdout frozen

- **Action performed:** Built exactly 450 original objectively scored prompts—150 arithmetic, 150 format, and 150 instruction—and froze their stable IDs, expected answers, scorer assignments, generation settings, and component hashes before any new adapter training. Audited the suite against seven prior retention suites, all four frozen synthetic splits, and the canonical 904-item development partition using exact normalized text and contiguous 12-token windows.
- **Result:** All 450 references self-score, all normalized prompts and IDs are unique, and there are zero ambiguous references. The audit covered 3,314 prior prompts and found zero exact overlaps and zero 12-token overlaps. Suite SHA-256 is `4f49c42cbae8ce7b5029192786f8ff493a4cc445f940063298e0bd22392b6ef9`; answer SHA-256 is `c53aff64ef1ac9e50bcfdd76be54bf33d57902ccacc779003ddff68188dff3e3`; scorer SHA-256 is `171f1a5abfbb4e0ece66732194134fadce9214d6a3ea4eeb45b59146fd9c1137`; content-free evidence SHA-256 is `0d8ea3b700649bad168446d9e830d434b1131220d7771a680cc9716825e460eb`.
- **Gate status:** **PASSED.** The content-bearing suite and local audit specification are ignored; the trackable artifact contains only identities, hashes, counts, and audit outcomes. No adapter output was read and no sealed-final content was opened.
- **Errors or uncertainty:** Two artifact-freeze launches failed closed before writing files: PowerShell removed quotes from repeated inline-JSON arguments, then the first Windows-safe inventory named the frozen synthetic field `question` rather than its actual `rendered_question`; a third launch correctly rejected a development-manifest/config hash mismatch. The final run used the manifest's actual bound smoke configuration and passed. These were interface/inventory corrections made before artifact creation; suite content and audit policy never changed.
- **Next action:** Evaluate the untouched pinned base on all 450 prompts exactly once, inspect all base failures for objective-reference/scorer defects without altering valid failures, apply the 60/60/60 and 250-overall usability gate, and freeze the base-correct IDs before any adapter training.

### 2026-07-21 — Milestone 9 independent holdout base gate failed

- **Action performed:** Evaluated the untouched pinned base exactly once on all 450 frozen holdout prompts with no adapter, then reconstructed the evaluation and applied the predeclared instrument-usability gate. Began an exhaustive read-only inspection of all base-failure rows; the suite, references, and scorers remain frozen.
- **Result:** The base scored arithmetic `84/150`, format `27/150`, and instruction `30/150`, for `141/450` overall. It produced 405 extractable responses, 115 exact-format responses, zero prompt echoes, zero question-generating outputs, 45 scorer-classified malformed outputs, and zero backend failures. Base-result summary SHA-256 is `6da14d0932255181d0ed9f59559b8f02d803dce72543bb4e743c0487724160b1`; gate summary SHA-256 is `e1bdc1cc14f2e126b8fb43f310b009b47bfef32d31795686259d49c8913d3f8a`.
- **Resources:** Model load took `2.712` seconds and generation took `193.598` seconds (`2.324` examples/second), consuming 30,248 input and 11,361 output tokens. Peak allocated/reserved GPU memory was 3,554,672,640/3,617,587,200 bytes.
- **Gate status:** **FAILED.** Arithmetic passes its minimum (`84 >= 60`), but format (`27 < 60`), instruction (`30 < 60`), and overall (`141 < 250`) fail. Backend and frozen-reference/scorer integrity checks pass.
- **Errors or uncertainty:** No backend error occurred. The failure is instrument usability for this base under exact-scoring prompts, not evidence about a trained adapter. A base-correct final-holdout subset was therefore not frozen.
- **Next action:** Per the stop rule, do not implement or run replay/KL training, do not train any of the six arms, do not perform retention selection, and do not access GSM1K. Complete the failure audit, documentation, repository verification, accurate stop commit, and push.

### 2026-07-21 — Milestone 9 exhaustive base-failure audit

- **Action performed:** Inspected all 309 failing base rows individually while keeping the 450 frozen prompts and references unchanged. Independently recomputed every prompt-to-reference result, reran every objective scorer, verified row/ID alignment and response hashes, and assigned content-free failure categories.
- **Result:** Prompt/reference recomputation errors, reference self-score failures, raw-score mismatches, ambiguous prompts, defective references, and defective scorers are all zero. The 309 failures comprise 66 arithmetic, 123 format, and 120 instruction items; 264 were extractable-but-wrong under their objective contract and 45 were malformed. Cause totals are 36 wrong arithmetic results, 15 omitted final subtractions, 15 correct values with noncanonical terminal formatting, 30 correct JSON objects wrapped in forbidden code fences, 90 punctuation/whitespace violations, 46 ordering errors, and 77 literal omissions/additions/mutations.
- **Gate status:** **AUDIT PASSED; INSTRUMENT-USABILITY GATE REMAINS FAILED.** Every base failure is genuine under an objectively valid frozen prompt/reference/scorer. Failure-ID SHA-256 is `f842092448be7f9cb28e65164ea8889a661acc4030eda70f9cc81878cdf187b1`; content-free classification SHA-256 is `72d6080c8e54ae61137196e5f5271e5fd8adde8de9dc30f99a2957133e89e386`.
- **Errors or uncertainty:** Natural-language exact-output prompts impose strict contracts, but those contracts were explicit and objectively scored; changing them after observing base failures is prohibited. Codex inspection is not human language review of the synthetic dataset.
- **Next action:** Finish repository-wide verification, create an accurate stopped-experiment commit, push it, confirm clean `0/0` synchronization, and wait for a new architectural decision.

### 2026-07-21 - Milestone 9 stopped-result verification passed

- **Action performed:** Reconstructed the two frozen 500-example datasets and all four splits, the pinned untouched-base snapshot, replay anchor, base-anchor evaluation, 83-record replay corpus, 450-item independent holdout, untouched-base holdout evaluation, usability gate, and exhaustive failure audit. Ran Ruff formatting and linting, strict Mypy, all unit and integration tests, both environment dependency checks, whitespace validation, hash/self-hash replay, exact and contiguous 12-token development-content scans, high-confidence secret scanning, sealed-path status checks without opening sealed-final content, ignored raw/model/cache/environment tracking checks, content-free result checks, tracked-file size review, and remote-divergence review.
- **Result:** Every published dataset, split, model, anchor, replay, holdout, evaluation, gate, and audit identity reconstructs exactly. Ruff is clean across 210 files; strict Mypy reports no issues in 125 source files; 504 unit tests and 7 integration tests pass; both environments have no broken requirements; and `git diff --check` passes. The holdout rescore remains arithmetic `84/150`, format `27/150`, instruction `30/150`, and overall `141/450`, with zero backend failures and zero objective prompt/reference/scorer defects. Across the 21 candidate paths, high-confidence secrets, exact development hits, contiguous 12-token development hits, sealed-path changes, tracked raw/model/adapter/cache/environment artifacts, forbidden content-bearing result fields, and files at or above 1 MiB are all zero.
- **Resources:** Anchor and final-holdout generation together took `224.126` seconds after model loading, consumed 38,204 input and 13,094 output tokens, and peaked at 3,554,672,640/3,617,587,200 allocated/reserved GPU bytes. Peak process RAM was not instrumented. New ignored raw replay/holdout evidence uses 552,243 bytes; the six content-free tracked result files use 34,635 bytes; the unchanged pinned model repository is 3,098,973,447 bytes.
- **Gate status:** **FINAL VERIFICATION PASSED; INSTRUMENT-USABILITY GATE FAILED.** The accurate stopped result is ready to commit. Replay/KL schedules, adapters, retention selection, independent adapter holdout inference, and GSM1K evaluation were not created or run.
- **Errors or uncertainty:** The only residual local implementation artifact is ignored Python bytecode for a discarded, never-run schedule prototype; it is neither source evidence nor tracked content. Expected Windows line-ending notices are non-fatal. No sealed-final content was opened or hashed.
- **Next action:** Stage only the 21 verified content-free/code/test/documentation paths, repeat index-only safety checks, commit with an accurate stopped-gate subject, push normally, confirm a clean synchronized branch, and wait for the user's next architectural decision.

### 2026-07-21 - Milestone 10 starting-state verification

- **Action performed:** Verified clean synchronized `main` at `a0438698e37bdb4f0e4216225eb0d9b8f6c2dc8b`; reconstructed both frozen synthetic datasets and all four splits, the 83-record replay corpus, the unused 450-item retention holdout and untouched-base result, both existing base-correct retention subsets, the pinned base snapshot, and the frozen GSM1K evaluator contract. Checked the `.venv-training` package/CUDA stack, ignored artifact boundaries, human-review page, and sealed-path Git status without opening or hashing sealed-final content.
- **Result:** Every expected hash and count matches. The holdout base result remains arithmetic/format/instruction `84/27/30`, total `141`, with every adapter field null. CPython 3.12.10, PyTorch 2.5.1+cu121, Transformers 4.51.3, tokenizers 0.21.4, PEFT 0.15.2, TRL 0.17.0, bitsandbytes 0.49.2, Accelerate 1.7.0, CUDA, and the RTX 3080 are available; both project environments pass `pip check`.
- **Gate status:** **PASSED.** Freezing the untouched-base-correct subset is authorized.
- **Errors or uncertainty:** None. No source, artifact, package, or Git configuration changed during verification.
- **Next action:** Use the existing strict base-conditioned subset freezer on the already-recorded untouched-base output, then independently reload and verify the expected `84/27/30 = 141` content-free manifest.

### 2026-07-21 - Milestone 10 final retention subset frozen

- **Action performed:** Ran the existing strict `freeze_base_correct_subset` path against the already-frozen 450-item suite, tracked untouched-base summary, and ignored untouched-base raw packet. Selected only rows whose frozen scorer recorded Boolean correctness, preserving exact suite order, then reloaded the manifest against the original suite.
- **Result:** The immutable subset contains 141 unique IDs: 84 arithmetic, 27 format, and 30 instruction. Subset SHA-256 is `f56845076a1a59e5ca1a95466541339b56f026e945f86118caec307a690ee4ec`; ordered-ID SHA-256 is `daa294be1d17e38d11fc06d5451ad387cf9cc0c718726aa30af9b3b430782879`. The manifest contains IDs, sections, skills, source identities, and counts only; `adapter_outputs_read=false`, `prompt_reference_or_scorer_modified=false`, and `prompts_or_references_in_manifest=false`.
- **Gate status:** **PASSED.** Prompt-only GRPO dataset, schedule, and reward-contract work may begin.
- **Errors or uncertainty:** A first read-only validation command attempted to treat the loader's tuple rows as objects and failed after successfully loading all 141 records; a corrected manifest/count check passed. No artifact changed as a result.
- **Next action:** Implement the prompt-only local artifact contracts, freeze identical replay placement and 52-arm synthetic schedules, and add deterministic verifier-reward tests before any GPU compatibility run.

### 2026-07-21 - Milestone 10 reward contract and paired schedules frozen

- **Action performed:** Implemented `foundry-verifier-grpo-reward-v1` with exact synthetic and replay scoring, additive safety penalties, strict trusted-metadata types, and original calibration fixtures. Built prompt-only generic and targeted schedules from the frozen synthetic training splits and 83-item replay corpus using the pinned tokenizer and chat template. Complete prompts and reward metadata were written only to ignored packets; tracked manifests contain IDs, counts, hashes, and aggregate token evidence only.
- **Result:** Reward implementation/configuration/fixture/calibration SHA-256 values are `089650105e29ead3c4ad62f1e0e41263e6c2af5fb8a12cb2851644aca3599616`, `4a47359fa3129b1bfd79dd158ecb609177e9b1642a95368c106e016a1554a965`, `8ba02448a87d5fe8c412f0e7a66acad5b45b6c6e9237dd366eecc060fbe67bdc`, and `fc420f3cfd1737592a0ef49c8c835baff25fa7382642b13c4802ab5d18c5722c`. Each arm has 64 groups, 52 synthetic and 12 identical replay groups at the same positions, with four completions per group. Generic and targeted prompt totals are exactly `6,702/6,702` tokenizer tokens. Generic/targeted manifest hashes are `5848ed6640dda21752ab9692c8e531d9175314a7d5a472616dc19ad834a6351e` and `cb13d4d522746bdfa829c9a405defdb0eff0acbd23859dc7fe49457318cc1ccf`; schedule-summary SHA-256 is `23fede9132f53b7d32f354056c728fc68faa20586a9162e101834db34f71ca64`.
- **Gate status:** **PASSED.** All 66 focused config, reward, schedule, reference, and trainer-hook tests pass under the main development environment; Ruff and strict Mypy are clean. The schedule is frozen before model generation and satisfies exact group/completion parity and the at-most-one-percent prompt-token rule.
- **Errors or uncertainty:** The first schedule-generation process lacked the repository `src` directory on its process-local `PYTHONPATH` and stopped before writing artifacts; the corrected process-local launch succeeded without package or environment changes. `.venv-training` intentionally lacks development-only Ruff, Mypy, and pytest, so those checks run from `.venv` while training dependency checks remain in `.venv-training`.
- **Next action:** Verify the installed TRL PEFT reference mechanism and exact truncation metadata hook, then run the bounded RTX 3080 compatibility smoke. No counted training is authorized until that gate passes.

### 2026-07-21 - Milestone 10 reference-policy contract verified

- **Action performed:** Audited the installed TRL 0.17.0 and PEFT 0.15.2 sources and implemented fail-closed source contracts for the official PEFT reference path, LoRA-only trainability, deterministic adapter-disabled no-gradient reference passes, exception-safe adapter restoration, exact checkpoint saves, and exact completion-truncation flags routed through the unmodified stock reward/loss flow.
- **Result:** TRL trainer, GRPO configuration, and PEFT model source SHA-256 values are `425161a6e4f82ee7cc6d4d6ad3fe7e495db970289d28427f45e99368ac5e985a`, `83d53640316958da75c4bb73451f9562f235f886c0cc31a3c825de172c0e17cc`, and `ea36efc37191855bb14fbb1ecd6743148aaa13350fed4ee9a8582c2b7fa29696`. The audited stock generation-and-score method hash is `688cb0ed965eee96bd9a985fdd185f63f984ee81eaab7bbfec2519f21e06331b`. The installed official path uses the same quantized base with the active PEFT adapter disabled for reference log probabilities and does not construct a second full reference model.
- **Gate status:** **STATIC VERIFICATION PASSED.** Sixteen focused reference/truncation-hook tests pass, including zero/nonzero KL behavior, RNG preservation, state restoration on exceptions, completion-order preservation, and saves only at steps 16, 32, and 64.
- **Errors or uncertainty:** Physical RTX 3080 memory usage, forward/backward compatibility, optimizer state, and adapter save/reload remain unmeasured until the bounded compatibility smoke. TRL's `per_device_train_batch_size=4` with four generations represents one unique prompt group per optimizer step; the smoke therefore uses two update groups plus one separate generation/reward/reference-only replay group to honor both the two-step and three-group requirements without changing frozen training settings.
- **Next action:** Complete the strict runtime orchestration tests, then run exactly the predeclared two-step/three-group compatibility probe with beta `0.04` and stop immediately on any mechanical or memory gate failure.

### 2026-07-21 - Milestone 10 GRPO compatibility gate failed

- **Action performed:** Completed the strict runtime orchestration and 85 focused tests, then launched the single predeclared RTX 3080 compatibility smoke using G1 beta `0.04`, the generic schedule's first two synthetic groups for optimizer updates, its first replay group for the separate generation/reward/reference probe, four generations per group, NF4 QLoRA, and all frozen generation and determinism settings.
- **Result:** The pinned model loaded on CUDA and the fresh PEFT adapter was constructed without a second reference model or CPU offload. During the first group's first sampled generation, Transformers' top-p logits processor called CUDA cumulative summation. PyTorch 2.5.1+cu121 raised `RuntimeError` because `cumsum_cuda_kernel` has no deterministic implementation while `torch.use_deterministic_algorithms(True)` is active. No completion finished, no reward or reference-KL callback ran, no backward pass or optimizer step occurred, and no adapter or completion record was saved. Content-free failure-summary SHA-256 is `8b57b6284c1e7dccd978379162de9519b7af30addbbfb9eb4d5a95a7f2b439a6`.
- **Gate status:** **FAILED; HARD STOP ENFORCED.** Counted G1/G2 training, retention checkpoint evaluation, independent final retention, and GSM1K evaluation are not authorized.
- **Errors or uncertainty:** The incompatibility is between the frozen stochastic top-p decoder and strict CUDA deterministic-algorithm enforcement on the approved PyTorch stack. `warn_only=True`, temporarily disabling deterministic enforcement around sampling, greedy decoding, CPU sampling, a package upgrade, or a generation-policy change could permit execution, but each changes an approved reproducibility or runtime decision and was not attempted. The process exited before its success-only resource summary was written, so peak process RAM and peak allocated/reserved VRAM are unavailable; the partial ignored output contains one empty `trainer_state` child directory and zero files.
- **Next action:** Document the exact mechanical blocker and negative scope, run complete repository and safety verification, commit as an accurate stopped GRPO result, push to `origin/main`, confirm clean `0/0` synchronization, and wait for an explicit decision. Do not run another configuration or compatibility probe automatically.

### 2026-07-21 - Milestone 10 stopped-result verification passed

- **Action performed:** Reconstructed the 141-ID final-retention subset, both 64-group prompt schedules with the pinned tokenizer, reward configuration and calibration, installed TRL/PEFT reference-source contracts, frozen G1/G2 configuration, and compatibility-failure evidence. Ran repository-wide Ruff formatting/linting, strict Mypy, all unit/integration tests, both dependency checks, whitespace validation, exact and 12-token development-content scans, high-confidence secret scanning, protected evaluator/synthesis/package status checks, sealed-path status checks without opening sealed content, raw/model/cache/environment tracking checks, tracked-manifest content checks, candidate-size review, ignored-artifact checks, complete diff review, independent read-only worktree audit, and live remote-divergence verification.
- **Result:** Ruff is clean across 224 files; strict Mypy reports no issues in 132 source files; 590 unit tests and 7 integration tests pass; both environments have no broken requirements; and `git diff --check` passes. The 86-test focused GRPO suite passes. All subset, configuration, execution, reward, source, schedule, packet, manifest, and failure hashes reconstruct exactly. Across 29 candidate files, exact and contiguous 12-token matches against all 904 approved development questions, high-confidence secrets, protected evaluator/synthesis changes, sealed-path changes, and tracked raw/model/cache/environment/adapter/checkpoint/prediction artifacts are all zero. No candidate is at or above 1 MiB; the only two repository files above that size are pre-existing content-free synthesis artifacts.
- **Gate status:** **FINAL VERIFICATION PASSED; GRPO COMPATIBILITY GATE REMAINS FAILED.** The stopped protocol implementation and negative evidence are ready to commit and push. G1/G2 counted training, retention inference, and GSM1K remain prohibited.
- **Errors or uncertainty:** Independent audit found and resolved three evidence-only issues before final verification: the first failure-summary self-hash was stale after its error-message hash changed; the partial ignored output contains one empty child directory rather than no directory; and one schedule docstring failed to mention hidden trusted reward metadata. A first consolidated Mypy/Pytest command used unexpanded PowerShell wildcards, and a first inline content-scan command had a quoting syntax error; corrected explicit-path and environment-carried scripts passed without changing experimental evidence. Peak compatibility RAM/VRAM is unavailable because the process failed before its success-only summary write. Expected Windows line-ending notices are non-fatal.
- **Next action:** Stage only the 29 verified content-free/code/test/documentation paths, repeat index-only scope and leak checks, commit with an accurate compatibility-stop subject, push normally, confirm clean synchronized `main`, and wait for the user's explicit runtime-contract decision.

### 2026-07-21 - Milestone 10E starting-state verification

- **Action performed:** Verified repository root, clean synchronized `main` at `a5f3af31bc05c25871934259c6d17e3ed47b704a`, live `origin/main`, the exact CPython/PyTorch/CUDA/Transformers/tokenizers/PEFT/TRL/bitsandbytes/Accelerate stack, RTX 3080 and driver, pinned offline Qwen snapshot, ignored GRPO schedules and partial failure directory, all frozen retention/configuration/reward/schedule/failure hashes, and the 86 focused GRPO tests. Checked protected-path status without opening sealed-final content.
- **Result:** Starting-state gate passed. The approved model snapshot exists at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; CUDA reports the RTX 3080 with 10,736,893,952 bytes total memory and driver 610.47. The prior partial GRPO output still contains only one empty `trainer_state` directory, with no counted adapter or checkpoint. Every approved embedded SHA-256 reconstructs exactly and all 86 focused tests pass.
- **Gate status:** **PASSED.** The narrowly scoped `foundry-warning-only-top-p-replay-v1` compatibility contract may be implemented. No generation has run in this milestone yet.
- **Errors or uncertainty:** `.venv-training` intentionally has no development-only Pytest installation, so focused tests ran from the unchanged main `.venv`; the pinned training environment itself was used for runtime/version checks. No dependency or Git configuration changed.
- **Next action:** Implement a fail-closed warning-only region around the actual stochastic generation call, preserve strict deterministic enforcement everywhere else, and add source-pinned state/warning/restoration tests before any model generation.

### 2026-07-21 - Milestone 10E warning-only top-p contract implemented

- **Action performed:** Implemented `foundry-warning-only-top-p-replay-v1` as a source-pinned wrapper around the actual `GenerationMixin.generate` call used by the frozen TRL path. The wrapper requires strict deterministic entry, temporarily changes only that call to `torch.use_deterministic_algorithms(True, warn_only=True)`, captures and normalizes all warnings, admits only the known CUDA cumulative-sum warning, then restores strict error-producing enforcement in `finally`. It also captures Python/Torch RNG transitions, base metadata, exact LoRA tensor state, active/enabled adapter state, LoRA scaling, and every module's train/eval state.
- **Result:** The pinned Transformers generation and top-p source hashes match `40c8e7d6adef288cd86f567b40e91ec8e95e1e916cc774e14fa4c91ad9f1105f` and `ac1f86dbf01f392e3fd068c434fa30b785d55ad2f8943209185319f081445698`. Ruff and strict Mypy pass for the compatibility, replay-evidence, replay-runtime, trainer, and runtime modules. Forty-nine focused compatibility/replay/trainer/runtime tests pass, including an RTX 3080 fixture that reproduces the strict `cumsum_cuda_kernel` exception and executes the same CUDA operation under warning-only enforcement.
- **Gate status:** **IMPLEMENTATION AND WARNING-WHITELIST TEST GATES PASSED.** No model generation or optimizer update has run yet.
- **Errors or uncertainty:** A concurrent draft initially imported `ContextManager` from `collections.abc`, which CPython 3.12 does not provide; it was corrected to `typing.ContextManager` before the focused suite. A NumPy RNG-state fixture initially exposed an unsupported ndarray to the generic state hasher; exact ndarray dtype/shape/bytes handling was added before runtime use. These were pre-generation implementation corrections and did not alter a measured model result.
- **Next action:** Freeze the content-free implementation, normalization, whitelist, fixture, and contract-test hashes, then run three exact generation-only replays in one process with a fresh base and adapter for each replay.

### 2026-07-21 - Milestone 10E preliminary replay excluded after contract audit

- **Action performed:** Loaded the pinned base snapshot and a fresh untrained G1 LoRA adapter three separate times inside one Python process. For each load, generated exactly two frozen synthetic groups and one frozen base-replay group with four stochastic completions per group, applying warning-only determinism only inside the stock `GenerationMixin.generate` call and strict deterministic enforcement for every other operation. Captured exact token IDs, decoded-output hashes, completion lengths, truncation flags, reward components, reference and policy log probabilities, per-token KL values, normalized warning hashes, Python/Torch/CUDA RNG transitions, and base/adapter state hashes.
- **Result:** All three 12-completion packets are byte-equivalent under the canonical packet contract and share SHA-256 `14ee42cb0bb96ba3df4f684612e7b618dab4def1018b96b359c1bda8590d2fa6`; the tracked summary SHA-256 is `7cd5707f15031ec65eaa9a7daf27400c2e2efb4fa3697020005ae4caed6eacc27`. Runtime per replay was `36.162`, `32.846`, and `33.011` seconds. Peak allocated VRAM was 4,599,902,720 bytes and peak reserved VRAM was 6,551,502,848 bytes; peak process RAM was 5,169,528,832 bytes.
- **Gate status:** **NOT USED AS OFFICIAL EVIDENCE.** The three packets matched, but an independent fail-closed review found that the draft runner did not yet bind the exact generic compatibility groups, prove three distinct child-process identities, require an externally effective `PYTHONHASHSEED`, reject multiple distinct allowed-looking warning messages, or include NumPy in each per-generation RNG transition. The obsolete tracked summary was removed; the ignored packets remain excluded from every gate.
- **Errors or uncertainty:** These were evidence-contract gaps, not an observed model-output mismatch. Transformers emitted its existing sliding-window/SDPA notice and Trainer emitted the existing empty-`label_names` notice; neither entered the captured generation warning stream. No optimizer step or counted training occurred.
- **Next action:** Harden the warning classifier and process/replay contract, rerun all focused tests, then create entirely new official same-process and fresh-process evidence from unused paths.

### 2026-07-21 - Milestone 10E stock cuBLAS transition diagnosed before official replay

- **Action performed:** Launched the hardened same-process replay from an unused ignored path after a child-process preflight proved `PYTHONHASHSEED=20260720` and prelaunch `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The first 12-completion replay completed with exact base-parameter evidence, then the second replay failed closed before model loading because the process environment no longer contained the prelaunch cuBLAS value. Inspected the pinned Transformers 4.51.3 source rather than bypassing the check.
- **Result:** `transformers.trainer_utils.enable_full_determinism`, called by the frozen `full_determinism=true` Trainer contract, deterministically changes `CUBLAS_WORKSPACE_CONFIG` to the other PyTorch-approved deterministic value `:16:8` at Trainer initialization. This stock transition explains why replay one passed the prelaunch check and replay two did not. The completed first-run packet used the exact frozen groups, contained 12 completion hashes, found only the canonical cumsum warning class (416 repeated warning instances), preserved exact base tensor hash `07efddeb5333d72687e5fbaa9b41b75921762c0395394a16c3839d84d27cb7c7`, and recorded two zero-variance groups; it remains excluded because the required three-run gate did not complete.
- **Gate status:** **FAILED CLOSED BEFORE OFFICIAL REPLAY EVIDENCE.** No optimizer step or counted training occurred. The failure is in the new environment-evidence model, not stochastic output identity or frozen Transformers behavior.
- **Errors or uncertainty:** Restoring `:4096:8`, suppressing the stock call, or accepting arbitrary values would alter or weaken the frozen runtime contract and was not done. Both `:4096:8` and `:16:8` are documented deterministic cuBLAS configurations, but only the exact source-pinned `:4096:8` prelaunch to `:16:8` Trainer transition is in scope.
- **Next action:** Bind and test that exact stock transition, require `:16:8` for every model operation, and rerun all three official same-process replays from a third unused path.

### 2026-07-21 - Milestone 10E replay contract stabilized

- **Action performed:** Bound the official replay and training runtimes to the exact source-pinned Transformers 4.51.3 full-determinism transition: the Python process must launch with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, then `transformers.trainer_utils.enable_full_determinism` (source SHA-256 `1893964197a05bfd07d1477815b58e42e883b9e64985f0e795b4562fc9f84834`) must activate `:16:8` before any model operation. Added stable process-start/active evidence, complete Python/NumPy/Torch CPU/CUDA RNG hashes, exact base tensor-byte evidence, fresh-process identity binding, and two-step caller integration.
- **Result:** The complete GRPO-focused slice passes `162` tests with `141` unrelated tests deselected; Ruff formatting/linting and strict Mypy pass for all compatibility/replay/runtime modules. A live no-model probe confirmed the exact `:4096:8` to `:16:8` stock transition. Warning-contract summary SHA-256 is `b1a040ac10cf754e820c8e237faaea65fb0aa55a560751247fd386ff1b5ee2fc`; its 19 contract tests and every implementation, fixture, whitelist, normalization, sampling, and pinned installed-source hash reconstruct exactly.
- **Gate status:** **PASSED.** The official same-process generation replay may run from a new unused ignored path. No counted training, backward pass, optimizer step, or adapter has been produced.
- **Errors or uncertainty:** Two preliminary official paths remain excluded: the first failed preflight before generation, and the second produced one generation-only packet before the then-incomplete environment contract rejected replay two. Neither contributes to the compatibility decision.
- **Next action:** Run three official generation-only replays in one process, then three more in distinct fresh processes. Stop verifier-GRPO immediately if either exact replay gate fails.

### 2026-07-21 - Milestone 10E official same-process replay failed exact gate; project stop

- **Action performed:** Ran the official frozen same-process generation replay three times from one unused ignored evidence directory. Each run loaded the pinned NF4 base and a fresh G1 LoRA initialization, reset the exact schedule and all RNG sources, and processed generic groups `g001`, `g002`, and `g005` with four completions each. No optimizer operation was requested. After the run, preserved the exact failure instead of excluding source identity from the gate.
- **Result:** All three replays completed: `3 x 12 = 36` completions. Their content-bearing diagnostic projections were equal after excluding source-contract drift: generated tokens and decoded hashes, lengths, truncation flags, rewards and components, zero-variance decisions, reference/policy log probabilities, KL, RNG transitions, base hash `07efddeb5333d72687e5fbaa9b41b75921762c0395394a16c3839d84d27cb7c7`, and LoRA tensor hash `1d1190d92abffcafeffda3f786aaa9d93fa4250e25c3ce4f0c2b4e135ca3b3da` matched. Each replay recorded `416` instances of only the approved normalized cumsum warning class; warning evidence SHA-256 is `0fc314eeaa033bbaf01b15e7763d925eb97a84a1f96842a5ba055019d225d8e7`, and no warning-only leak was detected.
- **Exact failure:** Packet SHA-256 values were `68ae4849b870d6d64232df83173f8fd560a28c91507383dd75abd2aa46c67d8c`, `80ad32513dc3a9f0253118f1a55d77eb30621abc4e19443e4013a114914794a7`, and `be3c8aa8f0f684ac3b8a740b3d00a0c6a3db7a979dcb473f5e3225f71177504e`. During the live replay, compatibility source changed from `da03f405c9ce4bc95509845f0e146b101b305cd2360c9682438f4a4fb21e6704` to `58358c3960c0a26f28caad2694fcd86f721c5b89490463976cabc46607f9a939`; replay-evidence source later changed from `4f5bd8ec9f996ca167e3697a9960515e98b06574e1cc445bd7a029918c6c9ce0` to `7cc32841c050d7a00702fdb598e4c43e0073d0fd0819395f46b344dfd6cd77e1`. Because source hashes are decision-bearing exact evidence, replay failed.
- **Artifacts:** The finalized compatibility slice has `20` contract cases and `165` focused GRPO tests. Final warning-contract summary SHA-256 is `eff84b9ec92715eeb74a6c74bcad5980dded9c4b5482012fd8e2438857f24598` (file SHA-256 `f473149d963b8a81bef69f4d13ce9f22ccfbf6b965a9b44fafad89eba84c90af`). Failure-summary SHA-256 is `8501b7681262ceca002659978c07c688a6f7baa45923ebb3c06e6134adabebe4` (file SHA-256 `ea9b7323e9565d2f2514c53849d53ebb503bb924d9b5e857923ef4676437e05b`).
- **Gate status:** **FAILED; VERIFIER-GRPO CLOSED.** The predeclared rule requires exact same-process replay. Diagnostic payload equality cannot override unequal source-bound packets. Fresh-process replay, first and duplicate two-step smokes, G1/G2 training, retention selection, independent retention, GSM1K, bootstrap, and signal gates were not run. Optimizer steps and saved adapters remain zero.

### 2026-07-21 - Milestone 10E stopped-result verification

- **Action performed:** Ran repository-wide Ruff formatting/linting, strict Mypy, all unit and integration tests, both environment dependency checks, whitespace validation, the 20-case warning contract, the 165-test focused GRPO slice, pinned Transformers source-hash reconstruction, both tracked artifact self-hashes, offline exact and contiguous 12-token scans against all 904 approved development questions, high-confidence secret scanning, protected evaluator and sealed-path status checks without opening sealed content, ignored raw/model/cache/environment tracking checks, candidate-size review, raw failure-packet inventory, full diff/status review, and live remote-divergence verification.
- **Result:** Ruff is clean across 232 files; strict Mypy reports no issues in 136 source files; 669 unit tests and 7 integration tests pass; the main environment has no broken requirements; and `git diff --check` passes. The warning-only contract's 20 tests and the complete 165-test GRPO slice pass. Across 20 candidate paths, exact-development hits, 12-token-development hits, high-confidence secret hits, protected evaluator changes, sealed-path changes, tracked raw/model/environment artifacts, and files at or above 1 MiB are all zero. Both tracked JSON self-hashes reconstruct. The ignored official failure evidence contains three packets totaling 654,912 bytes and no adapter, checkpoint, or model artifact. Local and live `origin/main` remain synchronized at the starting commit before the stop commit.
- **Gate status:** **VERIFICATION COMPLETE WITH ONE FROZEN-ENV METADATA WARNING; COMPATIBILITY GATE FAILED.** The failure implementation and content-free evidence are ready for the required accurate stop commit. No additional replay or training is authorized.
- **Errors or uncertainty:** The unchanged `.venv-training` fails `pip check` because installed `PyYAML 6.0.3` differs from the repository metadata pin `PyYAML==6.0.2`; the main environment passes. The approved environment inventory did not list PyYAML, and dependency installation or downgrade is expressly prohibited, so this pre-existing mismatch was recorded rather than changed. Peak RAM and VRAM for the official replay are unavailable because the exactness exception occurred before the success-only aggregate summary was written; command-wall runtime was 213.366 seconds.
- **Next action:** Stage only the 20 reviewed source, test, content-free result, and documentation paths; repeat index-only leak/scope checks; create `analysis: stop verifier GRPO after stochastic replay failure`; push normally; and confirm clean synchronized `main`. Then stop and wait for a new project-level decision.
- **Errors or uncertainty:** The scientific failure cause is concurrent shared-source drift during the official replay, not an observed model-output mismatch. The contract deliberately did not provide a post hoc exception for that cause. No second official attempt is authorized.
- **Next action:** Complete documentation, repository verification, the accurate stop commit, and push. Then stop the verifier-GRPO route for this project version; do not rerun, train, or evaluate GSM1K.

### 2026-07-21 - Milestone 10G runtime-root decoupling implemented

- **Action performed:** Inspected the complete GRPO runtime, replay, two-step, CLI, and focused-test call graph from clean synchronized `main` at `8f67e46262b7edafc57861aaf185efa345228179`. Added a frozen `GrpoRuntimePaths` contract that independently identifies detached source, primary repository, exact CPython executable, external writable artifacts, and the existing read-only model cache. Replaced source-relative interpreter lookup and repository-relative output checks with explicit canonical contract checks while preserving all generation, reward, reference, optimizer, schedule, retention, and evaluator behavior.
- **Path safety:** The contract rejects relative or alternate interpreters; source/primary/artifact/cache conflicts; traversal and symlink/junction escapes; non-source imports; source, root, executable, environment, command-template, or model-cache drift; non-ignored primary-repository caches; and artifacts outside the approved external root. Fresh child commands use the configured executable and receive the complete frozen environment with source-first `PYTHONPATH`, offline mode, `PYTHONNOUSERSITE=1`, and `PYTHONHASHSEED=20260720` before startup.
- **Result:** The unchanged focused suite passes exactly `165/165`; `17` new path-contract cases pass; the expanded focused suite passes `182/182`; and the full unit/integration run passes `693` tests. Ruff linting and strict Mypy pass for the changed implementation, both Python environments pass `pip check`, and whitespace validation passes. Approved CPython SHA-256 is `0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14`; planned V2 command-template SHA-256 is `6680c2c4d713882877d1c7e2ab1c47211ec07f2c84cee0464964e4de7b1d3498`. The content-free patch evidence self-hashes to `9d7e1c9994999ee568551e44fee9300a1711e54766de60e13332dd87c8ead3bb` (file SHA-256 `a3ee35db104c97beed57c5a3e8a8e7ddaa9b02f82019e0543b80a11eec786795`).
- **Diff scope:** Protected generator, reward, reference-policy, optimizer, schedule, retention, evaluator, synthesis, configuration, lock, and dependency paths have zero diff from the starting commit. The only implementation changes are the new path-contract module and the three GRPO orchestration runtimes, plus focused tests and required documentation.
- **Gate status:** **ORCHESTRATION IMPLEMENTED; NO SCIENTIFIC PROCESS STARTED.** Model generations, completions, reward calls, backward passes, optimizer steps, adapters, checkpoints, retention evaluations, GSM1K evaluations, and sealed-final access remain zero. The actual runtime-path-contract and source-manifest hashes are intentionally frozen outside Git after the new atomic commit because they bind its commit and full tracked tree.
- **Next action:** Complete the final repository and staged-index safety scans, commit exactly once as `fix: decouple GRPO runtime roots`, push and confirm synchronization, then create the untouched V2 worktree/runtime directories and freeze their external manifests before any model process.

### 2026-07-21 - Milestone 10G immutable replay failed closed

- **Action performed:** Pushed orchestration commit `b647a3dcadcab941359fbecab2b11c8f9f63cb8d`, created the detached V2 worktree without touching either historical non-V2 directory, and froze the external runtime-path and complete 491-file source manifests. Verified source-first import through the exact approved CPython, zero competing V2 Python/Codex command lines, clean source/primary worktrees, source-manifest stability, read-only model-cache identity, RTX 3080/CUDA evidence, and the exact process environment. Launched the official three-replay same-process command from unused external paths.
- **Frozen evidence:** V2 tree `099a9987df1b0a2d4da85eba33b4e22694ef2ab6`; runtime contract `2400654e155ba7be36aba99ffc4cf7588f80d726ffed59074f0f9955b948d953`; source manifest `72cd61b5f374f95bc7b0dbc1e51c0cafa81ca2cf3979d3d51421f2a1af4e2fab` (file `7d012f68...f4dd`); combined tracked source `79223089...d8d`; model manifest `5173393f...4006`; process environment `f888e1e7...644b`; command template `6680c2c4...3498`; exact replay command `beb8c90f...b2f7`.
- **Result:** The first frozen three-group replay completed its 12-completion in-memory generation/reward/reference/policy workload. During its `finally` block, the general path validator rejected `CUBLAS_WORKSPACE_CONFIG=:16:8`; it had frozen the required launch value `:4096:8` as a lifetime invariant even though stock Transformers' already audited full-determinism setup intentionally transitions to `:16:8` before model operations. Runtime was `79.741383` seconds. The exception occurred before `run_1.json` or the aggregate summary was written, so no prompt, completion, packet, warning/RNG record, or decoded output persisted.
- **Integrity after failure:** Source manifest, 491 tracked source files, model manifest, primary repository, and detached worktree remain unchanged. External evidence contains four files totaling `296,209` bytes and zero adapter, optimizer, scheduler, trainer-state, or checkpoint files. Peak RAM/VRAM, token, reward, warning, and KL aggregates are unavailable because the success-only packet/resource record was never persisted. Optimizer steps are zero.
- **Excluded preflight:** Before the official replay, two `--help` invocations accidentally lacked the frozen environment and failed immediately at module discovery with `ModuleNotFoundError`. They imported no Foundry module, loaded no model, wrote no artifact, and are excluded rather than treated as replay evidence.
- **Gate status:** **FAILED CLOSED; NO RETRY.** This is an orchestration validator defect, not an observed model-side replay mismatch. The authorization nevertheless forbids patching or rerunning after official replay failure. Fresh-process replay, both two-step smokes, G1/G2, retention, GSM1K, paired analysis, and the signal gate were not run. Failure-summary SHA-256 is `0a1c7085a95fef8138c06b17faaa8e0b5c0af195148012ca9a88c7a07a6d1eeb` (file `d38741f5e24c63279994b2cfd983cb2005c8a5e7d141a30d84dde96585163bb4`).
- **Next action:** Verify and publish the content-free immutable-replay failure as `analysis: stop verifier GRPO after immutable replay failure`, push normally, confirm synchronization, and stop the project route.

### 2026-07-21 - Milestone 10G failure-publication verification

- **Action performed:** Revalidated the content-free failure self-hash, external contract/source/model manifests, clean V2 source and primary worktrees, and zero adapter/checkpoint inventory. Ran repository-wide Ruff formatting/linting, strict Mypy, the full unit/integration suite, the unchanged and expanded GRPO-focused suites, main and exact-frozen-environment dependency checks, whitespace checks, exact and contiguous 12-token development-content scans against all 904 approved questions, high-confidence secret scans, protected scientific/source/dependency and sealed-path status checks, tracked raw/model/environment checks, candidate-size review, and diff/status review.
- **Result:** Ruff is clean across 234 files; strict Mypy reports no issues in 137 source files; all `693` unit/integration tests pass; the original focused slice passes `165/165`; and the expanded slice passes `182/182`. The main and exact V2 frozen environments both report no broken requirements, and `git diff --check` passes. Across eight candidate files, exact-development, 12-token-development, secret, protected-path, sealed-path, tracked raw/model/environment, and >=1 MiB candidate counts are all zero. Failure evidence reconstructs at summary SHA-256 `0a1c7085a95fef8138c06b17faaa8e0b5c0af195148012ca9a88c7a07a6d1eeb` and file SHA-256 `d38741f5e24c63279994b2cfd983cb2005c8a5e7d141a30d84dde96585163bb4`.
- **Errors or uncertainty:** A diagnostic `pip check` with the training interpreter pointed at the mutable primary `src` directory exposed its ignored editable metadata and reported installed PyYAML `6.0.3` versus repository pin `6.0.2`. The required exact V2 invocation has no such ignored metadata and passes. No dependency or environment was changed. Peak RAM/VRAM and per-replay token/reward/KL aggregates remain unavailable because the post-run validator raised before persistence.
- **Gate status:** **VERIFIED FAILURE READY TO PUBLISH.** No additional replay or scientific process is authorized.
- **Next action:** Stage exactly the eight reviewed content-free result/documentation files, repeat index-only checks, commit as `analysis: stop verifier GRPO after immutable replay failure`, push, confirm clean 0/0 synchronization, and stop.

### 2026-07-21 - Milestone 10H deterministic launch environment standardized

- **Authorization:** A new explicit project-level authorization opened one V3 experiment and left
  every V1/V2 source and runtime directory immutable. No historical evidence was reused or altered.
- **Action performed:** Inspected the installed Transformers 4.51.3 helper before patching and
  verified its five exact environment writes. Added a typed immutable process contract, replaced
  raw parent-environment copying with an explicit allowlist, standardized cuBLAS at `:16:8` before
  Python starts, and added fail-closed checks around import, deterministic initialization, CUDA,
  model loading, generation, backward, optimizer, cleanup, and exception exits.
- **Verification:** The helper file hashes to `33561736...a95e`, the function hashes to
  `18939641...4834`, deterministic initialization causes no effective environment change, the V3
  environment contract hashes to `1f80b141...38af`, all `198` focused GRPO tests pass, and all `709`
  repository tests pass. Ruff and strict Mypy pass. All 106 protected scientific/dependency files
  are unchanged from `ccbc88797441159f892bed28a336b625fc2ccab4`.
- **Gate status:** **ORCHESTRATION CORRECTION VERIFIED; NO MODEL PROCESS STARTED.** Generation,
  persisted completions, backward passes, optimizer steps, adapters, checkpoints, retention,
  GSM1K, paired analysis, and sealed-final access remain zero.
- **Next action:** Commit and push exactly as `fix: standardize GRPO deterministic environment`,
  confirm clean 0/0 synchronization, then create and freeze the new detached V3 experiment.

### 2026-07-21 - Milestone 10H V3 replay stopped before model loading

- **Action performed:** Published environment fix `2254b22aa10c9f024eebd56c1f1b98b9a3cf16ab`, created the clean detached V3 worktree, froze runtime contract `6154aecda902d6a4f9a9773a68f4da873d52e3474acb6cced10aee3a4291761a` and complete source manifest `f9f481186f3fb2e4e1c2c44b1d281069910f302a99738fd1a420930977e4c729`, then launched the official same-process replay with the explicit 30-field child-environment allowlist.
- **Result:** CUDA contract validation called `nvidia-smi` before model loading. The driver query succeeds under the parent environment with driver `610.47`, but the identical query under the exact allowlist exits `255` with `Failed to initialize NVML: Unknown Error`. The replay process stopped before any model load, generation, packet, reward, backward pass, optimizer step, adapter, or checkpoint.
- **Gate status:** **FAILED; STRICT STOP ENFORCED.** This is an orchestration/allowlist failure, not a model-side replay mismatch. Fresh-process replay, both two-step smokes, G1/G2, retention, GSM1K, paired analysis, and the signal gate were not run.
- **Evidence:** V3 source, model cache, environment hash, source-manifest file, and both repositories passed the post-failure audit unchanged. Content-free failure-summary SHA-256 is `b5f0e4b21b496b47a9ae5a93a42d9d9c39bb81b5e2fa7b4ddd36c7432464c2bf`.
- **Next action:** Preserve V3 and stop verifier-GRPO. No replay retry or further orchestration patch is authorized.

### 2026-07-21 - Milestone 10I direct CUDA-runtime preflight implemented

- **Authorization:** One new V4 experiment explicitly supersedes the V3 stop while preserving all V1-V3 source and runtime directories unchanged.
- **Action performed:** Split GPU evidence into a monitoring-only normal-parent `nvidia-smi` contract and an authoritative deterministic-child PyTorch CUDA contract. Removed every child driver query, added three-repeat fixed-tensor allocation/arithmetic/matmul/synchronization verification, prohibited NVML/pynvml in the child, and changed replay, smoke, and training resource measurement to public `torch.cuda` APIs.
- **Verification:** All `213` focused GRPO tests and all `724` repository tests pass under `PYTHONHASHSEED=20260720` and `CUBLAS_WORKSPACE_CONFIG=:16:8`. Ruff and strict Mypy pass. The host contract hashes to `18a87a86...1f68`, child contract to `ead57033...20ec`, and probe configuration to `1fc17a20...6775`; protected scientific and dependency paths changed: zero.
- **Gate status:** **ORCHESTRATION CORRECTION VERIFIED; V4 NOT YET CREATED.** No model load, generation, optimizer step, adapter, checkpoint, retention run, GSM1K run, or sealed-final access occurred. Patch-evidence SHA-256 is `712ab82ee6a97a0ec701c03282e10f0c6eb23ad9f6220d2056a475939981e8b5`.
- **Next action:** Publish exactly as `fix: validate GRPO GPU through CUDA runtime`, confirm clean 0/0 synchronization, then create and freeze only the new V4 source/runtime roots before collecting parent host evidence and launching the direct child CUDA gate.

### 2026-07-21 - Milestone 10I V4 compatibility stopped in first two-step smoke

- **Action performed:** Published GPU correction `a13c31b43a72c3bec205e440aaf7c424ac487d47`, created clean detached V4 tree `b938e97308d3e73493cf066ed7a656363657f4cb`, and froze runtime contract `a8543712...bc0a`, complete source manifest `dda8cf58...a8b8`, environment `0a5bd3bb...e55d`, and unchanged model manifest `5173393f...4006`. Parent `nvidia-smi` succeeded as monitoring-only evidence; the authoritative child CUDA probe passed with three identical `f8850fe4...e5af6` result hashes and no NVML/model load.
- **Replay result:** Three same-process and three distinct fresh-process generation replays matched exactly at common packet `084515f9...ee2f`; summaries hash to `319be850...043` and `de8ca110...dbb9`. The first complete two-step G1 smoke then failed during its first generation warning audit with `RuntimeError: generation emitted multiple distinct normalized warning classes`. The successful generation-only runs observed only the frozen CUDA-cumsum class; the training path also emitted gradient-checkpointing/use-cache and DynamicCache version messages.
- **Scientific stop:** The two-step process exited `1` at progress `0/2`, before backward, optimizer, packet, metadata, or adapter persistence. Optimizer steps and adapter/checkpoint files are zero. The duplicate smoke, counted G1/G2, retention, GSM1K, paired analysis, and signal gate were not run. This is a model-path compatibility failure, not an exact replay-packet mismatch, and the no-retry stop is enforced.
- **Integrity:** Post-failure validation re-proved clean source/primary worktrees, exact source/environment/interpreter/model identities, no CPU fallback, and no sealed-final access. A parent Codex shell telemetry timeout during the same-process run lost only stdout/stderr capture; the original child was not relaunched and its three packets and success summary validate exactly.
- **Evidence:** Content-free V4 failure-summary SHA-256 is `164d3e35828758d4eff77b21919b9b3b28dee6238135478fcd2b2e5e024c6f91` (file `7b6a05ff...3405`).
- **Next action:** Publish exactly as `analysis: stop verifier GRPO after V4 replay failure`, push, verify clean 0/0 synchronization, preserve V4, and stop verifier-GRPO.

### 2026-07-22 - Milestone 10J immutable training-warning audit failed closed

- **Action performed:** Revalidated primary commit `47bbe91e7bc0367be9abec68bfe113b4787c5d43`, detached V4 commit `a13c31b43a72c3bec205e440aaf7c424ac487d47`, tree `b938e973...f4cb`, stderr `e86b8e9d...3459`, source manifest `dda8cf58...a8b8`, and all six common packets `084515f9...ee2f`. Both V4 summary hashes reconstruct exactly. No model was loaded or generated.
- **Warning audit:** Four stderr warning classes are exactly recoverable, each with count one: `transformers-qwen2-sliding-window-sdpa-unimplemented-v1` (E), `transformers-peft-empty-label-names-informational-v1` (B), `transformers-qwen2-gradient-checkpoint-use-cache-transition-v1` (C, not accepted), and `transformers-dynamic-cache-torch-export-version-uncertainty-v1` (E). The approved CUDA-cumsum Class A remains proven only by the successful generation-only evidence at `416` occurrences per complete replay.
- **Fatal evidence gap:** The V4 warning auditor raised only after capturing at least two distinct Python warning classes, but its exception path did not serialize their messages, categories, sources, hashes, class IDs, or counts. The stderr cannot contain them because `warnings.catch_warnings(record=True)` retained them in memory. The exact additional class is therefore UNKNOWN under the authorization.
- **Gate status:** **FAILED; STRICT AUDIT STOP ENFORCED.** No source/dependency/scientific change, phase-warning contract, equivalence fixture, V5 worktree/runtime root, replay, optimizer step, adapter, checkpoint, retention run, GSM1K run, or sealed-final access occurred.
- **Evidence:** Content-free warning-audit SHA-256 is `a3e4d1ca40c3fb3f9fe984d3a019ed064a6ba96394a69b009257a248eebf1602`; classification SHA-256 is `37f564cf5a73e91a196496c00c31b0822c44a2b4c84e519b5628d5135479ad74`.
- **Verification:** All `724` repository tests pass under `PYTHONHASHSEED=20260720`; Ruff lint, Ruff formatting, and strict Mypy pass. The diff contains documentation plus one content-free audit artifact only.
- **Next action:** Publish exactly as `analysis: stop verifier GRPO after training-warning audit`, push, confirm clean `0/0` synchronization, preserve V1-V4, and stop.

### 2026-07-22 - Milestone 11 Phase 1 research package consolidated

- **Authorization:** Close Foundry Phase 1 from the frozen evidence at commit
  `20409ba41dc99bb1e6300b53d9ad9b3db1431722` without training, inference, generation, replay,
  dataset generation, GSM evaluation, sealed evaluation, or scientific-source changes.
- **Action performed:** Reconstructed every published aggregate and referenced evidence hash;
  consolidated the final report, reproducibility and architecture documentation, milestone index,
  Phase 2 directions, machine-readable summary, experiment timeline, consistency table, figure
  data, deterministic SVG renderer, five accessible figures, README, and release-specific tests.
  Checked both approved locations for the human-review export; neither exists, so genuine review
  remains explicitly pending.
- **Scientific result:** Base, generic SFT, and targeted SFT scored `521/814`, `387/814`, and
  `414/814`. Targeted beat generic by `27` questions (`3.316953` percentage points; paired 95%
  interval `[1.351351, 5.282555]`) while trailing base by `107`. Retention passed for both SFT arms.
  Contrastive scale transfer failed. Verifier-GRPO generation replay was exact, but the warning
  audit could not certify a training step; optimizer steps remain zero.
- **Verification:** The release-specific suite passes `23/23`; the full repository suite passes
  `747/747`; Ruff format and lint pass; strict Mypy reports no issues in `139` source files; both
  verified environments report no broken requirements; deterministic figure regeneration and
  whitespace checks pass. Development-content, secret, sealed-access, forbidden-artifact, hash,
  arithmetic, cross-file-consistency, and candidate-size audits are clean.
- **Gate status:** **PHASE 1 READY FOR ATOMIC PUBLICATION, PROVISIONAL PENDING GENUINE HUMAN
  REVIEW.** No sealed-final access occurred, and no production-training recommendation is made.
- **Next action:** Commit the reviewed closeout as
  `docs: publish Foundry phase 1 research results`, push `main`, verify clean `0/0`
  synchronization, create no tag, and stop.

### 2026-07-22 - Phase 2 Milestones 12A-12D Stage A repository gate passed

- **Authorization:** Open one gated Phase 2 experiment that tests Foundry's curriculum-selection
  system on vetted human-written math problems while preserving the frozen Phase 1 evaluator and
  evidence.
- **Repository verification:** Root is `C:/Users/Admin/Projects/Foundry`; branch is `main`; local
  and `origin/main` both equal `f4ee93afa4c2be52ca21aef8ca16dbf5827b4a99`; ahead/behind is
  `0/0`; worktree is clean. The Phase 1 report and summary exist.
- **Frozen identities:** Base result is `521/814` with `752/814` extractable. Model revision is
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; benchmark revision is
  `bc09569d09a614b9b530edc7f076fb214ac10493`. Evaluator config, development manifest,
  prompting source, and extractor source file hashes are `fa069dcc...d82d`,
  `09ef0acc...7d37`, `40584b1c...12a8`, and `aca21fc5...8fd`.
- **Boundary checks:** Sealed-final access/evaluation remain false. Prohibited raw-data, model,
  adapter, checkpoint, environment, cache, and binary-model tracked-path hits are zero. Existing
  ignore rules cover `data/` and `results/raw/` before any external download.
- **Gate status:** **PASSED.** No external corpus was downloaded and no model process was started
  in Stage A.
- **Next action:** Inspect the official ASDiv repository, pin one exact commit and tree, verify its
  license/citation/schema/count, then download only into the approved ignored path.

### 2026-07-22 - Phase 2 Stage B ASDiv source and provenance gate passed

- **Official source:** `https://github.com/chaochun/nlu-asdiv-dataset.git`; default branch
  `master`; detached commit `883f90a9a65bf00304ba8f37423910fe743abc47`; tree
  `2c3e8723c68436a2a6697329edfdf7fbd44e52ac`.
- **Verification:** The pinned README identifies ASDiv V1.0, `2,305` English math word problems,
  the expected `ID`/`Grade`/`Source` attributes and five required fields, the ACL 2020 citation,
  and CC BY-NC 4.0. Raw XML hash is `ef890406...c4929`; README raw-blob hash is
  `929249a7...5783`.
- **Storage:** The clean detached clone is under ignored
  `data/external/phase2_vetted/asdiv`. `data/` and `results/raw/` ignore checks passed before
  cloning. No raw corpus content entered a tracked path or progress message.
- **Line-ending audit:** ASDiv XML is byte-identical to its Git blob. Windows checkout conversion
  changes only README/fold text bytes; platform-neutral raw Git-blob identities are recorded, and
  folds are not experiment inputs.
- **Gate status:** **PASSED.** MathQA was not activated. No model process was started.
- **Next action:** Implement the safe exact ASDiv formula/answer verifier, deterministic family
  classifier, local content records, and exhaustive rejection accounting.

### 2026-07-22 - Phase 2 Stage C ASDiv exact-verification gate passed

- **Implementation:** Added a restricted recursive-descent formula parser using `Fraction`, bounded
  exact operations, independent answer extraction, deterministic unit compatibility, program
  hashing, replay checks, and a frozen non-LLM solution-type family map. Python `eval`, variables,
  arbitrary functions, and remote code are not used.
- **Measured result:** `2,305` source rows; `1,497` mathematically verified; `1,452` supported;
  `45` verified but unsupported; `808` rejected. Supported families are `1,126` bookkeeping,
  `118` rate/ratio/percentage/average, and `208` constraint/distribution/discrete.
- **Rejections:** `484` unknown formula grammar, `275` non-single equality, `36` unit
  incompatibility, `7` internal formula equality disagreement, `4` non-integer exponent, `1`
  unexpected token, and `1` non-single answer. No accepted formula/answer disagreement remains.
- **Determinism:** Two complete output directories reproduce summary `6c45b435...895d`, all-row
  `119546be...d7f2`, and supported-row `6478aa3e...c016`. Duplicate IDs and parser nondeterminism
  are zero. All complete records remain under ignored `results/raw/phase2_vetted_corpus`.
- **Verification:** Phase 2 unit tests pass `32/32`; strict Mypy passes for the new package. The
  source XML hash and count are asserted before parsing.
- **Gate status:** **PASSED.** The supported count exceeds `1,000`. No model process was started.
- **Next action:** Implement and run exact, 12-token, number-neutral, formula-structure, MiniLM
  semantic, source, Phase 1 synthetic, and candidate-duplicate screening.

### 2026-07-22 - Phase 2 Stages D-E contamination and capacity preflight passed

- **Inputs:** `1,452` supported ASDiv rows, the approved `904` development questions, all `1,000`
  Phase 1 synthetic questions, and pinned local MiniLM revision `1110a243...d41`.
- **Contamination result:** `73` development-semantic rejects at the fixed `>=0.75` threshold;
  `1,379` clean rows. Exact, 12-token, number-neutral, operation-structure, source-reference,
  candidate duplicate, and Phase 1 synthetic match counts are zero. Unresolved semantic candidates
  are zero, and the fixed 30-example semantic replay is exact.
- **Determinism:** Two full screens reproduce summary `0bf877c4...bdc5`, evidence
  `99cb38aa...a631`, and clean rows `8d99a1de...eaac`.
- **Capacity:** Clean family counts are `1,076` bookkeeping, `111` rate/ratio, and `192` discrete.
  ASDiv-only combined rate deficits are `59`, `30`, and `3` at per-arm sizes `300`, `250`, and
  `200`. Other census dimensions cover grades 1-6, operation/depth buckets, integer and terminating
  decimal answers, magnitude buckets, and token-length buckets. Summary is `16260814...ba00`.
- **Gate status:** **PASSED TO REQUIRED BASE EVALUATION.** ASDiv alone is not structurally eligible,
  but MathQA cannot activate until actual base-failure counts prove the 200-per-arm limit. No Qwen
  model process has run; MiniLM contamination inference ran CPU-only.
- **Next action:** Freeze the base-pool inference configuration and evaluate all `1,379` clean ASDiv
  candidates with greedy Qwen decoding, reporting 25/50/75/100% progress.

### 2026-07-22 - Phase 2 Stages F-G base-pool evaluation and MathQA fallback passed

- **ASDiv base result:** processed `1,379/1,379`; zero backend failures; exact fixed 30-row replay;
  `1,167` correct (`84.6265%`); `1,253` extractable (`90.8629%`). Base-failed families are
  `152` bookkeeping, `22` rate/ratio, and `38` discrete. Aggregate hash is
  `3eb702d1...7bd8`; prediction-content hash is `478740b2...434e`.
- **Fallback activation:** ASDiv cannot support the `200`-per-arm quotas from actual base failures,
  so the explicitly authorized MathQA fallback activated. Official train Parquet revision is
  `fafb9f7e...f2d6`; train artifact SHA-256 is `c16335ea...4a99`; validation/test access and
  rationale loading are false.
- **MathQA verification and contamination:** `15,468/29,837` rows passed exact program/option
  agreement; the frozen pre-inference selector chose `5,000`; `71` contamination candidates were
  rejected and `4,929` remained. Selected-row hash is `02fb19a8...3d45`; clean-row hash is
  `93d4d250...f418`; unresolved and cross-source matches are zero.
- **MathQA base result:** processed `4,929/4,929`; zero backend failures; exact 30-row replay;
  `2,363` correct (`47.9408%`); `3,787` extractable (`76.8310%`). Base-failed families are
  `1,214/1,136/216`. Aggregate hash is `5659a547...a80b` and prediction-content hash is
  `8a1d0967...e750`.
- **Operational note:** A sequential timing probe was stopped after `20` durable rows before the
  official run. Batch-20 equivalence then failed on `3/20`, so batching was rejected and removed.
  The official one-at-a-time run reproduced all 20 probe outputs apart from timing and completed
  in `49,233.8` seconds under the frozen environment.
- **Gate status:** **PASSED.** Continue only to the predeclared matched-size gate.

### 2026-07-22 - Phase 2 Stage H matching gate failed; experiment stopped

- **Eligible pool:** `2,778` combined base failures; stable question/program deduplication removed
  `62` duplicate latent programs and retained `2,716` unique eligible rows.
- **Method:** targeted stable coverage used formula structure, solution type/category, operation
  count, answer type, and available grade/difficulty. Generic selection enforced the frozen
  balanced family quotas, disjoint role partition, exact source composition, question/program
  uniqueness, and no number-neutral five-token-shingle near duplicate. Final pair assignment used
  an exact deterministic Hungarian solution over the frozen selected sets.
- **Size 300:** failed categorical balance because the `10_to_99` magnitude level differed by
  `0.06`; formula-depth SMD was `0.140411`.
- **Size 250:** failed categorical balance because the `10_to_99` magnitude level differed by
  `0.056`; formula-depth SMD was `0.137710`.
- **Size 200:** categorical gate and exact source composition passed. Numerical SMD failed only
  for formula depth (`0.113895`) and operation count (`0.108765`); the maximum is `0.10`.
- **Gate status:** **FAILED; STOPPED.** Stop-result SHA-256 is `1b169ab5...650f`. No targets,
  splits, schedules, training, retention, GSM1K adapter evaluation, paired analysis, or signal
  decision occurred. Optimizer steps, adapters, and checkpoints are zero. Sealed-final access is
  false. No Stage K publication commit or push was created.

### 2026-07-23 - Milestone 12E interrupted-worktree recovery and stop verification

- **Repository audit:** recovered only `C:\Users\Admin\Projects\Foundry`, on `main` at
  `f4ee93afa4c2be52ca21aef8ca16dbf5827b4a99`, equal to `origin/main` at `0/0`.
  The intentional Phase 2 implementation and documentation were present, but
  `src/foundry/phase2/matching_repair.py` was absent. Three stale system-Python processes running
  `_tmp12e_matching_repair.py` were stopped before verification.
- **Scratch preservation:** twelve untracked `_tmp12e*` files were hashed and moved without
  modification to the external runtime-artifact archive
  `C:\Users\Admin\Projects\Foundry-grpo-runtime\milestone12e-recovery-scratch`. Their SHA-256
  values, in filename order, are `cc5e59a7...3000`, `101fdbdf...e700`, `c8b823e2...3111`,
  `b477f9af...e9d4`, `27ce89a2...cd2d`, `d2829ab8...fd9e`, `c7a88776...8504`,
  `2a959d1c...163b`, `85f348d7...8b6d`, `1f4949a9...9482`, `d4ec744a...80ab`, and
  `a1870761...bfd0`. The non-authoritative lead was generic removal
  `mathqa-train-26455` and addition `mathqa-train-28853`; it remains unaccepted pending an
  independent canonical repair search.
- **Frozen-stop reconstruction:** canonical project code reproduced `2,778` base failures,
  `2,716` deduplicated eligible candidates, exact 200-row arms, required family quotas,
  `97` ASDiv plus `103` MathQA rows per arm, and zero cross-arm source-ID, exact-question,
  normalized-question, latent-program, or near-duplicate overlap. SMDs reproduced exactly as
  question tokens `0.0`, base output tokens `0.022589325494615863`, formula depth
  `0.11389459246177541`, and operation count `0.10876528809635315`.
- **Evidence verification:** ASDiv and MathQA source revisions, licenses, mathematical
  verification, contamination summaries, formula/program and semantic replays, untouched-base
  prediction hashes, both 30-example base replays, and attempted-assignment hashes passed.
  No source parsing, contamination inference, model inference, target construction, training,
  retention, GSM1K adapter evaluation, or sealed-final access occurred.
- **Repository verification:** Ruff format and lint passed; strict Mypy passed for 146 source
  files; all 77 Phase 2 tests, 740 other unit tests, and 7 integration tests passed. All 824 tests
  collect in one invocation after packaging the Phase 2 test directory. `pip check`,
  `git diff --check`, raw-data containment, high-confidence secret scanning, the exact and
  12-token scan against all 904 development questions, and the tracked-size review passed.
- **Environment note:** the required project interpreter is CPython `3.12.10` in `.venv`; its
  PyYAML remains `6.0.2`, and `pip check` reports no broken requirements. The interrupted
  non-authoritative `pip install pyyaml` did not alter this verified project environment.
- **Gate status:** the original Stage H result remains an accurate stopped experiment. The next
  authorized action, after publishing this evidence, is to freeze repair inputs and independently
  evaluate deterministic legal replacements without changing any scientific gate.

### 2026-07-23 - Milestone 12E matching repair and dataset freeze passed

- **Repair input:** `209` ASDiv plus `2,507` MathQA eligible failures; canonical normalized-question
  covariates and all other frozen fields bind to input hash `0e6332e2...5979`.
- **Search:** exhaustive deterministic single-row search checked `155,301`, found `152,226` legal
  and `1,979` passing replacements, and selected generic `mathqa-train-26455` to
  `mathqa-train-28853`. Two-row and global fallback stages did not run.
- **Matching result:** SMDs are question `0.0028892934`, base output `0.0075164765`, formula depth
  `0.0870898715`, and operations `0.0680561998`; categorical maximum is `0.05`. Source composition
  remains `97/103` per arm and all quota, duplicate, near-duplicate, and contamination gates pass.
  Matching evidence is `004d338b...d5b5`; full replay is byte-identical.
- **Targets and splits:** all 400 formula/program-derived targets replay exactly, contain one
  calculation line plus one terminal answer and EOS, and stay at or below 58 assistant tokens.
  Each arm split deterministically into `180` training and `20` validation rows with zero exact,
  normalized, program, or cross-arm overlap. Dataset identity `ee18f7f9...dc31` replays
  byte-identically.
- **Verification:** Ruff, strict Mypy, and all 83 Phase 2 tests pass, including changed-input,
  changed-threshold, scratch-independence, target replay, split, and byte-replay cases. Raw
  questions, completions, predictions, and source data remain ignored.
- **Next action:** publish `data: freeze vetted human-written curriculum pools`, then run V1
  REPLAY25 generic and targeted training under the frozen 64-step protocol.

### 2026-07-23 - Milestone 12E stopped before V1 on the frozen environment boundary

- **Published dataset:** commit `040129322c31da3464add4d108f0f771d0bdda1f` is synchronized
  `0/0` with `origin/main`; the worktree was clean before the training preflight.
- **Required interpreter:** `C:\Users\Admin\Projects\Foundry\.venv\Scripts\python.exe`, CPython
  `3.12.10`. PyTorch `2.5.1+cu121` sees the RTX 3080 with `10,736,893,952` bytes of VRAM;
  Transformers is `4.46.3`; `pip check` passes.
- **Blocking discrepancy:** PEFT, bitsandbytes, and TRL are absent from the required environment.
  The recovery authorization prohibits installing or modifying packages and requires this
  interpreter, so no alternate environment was used.
- **Scientific activity:** zero model loads, optimizer steps, schedules executed, adapters,
  checkpoints, retention evaluations, GSM1K adapter evaluations, or sealed-final access.
- **Decision:** stop before V1. Matching and dataset gates remain passed; training, retention, and
  signal gates are not reached. Resume only with explicit authorization for a pinned compatible
  training environment.

### 2026-07-23 - Milestone 12F-A stopped at the training-environment gate

- **Action performed:** Verified clean synchronized `main` at
  `a97972a162453a3c22b68b59c598048502d2b284`, reconstructed the frozen matching and dataset
  evidence with six focused tests, confirmed zero Phase 2 adapters/checkpoints and no active model
  process, and audited the authorized `.venv-training` interpreter, packages, CUDA runtime, GPU,
  process environment, and local model cache.
- **Result:** CPython 3.12.10, torch 2.5.1+cu121, CUDA 12.1, Transformers 4.51.3,
  tokenizers 0.21.4, PEFT 0.15.2, TRL 0.17.0, bitsandbytes 0.49.2, Accelerate 1.7.0, and the
  NVIDIA GeForce RTX 3080 all match. The interpreter SHA-256 is
  `0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14`; the 57-package
  inventory SHA-256 is `2d4dbf699b73b53206d96687f1381ec22dac8a2d1575b0a43791627b9b43b2c8`.
- **Stop gate:** With the required repository `src` root visible, `.venv-training` `pip check`
  reports that `foundry-post-training 0.1.0` requires `PyYAML==6.0.2`, but the environment has
  `PyYAML 6.0.3`. The authorization prohibits dependency repair, so schedule construction, smoke,
  training, retention, and GSM1K did not run.
- **Errors or uncertainty:** One provisional offline model load exposed that the official Qwen
  template places a post-EOS newline that the probe must mask. No generation, backward pass,
  optimizer step, adapter save, retention inference, or benchmark access occurred. The corrected
  probe was not run because `pip check` failed first.
- **Next action:** Obtain explicit authorization either to reconcile the training environment with
  the frozen Foundry dependency metadata or to define a scientifically justified exception. Then
  restart Milestone 12F-A from Stage A.

### 2026-07-23 - Milestone 12F-A1 PyYAML exception passed; native CUDA probe stopped

- **Action performed:** Reproduced the one-line `.venv-training` `pip check` discrepancy with exit
  code 1 and verified that `.venv` passes normally. Froze both PyYAML installations, hashed their
  Python/compiled package files and metadata, and independently audited all 31 tracked YAML files
  twice under PyYAML 6.0.2 and 6.0.3. Compared canonical `safe_load` results and the real
  assistant-only, QLoRA, and token-matched loader projections.
- **PyYAML result:** Every source and parsed hash, every typed-loader outcome, and both repeat audits
  match. The narrow exception passed with evidence SHA-256
  `dd9413331fc76c41f1e30b2bb4697abeabde0a97452196d225619329fcee810a`.
- **Environment result:** The offline NF4 model loaded without CPU offload and paged AdamW 8-bit was
  constructed. The first forward stopped before producing a loss because deterministic CuBLAS
  requires `CUBLAS_WORKSPACE_CONFIG` to be set before Python starts. The launch environment omitted
  that variable.
- **Stop accounting:** One model load occurred. No generation, completed forward loss, backward
  pass, optimizer state/update, schedule, adapter/checkpoint save, retention evaluation, GSM1K
  evaluation, or sealed-final access occurred. Neither environment nor dependency metadata changed.
- **Next action:** Interpret the launch-environment blocker and, if authorized, restart the native
  CUDA probe from a fresh process with an explicitly frozen prelaunch
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`; otherwise preserve this stop.
