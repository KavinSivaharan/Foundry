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
