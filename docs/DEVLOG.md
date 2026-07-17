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
