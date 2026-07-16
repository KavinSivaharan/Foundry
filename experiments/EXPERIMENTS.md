# Foundry Experiment Log

No model experiment has run yet. Planning, repository inspection, and literature/model-card research are not recorded as benchmark experiments and do not have fabricated scores.

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
