# Phase 2 research directions

This document ranks research options from the Phase 1 evidence. It is a decision aid, not an
authorization to run any experiment. Every option requires a separately scoped protocol, fresh
budgets, predeclared gates, and a new commit boundary.

## Ranking criteria

Options rank higher when they address an observed Phase 1 failure directly, can be falsified with a
bounded experiment, preserve the benchmark firewall, and avoid treating the positive
targeted-versus-generic direction as a base-model improvement.

## 1. Reimplement verifier-GRPO on a Linux-native pinned stack

**Hypothesis.** A Linux-native CUDA stack, independently pinned and audited before scientific model
execution, can support the frozen sampling, checkpointing, reference-policy, and warning-evidence
requirements through at least a bounded backward/optimizer smoke.

**Supporting evidence.** Phase 1 already froze matched prompt schedules, executable rewards, a
single-base reference-policy design, and exact same-machine generation replay. The blocker was not a
reward or schedule mismatch. It was an interaction among deterministic CUDA sampling, gradient
checkpointing, attention/cache behavior, and incomplete warning capture on the Windows stack.

**Risks.** Linux may expose different kernels or warnings rather than remove the problem. Recreating
the environment can silently change tokenizer, generation, TRL, PEFT, or PyTorch behavior. A
successful two-step smoke would establish compatibility only, not model benefit.

**Compute requirement.** One Linux CUDA host with at least the effective memory available on the RTX
3080; environment construction and source audit; no more than the predeclared generation replay and
two-step smoke before a new authorization. A larger GPU is optional but must not be used to change the
scientific configuration implicitly.

**Falsifier.** The option is rejected if exact replay fails, warning capture is incomplete, reference
state does not restore, gradients are non-finite, memory exceeds the bound, or any unclassified
training-path warning appears before the first certified optimizer step.

**Must remain frozen.** Base/model revision, tokenizer, prompt schedules, replay positions, reward
implementation/configuration, G1/G2 betas, sampling parameters, group counts, seed, reference-policy
semantics, retention gates, benchmark firewall, and zero access to sealed-final content.

## 2. Train with replay/KL retention in the objective

**Hypothesis.** Policy optimization with shared base-replay examples and an explicit KL or retention
term can preserve demonstrated base behavior better than post-hoc admission alone while retaining the
targeted curriculum direction.

**Supporting evidence.** Targeted beat generic, but both common-scaled SFT adapters remained far below
the base. The unscaled and contrastive adapters showed shared drift, suggesting that after-the-fact
retention filtering does not prevent the damaging update. Phase 1 froze 83 scorer-correct base replay
behaviors and verified a reference-policy design.

**Risks.** Replay can overfit a small behavior set, KL can suppress useful learning, and the existing
absolute holdout instruments were not usable for the pinned base. Training before replacing that
instrument would repeat the Milestone 9 error. Reward/KL scale selection could introduce substantial
researcher degrees of freedom.

**Compute requirement.** First, a no-training study to build and validate a new retention instrument
on the untouched base. Then a bounded two-arm smoke with identical replay ratios and a small,
predeclared beta grid. Only a passing smoke should unlock full scheduled runs.

**Falsifier.** Reject the approach if the untouched base cannot pass the new instrument, if replay/KL
does not improve retention relative to an otherwise identical no-replay control, or if targeted no
longer exceeds generic under the same constraint.

**Must remain frozen.** Development evaluator, 83 replay identities and scorers, targeted/generic
datasets, source-corpus sizes, per-arm token/group budgets, base revision, seed policy, comparison
order, retention-instrument freeze order, and sealed-final isolation.

## 3. Constrain or orthogonalize adaptation updates

**Hypothesis.** Limiting update components aligned with generic/shared drift can preserve instruction
behavior while retaining targeted-specific signal.

Candidate methods include projected gradients, low-rank subspace constraints, orthogonalization
against a base-retention gradient basis, or norm-bounded task-vector updates. These are alternatives,
not a menu to tune post hoc.

**Supporting evidence.** Generic and targeted updates had high cosine similarity (0.9399), while the
targeted-minus-generic component was only 34.61% of the targeted update norm. Exact subtraction was
numerically valid but failed retention at every tested scale. This suggests that useful differential
signal exists but cannot simply be added as an unconstrained task vector.

**Risks.** Projection objectives may remove the same features needed for arithmetic improvement.
Subspace estimates can be unstable and expensive. A retention gradient basis could itself overfit the
instrument. Repeated projection variants would create a large search space.

**Compute requirement.** A small offline tensor-analysis phase followed by one bounded adapter smoke
per predeclared method. Full training is justified only if an independently frozen retention protocol
passes for both arms.

**Falsifier.** Reject a method if its update equivalence/state-restoration checks fail, if retention is
no better than the scale-0.50 reference pair at matched effective norm, or if its targeted-minus-generic
advantage disappears.

**Must remain frozen.** Source adapters and their hashes, mathematical update definition, norm budget,
layer/module inventory, retention suites and thresholds, evaluation prompt/parser, data arms, and
bootstrap method.

## 4. Test a larger or more instruction-stable base model

**Hypothesis.** A base with greater instruction stability or capacity may absorb a targeted arithmetic
update without the severe role/format drift observed in the 1.5B model.

**Supporting evidence.** The 1.5B base was sensitive to full-sequence and assistant-only SFT, and even
short retention-safe updates required scale reduction. Phase 1's data and evaluator machinery can
support a controlled transfer study once the new model has its own trusted baseline.

**Risks.** A larger model changes baseline accuracy, failure taxonomy, target distribution, memory
requirements, and possibly the relevance of the existing synthetic curriculum. Reusing the old
taxonomy without re-audit would be invalid. Higher baseline accuracy may reduce measurable failure
counts while increasing evaluation cost.

**Compute requirement.** Hardware sized for quantized training and full 814-example evaluation of the
candidate. Budget must include a fresh evaluator compatibility smoke, complete baseline audit, new
failure taxonomy, matched data validation, training, and retention—not only the training run.

**Falsifier.** Reject the candidate if the evaluator cannot be trusted, if the base fails the
retention instrument, if matched short-run SFT shows the same collapse pattern, or if targeted does
not outperform generic under the new model's frozen comparison.

**Must remain frozen.** The model revision selected before any output, benchmark and development IDs,
evaluator acceptance rules, data-generation/verifier contracts, arm-matching policy, seed count,
retention freeze order, and sealed-final firewall. The old base remains the Phase 1 reference and must
not be retroactively replaced.

## 5. Extend beyond arithmetic only after a base-improving result

**Hypothesis.** The Foundry workflow may generalize to another objectively verifiable domain after it
demonstrates a retention-safe improvement over an untouched base in arithmetic.

**Supporting evidence.** Phase 1 successfully implemented evaluation, failure classification,
verified generation, matched controls, and content-free evidence handling. It did not demonstrate an
absolute model improvement. Expanding now would multiply unresolved adaptation and measurement risks.

**Risks.** New domains may lack exact symbolic verifiers, invite learned-judge bias, increase
contamination risk, and obscure whether a failure is in data, evaluation, or training. Premature scope
expansion could turn a clear negative result into an unbounded search.

**Compute requirement.** None until the prerequisite arithmetic result exists. Afterward, begin with a
small domain-selection and evaluator-calibration study; do not assume the arithmetic model, data, or
retention thresholds transfer.

**Falsifier.** The option remains deferred if no arithmetic arm beats its untouched base under a
second seed, genuine human review, retention admission, and a separately authorized confirmatory
evaluation. A candidate domain is rejected if it cannot support objective scoring and contamination
controls comparable to Phase 1.

**Must remain frozen.** The prerequisite claim criteria, arithmetic evidence, benchmark firewall,
objective-verifier requirement, matched-control principle, content-free publication boundary, and
sealed-evaluation policy.

## Recommended decision sequence

1. Decide whether a separately funded Linux-native compatibility phase is warranted.
2. Before any new optimization, validate a retention instrument that the untouched base can pass.
3. Choose exactly one retention-preserving update hypothesis and freeze its falsifier.
4. Require at least two seeds and genuine human review before any sealed evaluation request.
5. Expand models or domains only after the arithmetic path yields an absolute base improvement.

Phase 1 does not authorize any of these steps. Its next project-level decision is whether to open a
new Phase 2 with new scientific and compute authorization.
