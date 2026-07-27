# Phase 2 vetted-corpus result

## Current status

Fast-Track Phase 2 Milestones 12A-12D originally stopped at Stage H. Milestone 12E preserved that
result, then performed the separately authorized deterministic matching-only repair. One generic
row replacement passed every unchanged gate, and deterministic formula-derived targets plus exact
180/20 splits were frozen. A later pinned training-environment authorization completed V1 REPLAY25
and V2 REPLAY40, but every checkpoint initially failed the anchor retention gate on the same
question-generation decision.

Milestone 13A proved that repeated decision was a general deterministic scorer defect rather than
adapter-generated questioning. Corrected offline replay selected existing V1 step 64. Generic and
targeted then ran exactly once on the untouched independent final subset and both failed the
unchanged retention gate: each preserved `75/84` arithmetic items and had a maximum failure-family
concentration of five. No adapter pair is retention-approved, no adapter was evaluated on GSM1K,
and sealed-final content was not accessed.

## Frozen starting point

- Phase 1 release: `f4ee93afa4c2be52ca21aef8ca16dbf5827b4a99`
- Untouched base: `Qwen/Qwen2.5-1.5B-Instruct`
- Model revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Frozen GSM1K development result: `521/814` correct; `752/814` extractable
- Benchmark revision: `bc09569d09a614b9b530edc7f076fb214ac10493`
- Sealed-final access: false

## Primary source freeze

- Corpus: ASDiv V1.0
- Official repository: `https://github.com/chaochun/nlu-asdiv-dataset.git`
- Commit: `883f90a9a65bf00304ba8f37423910fe743abc47`
- Tree: `2c3e8723c68436a2a6697329edfdf7fbd44e52ac`
- Raw XML SHA-256: `ef8904068482919ac48c8eeaaf6df344b8a308ba66d048c2d4d87eab82dc4929`
- Official source count: `2,305`
- License: CC BY-NC 4.0; non-commercial research use
- Fallback: activated; official MathQA train Parquet revision
  `fafb9f7ee5b9ec4da9499f9c4177a4c91389f2d6`

## Research question

Can Foundry select a more useful training curriculum from a vetted human-written problem pool than
a matched generic selector while preserving the untouched base model's existing capabilities?

The human-written wording will not be rewritten. Both arms will use verified labels, disjoint
base-failed examples, matched non-curriculum covariates, identical training architecture, identical
token budgets, shared replay, the same seed, and one retention-only checkpoint rule.

## Gate ledger

| Stage | Status | Evidence |
|---|---|---|
| A: Phase 1 repository | Passed | Clean synchronized release; frozen hashes and sealed boundary verified |
| B: ASDiv source and license | Passed | Official commit/tree, README, CC BY-NC 4.0, schema, and 2,305 count verified |
| C: Formula verification | Passed | 1,452 supported verified; zero accepted disagreement, duplicate ID, or nondeterminism |
| D: Contamination | Passed | 1,379 clean; 73 semantic rejects; zero unresolved or lexical/duplicate overlaps |
| E: Capacity | Passed preflight | ASDiv-only has no eligible size; 200-per-arm rate quota is short by 3 |
| F: ASDiv base-pool evaluation | Passed | 1,379 processed; zero backend failures; exact replay |
| G: MathQA fallback | Passed | 4,929 clean processed; zero backend failures; exact replay |
| H: Original matched-size selection | Failed; preserved | Size 200 exceeded depth and operation SMD 0.10 gates |
| 12E repair | **Passed** | One exhaustive legal generic replacement; all frozen matching gates pass |
| I-K: Targets, splits, freeze | **Passed** | Deterministic targets and 180/20 splits replay byte-identically |
| L-N: Frozen training environment | **Passed after authorization** | Native Windows `.venv-training` compatibility and deterministic launch gates passed |
| O-P: V1 REPLAY25 | **Completed** | Both arms ran 64 steps/64,000 tokens; checkpoints 16/32/64 |
| Q-R: V1 retention | Failed under old scorer | All cells passed offline after general scorer correction |
| V2 REPLAY40 and retention | **Completed** | Both arms ran; all cells passed corrected offline replay |
| 13A scorer adjudication | **Scorer defect corrected** | Exactly 12/4,764 decisions changed; only `question_generation` |
| 13A independent final | **Failed** | Both arms: `75/84` arithmetic; maximum failure family `5` |
| T-U: GSM1K and signal gate | Not run | No retention-approved pair exists |

No claim of Phase 2 improvement is supported. Training completed, but independent retention rejected
both selected adapters before benchmark evaluation.

## ASDiv verification result

| Measure | Count |
|---|---:|
| Source rows | 2,305 |
| Mathematically verified | 1,497 |
| Supported and eligible before contamination | 1,452 |
| Verified but unsupported family | 45 |
| Rejected | 808 |
| Duplicate source IDs | 0 |
| Parser nondeterminism | 0 |
| Formula/answer disagreements among accepted rows | 0 |

Supported family counts before contamination are `1,126` bookkeeping, `118` rate/ratio, and `208`
discrete. Major deterministic rejections are `484` unknown/unsupported formula grammars, `275`
multi-equality formulas, `36` unit incompatibilities, and `7` internally inconsistent formula
equalities. Rejected inconsistencies are resolved by exclusion; they are not admitted labels.

The full verification replay reproduced summary SHA-256 `6c45b435...895d`, all-row SHA-256
`119546be...d7f2`, and supported-row SHA-256 `6478aa3e...c016` exactly.

## Contamination result

All `1,452` supported rows were screened against the frozen 904-question development inventory,
the 1,000 Phase 1 synthetic questions, and each other. Seventy-three candidates reached or
exceeded the fixed `0.75` development semantic threshold and were rejected. Exact, contiguous
12-token, number-neutral, operation-structure, source-reference, candidate duplicate, and Phase 1
synthetic overlap counts are zero. No manual-review band exists and unresolved semantic candidates
are zero.

The remaining `1,379` rows comprise `1,076` bookkeeping, `111` rate/ratio, and `192` discrete
examples. Complete replay reproduced summary `0bf877c4...bdc5`, evidence
`99cb38aa...a631`, and clean rows `8d99a1de...eaac`.

## ASDiv-only structural capacity

| Per-arm size | Combined rate requirement | Clean rate rows | Structural result |
|---:|---:|---:|---|
| 300 | 170 | 111 | Ineligible; deficit 59 |
| 250 | 141 | 111 | Ineligible; deficit 30 |
| 200 | 114 | 111 | Ineligible; deficit 3 |

Capacity summary SHA-256 is `16260814...ba00`. The later untouched-base evaluation proved actual
ASDiv failure capacity insufficient and activated the predeclared MathQA fallback.

## Untouched-base pool evaluation and fallback

The untouched base scored `1,167/1,379` on clean ASDiv (`84.6265%`) with `1,253` extractable,
zero backend failures, and exact fixed replay. Its `152/22/38` base failures could not support the
minimum quotas, so the predeclared MathQA fallback activated.

Only the official MathQA training artifact was used. Exact program execution accepted
`15,468/29,837`; the frozen pre-inference selector retained `5,000`; contamination screening
rejected `71` and left `4,929`. The untouched base scored `2,363/4,929` (`47.9408%`) with `3,787`
extractable, zero backend failures, exact replay, and `1,214/1,136/216` family failures. Validation,
test, natural-language rationales, remote code, GSM1K selection signals, and sealed-final content
were never accessed.

## Stage H stop result

The fixed selector evaluated the authorized sizes in descending order. Size `300` failed with
formula-depth SMD `0.140411` and a `0.060` magnitude-level difference. Size `250` failed with
formula-depth SMD `0.137710` and a `0.056` magnitude-level difference. Size `200` achieved exact
source composition and passed every categorical per-level limit, but formula-depth SMD was
`0.113895` and operation-count SMD was `0.108765`, both above the fixed `0.10` maximum.

The experiment therefore stopped before assistant-target construction. Selection stop SHA-256 is
`1b169ab5bf62c1f790e739a645f2eb26bee3c4a18f7af4f9159014e62615650f`. There is no selected
experiment size, target-format hash, split hash, training measurement, retention result, adapter
hash, adapter GSM1K score, paired interval, or Phase 2 signal-gate decision. The only valid next
action at that historical boundary was interpretation or a separately authorized experiment.

## Milestone 12E matching repair and dataset freeze

The repaired input contains `209` eligible ASDiv failures and `2,507` eligible MathQA failures at
freeze hash `0e6332e2...5979`. Exhaustive deterministic single-row search checked `155,301`
replacements, found `152,226` legal and `1,979` passing candidates, and selected the lexicographic
optimum: remove generic `mathqa-train-26455` and add `mathqa-train-28853`. No two-row or global
fallback search ran.

The repaired SMDs are question tokens `0.0028892934`, base output tokens `0.0075164765`, formula
depth `0.0870898715`, and operation count `0.0680561998`; categorical maximum is `0.05`. Source
composition remains `97` ASDiv and `103` MathQA per arm, all family quotas remain exact, and every
source-ID, exact-question, normalized-question, latent-program, near-duplicate, and contamination
gate passes. Matching evidence is `004d338b...d5b5`.

Target format `4239aad3...55a2` produces one concise formula-derived calculation line, exactly one
terminal `Final answer:` line, and one final EOS. All 400 targets replay their formula/program and
canonical answer; maximum assistant length is `58` tokens. Each arm is deterministically split
`180/20` with no exact, normalized, program, or cross-arm overlap. Dataset identity is
`ee18f7f9...dc31`; complete matching and dataset reconstruction both replay byte-identically.

## Historical training-environment stop and authorized resolution

The required interpreter `C:\Users\Admin\Projects\Foundry\.venv\Scripts\python.exe` is CPython
3.12.10 with CUDA-enabled PyTorch `2.5.1+cu121` on the RTX 3080. It has Transformers `4.46.3` and
passes `pip check`, but PEFT, bitsandbytes, and TRL are absent. The recovery authorization forbids
installing or modifying packages and explicitly requires this environment, so Foundry did not
switch to another interpreter or alter dependencies. V1/V2 training, retention, and adapter GSM1K
evaluation did not run under that recovery authorization.

A later explicit authorization selected `.venv-training`, reconciled its frozen PyYAML metadata
exception, and audited deterministic launch, Windows operational variables, argv transport, CUDA,
scheduler semantics, and the first positive-learning-rate update. V1 and V2 then trained without
changing the dataset, schedules, optimizer, scorer thresholds, or benchmark firewall.

## Milestone 13A scorer adjudication and independent final

The published V1/V2 stop contained twelve anchor question-generation failures: V1 and V2,
generic and targeted, and steps 16/32/64. All twelve used one prompt and one byte-identical response.
The untouched base response was also byte-identical. The response completed the requested
arithmetic task and used `?` only as a displayed mathematical unknown. The old scorer's
literal-question-mark-anywhere rule therefore created a false decision.

The corrected general scorer distinguishes protected mathematical placeholders, valid JSON, code,
and prompt-supplied quotations while continuing to reject unprotected question marks and generated
question/problem headers. Fifteen nonbenchmark fixtures, focused mutations, tamper checks, and the
old-versus-new projection passed. Exactly twelve of 4,764 stored decisions changed, and the only
changed field was `question_generation`. Corrected scorer source and configuration SHA-256 values
are `0cc6f583...b530b91` and `87a14082...f2c7`.

Two byte-identical offline rescoring runs re-applied every unchanged retention threshold to all 24
V1/V2 arm/checkpoint/subset cells. All passed, so the original V1-first/latest hierarchy selected
V1 step 64. Generic and targeted adapter SHA-256 values are `fe3c0f5a...18a73d` and
`7f15edf4...da0bd`.

Each selected adapter then ran once, generic followed by targeted, on independent subset
`f5684507...e4ec`. Both results were:

| Gate measure | Generic | Targeted | Requirement |
|---|---:|---:|---:|
| Overall preservation | 131/141 (92.9078%) | 131/141 (92.9078%) | at least 90% |
| Arithmetic preservation | 75/84 (89.2857%) | 75/84 (89.2857%) | at least 90% |
| Format preservation | 26/27 (96.2963%) | 26/27 (96.2963%) | at least 90% |
| Instruction preservation | 30/30 (100%) | 30/30 (100%) | at least 90% |
| Wilson lower bound | 87.4373% | 87.4373% | at least 85% |
| Question generation | 0 | 0 | exactly 0 |
| Prompt echo | 0 | 0 | at most 2% |
| Backend failures | 0 | 0 | exactly 0 |
| Maximum failure family | 5 | 5 | at most 3 |

Both arms fail arithmetic preservation and failure-family concentration. Decision SHA-256 is
`0746305653d6d23674f8df1652ec07be6585d1fda2f0d4dd3b3b70f6efe79741`. The success-only adapter
freeze commit and Milestone 12F-B are not authorized. The next project-level action is a separately
authorized interpretation of this independent retention failure; GSM1K remains not run.
