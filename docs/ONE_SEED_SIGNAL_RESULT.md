# One-Seed Signal Experiment: Token-Matched Result

> **Provisional one-seed result pending stratified human language review and second-seed
> confirmation.**

## Status

Foundry preserved the original parity-failed adapters as negative controls, froze a token-matched
whole-example schedule, retrained both arms from the same base, passed final actual-token parity,
and evaluated generic then targeted on the frozen 814-example development set. The one-seed signal
gate failed: generic scored 15 and targeted 14, compared with 521 for the frozen base. Both adapters
also lost most terminal-answer extractability, so the run does not provide a valid test of the
targeted-versus-generic curriculum hypothesis.

## Frozen data and language status

- Policy: `matched-template-signal-v1`; policy SHA-256 `7e56acfa...3518`.
- Attempts: 550 targeted and 550 generic; acceptances: 500 and 500.
- Family counts: targeted `275/117/108`; generic `167/167/166`.
- Splits: 450 training and 50 synthetic validation per dataset.
- Output-contract examples: 100 per dataset.
- False labels, verifier disagreements, deterministic language defects, exact/latent overlaps, and
  unresolved contamination cases: zero.
- Codex language audit: 1,000 high-confidence approvals; genuine stratified human review remains
  pending at `results/raw/foundry_500x2_signal_review/`.

## Frozen recipe and compatibility smoke

Recipe `foundry-qwen2.5-1.5b-signal-qlora-v1` has SHA-256
`4a9c6043f72d4f5b83dad774ffcd208e17f8c9738c9b34b0ab06919ba2620590`. It uses the pinned
Qwen2.5-1.5B revision, NF4 double quantization, rank-16 LoRA on seven projection families, 512-token
unpacked inputs, effective batch eight, paged AdamW 8-bit, a cosine schedule, seed `20260720`, and
200 optimizer steps. The 32-step smoke passed forward, backward, optimization, finite loss, adapter
save/reload, and inference at 3,741,319,168 bytes peak reserved VRAM.

## Final adapter training

| Measurement | Generic control | Targeted |
|---|---:|---:|
| Optimizer steps | 200 | 200 |
| Examples processed | 1,600 | 1,600 |
| Padded model-input tokens | 819,200 | 819,200 |
| Non-padding loss tokens | 271,396 | 306,766 |
| Initial logged loss | 3.1699 | 2.7859 |
| Final logged loss | 0.1179 | 0.1199 |
| Final synthetic-validation loss | 0.153627 | 0.144995 |
| Runtime seconds | 641.366 | 645.737 |
| Peak reserved VRAM bytes | 3,577,741,312 | 3,577,741,312 |
| Peak process RSS bytes | 1,542,365,184 | 1,542,606,848 |
| Adapter bytes | 89,796,953 | 89,796,953 |
| Adapter SHA-256 | `36b19165...e3ac` | `217a9bcf...406e` |

Both adapters reload offline on CUDA with zero offloaded parameters and exact directory hashes.
All recipe, package, base, seed, step, batch, padded-token, sequence-length, validation-frequency,
and checkpoint-rule fields match.

## Parity decision

The absolute difference is 35,370 non-padding tokens. Relative to the larger run, that is
11.5299%, so the frozen `<=2%` parity gate fails. The unequal loss-token exposure is consistent with
the targeted curriculum containing more bookkeeping examples and longer rendered/trace sequences.
No data was trimmed or regenerated after training, as required.

The frozen 814-example development evaluator was not run for either adapter. Consequently:

- generic development accuracy: not measured;
- targeted development accuracy: not measured;
- category-level changes: not measured;
- one-seed signal gate: not evaluated;
- sealed-final evaluation: not run.

## Token-matched v2 protocol

> **Provisional one-seed result pending stratified human language review and second-seed
> confirmation.**

The deterministic census found 77,348 unique-record tokens in generic and 87,317 in targeted, with
zero truncation. Fixed-occurrence Method A failed at an exact best-case 9.4343% difference.
Whole-example token-budgeted Method B passed: its 200-step schedules contain 271,292 generic and
271,150 targeted tokens, a 0.05234% difference. The v2 recipe SHA-256 is
`df7c7b8d7b402683a550fb11ebbe4ceb633ed47597c98ea661affa7876d6fa54`.

The four-step parity smoke passed at 5,464 versus 5,440 actual tokens (0.43924%), finite losses and
gradients, identical optimizer/scheduler counts, and successful offline reload.

## Token-matched v2 retraining and final parity

| Measurement | Generic control | Targeted |
|---|---:|---:|
| Optimizer/scheduler steps | 200 | 200 |
| Scheduled occurrences | 1,578 | 1,398 |
| Actual loss-bearing tokens | 271,292 | 271,150 |
| Padded tokens | 807,936 | 715,776 |
| Initial token-weighted loss | 3.262921 | 3.139081 |
| Final token-weighted loss | 0.149672 | 0.142186 |
| Mean token-weighted loss | 0.420702 | 0.390246 |
| Final synthetic-validation loss | 0.175241 | 0.165387 |
| Runtime seconds | 634.798 | 569.049 |
| Peak reserved VRAM bytes | 3,577,741,312 | 3,577,741,312 |
| Peak process RSS bytes | 1,477,734,400 | 1,478,729,728 |
| Adapter bytes | 89,796,953 | 89,796,953 |
| Adapter SHA-256 | `c039612d...5df1` | `b4a2e55d...b02e` |

Scheduled and actual loss tokens match exactly in each arm. The pair differs by 142 tokens, or
0.052342%, below the frozen 0.5% maximum. Base revision, packages, recipe, seed, optimizer,
scheduler, LoRA configuration, sequence length, step count, and final-checkpoint rule all match;
both adapters reload offline and no development record was exposed during training.

## Frozen development results

| Measurement | Frozen base | Generic control | Targeted |
|---|---:|---:|---:|
| Correct | 521/814 | 15/814 | 14/814 |
| End-to-end accuracy | 64.0049% | 1.8428% | 1.7199% |
| Extractable answers | 752/814 | 167/814 | 180/814 |
| Extractability | 92.38% | 20.5160% | 22.1130% |
| Exact-format compliant | 130/814 | 137/814 | 157/814 |
| Accuracy among extractable | 69.2819% | 8.9820% | 7.7778% |
| Extractable but wrong | 231 | 152 | 166 |
| Unextractable | 62 | 647 | 634 |
| Truncated | 3 | 2 | 3 |
| Backend failures | 0 | 0 | 0 |
| Evaluation seconds | 3,160.074 | 1,357.107 | 1,359.252 |
| Peak reserved VRAM bytes | 3,353,346,048 | 3,512,729,600 | 3,512,729,600 |

Both adapter runs used the unchanged config, prompt, canonical extractor, manifest, model revision,
greedy decoding, and 768-token limit. Generic changed by -506 correct answers versus base; targeted
changed by -507 versus base and -1 versus generic.

## Paired and category analysis

Targeted wins 11 examples that generic misses; generic wins 12 that targeted misses, so targeted's
net paired advantage is -1. Generic fixes one base failure and breaks 507 base successes; targeted
fixes two and breaks 509. With 10,000 paired bootstrap replicates, fixed seed `20260720`, and stable
ID ordering, targeted accuracy minus generic accuracy is -0.12285 percentage points with a 95%
interval of **[-1.22850, +0.98280] percentage points**.

The frozen taxonomy covers exactly the base model's 293 development failures. On its three selected
reasoning categories, both adapters fix zero examples: bookkeeping 0/68, rate/ratio 0/28, and
discrete 0/27. Across the 170 untargeted taxonomy rows, generic fixes one and targeted fixes two;
because the taxonomy contains base failures only, this aggregate has no base decline. Category
claims do not extend to the 521 base-success rows, which the frozen failure taxonomy does not label.

## One-seed signal decision

| Frozen clause | Result |
|---|---|
| Targeted at least 529/814 | **Failed:** 14 |
| Targeted at least four over generic | **Failed:** -1 |
| Targeted extractability at least 91.38% | **Failed:** 22.11% |
| Zero targeted backend failures | Passed |
| No >2-point decline on aggregate untargeted taxonomy set | Passed within frozen failure-only scope |
| Final training parity passed | Passed: 0.052342% |

Overall: **FAILED**. The shared collapse in terminal-answer extractability is the narrowest direct
evidence: both training arms failed to retain the base model's instruction/output-contract behavior.
That common failure dominates the curriculum contrast. No tuning, retraining, second seed,
sealed-final evaluation, or GRPO is authorized or recommended automatically.

## Human-review status and limitations

The stratified language review remains pending. The existing ignored assisted review page is:

`file:///C:/Users/Admin/Projects/Foundry/results/raw/foundry_500x2_signal_review/codex_assisted_review.html`

Codex inspection is not genuine human approval. This is one training seed, the failure taxonomy
labels base failures rather than all 814 examples, and the common post-training behavior failure
prevents a clean conclusion about targeted versus generic synthetic curricula.

## Next decision

Decide whether to approve a narrow diagnosis of SFT label scope, completion formatting, and adapter
instruction retention. Do not approve a second seed until the shared behavior collapse is explained.
Separately, the user may complete and export the pending stratified language review. No subsequent
milestone begins automatically.

## Milestone 8E diagnosis and retention stop

The approved diagnosis found that all prior training transcripts made system and user text
loss-bearing and that only 200/1,000 targets ended in the evaluator-aligned terminal contract.
Adapter loading, disabling, re-enabling, LoRA inventory/scaling, and untouched-base restoration all
passed. These findings classify the shared failure under causes 1 and 3; neither is claimed as the
sole cause.

The corrected `foundry-assistant-only-sft-v3` format hash is
`3ffba98610f0575e49e686c6d036e2c18963f3d9411b1f682fe07c009b535329`. Its full schedules contain
90,000 generic and 89,995 targeted assistant tokens (0.00556% difference); the two predeclared
32-step prefixes contain exactly 14,404 tokens each. Recipe `2e-4` failed multiple retention
requirements. Recipe `5e-5` failed because generic instruction accuracy was 13/15 and targeted
arithmetic was 25/30. The deterministic gate summary is
`5d1d5f0d31fce710836bd398baeeb469bc4d36c9888476ae38f9721f37671201`.

No recipe was selected. Full retraining, common-checkpoint selection, final parity, corrected
development evaluation, paired analysis, and a new signal decision were not run. The frozen base
and Milestone 8D numbers above remain historical evidence of the collapse, not a valid curriculum
comparison. Human review is still pending at
`file:///C:/Users/Admin/Projects/Foundry/results/raw/foundry_500x2_signal_review/codex_assisted_review.html`.

## Fast-Track 8F-H retention ladder and validation stop

> **Provisional one-seed result pending stratified human language review and second-seed
> confirmation.**

The v3 target audit found procedural trace style in 376/500 generic and 419/500 targeted
completions. Deterministic concise-v4 reconstructed all 1,000 records with zero rejection and a
maximum of 41 assistant tokens. Two original ignored 90-prompt suites were frozen; the untouched
base passed both.

The ladder compared exactly four pairs at steps 8/16/24/32 with 14,400 tokens per arm. Only
Variant A (v3, `5e-5`) had common calibration passes, so the hierarchy selected step 32 without
GSM1K access. Disjoint validation then failed: generic scored 45/45 arithmetic, 20/20 format,
21/25 instruction, 90/90 extractable; targeted scored 44/45, 20/20, 21/25, 89/90. Both instruction
scores are 84%, below the fixed 90% gate.

No 200-step retraining, final-holdout adapter evaluation, parity promotion, GSM1K rerun, paired
analysis, or new one-seed signal decision exists. The frozen historical base/generic/targeted
numbers above remain evidence of prior collapse, not a corrected curriculum comparison. The next
decision must diagnose the training method without reusing validation for selection; human review
remains pending at the same local URL.

## Fast-Track 8I–8K powered-adjudication stop

> **Provisional one-seed result pending stratified human language review and second-seed
> confirmation.**

The old 25-item instruction failure was audited as 21 shared passes, two shared failures, and two
base-only successes lost by both adapters. All four adapter failures are genuine instruction
noncompliance; prompt/scorer defects are zero. The powered artifacts were frozen before any adapter
evaluation, but the untouched base failed the adjudication suite's prerequisite usability gate:
`84/100` arithmetic, `48/100` format, `55/100` instruction, and `268/300` extractable.

Accordingly, neither existing A/32 adapter was evaluated on the new suite, the untouched anchor
holdout was not consumed, the shared-anchor fallback was not trained, and GSM1K was not rerun. No
selected branch, new adapter hash, paired bootstrap interval, or new signal-gate result exists. The
historical collapsed one-seed numbers remain non-comparative evidence; the next decision must address
the unusable powered instrument or stop SFT adaptation without treating development or validation as
a tuning set.

## Milestone 8L base-conditioned retention adjudication

> **Provisional one-seed result pending stratified human language review and second-seed
> confirmation.**

`foundry-base-conditioned-retention-v1` measures only preservation of prompts that the untouched
base answered correctly under the frozen scorer. The adjudication subset contains 187 IDs:
84 arithmetic, 48 format, and 55 instruction; subset SHA-256 is
`c76df74b911b96ca43c2663a123e41347fd544bf6644f15522ccaad7b77099e1`.

The untouched base was evaluated exactly once on the frozen anchor holdout and scored 96/100
arithmetic, 60/100 format, and 54/100 instruction (210/300 overall), with 283/300 extractable,
zero echo, zero backend failures, and one question-generation output. The 210 correct IDs were
frozen as the independent holdout subset; its SHA-256 is
`36be91d08f2ab0e05c491094c53965d1aa4f989a730347768877a2548a62c7a9`.

Generic and targeted each preserved 181/187 adjudication items (96.79%; Wilson lower 93.18%), with
84/84 arithmetic, 43/48 format, and 54/55 instruction. Each emitted one question-generation
output. On holdout, generic preserved 197/210 (93.81%; Wilson lower 89.70%) and targeted preserved
200/210 (95.24%; Wilson lower 91.46%). Their section scores were generic 90/96, 53/60, 54/54 and
targeted 93/96, 53/60, 54/54. Generic emitted one question-generation output; targeted emitted zero.
Echo and backend failures were zero in all four cells.

The retention gate failed because both adapters were below 90% format preservation on both subsets;
the adjudication cells and generic holdout also violated the zero-question-generation clause. The
pair is `failed_base_conditioned_retention`, so GSM1K was not run and no paired/category signal
analysis was authorized. The historical frozen base remains 521/814; no new generic or targeted
benchmark result exists. Human review remains pending at
`file:///C:/Users/Admin/Projects/Foundry/results/raw/foundry_500x2_signal_review/codex_assisted_review.html`.

The exact next decision is whether to stop the project or approve interpretation of the negative
adaptation result. Another SFT method, second seed, and sealed-final evaluation are not recommended
or authorized.

## Milestone 8M common-scale retention calibration

> **Provisional one-seed result pending stratified human language review and second-seed
> confirmation.**

The historical full-strength failure remains valid. A reversible runtime mechanism now applies one
common factor to all 196 active LoRA modules while leaving adapter tensors, base parameters, and
checkpoint files unchanged. Scale 0.0 exactly matches the untouched-base diagnostic output and
scale 1.0 exactly matches each unscaled adapter; scaling-source SHA-256 is
`1e0506ce89a65ab2699f514730eec0437788fe35d55eeb31e299bdd60fa5ceff`.

An original final holdout was frozen before scaled-adapter exposure: 450 prompts split equally
across arithmetic, format, and instruction, suite SHA-256
`b856c8ce8e56d98eb7e3fbffdead07ffde7091ab2a20abe5a22ada598136353e`. The untouched base scored
`112/150`, `127/150`, and `79/150` (318/450 overall), so the 318 correct IDs were frozen as subset
`0884923ce7ab39f1080282dab0ce51aff7063270d6c97f5c1d70370256012ded`.

Scale 1.00 reconstructed the prior four-cell failure. Scale 0.75 passed both adjudication cells but
failed the anchor holdout zero-question-generation rule. Scale 0.50 passed all four selection
cells: generic/targeted adjudication `182/187` and `183/187`; both anchor holdout `205/210`.
Independent validation then passed at generic `314/318` and targeted `315/318`, with format
`127/127` for both and no echo, question generation, malformed output, or backend failure. Scale
0.25 was not run. Decision SHA-256 is
`6f3e7a29dfbb184f5b6b5eb09fd52060c3c2465c5da2343f85d62d05f8589cc7`.

The retention decision was committed and pushed before generic then targeted were evaluated once at
common scale `0.50` on the unchanged 814-ID evaluator. The base was not rerun.

### Common-scale frozen development result

| Measurement | Frozen base | Generic scaled | Targeted scaled |
|---|---:|---:|---:|
| Correct | 521/814 | 387/814 | 414/814 |
| End-to-end accuracy | 64.0049% | 47.5430% | 50.8600% |
| Extractable | 752/814 | 768/814 | 767/814 |
| Extractability | 92.38% | 94.3489% | 94.2260% |
| Exact-format compliant | 130/814 | 482/814 | 479/814 |
| Accuracy among extractable | 69.2819% | 50.3906% | 53.9765% |
| Extractable but incorrect | 231 | 381 | 353 |
| Unextractable | 62 | 46 | 47 |
| Truncated | 3 | 2 | 3 |
| Backend failures | 0 | 0 | 0 |
| Delta versus base | - | -134 | -107 |

Both runs preserve the frozen manifest, prompt, extractor, model revision, greedy 768-token
generation, and adapter hashes. All 196 LoRA scaling values and adapter/base signatures restore
exactly. Generic used 113,720 input and 139,720 output tokens; targeted used 113,720 and 145,865.
Peak allocated/reserved VRAM was 3,248,531,968/3,512,729,600 bytes in both arms. Generic measured
4,412.066 seconds. Targeted's measured 33,838.811-second wall interval includes an observed host
suspension/long scheduling pause and must not be interpreted as active GPU compute throughput.

### Common-scale paired result and final gate

Targeted wins 47 rows generic misses; generic wins 20 rows targeted misses, for a net targeted win
of 27. Generic fixes 54 base failures and breaks 188 base successes; targeted fixes 58 and breaks
165. The paired targeted-minus-generic point estimate is +3.3170 points; the 10,000-replicate 95%
interval is **[+1.3514, +5.2826] points** with seed `20260720`.

On the frozen failure taxonomy, targeted versus generic changes are: bookkeeping `14/68` versus
`12/68`; rate/ratio `4/28` versus `5/28`; discrete `4/27` versus `3/27`; arithmetic execution
`3/22` versus `2/22`; output format `20/69` versus `18/69`; interpretation `8/53` versus `11/53`;
time/unit `5/24` versus `3/24`; ambiguity risk `0/2` for both. Across the 170 untargeted taxonomy
rows, targeted fixes 36 and generic 34. The taxonomy covers base failures only and cannot describe
regressions among the 521 base-success rows.

The one-seed signal gate **fails solely because targeted has 414 rather than at least 529 correct**.
Targeted is at least four above generic, extractability exceeds 91.38%, backend failures are zero,
the frozen untargeted-taxonomy clause passes, actual assistant-token parity is exactly
14,400/14,400, and common-scale retention passes on all three subsets. Decision SHA-256 is
`2b4f39b542ebe16a4cdfd4835856b9965de9dc04c2384fffaf12a064d736a0ed`.

The narrow conclusion is that targeted data outperformed matched generic data within this method,
but neither scaled adapter retained acceptable absolute GSM1K capability. Human review remains
pending at
`file:///C:/Users/Admin/Projects/Foundry/results/raw/foundry_500x2_signal_review/codex_assisted_review.html`.
Do not tune the scale, retrain, run a second seed, or access sealed-final automatically. Complete
the pending human review, then decide whether to stop or separately approve a materially different
retention-preserving architecture.

## Milestone 8N contrastive curriculum task-vector result

> **Provisional one-seed result pending stratified human language review and second-seed
> confirmation.**

The unchanged Variant A step-32 adapters were compatible for exact PEFT composition. Their dense
updates have global Frobenius norms `1.6918784364` for generic and `1.6980775191` for targeted,
with cosine similarity `0.9399098552`. The exact targeted-minus-generic differential has norm
`0.5876302228`, or `34.6056%` of the targeted update norm. These measurements support the
predeclared interpretation that the two adapters contain a large shared update plus a smaller
curriculum-specific differential; they do not by themselves establish benchmark improvement.

The differential was represented without retraining as the reversible rank-32
`targeted_minus_generic_v1` LoRA adapter. Its ignored local artifact SHA-256 is
`84f02df1cbc5ec1015d096164dbfe3833e166a14eda9ffadf62b5d2d2527c961`. Layerwise reconstruction
against `Delta_targeted - Delta_generic` passed: maximum absolute error was
`1.7462298274e-10` and maximum relative Frobenius error was `2.935335e-7`. The independent FP32
functional check also passed, with maximum logit error `5.626678e-5` and relative logit error
`1.959386e-6`. Source-adapter and base state remained unchanged.

Retention-only scale selection evaluated exactly the predeclared descending ladder:

| Contrastive scale | Adjudication | Anchor holdout | Decision |
|---:|---:|---:|---|
| 1.00 | 181/187; passed | 204/210; failed with 1 question-generation output | Failed |
| 0.75 | 182/187; passed | 207/210; failed with 2 question-generation outputs | Failed |
| 0.50 | 183/187; passed | 207/210; failed with 1 question-generation output | Failed |
| 0.25 | 184/187; passed | 208/210; failed with 2 question-generation outputs | Failed |

All four anchor cells failed the unchanged zero-question-generation clause. Consequently, no
contrastive scale was selected, the independent final holdout was not exposed to the adapter, and
GSM1K development was not run. The selection decision SHA-256 is
`b41d975f342820ac34ca693d599677994e3f272243c114c313605beb020ad49a`.

The exact arithmetic and numerical-equivalence gates passed, but the retention gate did not. The
contrastive adapter-arithmetic route is therefore closed for this project version: do not tune a
scale using GSM1K, retry another merge algorithm, retrain, or run a second seed automatically.
Human language review remains pending at
`file:///C:/Users/Admin/Projects/Foundry/results/raw/foundry_500x2_signal_review/codex_assisted_review.html`.
The next decision is whether to stop adaptation or separately design a materially different
retention-preserving approach such as KL/replay regularization or verifier-reward optimization.

## Milestone 9 base-behavior replay experiment stopped before training

> **Provisional one-seed result pending stratified human language review and second-seed
> confirmation.**

The approved replay architecture first evaluated the untouched base on the existing 120-item shared
anchor. It froze the 83 scorer-correct actual base outputs—40 arithmetic, 20 format, and 23
instruction—as a single shared replay corpus. No predefined gold response was substituted for a
base output.

A newly frozen 450-item independent retention holdout then passed self-score, ambiguity, and
disjointness checks, including zero exact and 12-token overlap against 3,314 prior prompts. The base
scored only 84/150 arithmetic, 27/150 format, and 30/150 instruction (141/450 overall), below the
declared 60-per-category and 250-overall usability requirements. There were zero backend failures,
prompt echoes, or question-generating outputs.

The experiment therefore stopped before replay/KL scheduling or training, before any of the six
adapters existed, and before retention selection or GSM1K. There is no new targeted-versus-generic
benchmark result. The exact next decision is whether to stop conventional adaptation or separately
approve a new architecture with a new independently frozen retention instrument; the observed
holdout cannot be revised as a response to its base score.

## Milestone 10 verifier-reward GRPO stopped at compatibility

> **Provisional one-seed result pending stratified human language review and second-seed
> confirmation.**

The verifier-GRPO protocol froze a base-conditioned 141-item final-retention subset and paired
prompt-only schedules with 64 groups and 256 planned completions per arm. Both arms contain the same
12 replay groups, 52 synthetic groups, and exactly 6,702 model-visible prompt tokens. The
deterministic reward and adapter-disabled base-reference contracts passed their focused tests.

The G1 compatibility probe failed before its first completion. Frozen top-p `0.95` sampling invoked
CUDA cumulative summation while strict deterministic algorithms were enabled; PyTorch 2.5.1+cu121
reported that this kernel has no deterministic implementation. Consequently, completions, rewards,
optimizer steps, G1/G2 adapters, retention results, and new GSM1K results are all zero or absent.
This is a runtime-contract incompatibility, not a new curriculum comparison.

The earlier one-seed benchmark result and its provisional label remain unchanged. Human review is
pending at
`file:///C:/Users/Admin/Projects/Foundry/results/raw/foundry_500x2_signal_review/codex_assisted_review.html`.
Before another compatibility probe, the user must explicitly approve a frozen way to reconcile
stochastic sampling with deterministic execution, or stop verifier-GRPO; no deterministic or
sampling control may be weakened silently.

## Milestone 10E warning-only replay ended verifier-GRPO

The approved warning-only contract successfully limited relaxed deterministic enforcement to the
frozen top-p generation call. Three official same-process replays each completed `12` outputs using
the same two synthetic groups and one replay group. All observed warning classes were the approved
CUDA cumsum warning, no warning-only state leaked, and the diagnostic model/evidence payloads were
otherwise identical.

Exact replay still failed. Shared compatibility source changed between replay 1 and replay 2, and
shared replay-evidence source changed between replay 2 and replay 3, yielding three different exact
packet hashes. The failure-summary SHA-256 is
`8501b7681262ceca002659978c07c688a6f7baa45923ebb3c06e6134adabebe4`; warning-contract summary
SHA-256 is `eff84b9ec92715eeb74a6c74bcad5980dded9c4b5482012fd8e2438857f24598`.

The predeclared stop rule therefore closed verifier-GRPO. Fresh-process and two-step replay, G1/G2,
retention, and GSM1K were not run. No adapter or new benchmark result exists. The earlier one-seed
curriculum comparison and its provisional human-review/second-seed label remain unchanged.

## Milestone 10G orchestration correction does not create a signal result

The newly authorized source/interpreter/artifact-root decoupling changes only how the immutable
experiment is launched and where writable evidence is stored. It does not alter the frozen sampler,
rewards, reference policy, optimizer, schedules, retention thresholds, or GSM1K evaluator. The
original `165` focused GRPO tests and all `17` new path-contract tests pass, but no model process has
run in this phase.

Accordingly, the one-seed GRPO signal gate remains **not reached**. There is no new adapter,
retention result, or generic-versus-targeted benchmark comparison. The historical base remains
`521/814`; the existing result remains **Provisional one-seed result pending stratified human
language review and second-seed confirmation**, with review still pending at the frozen local URL.

### Milestone 10G remained pre-signal after immutable replay failure

The V2 experiment reached one official 12-completion generation-only replay but failed in post-run
orchestration validation before writing or comparing its packet. The failure was the new validator's
incorrect lifetime treatment of the frozen cuBLAS `:4096:8` to `:16:8` transition. It was not a
targeted-versus-generic result and exposed no model-side replay difference.

The hard stop leaves G1/G2 optimizer steps, adapters, retention results, generic/targeted GSM1K
predictions, paired effects, bootstrap intervals, and signal decisions absent. The GRPO one-seed
signal gate is therefore **not reached**, and the historical provisional label and pending human
review remain unchanged.

### Milestone 10H remains pre-signal during the environment correction

The explicitly authorized V3 correction standardizes all deterministic environment values before
Python launch and does not change any scientific input or decision threshold. The correction phase
passed `198` focused GRPO tests and `709` repository tests, but ran no model generation, optimizer,
retention, or GSM1K process.

The GRPO one-seed signal gate therefore remains **not reached** at this publication point. The
historical base result remains `521/814`, and the existing result remains **Provisional one-seed
result pending stratified human language review and second-seed confirmation**.

The subsequently launched V3 same-process gate failed before model loading because `nvidia-smi`
could not initialize NVML under the exact child-environment allowlist. No completion, adapter,
retention score, or GSM1K prediction was produced. The GRPO one-seed signal gate is therefore
**not reached**, and the historical `521/814` base result was not rerun.

### Milestone 10I remains pre-signal during the CUDA-runtime correction

The newly authorized V4 source correction changes only GPU orchestration: parent NVML monitoring is
separate from an authoritative direct PyTorch CUDA child probe. No model process, adapter, retention
evaluation, or GSM1K evaluation ran while implementing it. The GRPO one-seed signal gate remains
**not reached**; the historical frozen base remains `521/814` and was not rerun.

### Milestone 10I stopped before a trainable compatibility pass

V4 passed direct CUDA compute and exact same/fresh generation replay, then failed the first complete
two-step smoke because the training generation path emitted multiple normalized warning classes.
No optimizer step, adapter, retention result, or new GSM1K prediction exists. The GRPO one-seed
signal gate is therefore **not reached**; the historical base remains `521/814` and was not rerun.
