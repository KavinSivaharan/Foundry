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
