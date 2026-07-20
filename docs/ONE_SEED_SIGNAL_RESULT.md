# One-Seed Signal Experiment: Training-Parity Stop

> No targeted-versus-generic development result exists. Evaluation stopped at the frozen training-token parity gate.

## Status

Foundry generated and automatically validated 500 targeted and 500 matched generic-control
examples, created deterministic 450/50 training/validation splits, validated native RTX 3080 QLoRA,
and trained one adapter per curriculum using the same published 200-step recipe. The intended
one-seed development comparison did not run because the loss-bearing token counts differed by
11.5299%, above the predeclared 2% maximum.

This is not a negative model-quality result. It is an experimental-control failure discovered before
benchmark exposure.

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

## Next decision

The narrowest valid next step is a separately approved experimental redesign that freezes equal
loss-bearing token budgets before training, then retrains both conditions from the same base. The
current adapters must not be benchmark-compared or presented as evidence for or against Foundry's
targeted-data hypothesis. The pending stratified human language review remains independently useful.
