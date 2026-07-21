# Verifier-reward GRPO result

> **Status:** Compatibility gate failed; hard stop enforced.
>
> **Benchmark label:** Provisional one-seed result pending stratified human language review and
> second-seed confirmation.

## Plain-English result

Foundry successfully prepared a verifier-reward GRPO experiment without exposing answers to the
model. It froze matched generic and targeted prompt schedules, implemented executable rewards, and
verified a memory-safe reference-policy design. The first GPU compatibility generation then hit a
mechanical conflict: the approved decoder uses stochastic top-p sampling, but the pinned PyTorch
CUDA stack has no deterministic implementation of a cumulative-sum operation that sampling needs.

The failure occurred before the model finished one completion. No reward was computed, no gradient
was applied, and no adapter was trained. The approved stop rule therefore blocks all G1/G2 training,
retention selection, independent final retention, and GSM1K evaluation.

## Closed conventional-SFT context

The conventional SFT lineage remains closed. Earlier common-scale SFT adapters preserved retention
only after reducing adapter contribution, then scored below the untouched base on frozen GSM1K.
Exact targeted-minus-generic adapter arithmetic also failed the independent retention selection
rule. Milestone 9 froze 83 demonstrated untouched-base replay behaviors but stopped before training
because its 450-item absolute holdout was not usable under the original base-coverage gate.

Milestone 10 did not reopen or rerun any SFT recipe. Every potential GRPO adapter was specified to
start from the untouched `Qwen/Qwen2.5-1.5B-Instruct` revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

## Base replay and final retention artifacts

The shared replay corpus remains 83 scorer-correct untouched-base behaviors:

- arithmetic: 40;
- format: 20;
- instruction: 23;
- replay corpus SHA-256:
  `b511129f89ce450014b78698e9e439bdaa0947657f301c3e99b2a9955b7ab4d1`;
- replay manifest SHA-256:
  `27ccd1c22bd321d17418ca346e1b3b4022fd696fdde07550e1c56f2864efde18`;
- replay-format SHA-256:
  `758dc1f35020e88e04c425b6106e54ea2f577f547afa4762ade9923762af6d66`.

The untouched base had answered 141 rows correctly on the unused 450-item final retention holdout.
Milestone 10 froze those IDs, in original suite order, before any new adapter training:

| Section | Frozen base-correct rows |
| --- | ---: |
| Arithmetic | 84 |
| Format | 27 |
| Instruction | 30 |
| Total | 141 |

Subset SHA-256 is
`f56845076a1a59e5ca1a95466541339b56f026e945f86118caec307a690ee4ec`; ordered-ID SHA-256 is
`daa294be1d17e38d11fc06d5451ad387cf9cc0c718726aa30af9b3b430782879`.
The manifest contains no prompts or references, read no adapter outputs, and changed no scorer.

## Prompt-only GRPO schedules

Complete prompts and reward-side metadata remain under ignored raw storage. The tracked schedules
contain only stable IDs, counts, hashes, categories, positions, and token aggregates. No reasoning
trace or assistant target is model-visible.

| Property | Generic | Targeted |
| --- | ---: | ---: |
| Total groups | 64 | 64 |
| Synthetic groups | 52 | 52 |
| Replay groups | 12 | 12 |
| Planned completions | 256 | 256 |
| Prompt tokens | 6,702 | 6,702 |

The 12 replay IDs, categories, positions, and ordering are identical. Synthetic IDs are unique
within each arm and disjoint across arms. Targeted synthetic allocation is 29 bookkeeping, 12 rate,
and 11 discrete groups; generic allocation is 17, 17, and 18. No model output or reward influenced
selection.

- generic ignored prompt-packet SHA-256:
  `67f48ebc3a310c0cb0db882b46759e993f3ad99faec9b7309d336d7b97f44400`;
- targeted ignored prompt-packet SHA-256:
  `1d13acb49aeea00bf2c68b6488d225796ee95411f6a79343d969731b6f1a2286`;
- generic manifest SHA-256:
  `5848ed6640dda21752ab9692c8e531d9175314a7d5a472616dc19ad834a6351e`;
- targeted manifest SHA-256:
  `cb13d4d522746bdfa829c9a405defdb0eff0acbd23859dc7fe49457318cc1ccf`;
- schedule-summary SHA-256:
  `23fede9132f53b7d32f354056c728fc68faa20586a9162e101834db34f71ca64`.

## Deterministic reward contract

`foundry-verifier-grpo-reward-v1` keeps trusted answers and scorer metadata outside the visible
prompt. Synthetic completions receive +1.00 for an exact extracted canonical answer, +0.10 for one
safely extractable terminal answer, and +0.05 for the exact terminal contract. Replay completions
receive +1.00 from the frozen prompt-specific scorer and +0.05 for its exact output contract.
Truncation, prompt-echo/question-generation behavior, and conflicting answers receive the frozen
additive safety penalties. There is no LLM judge, learned reward, benchmark reward, response-length
reward, style reward, or category multiplier.

- implementation SHA-256:
  `089650105e29ead3c4ad62f1e0e41263e6c2af5fb8a12cb2851644aca3599616`;
- configuration SHA-256:
  `4a47359fa3129b1bfd79dd158ecb609177e9b1642a95368c106e016a1554a965`;
- fixture SHA-256:
  `8ba02448a87d5fe8c412f0e7a66acad5b45b6c6e9237dd366eecc060fbe67bdc`;
- calibration SHA-256:
  `fc420f3cfd1737592a0ef49c8c835baff25fa7382642b13c4802ab5d18c5722c`.

The fixtures cover correct, wrong, malformed, conflicting, echoed, truncated, synthetic, and replay
cases and prove deterministic additive scoring without benchmark access.

## Reference-policy and runtime contracts

The installed TRL 0.17.0 PEFT path uses the same quantized base with the active adapter disabled for
reference log probabilities. It does not allocate a second full reference model. Reference passes
run without gradients, base parameters remain frozen, adapter state restores even after exceptions,
and controlled zero/nonzero adapter tests produce the expected KL behavior.

Audited source SHA-256 values are:

- TRL GRPO trainer:
  `425161a6e4f82ee7cc6d4d6ad3fe7e495db970289d28427f45e99368ac5e985a`;
- TRL GRPO configuration:
  `83d53640316958da75c4bb73451f9562f235f886c0cc31a3c825de172c0e17cc`;
- PEFT model implementation:
  `ea36efc37191855bb14fbb1ecd6743148aaa13350fed4ee9a8582c2b7fa29696`;
- stock generation-and-score method:
  `688cb0ed965eee96bd9a985fdd185f63f984ee81eaab7bbfec2519f21e06331b`.

The exact truncation hook derives EOS-absence flags from stock completion IDs while leaving the
stock generation, reward aggregation, KL, and loss path intact. Checkpoint handling is restricted to
steps 16, 32, and 64.

The frozen runtime configuration has SHA-256
`01515d186f2485662ea20ef0b444902bdf368a2b4a8cde335f34bfe1b9222bda`.
G1 (`beta=0.04`) and G2 (`beta=0.10`) execution hashes are
`d7023bf6705702a39dfe8d8718db264f6b2c0e2e211753145ad71e2368f4f4c0` and
`e31d814fc4bcd9fa94e6b74f48992bb79ec70bcf678e56e620277ea19dbe7bd8`.
The runtime fails closed on packet/config/source mismatch, non-finite reward, group-count mismatch,
missing reward variance, CPU offload, incorrect trainability, state-restoration failure, or RTX 3080
memory-gate failure.

## Compatibility smoke

The predeclared G1 probe selected two synthetic groups for two optimizer updates and one replay group
for a separate generation/reward/reference-only check. It requested four generations per group with
temperature `0.8`, top-p `0.95`, top-k `50`, and strict deterministic PyTorch algorithms.

The model loaded in NF4 on the RTX 3080, and a fresh LoRA adapter was created without CPU offload or
a second reference model. During the first group's first generation, Transformers'
`TopPLogitsWarper` invoked CUDA cumulative summation. PyTorch 2.5.1+cu121 raised:

`cumsum_cuda_kernel does not have a deterministic implementation, but
torch.use_deterministic_algorithms(True) is active`

Measured outcome:

| Item | Result |
| --- | ---: |
| Predeclared groups | 3 |
| Completed groups | 0 |
| Completed outputs | 0 |
| Reward calls | 0 |
| Reference-KL calls | 0 |
| Backward passes | 0 |
| Optimizer steps | 0 |
| Saved adapters | 0 |
| CPU offload | No |

No completion evidence was written; the partial ignored output contains one empty `trainer_state`
child directory and zero files. Because the
process stopped before the success-only resource summary, peak RAM and peak allocated/reserved VRAM
are unavailable. Failure-summary SHA-256 is
`8b57b6284c1e7dccd978379162de9519b7af30addbbfb9eb4d5a95a7f2b439a6`.

## Gates not reached

- G1 generic training: not run;
- G1 targeted training: not run;
- G1 retention at steps 16/32/64: not run;
- G2 generic or targeted training: not run;
- common checkpoint selection: not run;
- independent 141-item final retention: not run;
- generic GSM1K: not run;
- targeted GSM1K: not run;
- category analysis and paired bootstrap: not run;
- one-seed GRPO signal gate: not reached.

The frozen historical base remains `521/814` correct with `752/814` extractable. It was not rerun.
No new adapter or benchmark result exists, and no inference may be interpreted as a GRPO comparison.

## Safety, human review, and limitations

No generated question, synthetic dataset, prompt packet, model output, adapter, checkpoint,
prediction, model cache, secret, or sealed-final content is included in this report. The development
evaluator was not invoked and sealed-final remained untouched.

The stratified human language review is still pending at:

`file:///C:/Users/Admin/Projects/Foundry/results/raw/foundry_500x2_signal_review/codex_assisted_review.html`

Codex recommendations are advisory and do not replace genuine user decisions. Any historical
benchmark interpretation remains **Provisional one-seed result pending stratified human language
review and second-seed confirmation.**

## Exact next action

Stop and request an explicit user decision. Continuing verifier-GRPO requires a separately approved,
predeclared reconciliation between stochastic top-p sampling and the deterministic guarantees that
the installed CUDA stack can actually provide. Treating the exception as a warning, disabling
deterministic enforcement, changing decoding, sampling on CPU, or changing the dependency stack are
materially different contracts and were not attempted. No option should be adopted silently or
selected after observing rewards. The alternative is to stop the verifier-GRPO route.
