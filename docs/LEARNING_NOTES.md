# Foundry Learning Notes

Last updated: 2026-07-17

These notes explain the ideas in the context of Foundry's proposed first experiment. They describe the plan, not completed implementation or measured results.

## Open-weight models

An open-weight model lets us download the learned numerical weights and run the model ourselves. This is different from calling a hosted model through an API, where the provider keeps the weights and training system private. "Open-weight" also does not automatically mean that every training example, training script, or design decision is public.

For Foundry, local weights are essential because SFT and GRPO change model parameters. The proposed `Qwen/Qwen2.5-1.5B-Instruct` checkpoint is distributed under Apache-2.0 according to its [model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). Before an experiment, Foundry will record an immutable model revision rather than downloading whatever `main` points to that day.

## Parameters and VRAM

A parameter is one learned number in a neural network. A 1.54-billion-parameter model has about 1.54 billion such numbers. Parameter count matters because weights, activations, gradients, and optimizer state all consume memory.

Storing 1.54 billion weights at 16 bits takes roughly 3.08 GB before file/runtime overhead. Full fine-tuning would also keep gradients and optimizer states for essentially all of those weights, quickly exceeding a 10 or 12 GB RTX 3080. QLoRA instead stores the frozen base model in 4-bit form and trains a comparatively small adapter. The raw 4-bit weight calculation is about 0.77 GB, but that is not the total training footprint. Quantization metadata, temporary buffers, adapter and optimizer state, CUDA kernels, and especially activations still consume VRAM.

Foundry therefore treats 6–9 GB peak VRAM for a short-sequence 1.5B QLoRA run as an estimate, not a promise. The first smoke run must measure actual peak memory on the user's exact card. Sequence length, micro-batch size, gradient checkpointing, and software versions can move the result substantially.

## Quantization

Quantization stores weights with fewer bits. Moving from 16-bit weights to 4-bit weights cuts their raw storage by about four times. The arithmetic used by the GPU is not simply "4-bit training" end to end: weights are dequantized into a higher-precision compute type for matrix operations.

For Foundry, the planned QLoRA configuration uses a 4-bit format such as NF4 for the frozen Qwen base model while adapter calculations use a supported higher precision. This makes local training feasible but introduces another variable that must be pinned and tested. Quantization can affect numerical behavior, and a model evaluated in one representation must not be silently compared with a different base representation.

The [Transformers bitsandbytes documentation](https://huggingface.co/docs/transformers/quantization/bitsandbytes) describes 4-bit QLoRA loading and supported hardware. Foundry will pin known-compatible versions instead of assuming current documentation matches the installed environment.

## LoRA and QLoRA

LoRA, or Low-Rank Adaptation, freezes the original model and inserts small trainable matrices into selected layers. During inference, those matrices adjust the base model's behavior. Because only the adapters are trained, gradient and optimizer memory are far smaller than in full fine-tuning.

QLoRA combines LoRA with a quantized frozen base model. In Foundry's proposed SFT run:

1. Qwen's base weights are loaded in 4-bit form and remain frozen.
2. LoRA adapter weights are attached to selected linear layers.
3. SFT updates only those adapter weights.
4. The saved artifact is an adapter plus configuration, not a second full copy of the base model.

This is appropriate for an RTX 3080 and makes experiments cheaper to store and compare. It does not guarantee improvement; data quality, objective, prompt formatting, and hyperparameters still determine what the adapter learns. [TRL's PEFT integration](https://huggingface.co/docs/trl/peft_integration) and [PEFT's LoRA reference](https://huggingface.co/docs/peft/main/package_reference/lora) provide the intended library path, subject to version pinning and smoke tests.

## Supervised fine-tuning

Supervised fine-tuning teaches a model to produce a desired response for an input. Each Foundry arithmetic example would contain a new word problem plus a verified solution and final answer. The training loss rewards the model for assigning higher probability to the verified target tokens.

SFT is the right first intervention because it directly tests the project's main premise: can failures tell us what verified examples to create, and can those examples improve held-out behavior? If SFT cannot produce a controlled improvement, adding reinforcement learning would make debugging harder rather than fix the underlying evidence problem.

Foundry will not use GSM1K examples or answers as SFT targets. The benchmark is for measurement. SFT data will be generated from independent structured arithmetic programs aimed at aggregate failure categories.

## Synthetic data generation

Synthetic data is created by software or a model rather than copied from a human-authored corpus. Synthetic does not mean automatically correct.

For Foundry, a safe first generator starts with an executable specification such as:

- entities and units;
- integer or rational variables with allowed ranges;
- a sequence of arithmetic operations;
- constraints that guarantee a valid positive-integer answer;
- a language template that renders the specification as a word problem.

If the base model struggles with rates, the generator samples new rate programs and renders new stories. It does not take a failed benchmark question and swap names or numbers. The distinction matters: close rewrites would teach the test rather than the underlying skill.

Template generation is less linguistically diverse than API-based paraphrasing, but it is cheap, reproducible, and easier to verify. A future paraphrasing stage would need separate approval and would remain behind the same correctness and overlap filters.

## Automatic data verification

Automatic verification decides whether a generated example is safe enough to train on. A single model saying "looks correct" is not strong verification.

The proposed arithmetic verifier uses several independent checks:

1. Recompute the expected result from the structured specification using exact arithmetic, not floating-point approximations.
2. Independently validate every constraint, intermediate value, unit conversion, and final positive-integer requirement.
3. Confirm the rendered problem exposes all required quantities and does not accidentally reveal or contradict the answer.
4. Require the generated rationale's intermediate calculations and final answer to agree with the independent solver.
5. Reject malformed, ambiguous, duplicate, or suspiciously similar examples.
6. Record an explicit acceptance or rejection reason for audit.

Verification reduces label noise; it cannot prove that every natural-language problem has only one reasonable interpretation. Foundry should reject uncertain examples instead of maximizing dataset size.

## Hard-negative mining

A hard negative is an example that looks plausible to the model but exposes a specific mistake. In arithmetic, a model may consistently add before applying a percentage, mishandle "twice as many," or convert feet to inches in the wrong direction.

Foundry's first SFT loop will use hard-negative mining as a targeting idea, not by training incorrect answers as correct. A failure category causes the generator to produce more valid examples near that decision boundary, including carefully chosen distractors and operation orders. The verified correct solution remains the SFT label.

Later preference or GRPO work could compare a correct solution against a plausible wrong one, but only if the reward and data format are explicitly designed and approved.

## Benchmark contamination

Benchmark contamination occurs when evaluation questions or close variants appear in model training data. A contaminated model may score well because it remembers examples, not because it learned the intended capability.

This is particularly important to Foundry because its loop deliberately reacts to benchmark failures. If generated training examples are close rewrites of the benchmark, the system can optimize the score without becoming generally better. The GSM1K paper was created to examine overfitting on the older GSM8K benchmark, but GSM1K's current public availability means it is not permanently immune to future contamination.

Foundry's safeguards are:

- pin the model and dataset revisions;
- never train on benchmark examples or labels;
- keep benchmark examples out of the synthesis input;
- generate from independent program families;
- reject exact normalized overlap;
- reject high lexical and semantic similarity;
- group related synthetic templates before train/validation splitting;
- keep a final benchmark portion sealed from all tuning decisions.

An overlap checker may inspect benchmark prompts only to reject a generated example. It must not return those prompts to the generator.

## Train, development, and test splits

The three splits answer different questions:

- **Training data** updates the LoRA adapter. In Phase 1 it consists only of independently generated and verified examples.
- **Development data** guides engineering decisions. Foundry evaluates base and candidate models on this benchmark portion, studies failure categories, and chooses what to improve. It is never used as SFT text, but repeated decisions can still overfit to it.
- **Final test data** is sealed until the candidate and evaluation settings are frozen. It answers whether the improvement transfers to examples that did not guide development.

The GSM1K public dataset presents one test collection. Foundry proposes creating a deterministic development/final partition before any model result is seen. The split seed or stable-hash rule and example IDs will be recorded. Exact counts will be chosen only after the pinned revision is inspected.

Synthetic data needs its own train/validation split. Closely related variants must be grouped by generator template or program family before splitting; otherwise a validation example can be almost identical to training and give a misleadingly optimistic signal.

## Evaluation methodology

A fair base-versus-adapter comparison changes one main variable: the adapter. Both models must use:

- the same pinned base weights and quantization path;
- the same benchmark revision and example IDs;
- the same system/user prompt and chat template;
- the same deterministic decoding and maximum output length;
- the same final-answer parser and exact-match normalization;
- the same evaluator code and preferably the same hardware/software environment.

Every per-example prediction is stored before aggregate scoring. This makes parser errors, invalid outputs, wins, losses, and category regressions auditable. The final report must state sample counts and uncertainty, not only a rounded percentage.

A score seen during development is not the final proof. The primary claim comes from a paired comparison on the sealed final split after the adapter and checkpoint rule are frozen.

## Reward functions

A reward function converts model behavior into a number used by reinforcement learning. For a later arithmetic GRPO experiment, the simplest reward might grant 1 for an exact verified final answer and 0 otherwise. Additional rewards could target valid answer formatting or internally consistent executable steps.

The reward defines what the optimizer will pursue, not necessarily what the developer intended. Before GRPO, Foundry must state which behavior is desired and demonstrate that the reward measures it. If SFT is still improving, a more complex optimizer is not justified.

## Reward hacking

Reward hacking happens when the model earns reward through a shortcut that violates the spirit of the task. Arithmetic examples include:

- exploiting a permissive parser so any output is interpreted as the expected number;
- printing many numbers and hoping one is selected;
- copying a number exposed accidentally in tool output or metadata;
- producing an exact final answer with nonsensical reasoning if the reward claims to value reasoning;
- learning quirks of generated templates rather than general arithmetic.

Foundry would test the reward with adversarial outputs, strict single-answer parsing, hidden verifier cases, length limits, and separate final evaluation. Reward code must be unit-tested independently from training. A reward that cannot survive these tests is not ready for GRPO.

## GRPO

Group Relative Policy Optimization samples a group of responses for the same prompt, scores each response, and increases the relative probability of higher-reward responses. It avoids training a separate value model in the way some other reinforcement-learning methods require, but it still needs multiple generations and backpropagation, making it substantially more expensive than one SFT target per prompt.

For Foundry, GRPO is useful only if:

1. SFT has already produced a reproducible gain.
2. SFT then plateaus on a behavior that benefits from outcome exploration rather than more verified demonstrations.
3. The reward is objective, tested, and difficult to exploit.
4. The RTX 3080 memory/runtime estimate and any cloud cost are approved.
5. A predeclared result threshold justifies the added complexity.

Exact arithmetic answers offer a possible future reward, but they do not automatically validate reasoning quality. GRPO should not be added for branding or novelty.

## Statistical uncertainty and reproducibility

If a model gets 620 of 1,000 examples correct, 62% is an estimate from a finite sample, not the model's exact universal ability. A few examples changing outcome can move the score by several points. Foundry will report a confidence interval and use paired analysis because base and candidate answer the same examples.

The paired view is important. We need to know how many failures were fixed and how many previously correct answers regressed. A bootstrap over example-level paired outcomes estimates uncertainty in the improvement; McNemar's test provides another check on asymmetric wins and losses.

Reproducibility also includes training randomness. Data order, dropout, CUDA kernels, and adapter initialization can change a result. Foundry will record all seeds and run at least two approved SFT seeds before claiming the training procedure is reproducible. Complete bit-for-bit determinism may reduce performance or remain unavailable on some CUDA operations, so any nondeterminism will be documented rather than hidden.

Each experiment record will pin model and dataset commits, code commit, package versions, configuration, prompt hash, split manifest, hardware, seeds, runtime, cost, outputs, and checkpoint. Estimates and measured values will always be labeled separately.

## Identifier-only manifests

A manifest is a list that says which benchmark records belong to a run. Foundry's GSM1K manifests deliberately do not contain the questions or answers. Each entry contains:

- the row position inside one exact dataset revision; and
- a SHA-256 identifier derived from the dataset name, immutable revision, configuration, split, and row position.

The partition is created by hashing every identity together with the fixed seed `foundry-gsm1k-v1`, ranking those hashes, and assigning exactly 301 rows to the sealed-final portion. The other 904 rows become development data. Running the algorithm again produces byte-for-byte equivalent semantic manifests.

Each manifest also includes its own digest and the evaluation-config digest. Loading code recomputes those values and rejects edited, overlapping, incomplete, or wrong-revision files. This is not encryption: somebody can still see row positions. The purpose is reproducibility and accidental-use prevention, not hiding a public benchmark.

The sealed-final loader requires an explicit override. The real-model smoke command refuses the final manifest entirely. This guard cannot replace project discipline, but it makes accidental final-set evaluation harder.

## Why Foundry uses a strict final-answer contract

Math responses often contain many intermediate numbers. A parser that simply takes the last number can score the wrong thing and is easy to exploit. Foundry's prompt therefore requires one exact last line:

```text
Final answer: <integer>
```

The parser accepts signs, valid thousands separators, integral decimal spellings such as `42.0`, and a LaTeX boxed integer. It rejects `42.5`, malformed commas, appended units, multiple final-answer lines, or commentary after the answer. Earlier reasoning numbers are ignored.

This means a mathematically correct answer in the wrong format can be marked invalid. Foundry reports that invalid rate separately. Keeping the rule narrow is preferable to silently guessing what the model meant, especially when the same parser will compare the base and fine-tuned models.

## Dependency locks and machine portability

`pyproject.toml` describes the direct project dependencies and their exact versions. The lock files record the transitive versions selected by `pip-compile` under Python 3.12. Together they prevent a later install from silently pulling a different YAML parser, test runner, Transformers release, or dataset library.

The current machine is an Apple M2 Mac, not the intended NVIDIA desktop. A CUDA-enabled PyTorch wheel is hardware-specific, so the Mac could validate the framework and lock resolution but could not validate CUDA execution. On the RTX machine, PyTorch 2.5.1 must first come from PyTorch's official CUDA 12.1 wheel index. The remaining exact smoke dependencies then come from `requirements-smoke.lock.txt`.

A lock narrows software variation; it does not prove hardware compatibility. The RTX run must still record the GPU, VRAM, driver, CUDA runtime, peak allocation, and any out-of-memory failure.

## What the RTX 3080 smoke measured

The deferred Milestone 1 smoke ran on 2026-07-16 using Windows 11 Pro, CPython 3.12.10, PyTorch 2.5.1 with its CUDA 12.1 runtime, and an NVIDIA GeForce RTX 3080 with 10,240 MiB VRAM. The installed NVIDIA driver reported CUDA UMD 13.3, but that did not require a CUDA 13.3 PyTorch package: the approved CUDA 12.1 wheel loaded successfully and `torch.cuda.is_available()` returned true. A sufficiently new NVIDIA driver can run applications built against an older compatible CUDA runtime.

Float16 inference for the pinned 1.5B Qwen model peaked at 2,972.14 MiB allocated and 3,162 MiB reserved. This proves that the evaluation path fits comfortably on this card. It is encouraging for the proposed short-sequence QLoRA pilot because roughly 6.9 GiB remained outside PyTorch's peak reservation, but it is not a training-memory measurement. QLoRA adds quantization metadata, adapters, gradients, optimizer state, activations, and training kernels. A separately approved tiny training smoke is still required before claiming that the complete QLoRA configuration fits 10 GiB.

The ten-example run produced 2 exact correct answers, one validly parsed incorrect answer, and seven invalid-format outputs. All seven invalid outputs failed to include exactly one required `Final answer:` line. This is useful software and behavior evidence: CUDA generation, token accounting, scoring, and failure recording worked, while the fixed general-purpose model followed the strict response contract inconsistently. The 20% accuracy and 70% invalid rate from only ten examples are not reliable estimates of full-development performance and must not be treated as a benchmark conclusion.

The first-run model download/local cache occupied about 2.886 GiB, while the pinned GSM1K Hub and materialized dataset caches totaled about 0.74 MiB. Windows could not use Hugging Face's preferred cache symlinks, so the cache may consume more space than an equivalent Developer Mode or administrator-enabled setup. Foundry did not change Windows security or developer settings to remove that warning.

## Prompt-format calibration is separate from benchmark measurement

The RTX smoke showed that many model responses contained a plausible terminal answer but did not use Foundry's exact `Final answer: <integer>` line. Inspection confirmed the instruction survived Qwen's chat template and the parser accepted a compliant sanitized response. The seven original invalids were therefore model-format failures, not hidden prompt truncation or a parser implementation bug.

Changing a prompt after seeing development behavior is a form of tuning. Foundry reserved 30 development identifiers solely for prompt-format calibration and removed them from the 874 identifiers that could later report the main development baseline. This prevents the headline baseline from including the same examples used to choose its format instructions. Both subsets are deterministic, identifier-only, and tied to the canonical development-manifest digest.

Three greedy prompt runs illustrated why an absolute admission rule matters. The current prompt produced 16.67% valid outputs, a minimal wording revision produced 10.00%, and a stronger explicit contract produced 43.33%. The strongest prompt was better relative to the others and shorter on average, but a majority of its responses were still invalid. Selecting it merely because it was best would convert a failed calibration into a post-hoc success.

Strict parsing and prompt compliance answer different questions. The parser asks whether output satisfies a predeclared machine-readable contract. The prompt asks the model to produce that contract. Accepting every boxed, prose, unit, currency, or Markdown ending would raise the apparent valid rate by changing the scorer rather than improving compliance. No parser change was justified by this diagnosis.

Format compliance also differs from mathematical accuracy. Calibration recorded accuracy only to detect gross anomalies; it did not choose prompts based on correct-answer count. A future format-control proposal must reach at least 90% validity with no generation failures and reasonable output length before the 874-ID main baseline can begin.

## Deterministic answer extraction needs its own held-out validation

Exact-format compliance and mathematical answer extraction answer different questions. Compliance asks whether the model obeyed the literal `Final answer:` contract. Canonical extraction asks whether a deterministic grammar can identify one clear terminal integer without using an LLM judge, question-specific logic, or arbitrary last-number guessing. Foundry now reports both instead of allowing format failures to hide mathematical answers.

The calibrated extractor accepts explicit answer labels, standalone boxed or bold conclusions, terminal answer/result sentences, valid currency/sign/comma forms, and integral decimals. It rejects conflicts, malformed comma grouping, non-integral decimals, missing conclusions, and generations that reach the configured token limit. Its behavior is versioned and hashed, and the original strict parser remains unchanged.

Re-scoring existing predictions is not enough to validate an extractor because grammar rules were refined after observing those outputs. The original 90-output re-score looked strong—up to 96.67% extractable—and manual review found no false extractions. A separate 30-ID set, never used to design the grammar, fell to 76.67%. That gap is evidence of calibration overfit or insufficient coverage, even though the extractor remained conservative and made no audited false extraction.

The fresh result also separates three kinds of failure: model verbosity caused three token-limit rejections, two model answers were non-integral and therefore outside the benchmark's integer contract, and two clear integer answers used unsupported phrasing. Fixing only the grammar cannot reach the 90% gate because the best post-hoc upper bound is 25/30, or 83.33%. A trustworthy next attempt needs predeclared extraction and generation changes plus a new untouched admission set; the observed validation set cannot be reused as if it were still held out.
