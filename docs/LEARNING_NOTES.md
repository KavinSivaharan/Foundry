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

## Exact answer normalization can be safe and still miss the coverage gate

Milestone 1.7 changed the canonical path from integer-only extraction to exact terminal-number extraction without changing the strict compliance parser. `Decimal` and `Fraction` normalization lets the evaluator represent `42.5`, `3/2`, or `\frac{3}{2}` exactly. That matters even for an integer-answer benchmark: a clear non-integral answer is extractable but wrong, while an absent or ambiguous answer is unextractable. Conflating those outcomes hides model behavior and distorts coverage.

The admitted grammar requires explicit terminal structure. It supports signed and comma-grouped values, integral/non-integral decimals, ASCII and LaTeX fractions, currency and percentage wrappers, units, boxed/bold/inline-LaTeX/text forms, constrained conclusion sentences, assignments, and terminal equations. It still rejects malformed grouping, zero-denominator fractions, unfinished expressions, conflicting answer cues or `and`/`or` candidate lists, arbitrary intermediate values, missing conclusions, and any response marked truncated. The strict parser continues to ask the narrower question: did the model produce exactly one literal final line?

False-extraction risk grows fastest in free-form prose. A terminal sentence may repeat constraints, times, prices, or intermediate quantities after its intended answer. Foundry therefore does not generically take the first or last number. Wrapper rules require conclusion/answer cues plus syntactic structure, and global strong candidates are checked for conflict. All four newly accepted old-validation outputs and all 25 accepted final-validation outputs matched the model's clear terminal intent in manual review; no false extraction was found.

Increasing `max_new_tokens` from 512 to 768 resolved two of three known truncations, but one response still filled 768 tokens. The final untouched set also contained one 768-token truncation. A larger budget might reduce invalids, but it also increases runtime and output length and was outside the approved bound. The final run averaged 315.03 output tokens and took 125.10 evaluation seconds for 30 examples, so 768 was computationally reasonable even though it was not universally sufficient.

Most importantly, safe local improvements did not generalize to the admission threshold. The observed old-set re-score reached exactly 90%, while the last untouched set reached 83.33% with four clear false rejections and zero false acceptances. Adding those four wrappers afterward would make the same set look successful but would no longer be validation. The scientifically honest stop is to retain the evidence and ask whether to score unextractable outputs as wrong with coverage reported separately, or to reconsider the model/benchmark—not to keep tuning the evaluator against successive held-out sets.

## A larger run can overturn a small calibration premise

Milestone 2 used the extractor exactly as frozen and produced a reproducible 64.00% end-to-end development score. The separate metrics matter: 92.38% of outputs were classified extractable, 69.28% of those extracted answers were correct, and only 15.97% followed the literal required line. Counting unextractable answers wrong makes the end-to-end numerator unambiguous under the frozen evaluator, but it does not prove that every accepted value matches what the model intended.

A deterministic manual audit of 100 extractable-but-wrong records found two false acceptances. One terminal currency conclusion was preceded by a percentage that the extractor selected instead; another response expressed a loss whose sign was not preserved. Both still compared unequal to the benchmark target, so neither altered the 521-correct count. Even so, they disprove the stronger assumption that every remaining extraction error is a conservative rejection. The audit did not inspect the 521 records scored correct, so accidental matches cannot be ruled out.

This is why evaluator precision and benchmark accuracy need different evidence. A calibration audit can demonstrate precision on the outputs it reviewed, but a larger and more varied generation population can expose syntax interactions that were absent from calibration. Freezing prevents post-hoc score manipulation; it does not make the frozen component infallible. The correct response is to keep the exact result, disclose the limitation, and obtain explicit approval before either auditing the scored-correct population or using the taxonomy to design synthetic data.

The provisional 100-record taxonomy is useful but not exhaustive. Primary categories were multi-step bookkeeping or omission (28), target/language interpretation (18), constraint/distribution/discrete reasoning (15), time/unit/sequence reasoning (14), arithmetic execution (12), rate/ratio/percentage/average reasoning (12), and one benchmark-ambiguity or annotation-risk case. Some failures have plausible secondary causes, and the sample excludes the 62 unextractable outputs, so these counts describe the sampled extractable-wrong population rather than all model failures.

## Label-blind auditing separates model intent from score agreement

Milestone 2.1 audited every one of the 521 responses scored correct without showing the benchmark question or reference answer. The audit view showed only the completion, extracted value, exact source span, extraction rule, strict/canonical status, terminal context, completion metadata, token count, and generic suspicion flags. A reviewer therefore had to answer, “Is this clearly the value the model intended as its terminal answer?” rather than, “Did this value match the label?”

The order of operations is an important research control. All 521 intent decisions were completed first, then normalized and frozen with SHA-256 `669a866e984c35908bdb9e5443cb989733fd762d11bf62456387a25a5c12e14c`, and only then joined to the existing scoring records. That boundary prevents a correct label match from becoming evidence for intent and makes later classification changes detectable.

The full audit found 521 clearly intended extracted answers, zero false-positive correct answers, and zero ambiguous cases. Ninety correct responses satisfied the strict parser; 431 were accepted only by the canonical extractor. All 41 terminal contexts containing multiple numeric values used an explicit conclusion that distinguished the extracted answer from supporting calculations. Four responses used negative-intent words such as “lost” or “short,” but each asked for and stated a positive magnitude; none repeated the earlier lost-sign failure. The earlier percentage-plus-currency collision did not occur among correct-scored responses.

This makes the frozen 521/814 score trustworthy as a development baseline: the audited lower bound, upper bound, and exact adjusted accuracy all remain 64.0049%. It does not mean the extractor is perfect in general. The two false acceptances among sampled wrong responses remain real, the 62 unextractable responses still reflect conservative rejection, and the failure taxonomy still covers only a deterministic sample. The practical lesson is narrower: no accidental extractor match inflated the correct numerator, so the existing development taxonomy is reliable enough to inform a separately approved targeted synthetic-data design.

## Exhaustive failure review changes the curriculum decision

Milestone 3 expanded the provisional 100-record inventory to every one of the 293 development
failures. The complete counts were 69 output-format/extraction, 68 bookkeeping/omission, 53
target/language interpretation, 28 rate/ratio/percentage/average, 27 constraint/discrete, 24
time/unit/sequence, 22 arithmetic execution, and two benchmark-risk cases. Secondary tags overlap,
which is useful: a time problem can fail because one update was omitted, and a ratio problem can
also have a target-interpretation error. One primary label supports aggregate planning without
pretending root causes are always exclusive.

The most frequent reasoning category is not automatically the best first generator target.
Target/language interpretation was the second-largest reasoning group, but a template can silently
change which quantity is requested. That creates ambiguous labels and benchmark-copy risk even
when the arithmetic program is sound. Rate/ratio and discrete constraints were selected instead
because they support exact equations, inverse checks, and bounded enumeration. Prevalence matters,
but label trust and independent verification matter more.

Seven wrong-output false extractions were found across the full failure review, including the two
already known. They all remained scored wrong and therefore do not alter the audited correct
numerator. Their broader lesson is that output-contract data should be a separate shared track,
not folded into a reasoning category or used to change the frozen evaluator after results are
known.

## Executable structure is safer than synthetic prose as the source of truth

The first generator design starts with a latent exact program, computes its result, renders a
controlled question, and then checks the answer through a different method. This order is the
opposite of asking a model to write a question and trusting its label. The natural-language text
is a view of the program; it is never the authority.

Calling the same solver twice is not independent verification. The bookkeeping track pairs DAG
execution with a state ledger and conservation checks; rate/ratio pairs exact equation evaluation
with cross-multiplication or inverse substitution; discrete constraints pair constructive solving
with bounded brute force. Any disagreement rejects the candidate instead of using an LLM judge.
Exact `Fraction`-style normalization also avoids binary floating-point label drift.

Contamination is structural as well as lexical. Changing names and numbers does not make a copied
problem independent, so screening must compare normalized text, number-neutral templates, latent
program structure, n-grams, and semantic similarity. A missing semantic result cannot be treated
as permission to accept. Milestone 3 freezes thresholds but intentionally leaves the local encoder
unpinned; selecting and pinning that artifact is a necessary future decision, not a reason to
download one during a design-only milestone.

Finally, targeted data needs a matched generic control. Equal example counts, token budgets,
difficulty ranges, output-format examples, verifier rules, and training steps isolate the one
scientific variable Foundry cares about: whether choosing data from measured failures is better
than generating broad arithmetic practice. Without that control, an improvement could come from
additional training tokens rather than the failure-targeting loop.

## Exact labels do not rescue weak rendered training examples

Milestone 4 confirmed that independent exact verification can work while the overall data pipeline
still fails. All 120 candidates had agreeing verifier evidence and the manual audit found zero false
labels, yet only 24 passed the frozen contamination gates. Limited controlled-template diversity
caused 25 number-neutral template rejections and 50 five-token-overlap rejections. Fourteen more
generated-to-generated near matches were conservatively rejected after manual review.

The audit also exposed a separate source-of-truth boundary. The bookkeeping arithmetic programs
were correct, but four accepted questions named one inventory object initially and different
objects in later updates without establishing that the number represented a common total. A human
can infer the intended addition, but the prose does not faithfully establish the latent state.
One accepted discrete-capacity item also had singular/plural and tied-constraint difficulty defects.
Those are invalid training examples even though their numeric labels are right.

This is why a generator readiness gate needs both label correctness and rendered-example validity.
The semantic encoder behaved acceptably and the contamination pipeline rejected repetition rather
than hiding it. The right response is not to lower thresholds; it is to repair rendering semantics,
grammar, constraint construction, and template diversity, then test a fresh fixed sample. Full
generation remains blocked until that new evidence passes the original gates.

## Typed metadata improves safety, but controlled prose still needs adversarial auditing

Milestone 4.1 showed a large, real improvement: one explicit object family and unit now flows
through every bookkeeping ledger, transfers name compatible endpoints, discrete capacities are
untied, all 120 number-neutral templates are distinct, and acceptance rose from 24/120 to 86/120
without changing contamination thresholds. Exact labels and independent verifiers remained
correct for every candidate.

The fresh audit also showed why typed metadata is not a complete grammar model. Eleven accepted
questions still contained defects the deterministic rules did not represent: plural nouns used as
awkward attributive modifiers, a grouping phrase with incorrect noun form, a duplicated weighted
average group, a conclusion asking for a count when the latent target was a mean, a rate missing
its explicit interval denominator, an elided discrete object noun, and the irregular plural
`shelfs`. All were understandable to a human, but training-data admission requires faithful,
natural text rather than merely recoverable intent.

The negative result is informative. Template diversity eliminated exact number-neutral and latent
structure rejections, but semantic screening still identified 28 same-scenario/same-family near
matches and five-token screening found six more. A finite renderer inventory can therefore satisfy
hash diversity while remaining semantically repetitive. The project must keep content quality,
contamination diversity, and exact label verification as separate gates; success on one cannot
substitute for another.

## A typed grammar can preserve semantics without providing natural or scalable language

Milestone 4.2 made the semantic boundary substantially stronger. Generators now emit typed problem
IR, and a centralized compiler records explicit noun morphology, answer-target kinds, rate
denominators, and a one-to-one map from semantic nodes to clauses. That architecture rejected all
eleven prior defect classes and produced 900/900 internally valid renders with no false labels.

The stress audit exposed the remaining distinction: internal well-formedness is not the same as
natural English. Normalizing every request into a question mark turned imperative requests such as
`Determine ...` into unnatural direct questions. More importantly, distinct render signatures did
not imply diverse language: 99 number-neutral templates repeated and nearly every render's closest
generated neighbor exceeded the semantic rejection threshold. A finite rule system can multiply
choice metadata faster than it creates meaningfully different surface language.

The evidence supports an architectural boundary rather than more patches. Exact procedural programs
and dual verifiers remain the trustworthy label source. A future surface-realization model, if
approved, should receive only typed synthetic semantics, produce wording rather than labels, and be
accepted only after deterministic round-trip recovery of every semantic node, target, quantity,
unit, and relation. The existing contamination scanner then remains the final diversity firewall.

## Fluent realization can be useful only when it is value-blind and reject-only

Milestone 5A turns the architectural boundary into a concrete contract. The local model sees typed
roles and immutable placeholder tokens, not names, numbers, units, or answers. It proposes a JSON
template and a clause-to-semantic-node map. Deterministic code checks exact slot inventory,
occurrence counts, node coverage, target/intent equality, rate denominators, and discourse order
before inserting real values. The original exact program and independent verifier still determine
the label. A model failure can therefore reduce yield, but it cannot legitimately establish
correctness.

Round-trip validation is a stack, not a second request to the same model. Schema, slots, nodes,
targets, units, filled surfaces, exact execution, independent execution, answer contracts,
language rules, contamination checks, and answer-blind human audit each cover a different failure
mode. An LLM reverse parse may add a conservative rejection signal later, but asking a model to
approve its own wording would not be independent evidence.

Semantic similarity also needs role-specific interpretation. MiniLM's sentence embeddings capture
broad meaning and topic; they do not encode exact latent-program identity. A same-skill pair can be
close yet independently generated, while a benchmark paraphrase remains dangerous. Foundry will
therefore retain strict 0.75/0.82 MiniLM screening against development questions but calibrate a
separate generated-peer semantic policy on original fixtures before any new realization smoke.

## A reject-only LLM boundary can protect labels while still producing zero usable data

Milestone 5B validated the safety argument and falsified the yield assumption at the same time. The
pinned Qwen3 runtime never saw values, answers, or benchmark questions; procedural execution and an
independent verifier still agreed on every label; and the deterministic validator rejected all 360
beams. That produced zero false labels and zero invalid acceptances—but also zero clean IRs.

The verbose output contract created two coupled problems. First, the model often spent its budget
copying the placeholder inventory and clause map: 160 of 179 unparsed outputs reached 256 tokens
before closing valid JSON. Second, even valid JSON usually reduced the task to the target question,
omitting the events it was supposed to express. Some alternatives preserved all placeholders by
echoing imperative instructions in run-on or all-caps form, which is structurally faithful but not
natural training text.

This separates three notions that are easy to conflate:

1. **Label safety:** exact programs and independent verifiers prevent an LLM from inventing labels.
2. **Semantic safety:** strict slots and node coverage prevent incomplete wording from being
   accepted.
3. **Practical yield:** enough outputs must still be natural and complete to justify dataset-scale
   generation.

Milestone 5B passed the first two only because the second operated conservatively; it failed the
third completely. Exact replay proves the failure is reproducible, not random. The proper next move
is not to loosen coverage or repair model text after generation. A future design would need a much
shorter declarative protocol whose semantic coverage is derived rather than redundantly echoed, and
that design must be frozen on original fixtures before another model run.

## Removing metadata does not give a small model compositional syntax

Milestone 5C tested the shortest safe version of the same idea. Qwen returned only ordered event
tags, immutable data placeholders, and semantic-anchor tokens. Deterministic code retained every
mapping, phrase replacement, target, label, verifier, and screen. This solved the serialization
problem: all 90 beams parsed as tags, most completed far below the 384-token cap, and 87 preserved
the complete immutable-token assignment.

The result still had zero usable yield. Qwen mostly copied each supplied token list, then placed the
semantic anchor at the end. A human could see which tokens had been supplied, but the output did not
grammatically bind a quantity to an entity, an origin to a destination, or a target to its question.
Token presence is therefore weaker than semantic preservation: predicate-argument structure is a
required part of meaning, not optional style.

This falsifies the hypothesis that prompt compression alone can make Qwen3-1.7B a reliable surface
compiler. The safe validator again converted model weakness into rejection rather than label
corruption, and exact replay shows the result is stable. Further Qwen3 prompt edits would tune the
interface against observed failures without evidence that model capacity is adequate. The cleanest
next experiment, if approved, is to hold the compact protocol fixed and test one stronger local
model; otherwise the project should stop this realization route.

## Model capacity did not rescue the frozen compact protocol

Qwen3-4B-Instruct-2507 had enough memory headroom for the exact three-beam experiment, but more
parameters did not produce a reliable compiler. Relative to Qwen3-1.7B, tag parsing fell from 90/90
to 71/90, complete placeholder preservation fell from 87/90 to 47/90, and clean yield stayed 0/30.
The larger model often substituted invented values or nouns for opaque placeholders; beams that did
retain tokens still placed predicates after argument lists instead of composing natural clauses.

This controlled negative result distinguishes model-capability failure from prompt-tuning failure:
the IRs, prompt, decoding, validators, and gates were unchanged. The deterministic safety stack again
did its job by rejecting every invalid surface without corrupting labels. The appropriate next
architecture is not another live realizer. It is a finite, offline template bank whose language is
manually vetted before deterministic slot filling, while procedural mathematics and dual verification
remain the source of truth.

## A finite bank still needs a language compiler, not metadata interpolation

Milestone 6A confirms that removing live generation eliminates model drift without automatically
solving natural-language quality. Exact typed IR, stable plans, and perfect verifier agreement gave
118 automatic passes, yet complete Codex inspection found 13 invalid or unnatural surfaces. The
common cause was treating internal frame labels as if they were already natural noun phrases and
forming ordinals mechanically.

The lesson is that template review must cover composed outputs, not only isolated sentence plans or
typed slots. A deterministic validator can prove coverage, units, target identity, and punctuation,
but it cannot infer that “record record,” “1th,” or a malformed compound sounds wrong unless those
properties are represented explicitly. Human review remains a separate scientific gate; Codex
inspection is useful pre-screening but must not be mislabeled as human vetting.

## Surface provenance turns composition assumptions into testable invariants

Milestone 6B fixed classes rather than strings. Internal frame and role IDs now stay in metadata;
English comes only from approved surface lexemes. A noun phrase has one typed head, optional typed
modifiers, and explicit morphology. Numeric ordinals use the English 11/12/13 exception rather than
blindly appending `th`. Each emitted token is attributed to grammar, an approved lexeme, an entity,
a quantity, a unit, morphology, punctuation, or approved context, while each semantic node must be
realized exactly once.

The first 2,320-render sweep showed why broad expansion matters: it found repeated event clauses and
an over-broad identifier substring rule that a small happy-path test missed. Sequence-aware event
phrasing and word-boundary matching resolved both across the bank. The final sweep passed 2,320/2,320
and the fresh smoke safely rejected four duplicates while accepting 116 candidates.

The remaining distinction is crucial: deterministic validity is not human naturalness. Codex
inspection can pre-screen synthetic wording, but only the user's packet decisions can change the
review state from pending. A high automatic yield is evidence that review is worth doing, not a
substitute for it.

## Human review can pass language repair while runtime diversity still blocks the packet

The genuine Milestone 6B review rejected half the packet even though every question had passed typed
composition checks. That evidence was most useful at the sentence-plan level: direct relationship
and direct constraint wording survived, while update-log, register, transfer-record, vague ratio,
and workflow-like constructions repeatedly failed. Replacing those families and expanding every plan
across ten fixtures produced a clean static language result.

The v3 smoke exposed a different failure mode. It had no mathematical, grammar, target, verifier, or
contamination errors, yet 15 questions were conservatively rejected because their filled wording
collided after number normalization. Static uniqueness of plan metadata is therefore not sufficient
to predict runtime surface diversity when the same scenario vocabulary and sentence structures can
coincide. The right response is to preserve the negative result and require separate approval for a
selection/diversity change—not rerun seeds until one crosses the gate or weaken duplicate screening.

## A better allocator cannot manufacture missing language capacity

The full-capacity audit separates a scheduling defect from a finite vocabulary defect. Fifteen v3
collisions did show allocator imbalance: the same sentence-plan identity was reused across frame
metadata. The latent collision likewise came from the seed schedule. Those observations explain the
small smoke, but fixing them would not make the bank ready for 8,000 examples.

The planned targeted and generic datasets require 10,003 fixed attempts after applying the 125%
budget inside each stratum. Only 232 active plan signatures exist. Counting domain choices expands
that to 3,728 signatures, but the unchanged number-neutral filter sees only 1,176 distinct surfaces:
768 bookkeeping, 88 rate/ratio, and 320 discrete. Numeric values do not enlarge those pools because
the control intentionally removes numbers before comparison.

Capacity must therefore be measured before allocator implementation. The next scientifically valid
step is not a cleverer permutation or a weaker screen; it is independent language-bank expansion
that adds genuinely different sentence plans, lexical scenarios, and structures. Those additions
must be human reviewed, then re-enumerated against the same per-category, difficulty, output-track,
and cross-dataset uniqueness contract before any smoke or full schedule is frozen.
