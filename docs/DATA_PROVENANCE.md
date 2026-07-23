# Phase 2 data provenance

## Status

ASDiv V1.0 is the verified primary source for the Phase 2 vetted-corpus experiment. The source is
pinned before parsing or model evaluation. MathQA is an authorized fallback only; it has not been
activated, downloaded, or inspected beyond the authorization's stated source identity.

## Primary source: ASDiv V1.0

| Field | Frozen value |
|---|---|
| Source name | Academia Sinica Diverse Math Word Problem Dataset (ASDiv) V1.0 |
| Official repository | `https://github.com/chaochun/nlu-asdiv-dataset.git` |
| Default branch at verification | `master` |
| Commit | `883f90a9a65bf00304ba8f37423910fe743abc47` |
| Git tree | `2c3e8723c68436a2a6697329edfdf7fbd44e52ac` |
| Dataset XML | `dataset/ASDiv.xml` |
| Official problem count | `2,305` |
| License | CC BY-NC 4.0 |
| Download timestamp | `2026-07-22T16:45:27.9345737Z` |
| Ignored local root | `data/external/phase2_vetted/asdiv` |

The repository README is the source's license statement; the repository does not contain a
separate license file. This project therefore treats all ASDiv use as non-commercial research use
and does not claim that the source problems or worksheets are commercially reusable.

## Official source-file identities

The canonical SHA-256 values below hash the raw bytes stored in the pinned Git blobs, before any
platform line-ending conversion. Git blob SHA-1 and byte length provide an independent identity.

| Path | Git blob SHA-1 | Bytes | Raw-blob SHA-256 |
|---|---|---:|---|
| `README.md` | `020d1d27a6a4b346855e30962a457c09def3bc8d` | 3,643 | `929249a72b2d0b6d3b144acd21886f7e728faa945b10f04b7ba2995138845783` |
| `dataset/ASDiv.xml` | `092be2256db15f64a11b2d95d958027c8e7fa432` | 947,401 | `ef8904068482919ac48c8eeaaf6df344b8a308ba66d048c2d4d87eab82dc4929` |
| `dataset/nfolds/asdiv-a/fold0.txt` | `89e87bf62615c10908cf95d5388d19af322b599e` | 2,618 | `1e4e8c670e461e9d59beee13e64e28d7f92e3be8ba6766d3bbeb40818e03e8d6` |
| `dataset/nfolds/asdiv-a/fold1.txt` | `bd94773507e22e94fd9f9048da494d8ee7ba4ef6` | 2,618 | `81bb372c89696d8d0a253d8ac73500208a801c35670112d682c2f2306ef35fd2` |
| `dataset/nfolds/asdiv-a/fold2.txt` | `483dfc9760ffccd3a25426ec19bedc1fed79c2c0` | 2,618 | `56ffc96a9eead77224f782d7c4c84205c21079b3588b08e4a68d2deabf50febc` |
| `dataset/nfolds/asdiv-a/fold3.txt` | `6b7aa933fb4eb8f3d239550d8a1c8cba1050049d` | 2,618 | `68733651260332fa6ae89155aa5e15d20a4b11833023712d0a9b2a2d06551e54` |
| `dataset/nfolds/asdiv-a/fold4.txt` | `fbf4f96a4acadc03f6a3f473512e04a290714697` | 2,926 | `103d2091eab4ee825e55353af817af006ddd2fee4948b0ef7fa1df0f6a33723b` |
| `dataset/nfolds/asdiv-w/fold0.txt` | `012cc11c2c1da25c73d6dd1cd4c5b933d6f945b9` | 4,884 | `6b00a5307b2236017cde5d39ce9e70862dab956874292492e3fa94482c454707` |
| `dataset/nfolds/asdiv-w/fold1.txt` | `e25eba93b6a5afb134cd4abe43f45225f8006e99` | 4,884 | `b6044e25fb77790fa39f9a1bc78af528436de4aeeec2505d8d5721e2f52b1182` |
| `dataset/nfolds/asdiv-w/fold2.txt` | `897d110a1a21c200f23bca8083392ea133a7120e` | 4,884 | `c6d7f167b3d8ede05bbff5e3a0f3cd9e420d104b8c18beaa35f13b8258c1e1a8` |
| `dataset/nfolds/asdiv-w/fold3.txt` | `763ad41eeb817d74f4059c75e6766269b66e4551` | 4,884 | `484cde36443fc4dc06afbf5628bc5d1ec901e55c9f0a6638d846fc620262be56` |
| `dataset/nfolds/asdiv-w/fold4.txt` | `66462ed85b0810bf50bf3e22ea6a5cd2843b26d8` | 5,819 | `e5e89a52d9d53ad9cd35587e6c41c6c3d3bf2ef1fdeb01f60c19639f193ca558` |

The ASDiv XML checkout is byte-identical to its raw Git blob. Windows Git converts line endings in
README and fold text files in the local checkout; those files are not experiment inputs. The raw
Git-blob values above remain the platform-neutral provenance identities.

## Schema and fields

Every ASDiv `Problem` has the attributes `ID`, `Grade`, and `Source`, plus `Body`, `Question`,
`Solution-Type`, `Answer`, and `Formula`. Phase 2 uses:

- `ID` as the stable source identity;
- `Grade` as a difficulty/matching covariate;
- `Source` only for provenance and duplicate/contamination screening;
- `Body` and `Question` to form the whitespace-normalized user prompt;
- `Solution-Type` and subtype attributes for deterministic family classification;
- `Answer` for an independent canonical-number and unit check;
- `Formula` for restricted exact execution and deterministic target construction.

The supplied N-fold files are recorded for source completeness but are deliberately ignored by the
experiment. No source split, model output, rationale, or external annotation outside the fields
above controls selection.

## Attribution and citation

ASDiv was constructed by the Natural Language Understanding laboratory, Institute of Information
Science, Academia Sinica. Phase 2 attributes the dataset to Shen-yun Miao, Chao-Chun Liang, and
Keh-Yih Su, *A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers*,
Proceedings of ACL 2020, pages 975-984. The official paper link is
`https://aclanthology.org/2020.acl-main.92/`.

## Storage and redistribution

Raw source files, normalized questions, answers, formulas tied to questions, selected records,
base predictions, targets, adapters, and checkpoints remain only under ignored paths:

- `data/external/phase2_vetted/asdiv`;
- `data/external/phase2_vetted/mathqa` if the fallback gate later activates;
- `results/raw/phase2_vetted_corpus`.

The Git repository may contain source URLs, revisions, licenses, attribution, hashes, stable source
IDs, content-free covariates/manifests, aggregate counts, implementation, tests, and documentation.
It must not redistribute ASDiv question or answer text.

## Contamination boundary

The frozen 904-question GSM1K development inventory is used only to reject contamination through
exact, 12-token, number-neutral template, formula/operation, semantic, and source screening. It is
not available to family classification, arm selection, matching, target construction, training, or
checkpoint selection. Phase 1 synthetic training questions are also screened only for overlap.

## MathQA fallback provenance

MathQA activated only after evaluated ASDiv base failures could not support the frozen
`200`-per-arm minimum. The source is the official `allenai/math_qa` namespace. Dataset-card
revision is `c4f1cc784c04c4957b50c97858f23893b633eea6`; the pinned train Parquet revision is
`fafb9f7ee5b9ec4da9499f9c4177a4c91389f2d6`; artifact SHA-256 is
`c16335ea4f7c9a8da44ccec52146d29e040582d2c11ca712fcfa2dd0ee964a99`. Metadata identifies
Apache-2.0 and upstream AQuA-RAT/MathQA provenance.

Only `Problem`, `options`, `correct`, `annotated_formula`, `linear_formula`, and `category` were
loaded from the official train artifact. Natural-language rationales were deliberately not loaded
or used. Validation and test were not accessed; arbitrary remote code was not executed;
`trust_remote_code` remained false. Exact program/option agreement verified `15,468/29,837` rows.
The frozen pre-inference selection retained `5,000` at SHA-256 `02fb19a8...3d45`; contamination
screening retained `4,929` at SHA-256 `93d4d250...f418`.

MathQA and ASDiv full records remain under ignored local paths. No raw question, option, answer,
formula tied to a question, prediction, selected record, or adapter is tracked. Phase 2 stopped at
matching before any dataset freeze or training.
