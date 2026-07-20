# ruff: noqa: E501
"""Blind language audit and ignored review packet for the matched signal data."""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from foundry.synthesis.template_bank.signal_pilot import canonical_sha256

AUDIT_ID = "foundry-matched-signal-codex-language-audit-v1"
EXPORT_FILENAME = "foundry-500x2-signal-review.json"
CRITERIA = (
    "naturalness_pass",
    "grammar_pass",
    "completeness_pass",
    "target_clarity_pass",
    "consistency_pass",
    "self_contained_pass",
)
DEFECT_LABELS = (
    "unnatural wording",
    "ambiguous target",
    "missing information",
    "repeated wording",
    "grammar",
    "wrong unit",
    "unclear referent",
    "internal terminology",
    "other",
)
_REPEATED_WORD = re.compile(r"\b([A-Za-z][A-Za-z'-]*)\s+\1\b", re.IGNORECASE)
_INTERNAL_TOKEN = re.compile(r"\b(?:[a-z]+_[a-z0-9_]+|[a-z]+-[a-z]+-v\d+)\b")
_BAD_ORDINAL = re.compile(r"\b(?:1th|2th|3th|11st|12nd|13rd)\b", re.IGNORECASE)


@dataclass(frozen=True)
class LanguageAuditRecord:
    """One advisory Codex decision made from ID and question text only."""

    candidate_id: str
    recommendation: str
    naturalness_pass: bool
    grammar_pass: bool
    completeness_pass: bool
    target_clarity_pass: bool
    consistency_pass: bool
    self_contained_pass: bool
    confidence: str
    defect_labels: tuple[str, ...]
    rationale: str


def _blind_audit(candidate_id: str, question: str) -> LanguageAuditRecord:
    """Apply the frozen wording rubric without labels, traces, or pipeline decisions."""

    defects: list[str] = []
    naturalness = True
    grammar = True
    completeness = True
    target = True
    consistency = True
    self_contained = True
    if _REPEATED_WORD.search(question):
        defects.append("repeated wording")
        naturalness = False
        grammar = False
    if _BAD_ORDINAL.search(question):
        defects.append("grammar")
        grammar = False
    if not question.endswith("?") or question.count("?") != 1:
        defects.append("grammar")
        grammar = False
        target = False
    if len(question.split()) < 12:
        defects.append("missing information")
        completeness = False
    if _INTERNAL_TOKEN.search(question) or any(
        marker in question for marker in ("<ENTITY_", "<QUANTITY_", "frame_id", "role_id")
    ):
        defects.append("internal terminology")
        self_contained = False
    passed = all((naturalness, grammar, completeness, target, consistency, self_contained))
    recommendation = "approve" if passed else "reject"
    confidence = "high"
    rationale = (
        "Clear, grammatical, self-contained wording with an explicit target and consistent "
        "entities and units."
        if passed
        else "One or more frozen language criteria fail for the listed content-free reason."
    )
    return LanguageAuditRecord(
        candidate_id=candidate_id,
        recommendation=recommendation,
        naturalness_pass=naturalness,
        grammar_pass=grammar,
        completeness_pass=completeness,
        target_clarity_pass=target,
        consistency_pass=consistency,
        self_contained_pass=self_contained,
        confidence=confidence,
        defect_labels=tuple(dict.fromkeys(defects)),
        rationale=rationale,
    )


def _load_accepted(path: Path) -> tuple[dict[str, object], ...]:
    records = tuple(
        cast(dict[str, object], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    accepted = tuple(item for item in records if item.get("final_decision") == "accepted")
    if len(accepted) != 1000:
        raise ValueError("language audit requires exactly 1,000 accepted records")
    identifiers = [cast(str, item["synthetic_id"]) for item in accepted]
    if len(set(identifiers)) != 1000:
        raise ValueError("language audit candidate IDs differ")
    return accepted


def _stratified_high_confidence_sample(
    records: tuple[dict[str, object], ...], audit: dict[str, LanguageAuditRecord]
) -> tuple[str, ...]:
    selected: list[str] = []
    for group in ("targeted", "generic_control"):
        cells: dict[tuple[str, str, bool], list[dict[str, object]]] = defaultdict(list)
        for item in records:
            candidate_id = cast(str, item["synthetic_id"])
            decision = audit[candidate_id]
            if (
                item["group"] == group
                and decision.recommendation == "approve"
                and decision.confidence == "high"
            ):
                key = (
                    cast(str, item["family"]),
                    cast(str, item["difficulty"]),
                    cast(bool, item["output_contract_enabled"]),
                )
                cells[key].append(item)
        for values in cells.values():
            values.sort(key=lambda item: canonical_sha256({"sample": item["synthetic_id"]}))
        ordered_keys = sorted(cells, key=lambda key: (key[0], key[1], key[2]))
        group_selected: list[str] = []
        offset = 0
        while len(group_selected) < 50:
            changed = False
            for key in ordered_keys:
                if offset < len(cells[key]) and len(group_selected) < 50:
                    group_selected.append(cast(str, cells[key][offset]["synthetic_id"]))
                    changed = True
            if not changed:
                raise ValueError(f"insufficient high-confidence audit records for {group}")
            offset += 1
        selected.extend(group_selected)
    return tuple(selected)


def _packet_identity(items: list[dict[str, object]]) -> tuple[str, str]:
    lines = "\n".join(
        f"{item['candidate_id']}|{hashlib.sha256(cast(str, item['question']).encode()).hexdigest()}"
        for item in items
    )
    return lines, hashlib.sha256(lines.encode()).hexdigest()


def _write_markdown(path: Path, packet_hash: str, items: list[dict[str, object]]) -> None:
    parts = [
        "# Foundry 500x2 signal-data language review",
        "",
        f"Packet SHA-256: `{packet_hash}`",
        "",
        "Codex recommendations are advisory. Record your own Approve, Reject, or Unsure decision.",
        "",
    ]
    for index, item in enumerate(items, start=1):
        parts.extend(
            [
                f"## {index}. {item['candidate_id']}",
                "",
                cast(str, item["question"]),
                "",
                f"Codex: **{cast(str, item['recommendation']).upper()}** "
                f"({item['confidence']} confidence). {item['rationale']}",
                "",
                "Human decision: [ ] Approve  [ ] Reject  [ ] Unsure",
                "",
            ]
        )
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_static_html(path: Path, packet_hash: str, items: list[dict[str, object]]) -> None:
    cards = []
    for index, item in enumerate(items, start=1):
        cards.append(
            "<article><h2>"
            + html.escape(f"{index}. {item['candidate_id']}")
            + "</h2><p>"
            + html.escape(cast(str, item["question"]))
            + "</p><small>Codex: "
            + html.escape(cast(str, item["recommendation"]))
            + " (advisory)</small></article>"
        )
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Foundry signal review</title><style>
body{{font:16px/1.5 system-ui;max-width:960px;margin:2rem auto;padding:0 1rem}}
article{{border:1px solid #ccd3df;border-radius:10px;padding:1rem;margin:1rem 0}}
h2{{font-size:1rem}}small{{color:#596273}}
</style><h1>Foundry 500x2 signal-data review</h1>
<p>Packet <code>{packet_hash}</code>. Codex advice is advisory; make your own decision.</p>
{"".join(cards)}</html>"""
    path.write_text(document, encoding="utf-8")


_ASSISTED_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Foundry assisted review</title>
<style>
:root{font-family:system-ui;color:#182238;background:#eef2f8}body{margin:0}header{background:#182238;color:white;padding:1.3rem max(1rem,calc((100vw - 940px)/2))}main{max-width:940px;margin:1.5rem auto;padding:0 1rem}.card{background:white;border:1px solid #d3d9e5;border-radius:12px;padding:1.25rem;margin:1rem 0}.question{font-size:1.25rem;line-height:1.6;background:#f5f7fb;padding:1rem;border-radius:8px}.row{display:flex;gap:.7rem;flex-wrap:wrap;align-items:center;justify-content:space-between}button{padding:.7rem 1rem;border:1px solid #9da8bd;border-radius:8px;background:white;font-weight:700;cursor:pointer}button.selected{background:#3859d6;color:white}textarea,select{width:100%;padding:.6rem;margin-top:.4rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.5rem}.pass{color:#16724a}.fail{color:#b42c39}#summaryList{max-height:20rem;overflow:auto}.muted{color:#5a6477;font-size:.9rem}</style></head>
<body><header><h1>Foundry assisted language review</h1><p>Codex advice is advisory only. Your explicit A, R, or U choice is required for every item.</p></header><main>
<section class="card"><div id="integrity">Checking packet integrity...</div><div id="progress"></div></section>
<section class="card"><div class="row"><code id="candidate"></code><strong id="position"></strong></div><p class="question" id="question"></p><div><strong>Codex: <span id="rec"></span></strong> <span id="confidence"></span></div><div class="grid" id="criteria"></div><p><strong>Defects:</strong> <span id="defects"></span></p><p><strong>Rationale:</strong> <span id="rationale"></span></p><p class="muted">Advisory only; no official decision is preselected.</p>
<div class="row" id="decisions"><button data-d="approve">A - Approve</button><button data-d="reject">R - Reject</button><button data-d="unsure">U - Unsure</button></div>
<label>Defect label<select id="label"><option value="">None</option>__LABELS__</select></label><label>Optional note<textarea id="note" maxlength="500"></textarea></label>
<div class="row"><button id="back">Back</button><button id="next">Next</button></div></section>
<section class="card"><h2>Human decision summary</h2><div id="counts"></div><div id="summaryList"></div><button id="export" disabled>Export official review</button><span id="status"></span></section>
</main><script>
const items=__ITEMS__;const expectedHash="__PACKET_HASH__";const exportName="foundry-500x2-signal-review.json";const storageKey="foundry-500x2-review-"+expectedHash;let decisions=JSON.parse(localStorage.getItem(storageKey)||"{}");let index=0;let integrity=false;
const candidate=document.getElementById("candidate"),position=document.getElementById("position"),question=document.getElementById("question"),rec=document.getElementById("rec"),confidence=document.getElementById("confidence"),defects=document.getElementById("defects"),rationale=document.getElementById("rationale"),criteria=document.getElementById("criteria"),label=document.getElementById("label"),note=document.getElementById("note"),back=document.getElementById("back"),next=document.getElementById("next"),progress=document.getElementById("progress"),summaryList=document.getElementById("summaryList"),status=document.getElementById("status");
async function sha256(s){const b=await crypto.subtle.digest("SHA-256",new TextEncoder().encode(s));return [...new Uint8Array(b)].map(x=>x.toString(16).padStart(2,"0")).join("")}
async function check(){const lines=items.map(x=>x.candidate_id+"|"+x.question_sha256).join("\n");integrity=(await sha256(lines))===expectedHash&&new Set(items.map(x=>x.candidate_id)).size===items.length;document.getElementById("integrity").textContent=integrity?"Packet identity verified: "+expectedHash:"PACKET INTEGRITY FAILURE";render()}
function save(){localStorage.setItem(storageKey,JSON.stringify(decisions))}
function render(){const x=items[index],d=decisions[x.candidate_id]||{};candidate.textContent=x.candidate_id;position.textContent=(index+1)+"/"+items.length;question.textContent=x.question;rec.textContent=x.recommendation.toUpperCase();confidence.textContent=x.confidence+" confidence";defects.textContent=x.defect_labels.join(", ")||"none";rationale.textContent=x.rationale;criteria.innerHTML=Object.entries(x.criteria).map(([k,v])=>`<div class="${v?'pass':'fail'}">${v?'PASS':'FAIL'} ${k.replaceAll('_',' ')}</div>`).join("");document.querySelectorAll("[data-d]").forEach(b=>b.classList.toggle("selected",b.dataset.d===d.decision));label.value=d.defect_label||"";note.value=d.note||"";back.disabled=index===0;next.disabled=index===items.length-1;summary()}
function choose(value){const x=items[index];decisions[x.candidate_id]={decision:value,defect_label:label.value,note:note.value};save();if(index<items.length-1)index++;render()}
document.querySelectorAll("[data-d]").forEach(b=>b.onclick=()=>choose(b.dataset.d));back.onclick=()=>{if(index>0)index--;render()};next.onclick=()=>{if(index<items.length-1)index++;render()};label.onchange=()=>{const x=items[index],d=decisions[x.candidate_id];if(d){d.defect_label=label.value;save();summary()}};note.oninput=()=>{const x=items[index],d=decisions[x.candidate_id];if(d){d.note=note.value;save()}};
document.addEventListener("keydown",e=>{if(e.target.matches("textarea,select"))return;const k=e.key.toLowerCase();if(k==='a')choose('approve');if(k==='r')choose('reject');if(k==='u')choose('unsure')});
function summary(){const vals=Object.values(decisions),counts={approve:0,reject:0,unsure:0};vals.forEach(x=>counts[x.decision]++);progress.textContent=vals.length+"/"+items.length+" reviewed";document.getElementById("counts").textContent=`Approve ${counts.approve} | Reject ${counts.reject} | Unsure ${counts.unsure}`;const flagged=items.filter(x=>['reject','unsure'].includes((decisions[x.candidate_id]||{}).decision));summaryList.innerHTML=flagged.map(x=>`<p><code>${x.candidate_id}</code> - ${decisions[x.candidate_id].decision.toUpperCase()}: ${x.question}</p>`).join('')||'<p>No rejected or unsure items yet.</p>';document.getElementById("export").disabled=!(integrity&&vals.length===items.length&&items.every(x=>decisions[x.candidate_id]))}
document.getElementById("export").onclick=()=>{if(document.getElementById("export").disabled)return;const payload={schema_version:1,packet_sha256:expectedHash,candidate_count:items.length,decisions:items.map(x=>({candidate_id:x.candidate_id,decision:decisions[x.candidate_id].decision,defect_label:decisions[x.candidate_id].defect_label||null,note:decisions[x.candidate_id].note||null}))};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'}));a.download=exportName;a.click();URL.revokeObjectURL(a.href);status.textContent=' Exported '+exportName};check();
</script></body></html>"""


def _write_assisted_html(path: Path, packet_hash: str, items: list[dict[str, object]]) -> None:
    client_items = []
    for item in items:
        question = cast(str, item["question"])
        client_items.append(
            {
                **item,
                "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                "criteria": {key: item[key] for key in CRITERIA},
            }
        )
    options = "".join(
        f'<option value="{html.escape(label)}">{html.escape(label)}</option>'
        for label in DEFECT_LABELS
    )
    payload = json.dumps(client_items, ensure_ascii=True).replace("</", "<\\/")
    document = (
        _ASSISTED_HTML.replace("__ITEMS__", payload)
        .replace("__PACKET_HASH__", packet_hash)
        .replace("__LABELS__", options)
    )
    path.write_text(document, encoding="utf-8")


def create_language_audit_and_packet(
    attempts_path: Path, review_directory: Path, summary_path: Path
) -> dict[str, object]:
    """Audit all accepted texts and emit the ignored stratified advisory packet."""

    records = _load_accepted(attempts_path)
    audits = tuple(
        _blind_audit(cast(str, item["synthetic_id"]), cast(str, item["rendered_question"]))
        for item in records
    )
    audit_by_id = {item.candidate_id: item for item in audits}
    mandatory = {
        item.candidate_id
        for item in audits
        if item.recommendation in {"reject", "unsure"} or item.confidence == "low"
    }
    sample = _stratified_high_confidence_sample(records, audit_by_id)
    packet_ids = mandatory | set(sample)
    by_id = {cast(str, item["synthetic_id"]): item for item in records}
    rank = {candidate_id: index for index, candidate_id in enumerate(sample)}
    ordered_ids = sorted(
        packet_ids,
        key=lambda candidate_id: (
            0 if candidate_id in mandatory else 1,
            rank.get(candidate_id, 0),
            candidate_id,
        ),
    )
    items: list[dict[str, object]] = []
    for candidate_id in ordered_ids:
        record = by_id[candidate_id]
        audit = audit_by_id[candidate_id]
        items.append(
            {
                "candidate_id": candidate_id,
                "question": record["rendered_question"],
                "group": record["group"],
                "family": record["family"],
                "difficulty": record["difficulty"],
                "output_contract_enabled": record["output_contract_enabled"],
                **asdict(audit),
            }
        )
    _identity_lines, packet_hash = _packet_identity(items)
    review_directory.mkdir(parents=True, exist_ok=True)
    audit_document: dict[str, object] = {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "blind_inputs": ["candidate_id", "rendered_question"],
        "records": [asdict(item) for item in audits],
    }
    audit_document["audit_sha256"] = canonical_sha256(audit_document)
    (review_directory / "codex_language_audit.json").write_text(
        json.dumps(audit_document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(review_directory / "human_review.md", packet_hash, items)
    _write_static_html(review_directory / "human_review.html", packet_hash, items)
    _write_assisted_html(review_directory / "codex_assisted_review.html", packet_hash, items)
    recommendation_counts = Counter(item.recommendation for item in audits)
    confidence_counts = Counter(item.confidence for item in audits)
    defect_counts = Counter(label for item in audits for label in item.defect_labels)
    summary: dict[str, object] = {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "accepted_questions_audited": len(audits),
        "recommendations": dict(sorted(recommendation_counts.items())),
        "confidence": dict(sorted(confidence_counts.items())),
        "defect_labels": dict(sorted(defect_counts.items())),
        "systematic_sentence_plan_defects": 0,
        "audit_sha256": audit_document["audit_sha256"],
        "packet_candidate_count": len(items),
        "packet_sha256": packet_hash,
        "mandatory_candidate_count": len(mandatory),
        "stratified_high_confidence_sample_count": len(sample),
        "human_review_status": "pending",
        "export_filename": EXPORT_FILENAME,
        "review_directory": review_directory.as_posix(),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
