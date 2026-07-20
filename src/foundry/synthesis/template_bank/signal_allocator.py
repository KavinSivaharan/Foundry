"""Deterministic global allocation for the 2,504-attempt signal-first pilot."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from foundry.synthesis.contamination import (
    canonical_number_neutral_identity,
    normalized_text_sha256,
    number_neutral_identity_contract_sha256,
    require_number_neutral_identity,
)
from foundry.synthesis.generators import CandidateDraft
from foundry.synthesis.generators.bookkeeping import generate_bookkeeping
from foundry.synthesis.generators.discrete import generate_discrete
from foundry.synthesis.generators.rates import generate_rates
from foundry.synthesis.pipeline import _verify
from foundry.synthesis.schema import DifficultyLevel
from foundry.synthesis.taxonomy import FailureCategory
from foundry.synthesis.template_bank.bank import build_template_bank
from foundry.synthesis.template_bank.contracts import SentencePlanSpec, TemplateSpec
from foundry.synthesis.template_bank.difficulty_reallocation import (
    build_corrected_capacity_audit,
    corrected_subordinate_audit,
    load_difficulty_reallocation_config,
)
from foundry.synthesis.template_bank.renderer import render_with_template
from foundry.synthesis.template_bank.reuse import load_contract
from foundry.synthesis.template_bank.signal_pilot import (
    CATEGORY_ORDER,
    DIFFICULTY_ORDER,
    GROUP_ORDER,
    MODE_ORDER,
    SPLIT_ORDER,
    SignalPilotConfig,
    _derive_group_caps,
    canonical_sha256,
    load_signal_pilot_config,
)
from foundry.synthesis.template_bank.submode_policy import (
    balanced_matrix,
)
from foundry.synthesis.template_bank.surface_reuse import (
    derive_surface_caps,
    load_surface_reuse_config,
)

ALLOCATOR_VERSION = "foundry-signal-pilot-balanced-allocator-v1"

_CATEGORY_ENUM = {
    str(FailureCategory.MULTI_STEP_BOOKKEEPING): FailureCategory.MULTI_STEP_BOOKKEEPING,
    str(FailureCategory.RATE_RATIO_PERCENTAGE): FailureCategory.RATE_RATIO_PERCENTAGE,
    str(FailureCategory.CONSTRAINT_DISCRETE): FailureCategory.CONSTRAINT_DISCRETE,
}


@dataclass(frozen=True)
class SlotRequest:
    """One content-free quota position before latent and language assignment."""

    slot_index: int
    group: str
    group_index: int
    family: str
    family_index: int
    mode: str
    mode_index: int
    difficulty: str
    output_contract_enabled: bool
    future_split: str

    @property
    def slot_id(self) -> str:
        material = {
            "allocator": ALLOCATOR_VERSION,
            "slot_index": self.slot_index,
            "group": self.group,
            "family": self.family,
            "mode": self.mode,
            "difficulty": self.difficulty,
            "output": self.output_contract_enabled,
            "split": self.future_split,
        }
        return f"signal-slot-{self.slot_index:04d}-{canonical_sha256(material)[:12]}"


@dataclass(frozen=True)
class LatentCandidate:
    """Content-free identity of a generated latent program."""

    seed: int
    variant: int
    latent_sha256: str
    semantic_ir_sha256: str
    scenario_domain: str
    lexical_family: str
    target_type: str
    primary_evidence_sha256: str
    independent_evidence_sha256: str


@dataclass(frozen=True)
class PilotScheduleRecord:
    """One future candidate slot without rendered question or answer content."""

    slot_index: int
    slot_id: str
    group: str
    group_index: int
    family: str
    family_index: int
    mode: str
    mode_index: int
    difficulty: str
    output_contract_enabled: bool
    future_split: str
    latent_seed: int
    generator_variant: int
    latent_program_sha256: str
    semantic_ir_sha256: str
    target_type: str
    semantic_frame: str
    template_id: str
    sentence_plan_id: str
    scenario_domain: str
    lexical_family: str
    render_signature_sha256: str
    rendered_text_sha256: str
    number_neutral_sha256: str
    schedule_identity_contract_sha256: str
    candidate_identity_sha256: str
    primary_evidence_sha256: str
    independent_evidence_sha256: str
    verifier_agreement: bool


@dataclass(frozen=True)
class _SurfaceChoice:
    """One deterministic template option used by exact constrained matching."""

    template: TemplateSpec
    plan: SentencePlanSpec
    exact_sha256: str
    number_neutral_sha256: str
    plan_key: str
    scenario_key: tuple[str, str]
    frame_key: str


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: int
    initial_capacity: int


def _add_flow_edge(graph: list[list[_FlowEdge]], start: int, end: int, capacity: int) -> None:
    graph[start].append(_FlowEdge(end, len(graph[end]), capacity, capacity))
    graph[end].append(_FlowEdge(start, len(graph[start]) - 1, 0, 0))


def _dinic(graph: list[list[_FlowEdge]], source: int, sink: int) -> int:
    total = 0
    while True:
        level = [-1] * len(graph)
        level[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for edge in graph[node]:
                if edge.capacity and level[edge.to] < 0:
                    level[edge.to] = level[node] + 1
                    queue.append(edge.to)
        if level[sink] < 0:
            return total
        progress = [0] * len(graph)

        while True:
            amount = _send_flow(graph, source, sink, 10**9, level, progress)
            if not amount:
                break
            total += amount


def _send_flow(
    graph: list[list[_FlowEdge]],
    node: int,
    sink: int,
    available: int,
    level: list[int],
    progress: list[int],
) -> int:
    if node == sink:
        return available
    while progress[node] < len(graph[node]):
        edge = graph[node][progress[node]]
        if edge.capacity and level[edge.to] == level[node] + 1:
            amount = _send_flow(
                graph,
                edge.to,
                sink,
                min(available, edge.capacity),
                level,
                progress,
            )
            if amount:
                edge.capacity -= amount
                graph[edge.to][edge.reverse].capacity += amount
                return amount
        progress[node] += 1
    return 0


def _variant_for(family: str, mode: str, index: int) -> int:
    if family == CATEGORY_ORDER[0]:
        if mode == "grouping":
            return 5 * index + 4
        return 5 * (index // 4) + index % 4
    modes = MODE_ORDER[family]
    return modes.index(mode) + len(modes) * index


def _generate(
    family: str,
    *,
    seed: int,
    difficulty: str,
    variant: int,
    output_contract_enabled: bool,
) -> CandidateDraft:
    level = DifficultyLevel(difficulty)
    if family == CATEGORY_ORDER[0]:
        return generate_bookkeeping(
            seed=seed,
            difficulty=level,
            variant=variant,
            output_contract_enabled=output_contract_enabled,
        )
    if family == CATEGORY_ORDER[1]:
        return generate_rates(
            seed=seed,
            difficulty=level,
            variant=variant,
            output_contract_enabled=output_contract_enabled,
        )
    if family == CATEGORY_ORDER[2]:
        return generate_discrete(
            seed=seed,
            difficulty=level,
            variant=variant,
            output_contract_enabled=output_contract_enabled,
        )
    raise ValueError("unapproved family")


def _latent_hash(draft: CandidateDraft) -> str:
    return canonical_sha256(asdict(draft.latent_program))


def _mode(draft: CandidateDraft) -> str:
    return draft.latent_program.program_family.split(":", maxsplit=1)[-1]


def _seed(master_seed: str, family: str, mode: str, difficulty: str, index: int) -> int:
    material = f"{master_seed}:{family}:{mode}:{difficulty}:{index}"
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def _candidate_metadata(draft: CandidateDraft, variant: int) -> LatentCandidate:
    primary, independent, reasons = _verify(draft)
    if (
        reasons
        or not primary.success
        or not independent.success
        or primary.answer != independent.answer
        or primary.answer != draft.canonical_final_answer
        or primary.verifier_id == independent.verifier_id
        or primary.method_family == independent.method_family
    ):
        raise ValueError("scheduled latent program failed independent verification")
    domain = draft.problem_ir.domain
    return LatentCandidate(
        seed=draft.random_seed,
        variant=variant,
        latent_sha256=_latent_hash(draft),
        semantic_ir_sha256=draft.semantic_ir_sha256,
        scenario_domain=domain.domain_id,
        lexical_family=domain.item.lexeme.lexeme_id,
        target_type=str(draft.problem_ir.target.kind),
        primary_evidence_sha256=primary.evidence_sha256,
        independent_evidence_sha256=independent.evidence_sha256,
    )


def _stable_label_vector(counts: dict[str, int], material: str) -> list[str]:
    expanded = [
        (label, occurrence) for label, quantity in counts.items() for occurrence in range(quantity)
    ]
    expanded.sort(
        key=lambda item: hashlib.sha256(f"{material}:{item[0]}:{item[1]}".encode()).hexdigest()
    )
    return [label for label, _ in expanded]


def build_slot_requests(config_path: Path, policy_path: Path) -> tuple[SlotRequest, ...]:
    """Construct exact content-free slot margins before latent matching."""

    config = load_signal_pilot_config(config_path)
    if not policy_path.is_file():
        raise ValueError("frozen submode-policy configuration is missing")
    capacity = build_corrected_capacity_audit(config.difficulty_reallocation_path)
    if capacity["capacity_gate_passed"] is not True:
        raise ValueError("allocator cannot run before the revised capacity gate passes")
    audit = corrected_subordinate_audit(config.difficulty_reallocation_path)
    requests: list[SlotRequest] = []
    slot_index = 1
    for group in GROUP_ORDER:
        group_index = 0
        for family in CATEGORY_ORDER:
            quota = config.datasets[group].families[family]
            dataset_audit = cast(dict[str, dict[str, dict[str, object]]], audit["datasets"])
            family_audit = dataset_audit[group][family]
            subordinate = cast(dict[str, object], family_audit["attempt_subordinate_allocations"])
            difficulty_matrix = cast(dict[str, dict[str, int]], subordinate["difficulty"])
            output_matrix = balanced_matrix(
                quota.attempt_modes,
                {
                    "enabled": quota.output_contract_attempts,
                    "disabled": quota.attempts - quota.output_contract_attempts,
                },
            )
            split_matrix = balanced_matrix(
                quota.attempt_modes,
                {
                    "training": quota.training_attempts,
                    "synthetic_validation": quota.validation_attempts,
                },
            )
            family_index = 0
            for mode in MODE_ORDER[family]:
                count = quota.attempt_modes[mode]
                difficulties = _stable_label_vector(
                    difficulty_matrix[mode], f"{config.full_schedule_master_seed}:{group}:{mode}:d"
                )
                outputs = _stable_label_vector(
                    output_matrix[mode], f"{config.full_schedule_master_seed}:{group}:{mode}:o"
                )
                splits = _stable_label_vector(
                    split_matrix[mode], f"{config.full_schedule_master_seed}:{group}:{mode}:s"
                )
                for mode_index in range(count):
                    requests.append(
                        SlotRequest(
                            slot_index=slot_index,
                            group=group,
                            group_index=group_index,
                            family=family,
                            family_index=family_index,
                            mode=mode,
                            mode_index=mode_index,
                            difficulty=difficulties[mode_index],
                            output_contract_enabled=outputs[mode_index] == "enabled",
                            future_split=splits[mode_index],
                        )
                    )
                    slot_index += 1
                    group_index += 1
                    family_index += 1
    if len(requests) != 2_504:
        raise ValueError("global request schedule does not contain 2,504 slots")
    return tuple(requests)


def _pool_for_mode(
    *,
    family: str,
    mode: str,
    master_seed: str,
    pool_size: int,
) -> dict[str, dict[str, LatentCandidate]]:
    candidates: dict[str, dict[str, LatentCandidate]] = {}
    for difficulty in DIFFICULTY_ORDER:
        for index in range(pool_size):
            seed = _seed(master_seed, family, mode, difficulty, index)
            variant = _variant_for(family, mode, index)
            draft = _generate(
                family,
                seed=seed,
                difficulty=difficulty,
                variant=variant,
                output_contract_enabled=False,
            )
            if _mode(draft) != mode:
                raise ValueError("generator mode differs from the frozen request")
            latent = _latent_hash(draft)
            by_difficulty = candidates.setdefault(latent, {})
            if difficulty not in by_difficulty:
                # Verification is deferred until a member is actually selected.
                domain = draft.problem_ir.domain
                by_difficulty[difficulty] = LatentCandidate(
                    seed=seed,
                    variant=variant,
                    latent_sha256=latent,
                    semantic_ir_sha256=draft.semantic_ir_sha256,
                    scenario_domain=domain.domain_id,
                    lexical_family=domain.item.lexeme.lexeme_id,
                    target_type=str(draft.problem_ir.target.kind),
                    primary_evidence_sha256="",
                    independent_evidence_sha256="",
                )
    return candidates


def _match_mode_candidates(
    requests: tuple[SlotRequest, ...],
    candidates: dict[str, dict[str, LatentCandidate]],
) -> dict[str, list[LatentCandidate]]:
    required = Counter(request.difficulty for request in requests)
    difficulties = tuple(label for label in DIFFICULTY_ORDER if required[label])
    latent_hashes = tuple(sorted(candidates))
    source = 0
    difficulty_offset = 1
    latent_offset = difficulty_offset + len(difficulties)
    sink = latent_offset + len(latent_hashes)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    for difficulty_index, difficulty in enumerate(difficulties):
        node = difficulty_offset + difficulty_index
        _add_flow_edge(graph, source, node, required[difficulty])
        for latent_index, latent in enumerate(latent_hashes):
            if difficulty in candidates[latent]:
                _add_flow_edge(graph, node, latent_offset + latent_index, 1)
    for latent_index in range(len(latent_hashes)):
        _add_flow_edge(graph, latent_offset + latent_index, sink, 1)
    flow = _dinic(graph, source, sink)
    if flow != len(requests):
        raise ValueError(
            "fixed latent candidate pool cannot satisfy difficulty-isolated uniqueness"
        )
    selected: dict[str, list[LatentCandidate]] = {difficulty: [] for difficulty in difficulties}
    for difficulty_index, difficulty in enumerate(difficulties):
        node = difficulty_offset + difficulty_index
        for edge in graph[node]:
            if (
                latent_offset <= edge.to < sink
                and edge.initial_capacity == 1
                and edge.capacity == 0
            ):
                latent = latent_hashes[edge.to - latent_offset]
                selected[difficulty].append(candidates[latent][difficulty])
        selected[difficulty].sort(key=lambda item: item.latent_sha256)
        if len(selected[difficulty]) != required[difficulty]:
            raise ValueError("latent matching did not satisfy a difficulty margin")
    return selected


def _compatible_templates(
    family: str, mode: str, bank: tuple[TemplateSpec, ...]
) -> tuple[TemplateSpec, ...]:
    category = _CATEGORY_ENUM[family]
    result = [item for item in bank if item.reasoning_category == category]
    if family != CATEGORY_ORDER[0]:
        result = [item for item in result if item.semantic_frame.startswith(mode + ".")]
    return tuple(result)


def _choose_template_plan(
    *,
    request: SlotRequest,
    latent: LatentCandidate,
    draft: CandidateDraft,
    bank: tuple[TemplateSpec, ...],
    plan_cap: int,
    plan_scenario_cap: int,
    frame_cap: int,
    number_neutral_cap: int,
    plan_use: Counter[tuple[str, str, str]],
    plan_scenario_use: Counter[tuple[str, str, str, str]],
    frame_use: Counter[tuple[str, str, str]],
    number_neutral_use: Counter[tuple[str, str, str, str]],
    exact_hashes: set[str],
) -> tuple[TemplateSpec, SentencePlanSpec, str, str]:
    options: list[
        tuple[tuple[int, int, int, int, str], TemplateSpec, SentencePlanSpec, str, str]
    ] = []
    for template in _compatible_templates(request.family, request.mode, bank):
        frame_key = (request.group, request.family, template.template_id)
        if frame_use[frame_key] >= frame_cap:
            continue
        for plan in template.sentence_plan_variants:
            plan_key = (request.group, request.family, f"{template.template_id}:{plan.plan_id}")
            scenario_key = (
                request.group,
                request.family,
                f"{template.template_id}:{plan.plan_id}",
                latent.scenario_domain,
            )
            if (
                plan_use[plan_key] >= plan_cap
                or plan_scenario_use[scenario_key] >= plan_scenario_cap
            ):
                continue
            preview = render_with_template(draft, template, plan)
            exact_hash = normalized_text_sha256(preview.rendered_question)
            number_neutral_hash = canonical_number_neutral_identity(
                preview.rendered_question
            ).sha256
            number_neutral_key = (
                request.group,
                request.family,
                request.mode,
                number_neutral_hash,
            )
            if (
                exact_hash in exact_hashes
                or number_neutral_use[number_neutral_key] >= number_neutral_cap
            ):
                continue
            tie = canonical_sha256(
                {
                    "slot": request.slot_id,
                    "template": template.template_id,
                    "plan": plan.plan_id,
                }
            )
            if request.mode == "weighted_average":
                score = (
                    number_neutral_use[number_neutral_key],
                    plan_scenario_use[scenario_key],
                    frame_use[frame_key],
                    plan_use[plan_key],
                    tie,
                )
            else:
                score = (
                    plan_scenario_use[scenario_key],
                    number_neutral_use[number_neutral_key],
                    frame_use[frame_key],
                    plan_use[plan_key],
                    tie,
                )
            options.append((score, template, plan, exact_hash, number_neutral_hash))
    if not options:
        raise ValueError(
            "template allocator exhausted a frozen plan or scenario cap: "
            f"{request.group}/{request.family}/{request.mode}/"
            f"{latent.scenario_domain}; plan_cap={plan_cap}; "
            f"plan_scenario_cap={plan_scenario_cap}; frame_cap={frame_cap}"
        )
    _, template, plan, exact_hash, number_neutral_hash = min(options, key=lambda item: item[0])
    frame_use[(request.group, request.family, template.template_id)] += 1
    plan_use[(request.group, request.family, f"{template.template_id}:{plan.plan_id}")] += 1
    plan_scenario_use[
        (
            request.group,
            request.family,
            f"{template.template_id}:{plan.plan_id}",
            latent.scenario_domain,
        )
    ] += 1
    number_neutral_use[(request.group, request.family, request.mode, number_neutral_hash)] += 1
    exact_hashes.add(exact_hash)
    return template, plan, exact_hash, number_neutral_hash


def _surface_choices(
    request: SlotRequest,
    latent: LatentCandidate,
    bank: tuple[TemplateSpec, ...],
) -> tuple[_SurfaceChoice, ...]:
    draft = _generate(
        request.family,
        seed=latent.seed,
        difficulty=request.difficulty,
        variant=latent.variant,
        output_contract_enabled=request.output_contract_enabled,
    )
    choices: list[_SurfaceChoice] = []
    for template in _compatible_templates(request.family, request.mode, bank):
        for plan in template.sentence_plan_variants:
            preview = render_with_template(draft, template, plan)
            choices.append(
                _SurfaceChoice(
                    template=template,
                    plan=plan,
                    exact_sha256=normalized_text_sha256(preview.rendered_question),
                    number_neutral_sha256=canonical_number_neutral_identity(
                        preview.rendered_question
                    ).sha256,
                    plan_key=f"{template.template_id}:{plan.plan_id}",
                    scenario_key=(
                        f"{template.template_id}:{plan.plan_id}",
                        latent.scenario_domain,
                    ),
                    frame_key=template.template_id,
                )
            )
    return tuple(choices)


def _match_surface_choices(
    requests: tuple[SlotRequest, ...],
    latent_for_slot: dict[str, LatentCandidate],
    bank: tuple[TemplateSpec, ...],
    *,
    plan_cap: int,
    plan_scenario_cap: int,
    frame_cap: int,
    number_neutral_cap: int,
) -> dict[str, _SurfaceChoice]:
    """Exactly match a constrained submode when greedy ordering is insufficient."""

    options = {
        request.slot_id: _surface_choices(request, latent_for_slot[request.slot_id], bank)
        for request in requests
    }
    request_by_id = {request.slot_id: request for request in requests}
    scenario_frequency = Counter(
        latent_for_slot[request.slot_id].scenario_domain for request in requests
    )
    remaining = sorted(
        request_by_id,
        key=lambda slot_id: (
            -scenario_frequency[latent_for_slot[slot_id].scenario_domain],
            DIFFICULTY_ORDER.index(request_by_id[slot_id].difficulty),
            latent_for_slot[slot_id].scenario_domain,
            slot_id,
        ),
    )
    plan_use: Counter[str] = Counter()
    scenario_use: Counter[tuple[str, str]] = Counter()
    frame_use: Counter[str] = Counter()
    identity_use: Counter[str] = Counter()
    exact_use: set[str] = set()
    selected: dict[str, _SurfaceChoice] = {}
    visited = 0

    def allowed(choice: _SurfaceChoice) -> bool:
        return (
            plan_use[choice.plan_key] < plan_cap
            and scenario_use[choice.scenario_key] < plan_scenario_cap
            and frame_use[choice.frame_key] < frame_cap
            and identity_use[choice.number_neutral_sha256] < number_neutral_cap
            and choice.exact_sha256 not in exact_use
        )

    def identity_margin_feasible(pending: list[str]) -> bool:
        identities = sorted(
            {
                choice.number_neutral_sha256
                for slot_id in pending
                for choice in options[slot_id]
                if allowed(choice)
            }
        )
        source = 0
        slot_offset = 1
        identity_offset = slot_offset + len(pending)
        sink = identity_offset + len(identities)
        graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
        identity_index = {label: index for index, label in enumerate(identities)}
        for slot_index, slot_id in enumerate(pending):
            slot_node = slot_offset + slot_index
            _add_flow_edge(graph, source, slot_node, 1)
            for identity in sorted(
                {choice.number_neutral_sha256 for choice in options[slot_id] if allowed(choice)}
            ):
                _add_flow_edge(
                    graph,
                    slot_node,
                    identity_offset + identity_index[identity],
                    1,
                )
        for identity in identities:
            _add_flow_edge(
                graph,
                identity_offset + identity_index[identity],
                sink,
                number_neutral_cap - identity_use[identity],
            )
        return _dinic(graph, source, sink) == len(pending)

    def search(pending: list[str]) -> bool:
        nonlocal visited
        visited += 1
        if visited > 2_000_000:
            raise ValueError("deterministic surface matching search bound exceeded")
        if not pending:
            return True
        ranked: list[tuple[int, str, list[_SurfaceChoice]]] = []
        for slot_id in pending:
            available = [choice for choice in options[slot_id] if allowed(choice)]
            if not available:
                return False
            ranked.append((len(available), slot_id, available))
        _, slot_id, available = min(ranked, key=lambda item: (item[0], item[1]))
        request = request_by_id[slot_id]
        available.sort(
            key=lambda choice: (
                (
                    identity_use[choice.number_neutral_sha256]
                    if request.mode == "weighted_average"
                    else scenario_use[choice.scenario_key]
                ),
                (
                    scenario_use[choice.scenario_key]
                    if request.mode == "weighted_average"
                    else identity_use[choice.number_neutral_sha256]
                ),
                frame_use[choice.frame_key],
                plan_use[choice.plan_key],
                canonical_sha256(
                    {
                        "slot": slot_id,
                        "template": choice.template.template_id,
                        "plan": choice.plan.plan_id,
                    }
                ),
            )
        )
        next_pending = [item for item in pending if item != slot_id]
        for choice in available:
            selected[slot_id] = choice
            plan_use[choice.plan_key] += 1
            scenario_use[choice.scenario_key] += 1
            frame_use[choice.frame_key] += 1
            identity_use[choice.number_neutral_sha256] += 1
            exact_use.add(choice.exact_sha256)
            if identity_margin_feasible(next_pending) and search(next_pending):
                return True
            exact_use.remove(choice.exact_sha256)
            identity_use[choice.number_neutral_sha256] -= 1
            frame_use[choice.frame_key] -= 1
            scenario_use[choice.scenario_key] -= 1
            plan_use[choice.plan_key] -= 1
            del selected[slot_id]
        return False

    if not search(remaining):
        raise ValueError("exact deterministic surface matching is infeasible")
    return selected


def _frame_cap(
    *,
    request: SlotRequest,
    frame_count: int,
    config: SignalPilotConfig,
) -> int:
    allocation_total = (
        config.datasets[request.group].families[request.family].attempts
        if request.family == CATEGORY_ORDER[0]
        else config.datasets[request.group].families[request.family].attempt_modes[request.mode]
    )
    return (allocation_total + frame_count - 1) // frame_count


def build_full_schedule(
    config_path: Path,
    policy_path: Path,
    *,
    progress: bool = False,
) -> tuple[PilotScheduleRecord, ...]:
    """Build all 2,504 content-free records from fixed candidate pools."""

    config = load_signal_pilot_config(config_path)
    requests = build_slot_requests(config_path, policy_path)
    bank = build_template_bank()
    reuse = load_contract(config.reuse_config_path)
    difficulty_policy = load_difficulty_reallocation_config(config.difficulty_reallocation_path)
    surface_policy = load_surface_reuse_config(difficulty_policy.surface_policy_config)
    surface_caps = derive_surface_caps(config, surface_policy)
    latent_for_slot: dict[str, LatentCandidate] = {}
    scenario_use: Counter[tuple[str, str, str]] = Counter()
    scenario_difficulty_use: Counter[tuple[str, str, str, str]] = Counter()
    lexical_use: Counter[tuple[str, str, str]] = Counter()
    for family in CATEGORY_ORDER:
        for mode in MODE_ORDER[family]:
            current_requests = tuple(
                request for request in requests if request.family == family and request.mode == mode
            )
            pool = _pool_for_mode(
                family=family,
                mode=mode,
                master_seed=config.full_schedule_master_seed,
                pool_size=config.full_schedule_candidate_pool_per_mode_difficulty,
            )
            selected = _match_mode_candidates(current_requests, pool)
            complete_packages_used: set[str] = set()
            for difficulty in DIFFICULTY_ORDER:
                slots = sorted(
                    (request for request in current_requests if request.difficulty == difficulty),
                    key=lambda request: request.slot_id,
                )
                available = (
                    [
                        by_difficulty[difficulty]
                        for latent_sha256, by_difficulty in sorted(pool.items())
                        if difficulty in by_difficulty
                        and latent_sha256 not in complete_packages_used
                    ]
                    if mode == "complete_packages"
                    else list(selected.get(difficulty, []))
                )
                for request in slots:
                    latent = min(
                        available,
                        key=lambda item: (
                            scenario_difficulty_use[
                                (
                                    request.group,
                                    family,
                                    difficulty,
                                    item.scenario_domain,
                                )
                            ],
                            scenario_use[(request.group, family, item.scenario_domain)],
                            lexical_use[(request.group, family, item.lexical_family)],
                            item.latent_sha256,
                        ),
                    )
                    available.remove(latent)
                    if mode == "complete_packages":
                        complete_packages_used.add(latent.latent_sha256)
                    scenario_difficulty_use[
                        (
                            request.group,
                            family,
                            difficulty,
                            latent.scenario_domain,
                        )
                    ] += 1
                    scenario_use[(request.group, family, latent.scenario_domain)] += 1
                    lexical_use[(request.group, family, latent.lexical_family)] += 1
                    latent_for_slot[request.slot_id] = latent
            if progress:
                print(
                    json.dumps(
                        {
                            "stage": "latent_pool_matched",
                            "family": family,
                            "mode": mode,
                            "required": len(current_requests),
                            "unique_pool": len(pool),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    if len({item.latent_sha256 for item in latent_for_slot.values()}) != 2_504:
        raise ValueError("global latent-program uniqueness failed")
    plan_use: Counter[tuple[str, str, str]] = Counter()
    plan_scenario_use: Counter[tuple[str, str, str, str]] = Counter()
    frame_use: Counter[tuple[str, str, str]] = Counter()
    number_neutral_use: Counter[tuple[str, str, str, str]] = Counter()
    exact_hashes: set[str] = set()
    records: list[PilotScheduleRecord] = []
    surface_requests = sorted(
        requests,
        key=lambda request: (
            request.group,
            request.family,
            request.mode,
            (
                0
                if request.mode == "weighted_average" and request.difficulty in {"easy", "medium"}
                else 1
                if request.mode == "weighted_average"
                else 0
            ),
            DIFFICULTY_ORDER.index(request.difficulty),
            latent_for_slot[request.slot_id].scenario_domain,
            request.slot_id,
        ),
    )
    exact_matches: dict[str, _SurfaceChoice] = {}
    constrained_modes = (
        (CATEGORY_ORDER[1], "weighted_average"),
        (CATEGORY_ORDER[2], "complete_packages"),
    )
    for group in GROUP_ORDER:
        for family, mode in constrained_modes:
            current = tuple(
                request
                for request in surface_requests
                if request.group == group and request.family == family and request.mode == mode
            )
            example = current[0]
            caps = _derive_group_caps(config, reuse.identity_inventory, group, family)
            attempt_caps = cast(dict[str, int], caps["attempt_caps"])
            frame_count = len(_compatible_templates(family, mode, bank))
            try:
                mode_matches = _match_surface_choices(
                    current,
                    latent_for_slot,
                    bank,
                    plan_cap=attempt_caps["sentence_plan"],
                    plan_scenario_cap=attempt_caps["plan_scenario_domain"],
                    frame_cap=_frame_cap(
                        request=example,
                        frame_count=frame_count,
                        config=config,
                    ),
                    number_neutral_cap=surface_caps[group][family][mode].max_attempts_per_identity,
                )
            except ValueError as error:
                raise ValueError(
                    f"surface matching failed for {group}/{family}/{mode}: {error}"
                ) from error
            exact_matches.update(mode_matches)
    for request in surface_requests:
        source = latent_for_slot[request.slot_id]
        draft = _generate(
            request.family,
            seed=source.seed,
            difficulty=request.difficulty,
            variant=source.variant,
            output_contract_enabled=request.output_contract_enabled,
        )
        latent = _candidate_metadata(draft, source.variant)
        caps = _derive_group_caps(
            config,
            reuse.identity_inventory,
            request.group,
            request.family,
        )
        attempt_caps = cast(dict[str, int], caps["attempt_caps"])
        frame_count = len(_compatible_templates(request.family, request.mode, bank))
        frame_cap = _frame_cap(
            request=request,
            frame_count=frame_count,
            config=config,
        )
        matched = exact_matches.get(request.slot_id)
        if matched is None:
            template, plan, rendered_text_hash, number_neutral_hash = _choose_template_plan(
                request=request,
                latent=latent,
                draft=draft,
                bank=bank,
                plan_cap=attempt_caps["sentence_plan"],
                plan_scenario_cap=attempt_caps["plan_scenario_domain"],
                frame_cap=frame_cap,
                number_neutral_cap=surface_caps[request.group][request.family][
                    request.mode
                ].max_attempts_per_identity,
                plan_use=plan_use,
                plan_scenario_use=plan_scenario_use,
                frame_use=frame_use,
                number_neutral_use=number_neutral_use,
                exact_hashes=exact_hashes,
            )
        else:
            template = matched.template
            plan = matched.plan
            rendered_text_hash = matched.exact_sha256
            number_neutral_hash = matched.number_neutral_sha256
            if rendered_text_hash in exact_hashes:
                raise ValueError("exact constrained surface matching produced a duplicate")
            frame_use[(request.group, request.family, template.template_id)] += 1
            plan_use[(request.group, request.family, matched.plan_key)] += 1
            plan_scenario_use[
                (
                    request.group,
                    request.family,
                    matched.plan_key,
                    latent.scenario_domain,
                )
            ] += 1
            number_neutral_use[
                (request.group, request.family, request.mode, number_neutral_hash)
            ] += 1
            exact_hashes.add(rendered_text_hash)
        render_signature = template.render_signature_hash(plan)
        candidate_identity = canonical_sha256(
            {
                "latent": latent.latent_sha256,
                "template": template.template_id,
                "plan": plan.plan_id,
            }
        )
        records.append(
            PilotScheduleRecord(
                slot_index=request.slot_index,
                slot_id=request.slot_id,
                group=request.group,
                group_index=request.group_index,
                family=request.family,
                family_index=request.family_index,
                mode=request.mode,
                mode_index=request.mode_index,
                difficulty=request.difficulty,
                output_contract_enabled=request.output_contract_enabled,
                future_split=request.future_split,
                latent_seed=latent.seed,
                generator_variant=latent.variant,
                latent_program_sha256=latent.latent_sha256,
                semantic_ir_sha256=latent.semantic_ir_sha256,
                target_type=latent.target_type,
                semantic_frame=template.semantic_frame,
                template_id=template.template_id,
                sentence_plan_id=plan.plan_id,
                scenario_domain=latent.scenario_domain,
                lexical_family=latent.lexical_family,
                render_signature_sha256=render_signature,
                rendered_text_sha256=rendered_text_hash,
                number_neutral_sha256=number_neutral_hash,
                schedule_identity_contract_sha256=number_neutral_identity_contract_sha256(),
                candidate_identity_sha256=candidate_identity,
                primary_evidence_sha256=latent.primary_evidence_sha256,
                independent_evidence_sha256=latent.independent_evidence_sha256,
                verifier_agreement=True,
            )
        )
    return tuple(sorted(records, key=lambda record: record.slot_index))


def _counter(records: tuple[PilotScheduleRecord, ...], *fields: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        key = "/".join(str(getattr(record, field)) for field in fields)
        counts[key] += 1
    return dict(sorted(counts.items()))


def validate_full_schedule(
    records: tuple[PilotScheduleRecord, ...], config_path: Path, policy_path: Path
) -> dict[str, object]:
    """Validate quota, isolation, identity, cap, and deterministic-order contracts."""

    config = load_signal_pilot_config(config_path)
    if len(records) != 2_504 or tuple(record.slot_index for record in records) != tuple(
        range(1, 2_505)
    ):
        raise ValueError("schedule count or stable order differs")
    unique_fields = (
        "slot_id",
        "latent_program_sha256",
        "semantic_ir_sha256",
        "rendered_text_sha256",
        "candidate_identity_sha256",
    )
    for field in unique_fields:
        if len({getattr(record, field) for record in records}) != len(records):
            raise ValueError(f"schedule uniqueness failed for {field}")
    expected = build_slot_requests(config_path, policy_path)
    expected_by_id = {request.slot_id: request for request in expected}
    for record in records:
        request = expected_by_id.get(record.slot_id)
        if request is None:
            raise ValueError("schedule contains an unknown or tampered slot")
        for field in (
            "slot_index",
            "group",
            "group_index",
            "family",
            "family_index",
            "mode",
            "mode_index",
            "difficulty",
            "output_contract_enabled",
            "future_split",
        ):
            if getattr(record, field) != getattr(request, field):
                raise ValueError(f"schedule field {field} differs from the frozen request")
    reuse = load_contract(config.reuse_config_path)
    plan_counts = Counter(
        (record.group, record.family, record.template_id, record.sentence_plan_id)
        for record in records
    )
    scenario_counts = Counter(
        (
            record.group,
            record.family,
            record.template_id,
            record.sentence_plan_id,
            record.scenario_domain,
        )
        for record in records
    )
    number_neutral_counts = Counter(
        (record.group, record.family, record.mode, record.number_neutral_sha256)
        for record in records
    )
    difficulty_policy = load_difficulty_reallocation_config(config.difficulty_reallocation_path)
    surface_policy = load_surface_reuse_config(difficulty_policy.surface_policy_config)
    surface_caps = derive_surface_caps(config, surface_policy)
    for group in GROUP_ORDER:
        for family in CATEGORY_ORDER:
            caps = _derive_group_caps(config, reuse.identity_inventory, group, family)
            attempt_caps = cast(dict[str, int], caps["attempt_caps"])
            if (
                max(
                    count
                    for key, count in plan_counts.items()
                    if key[0] == group and key[1] == family
                )
                > attempt_caps["sentence_plan"]
            ):
                raise ValueError("sentence-plan cap exceeded")
            if (
                max(
                    count
                    for key, count in scenario_counts.items()
                    if key[0] == group and key[1] == family
                )
                > attempt_caps["plan_scenario_domain"]
            ):
                raise ValueError("plan-plus-scenario cap exceeded")
            for mode in MODE_ORDER[family]:
                if (
                    max(
                        count
                        for key, count in number_neutral_counts.items()
                        if key[0] == group and key[1] == family and key[2] == mode
                    )
                    > surface_caps[group][family][mode].max_attempts_per_identity
                ):
                    raise ValueError("submode-local number-neutral reuse cap exceeded")
    bank = build_template_bank()
    template_by_id = {item.template_id: item for item in bank}
    for record in records:
        if record.schedule_identity_contract_sha256 != (number_neutral_identity_contract_sha256()):
            raise ValueError("dry schedule identity contract differs")
        draft = _generate(
            record.family,
            seed=record.latent_seed,
            difficulty=record.difficulty,
            variant=record.generator_variant,
            output_contract_enabled=record.output_contract_enabled,
        )
        if (
            _latent_hash(draft) != record.latent_program_sha256
            or draft.semantic_ir_sha256 != record.semantic_ir_sha256
        ):
            raise ValueError("dry schedule latent reconstruction differs")
        template = template_by_id.get(record.template_id)
        if template is None:
            raise ValueError("dry schedule template is missing")
        plan = next(
            (
                item
                for item in template.sentence_plan_variants
                if item.plan_id == record.sentence_plan_id
            ),
            None,
        )
        if plan is None:
            raise ValueError("dry schedule sentence plan is missing")
        rendered = render_with_template(draft, template, plan)
        require_number_neutral_identity(rendered.rendered_question, record.number_neutral_sha256)
        if (
            normalized_text_sha256(rendered.rendered_question) != record.rendered_text_sha256
            or template.render_signature_hash(plan) != record.render_signature_sha256
        ):
            raise ValueError("dry schedule/runtime surface identity mismatch")
    if any(not record.verifier_agreement for record in records):
        raise ValueError("scheduled verifier disagreement")
    if any(
        key in asdict(records[0])
        for key in ("rendered_question", "canonical_answer", "solution_trace")
    ):
        raise ValueError("content-bearing field entered the dry schedule")
    schedule_hash = canonical_sha256([asdict(record) for record in records])
    payload: dict[str, object] = {
        "schema_version": 1,
        "schedule_id": "foundry-signal-first-dry-schedule-v1",
        "allocator_version": ALLOCATOR_VERSION,
        "signal_config_sha256": config.config_sha256,
        "record_count": len(records),
        "schedule_sha256": schedule_hash,
        "group_family_counts": _counter(records, "group", "family"),
        "group_family_mode_counts": _counter(records, "group", "family", "mode"),
        "group_family_difficulty_counts": _counter(records, "group", "family", "difficulty"),
        "group_family_output_counts": _counter(
            records, "group", "family", "output_contract_enabled"
        ),
        "group_family_split_counts": _counter(records, "group", "family", "future_split"),
        "target_type_counts": _counter(records, "family", "target_type"),
        "maximum_plan_use": max(plan_counts.values()),
        "maximum_plan_scenario_use": max(scenario_counts.values()),
        "maximum_number_neutral_use": max(number_neutral_counts.values()),
        "unique_slot_ids": len({record.slot_id for record in records}),
        "unique_latent_programs": len({record.latent_program_sha256 for record in records}),
        "unique_semantic_irs": len({record.semantic_ir_sha256 for record in records}),
        "unique_candidate_identities": len(
            {record.candidate_identity_sha256 for record in records}
        ),
        "unique_rendered_questions": len({record.rendered_text_sha256 for record in records}),
        "unique_number_neutral_identities": len(
            {record.number_neutral_sha256 for record in records}
        ),
        "schedule_identity_contract_sha256": number_neutral_identity_contract_sha256(),
        "difficulty_reallocation_policy_sha256": build_corrected_capacity_audit(
            config.difficulty_reallocation_path
        )["policy_sha256"],
        "surface_reuse_policy_sha256": surface_policy.config_sha256,
        "schedule_runtime_identity_mismatches": 0,
        "targeted_generic_latent_overlap": len(
            {record.latent_program_sha256 for record in records if record.group == GROUP_ORDER[0]}
            & {record.latent_program_sha256 for record in records if record.group == GROUP_ORDER[1]}
        ),
        "training_validation_latent_overlap": len(
            {
                record.latent_program_sha256
                for record in records
                if record.future_split == SPLIT_ORDER[0]
            }
            & {
                record.latent_program_sha256
                for record in records
                if record.future_split == SPLIT_ORDER[1]
            }
        ),
        "verifier_agreements": sum(record.verifier_agreement for record in records),
        "content_bearing_fields": False,
        "rendered_questions_persisted": False,
        "complete_dataset_generated": False,
        "schedule_gate_passed": True,
        "sealed_final_accessed": False,
    }
    payload["summary_sha256"] = canonical_sha256(payload)
    return payload


def write_full_schedule(
    config_path: Path,
    policy_path: Path,
    *,
    progress: bool = False,
) -> dict[str, object]:
    """Write ignored content-free records and tracked aggregate schedule evidence."""

    config = load_signal_pilot_config(config_path)
    records = build_full_schedule(config_path, policy_path, progress=progress)
    summary = validate_full_schedule(records, config_path, policy_path)
    config.full_schedule_raw_path.parent.mkdir(parents=True, exist_ok=True)
    config.full_schedule_raw_path.write_text(
        "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    config.full_schedule_summary_path.parent.mkdir(parents=True, exist_ok=True)
    config.full_schedule_summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
