"""Exact rate, ratio, percentage, and weighted-average generation."""

# ruff: noqa: E501  # controlled natural-language templates are kept readable as full clauses

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction

from foundry.synthesis.generators import (
    CandidateDraft,
    GeneratorVerification,
    candidate_id,
    exact_value,
    training_completion,
    verification_result,
)
from foundry.synthesis.object_units import (
    Countability,
    ObjectKind,
    QuantityUnit,
    validate_combination,
)
from foundry.synthesis.quality import (
    NounFormEvidence,
    RenderQualityMetadata,
    UnitTransitionEvidence,
)
from foundry.synthesis.schema import (
    DifficultyLevel,
    LatentProgramSpec,
    ProgramParameter,
    ProgramStep,
)
from foundry.synthesis.taxonomy import FailureCategory

GENERATOR_ID = "exact-rate-ratio-relations"
GENERATOR_VERSION = "2"


def _kind(family: str, singular: str, plural: str) -> ObjectKind:
    return ObjectKind(
        family=family,
        singular=singular,
        plural=plural,
        countability=Countability.DISCRETE,
        unit=QuantityUnit.ITEM,
        combination_key=f"{family}:count",
        transferable=False,
        supported_verbs=("produce", "inspect", "prepare", "deliver", "carry"),
        supported_containers=("batch",),
    )


@dataclass(frozen=True)
class _RateScenario:
    scenario_id: str
    setting: str
    operator: str
    object_kind: ObjectKind
    safe_context: str


_SCENARIOS = (
    _RateScenario(
        "print",
        "solar print room",
        "Ari",
        _kind("signal_card", "signal card", "signal cards"),
        "Color calibration is handled before production.",
    ),
    _RateScenario(
        "beacon",
        "coastal beacon lab",
        "Bea",
        _kind("culture_panel", "culture panel", "culture panels"),
        "A separate sensor logs room temperature.",
    ),
    _RateScenario(
        "lens",
        "mountain lens shop",
        "Chen",
        _kind("lens_blank", "lens blank", "lens blanks"),
        "Packaging occurs after the measured run.",
    ),
    _RateScenario(
        "nursery",
        "floating plant nursery",
        "Dina",
        _kind("sample_capsule", "sample capsule", "sample capsules"),
        "Water testing uses an independent schedule.",
    ),
    _RateScenario(
        "acoustics",
        "desert acoustics station",
        "Eli",
        _kind("echo_tag", "echo tag", "echo tags"),
        "The microphone check does not alter output.",
    ),
    _RateScenario(
        "loom",
        "orbital fabric loom",
        "Farah",
        _kind("woven_strip", "woven strip", "woven strips"),
        "Thread tension is recorded separately.",
    ),
    _RateScenario(
        "battery",
        "polar battery archive",
        "Gus",
        _kind("power_cell", "power cell", "power cells"),
        "Charging tests happen after counting.",
    ),
    _RateScenario(
        "kiln",
        "volcanic crystal kiln",
        "Hui",
        _kind("crystal_plate", "crystal plate", "crystal plates"),
        "Cooling time is outside this calculation.",
    ),
    _RateScenario(
        "greenhouse",
        "urban greenhouse",
        "Inez",
        _kind("plant_label", "plant label", "plant labels"),
        "Seed trays follow another workflow.",
    ),
    _RateScenario(
        "robot",
        "robot assembly bay",
        "Jamal",
        _kind("control_chip", "control chip", "control chips"),
        "Diagnostic cables use another counter.",
    ),
    _RateScenario(
        "bakery",
        "automated bakery",
        "Kira",
        _kind("bread_token", "bread token", "bread tokens"),
        "Oven cleaning is scheduled separately.",
    ),
    _RateScenario(
        "harbor",
        "harbor dispatch office",
        "Luis",
        _kind("cargo_tag", "cargo tag", "cargo tags"),
        "Vessel logs do not affect tag totals.",
    ),
    _RateScenario(
        "museum",
        "museum catalog room",
        "Mei",
        _kind("archive_card", "archive card", "archive cards"),
        "Exhibit labels use a separate batch.",
    ),
    _RateScenario(
        "clinic",
        "mobile clinic",
        "Noor",
        _kind("test_strip", "test strip", "test strips"),
        "Patient scheduling is not part of the count.",
    ),
    _RateScenario(
        "weather",
        "weather balloon workshop",
        "Omar",
        _kind("sensor_tab", "sensor tab", "sensor tabs"),
        "Forecast data is processed elsewhere.",
    ),
    _RateScenario(
        "library",
        "digital library lab",
        "Pia",
        _kind("scan_ticket", "scan ticket", "scan tickets"),
        "Book returns follow another queue.",
    ),
    _RateScenario(
        "aquarium",
        "aquarium food station",
        "Quin",
        _kind("feed_pouch", "feed pouch", "feed pouches"),
        "Tank inspection uses an independent form.",
    ),
    _RateScenario(
        "rail",
        "rail maintenance depot",
        "Rosa",
        _kind("inspection_clip", "inspection clip", "inspection clips"),
        "Tool checkout is logged separately.",
    ),
    _RateScenario(
        "studio",
        "animation studio",
        "Sami",
        _kind("render_tile", "render tile", "render tiles"),
        "Audio processing uses another system.",
    ),
    _RateScenario(
        "orchard",
        "orchard sorting shed",
        "Tala",
        _kind("grade_sticker", "grade sticker", "grade stickers"),
        "Crate washing is not part of the run.",
    ),
)

_RENDERER_FAMILIES = (
    "process_first",
    "operator_report",
    "result_request_first",
    "passive_measurement",
    "shift_summary",
    "comparison_note",
)
_PERCENTS = (10, 20, 25, 40, 50, 60, 75, 80)
TEMPLATE_FAMILIES = (
    "rate_total",
    "ratio_scale",
    "percentage",
    "weighted_average",
    "combined_rate",
)
RENDERING_VARIANTS_PER_FAMILY = len(_RENDERER_FAMILIES)
SCENARIO_DOMAIN_COUNT = len(_SCENARIOS)


def _mode(variant: int) -> str:
    return ("rate_total", "ratio_scale", "percentage", "weighted_average", "combined_rate")[
        variant % 5
    ]


def _render_rate_question(
    *, mode: str, family: str, scenario: _RateScenario, values: dict[str, object]
) -> tuple[str, tuple[str, ...], str]:
    plural = scenario.object_kind.plural
    setting = scenario.setting
    actor = scenario.operator
    if mode == "rate_total":
        rate = _payload_int(values, "rate")
        intervals = _payload_int(values, "intervals")
        forms = (
            (
                f"A calibrated process at the {setting} produces {rate} {plural} per cycle.",
                f"It completes {intervals} cycles.",
                f"How many {plural} are produced in total?",
            ),
            (
                f"{actor} reports {intervals} completed cycles at the {setting}.",
                f"Each cycle yields {rate} {plural}.",
                f"What is the combined output of {plural}?",
            ),
            (
                f"What total number of {plural} results from {intervals} cycles?",
                f"The {setting} makes {rate} {plural} during every cycle.",
                "Calculate the full production count.",
            ),
            (
                f"At the {setting}, {rate} {plural} are produced during each calibrated cycle.",
                f"Exactly {intervals} cycles are completed.",
                f"Determine how many {plural} leave the process.",
            ),
            (
                f"The shift summary lists a rate of {rate} {plural} for one cycle and {intervals} cycles run.",
                scenario.safe_context,
                f"How many {plural} belong in the output total?",
            ),
            (
                f"{actor} compares identical cycles at the {setting}.",
                f"There are {intervals} cycles, each contributing {rate} {plural}.",
                f"Find the total contribution of {plural}.",
            ),
        )
    elif mode == "ratio_scale":
        first = _payload_int(values, "first_part")
        second = _payload_int(values, "second_part")
        known = _payload_int(values, "known")
        forms = (
            (
                f"The {setting} prepares amber and cobalt {plural} in the ratio {first}:{second}.",
                f"The amber portion contains {known} {plural}.",
                f"How many {plural} are in the cobalt portion?",
            ),
            (
                f"{actor} records {known} amber {plural} in a batch whose amber-to-cobalt ratio is {first}:{second}.",
                "The ratio is exact.",
                "Determine the cobalt count.",
            ),
            (
                f"How large is the cobalt share of {plural}?",
                f"At the {setting}, amber and cobalt shares follow {first}:{second}, and the amber share is {known}.",
                "Compute the corresponding cobalt quantity.",
            ),
            (
                f"At the {setting}, {plural} are divided between amber and cobalt groups in an exact {first}:{second} ratio.",
                f"An amber group of {known} is observed.",
                "What cobalt group size matches it?",
            ),
            (
                f"The shift note gives {known} amber {plural} and the proportion {first} amber parts to {second} cobalt parts.",
                scenario.safe_context,
                f"Find the number of cobalt {plural}.",
            ),
            (
                f"{actor} scales two matched collections of {plural}.",
                f"The first-to-second ratio is {first}:{second}, with {known} in the first collection.",
                "What count belongs in the second collection?",
            ),
        )
    elif mode == "percentage":
        percent = _payload_int(values, "percent")
        base = _payload_int(values, "base")
        forms = (
            (
                f"A quality check at the {setting} examines {percent}% of a batch containing {base} {plural}.",
                "The percentage is applied exactly.",
                f"How many {plural} are examined?",
            ),
            (
                f"{actor} selects exactly {percent}% from {base} prepared {plural}.",
                "The rest remain untouched.",
                "What is the selected count?",
            ),
            (
                f"How many {plural} make up {percent}% of a batch of {base}?",
                f"The batch was independently prepared at the {setting}.",
                "Calculate the exact subset size.",
            ),
            (
                f"From {base} {plural} at the {setting}, a fraction of {percent}% is marked for inspection.",
                "No rounding is permitted.",
                "Determine the number marked.",
            ),
            (
                f"The shift summary lists {base} total {plural} and an inspection share of {percent}%.",
                scenario.safe_context,
                "Find the exact inspection total.",
            ),
            (
                f"{actor} compares the full batch with a {percent}% sample.",
                f"The full batch contains {base} {plural}.",
                f"What number of {plural} belongs to the sample?",
            ),
        )
    elif mode == "weighted_average":
        weights_raw = values["weights"]
        marks_raw = values["values"]
        if not isinstance(weights_raw, list) or not isinstance(marks_raw, list):
            raise ValueError("weighted rendering requires integer lists")
        weights = [int(item) for item in weights_raw]
        marks = [int(item) for item in marks_raw]
        joined = "; ".join(
            f"{weight} panels with {mark} marks each"
            for weight, mark in zip(weights, marks, strict=True)
        )
        forms = (
            (
                f"At the {setting}, a report lists {joined}.",
                "Every panel has equal weight within its group.",
                "What is the exact average number of marks per panel?",
            ),
            (
                f"{actor} combines panel groups: {joined}.",
                "The group sizes must weight their mark counts.",
                "Calculate the weighted mean of marks per panel.",
            ),
            (
                "What exact weighted average describes the panels?",
                f"The {setting} records {joined}.",
                "Report the mean marks per panel.",
            ),
            (
                f"At the {setting}, mark counts are measured across groups containing {joined}.",
                "The observations are pooled by panel count.",
                "Determine the weighted average.",
            ),
            (
                f"The shift summary contains {joined}.",
                scenario.safe_context,
                "Find the exact panel-weighted mark count.",
            ),
            (
                f"{actor} compares several unequal panel groups.",
                f"Their measurements are {joined}.",
                "What weighted mean gives marks per panel?",
            ),
        )
    else:
        first = _payload_int(values, "first_rate")
        second = _payload_int(values, "second_rate")
        intervals = _payload_int(values, "intervals")
        forms = (
            (
                f"Two channels at the {setting} deliver {first} and {second} {plural} per interval.",
                f"Both operate for {intervals} intervals.",
                f"How many {plural} arrive altogether?",
            ),
            (
                f"{actor} monitors two simultaneous streams for {intervals} intervals.",
                f"Their rates are {first} and {second} {plural} per interval.",
                "What combined total is delivered?",
            ),
            (
                f"What is the total delivery of {plural} from two channels over {intervals} intervals?",
                f"One channel contributes {first} per interval and the other contributes {second}.",
                "Calculate their shared output.",
            ),
            (
                f"At the {setting}, {plural} are delivered by two independent channels at rates {first} and {second} per interval.",
                f"Each channel runs for {intervals} intervals.",
                "Determine the aggregate arrival count.",
            ),
            (
                f"The shift summary lists {intervals} intervals and channel rates of {first} and {second} {plural}.",
                scenario.safe_context,
                "Find the total received from both channels.",
            ),
            (
                f"{actor} compares two equal-duration streams at the {setting}.",
                f"Across {intervals} intervals they provide {first} and {second} {plural} per interval.",
                f"How many {plural} do the streams provide together?",
            ),
        )
    selected = forms[_RENDERER_FAMILIES.index(family)]
    return " ".join(selected), selected[:-1], selected[-1]


def generate_rates(
    *,
    seed: int,
    difficulty: DifficultyLevel,
    variant: int,
    output_contract_enabled: bool,
) -> CandidateDraft:
    """Generate one exact rational-relationship candidate without external text."""

    if seed < 0 or variant < 0:
        raise ValueError("rate seed and variant must be non-negative")
    rng = random.Random(seed)
    mode = _mode(variant)
    occurrence = variant // 5
    scenario = _SCENARIOS[(variant * 7 + occurrence) % len(_SCENARIOS)]
    renderer_family = _RENDERER_FAMILIES[occurrence % len(_RENDERER_FAMILIES)]
    context = scenario.setting
    item = scenario.object_kind.plural
    parameters: list[ProgramParameter] = []
    steps: list[ProgramStep] = []
    trace: list[str] = []
    payload: dict[str, object] = {"mode": mode, "unit_check": "exact"}

    if mode == "rate_total":
        rate = rng.randint(3, 14)
        intervals = rng.randint(2, 7) + (2 if difficulty is DifficultyLevel.HARD else 0)
        answer_fraction = Fraction(rate * intervals)
        parameters.extend(
            (
                ProgramParameter("rate", exact_value(rate), "items/interval"),
                ProgramParameter("intervals", exact_value(intervals), "intervals"),
            )
        )
        steps.append(
            ProgramStep("total", "multiply", ("rate", "intervals"), exact_value(answer_fraction))
        )
        question = (
            f"At {context}, one calibrated cycle produces {rate} {item}. "
            f"The device completes exactly {intervals} cycles. What total number of {item} "
            "does it produce?"
        )
        trace.append(f"Multiply the exact per-cycle rate {rate} by {intervals} cycles.")
        payload.update(rate=rate, intervals=intervals)
        answer_symbol = "total"
    elif mode == "ratio_scale":
        first_part = rng.randint(2, 7)
        second_part = rng.randint(2, 9)
        if first_part == second_part:
            second_part += 1
        scale = rng.randint(3, 10)
        known = first_part * scale
        answer_fraction = Fraction(second_part * scale)
        parameters.extend(
            (
                ProgramParameter("first_part", exact_value(first_part), "ratio_part"),
                ProgramParameter("second_part", exact_value(second_part), "ratio_part"),
                ProgramParameter("known", exact_value(known), "items"),
            )
        )
        steps.extend(
            (
                ProgramStep("scale", "divide", ("known", "first_part"), exact_value(scale)),
                ProgramStep(
                    "paired_amount",
                    "multiply",
                    ("scale", "second_part"),
                    exact_value(answer_fraction),
                ),
            )
        )
        question = (
            f"At {context}, amber and cobalt {item} are prepared in the exact ratio "
            f"{first_part}:{second_part}. If the amber share contains {known}, how many "
            "belong to the cobalt share?"
        )
        trace.extend(
            (
                f"Divide {known} by the first ratio part {first_part} to obtain scale {scale}.",
                f"Multiply scale {scale} by the second part {second_part}.",
            )
        )
        payload.update(first_part=first_part, second_part=second_part, known=known)
        answer_symbol = "paired_amount"
    elif mode == "percentage":
        percent = _PERCENTS[variant % len(_PERCENTS)]
        base = rng.randint(4, 16) * 100
        answer_fraction = Fraction(base * percent, 100)
        parameters.extend(
            (
                ProgramParameter("base", exact_value(base), "items"),
                ProgramParameter("percent", exact_value(percent), "percent"),
                ProgramParameter("hundred", exact_value(100), "percent_base"),
            )
        )
        steps.extend(
            (
                ProgramStep("scaled", "multiply", ("base", "percent"), exact_value(base * percent)),
                ProgramStep(
                    "selected",
                    "divide",
                    ("scaled", "hundred"),
                    exact_value(answer_fraction),
                ),
            )
        )
        question = (
            f"At {context}, a quality scan examines {percent}% of an independently prepared "
            f"batch of {base} {item}. How many {item} are examined?"
        )
        trace.extend(
            (
                f"Represent {percent}% exactly as {percent}/100.",
                f"Multiply {base} by {percent}/100 without floating point.",
            )
        )
        payload.update(base=base, percent=percent)
        answer_symbol = "selected"
    elif mode == "weighted_average":
        group_count = 3 if difficulty is DifficultyLevel.HARD else 2
        weights = [rng.randint(2, 7) for _ in range(group_count)]
        values = [rng.randint(4, 18) for _ in range(group_count)]
        weighted_total = sum(weight * value for weight, value in zip(weights, values, strict=True))
        total_weight = sum(weights)
        answer_fraction = Fraction(weighted_total, total_weight)
        for index, (weight, value) in enumerate(zip(weights, values, strict=True), start=1):
            parameters.extend(
                (
                    ProgramParameter(f"weight_{index}", exact_value(weight), "panels"),
                    ProgramParameter(f"value_{index}", exact_value(value), "marks/panel"),
                )
            )
            steps.append(
                ProgramStep(
                    f"weighted_{index}",
                    "multiply",
                    (f"weight_{index}", f"value_{index}"),
                    exact_value(weight * value),
                )
            )
        parameters.append(ProgramParameter("total_weight", exact_value(total_weight), "panels"))
        steps.append(
            ProgramStep(
                "weighted_total",
                "sum_many",
                tuple(f"weighted_{index}" for index in range(1, group_count + 1)),
                exact_value(weighted_total),
            )
        )
        steps.append(
            ProgramStep(
                "weighted_mean",
                "divide",
                ("weighted_total", "total_weight"),
                exact_value(answer_fraction),
            )
        )
        groups = "; ".join(
            f"{weight} panels carry {value} marks each"
            for weight, value in zip(weights, values, strict=True)
        )
        question = (
            f"At {context}, {groups}. What is the exact weighted average number of marks per panel?"
        )
        trace.extend(
            (
                f"The exact weighted total is {weighted_total}.",
                f"Divide by the total weight {total_weight} to obtain the weighted average.",
            )
        )
        payload.update(weights=weights, values=values)
        answer_symbol = "weighted_mean"
    else:
        first_rate = rng.randint(2, 9)
        second_rate = rng.randint(2, 9)
        intervals = rng.randint(3, 8)
        answer_fraction = Fraction((first_rate + second_rate) * intervals)
        parameters.extend(
            (
                ProgramParameter("first_rate", exact_value(first_rate), "items/interval"),
                ProgramParameter("second_rate", exact_value(second_rate), "items/interval"),
                ProgramParameter("intervals", exact_value(intervals), "intervals"),
            )
        )
        steps.extend(
            (
                ProgramStep(
                    "combined_rate",
                    "add",
                    ("first_rate", "second_rate"),
                    exact_value(first_rate + second_rate),
                ),
                ProgramStep(
                    "combined_total",
                    "multiply",
                    ("combined_rate", "intervals"),
                    exact_value(answer_fraction),
                ),
            )
        )
        question = (
            f"At {context}, two independent channels deliver {first_rate} and {second_rate} "
            f"{item} per interval. Both run for {intervals} intervals. How many arrive altogether?"
        )
        trace.extend(
            (
                f"Add the two compatible rates to obtain {first_rate + second_rate} per interval.",
                f"Multiply that exact rate by {intervals} intervals.",
            )
        )
        payload.update(first_rate=first_rate, second_rate=second_rate, intervals=intervals)
        answer_symbol = "combined_total"

    question, rendered_clauses, conclusion = _render_rate_question(
        mode=mode,
        family=renderer_family,
        scenario=scenario,
        values=payload,
    )
    compatibility_errors = validate_combination((scenario.object_kind, scenario.object_kind))
    if compatibility_errors:
        raise ValueError("incompatible rate scenario: " + "; ".join(compatibility_errors))

    unit_transitions: tuple[UnitTransitionEvidence, ...]
    if mode in {"rate_total", "combined_rate"}:
        unit_transitions = (
            UnitTransitionEvidence(
                source_unit="items/interval",
                target_unit=QuantityUnit.ITEM,
                conversion_explicit=True,
            ),
        )
    elif mode == "percentage":
        unit_transitions = (
            UnitTransitionEvidence(
                source_unit="percent",
                target_unit=QuantityUnit.ITEM,
                conversion_explicit=True,
            ),
        )
    elif mode == "weighted_average":
        unit_transitions = (
            UnitTransitionEvidence(
                source_unit="marks/panel",
                target_unit="marks/panel",
                conversion_explicit=True,
            ),
        )
    else:
        unit_transitions = ()

    noun_forms: tuple[NounFormEvidence, ...] = ()
    if mode != "weighted_average":
        noun_forms = (
            NounFormEvidence(
                quantity=2,
                rendered_noun=scenario.object_kind.plural,
                object_kind=scenario.object_kind,
            ),
        )

    answer = exact_value(answer_fraction)
    trace_tuple = tuple(trace)
    program = LatentProgramSpec(
        program_family=f"{GENERATOR_ID}:{mode}",
        parameters=tuple(parameters),
        steps=tuple(steps),
        constraints=(
            "relationships use exact rational arithmetic",
            "units are dimensionally valid",
        ),
        answer_symbol=answer_symbol,
    )
    return CandidateDraft(
        candidate_id=candidate_id(GENERATOR_ID, seed, variant),
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        random_seed=seed,
        target_failure_category=FailureCategory.RATE_RATIO_PERCENTAGE,
        secondary_skill_tags=(mode, "exact_rational"),
        difficulty_level=difficulty,
        output_contract_enabled=output_contract_enabled,
        latent_program=program,
        rendered_question=question,
        deterministic_solution_trace=trace_tuple,
        canonical_final_answer=answer,
        training_completion=training_completion(
            trace_tuple, answer, output_contract_enabled=output_contract_enabled
        ),
        quality_metadata=RenderQualityMetadata(
            scenario_id=scenario.scenario_id,
            renderer_family=renderer_family,
            clauses=rendered_clauses,
            declared_entity_ids=("operator", "process", "batch"),
            referenced_entity_ids=("process",),
            pronoun_referent_ids=("process",),
            noun_forms=noun_forms,
            combination_groups=((scenario.object_kind, scenario.object_kind),),
            operations=(),
            unit_transitions=unit_transitions,
            target_symbol=answer_symbol,
            target_mentions=1,
            conclusion=conclusion,
            constraints_tied=False,
            grammar_complete=True,
        ),
        structure_signature={
            "generator": GENERATOR_ID,
            "mode": mode,
            "difficulty": difficulty,
            "scenario_family": scenario.scenario_id,
            "renderer_family": renderer_family,
            "step_operations": [step.operation for step in steps],
            "answer_symbol": answer_symbol,
        },
        verifier_payload=payload,
    )


def _payload_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"invalid integer payload field {key}")
    return value


def _payload_int_list(payload: dict[str, object], key: str) -> list[int]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ValueError(f"invalid integer-list payload field {key}")
    return [int(item) for item in value]


def _equation_answer(payload: dict[str, object]) -> Fraction:
    mode = payload.get("mode")
    if mode == "rate_total":
        return Fraction(_payload_int(payload, "rate") * _payload_int(payload, "intervals"))
    if mode == "ratio_scale":
        return Fraction(
            _payload_int(payload, "known") * _payload_int(payload, "second_part"),
            _payload_int(payload, "first_part"),
        )
    if mode == "percentage":
        return Fraction(_payload_int(payload, "base") * _payload_int(payload, "percent"), 100)
    if mode == "weighted_average":
        weights = _payload_int_list(payload, "weights")
        values = _payload_int_list(payload, "values")
        if len(weights) != len(values) or not weights:
            raise ValueError("weighted-average payload lengths differ")
        return Fraction(
            sum(weight * value for weight, value in zip(weights, values, strict=True)),
            sum(weights),
        )
    if mode == "combined_rate":
        return Fraction(
            (_payload_int(payload, "first_rate") + _payload_int(payload, "second_rate"))
            * _payload_int(payload, "intervals")
        )
    raise ValueError("unsupported rational-relation mode")


def verify_rate_equation(draft: CandidateDraft) -> GeneratorVerification:
    """Compute the answer directly from the exact mode equation."""

    try:
        answer = _equation_answer(draft.verifier_payload)
    except (ValueError, ZeroDivisionError) as error:
        return verification_result(
            verifier_id="rate-equation-v1",
            method_family="exact_rational_equation",
            answer=None,
            failure_reason="invalid_equation_payload",
            evidence_payload={"error": type(error).__name__},
        )
    return verification_result(
        verifier_id="rate-equation-v1",
        method_family="exact_rational_equation",
        answer=answer,
        failure_reason=None,
        evidence_payload={"mode": draft.verifier_payload.get("mode")},
    )


def verify_rate_inverse(draft: CandidateDraft) -> GeneratorVerification:
    """Check the answer with cross-products, inverse substitution, and units."""

    payload = draft.verifier_payload
    if payload.get("unit_check") != "exact":
        return verification_result(
            verifier_id="rate-inverse-v1",
            method_family="cross_product_inverse_units",
            answer=None,
            failure_reason="unit_consistency_failure",
            evidence_payload={},
        )
    try:
        candidate = draft.canonical_final_answer.fraction
        mode = payload.get("mode")
        if mode == "rate_total":
            valid = candidate / _payload_int(payload, "intervals") == _payload_int(payload, "rate")
        elif mode == "ratio_scale":
            valid = _payload_int(payload, "known") * _payload_int(
                payload, "second_part"
            ) == candidate * _payload_int(payload, "first_part")
        elif mode == "percentage":
            valid = candidate * 100 == _payload_int(payload, "base") * _payload_int(
                payload, "percent"
            )
        elif mode == "weighted_average":
            weights = _payload_int_list(payload, "weights")
            values = _payload_int_list(payload, "values")
            valid = candidate * sum(weights) == sum(
                weight * value for weight, value in zip(weights, values, strict=True)
            )
        elif mode == "combined_rate":
            valid = candidate / _payload_int(payload, "intervals") == (
                _payload_int(payload, "first_rate") + _payload_int(payload, "second_rate")
            )
        else:
            valid = False
    except (ValueError, ZeroDivisionError):
        valid = False
    if not valid:
        return verification_result(
            verifier_id="rate-inverse-v1",
            method_family="cross_product_inverse_units",
            answer=None,
            failure_reason="inverse_or_cross_product_failure",
            evidence_payload={"mode": payload.get("mode")},
        )
    return verification_result(
        verifier_id="rate-inverse-v1",
        method_family="cross_product_inverse_units",
        answer=candidate,
        failure_reason=None,
        evidence_payload={"mode": payload.get("mode"), "unit_check": True},
    )


def validate_rate_constraints(draft: CandidateDraft) -> tuple[str, ...]:
    """Reject unsafe or ambiguous rational candidates."""

    reasons: list[str] = []
    if draft.ambiguity_flags:
        reasons.append("ambiguous_target")
    if draft.canonical_final_answer.denominator <= 0:
        reasons.append("invalid_denominator")
    if draft.verifier_payload.get("unit_check") != "exact":
        reasons.append("unit_consistency_failure")
    return tuple(reasons)
