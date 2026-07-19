"""Source-controlled original sentence plans for the initial offline bank."""

# ruff: noqa: E501  # complete sentence plans remain readable as single literals

from __future__ import annotations

from foundry.synthesis.realization.ir import TargetKind
from foundry.synthesis.schema import DifficultyLevel
from foundry.synthesis.taxonomy import FailureCategory
from foundry.synthesis.template_bank.contracts import (
    SentencePlanSpec,
    SurfaceLexemeSpec,
    TemplateSpec,
)

TEMPLATE_BANK_VERSION = "foundry-template-bank-v2"
_DIFFICULTIES = (DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD)

_BOOKKEEPING_FRAMES = (
    "opening_inventory",
    "scheduled_transfers",
    "stock_log",
    "supply_ledger",
    "storage_updates",
    "shift_handover",
    "dispatch_record",
    "receiving_record",
    "allocation_log",
    "materials_register",
    "workshop_balance",
    "collection_changes",
    "equipment_register",
    "packing_inventory",
    "resource_movements",
    "cabinet_tally",
    "depot_balance",
    "closing_inventory",
)

_RATE_FRAMES = {
    "rate_total": ("steady_output", "timed_production", "interval_yield", "repeated_cycle"),
    "ratio_scale": ("paired_collections", "matched_batches", "proportional_sets", "scaled_groups"),
    "percentage": ("selected_share", "inspection_sample", "allocated_portion", "chosen_fraction"),
    "weighted_average": (
        "weighted_readings",
        "grouped_measurements",
        "combined_mean",
        "weighted_scores",
    ),
    "combined_rate": (
        "parallel_channels",
        "paired_streams",
        "joint_output",
        "simultaneous_processes",
    ),
}

_DISCRETE_FRAMES = {
    "two_type_allocation": (
        "two_design_build",
        "dual_recipe_plan",
        "mixed_package_order",
        "two_grade_allocation",
        "paired_type_schedule",
    ),
    "complete_packages": (
        "complete_container_count",
        "full_bundle_count",
        "whole_package_count",
        "complete_set_count",
        "filled_crate_count",
    ),
    "equal_distribution": (
        "equal_container_share",
        "balanced_location_split",
        "even_station_allocation",
        "uniform_group_share",
        "equal_batch_distribution",
    ),
    "dual_capacity": (
        "two_resource_capacity",
        "paired_supply_limit",
        "dual_material_build",
        "combined_constraint_capacity",
        "two_component_limit",
    ),
}

_SURFACE_PHRASES = {
    # Bookkeeping frames. These phrases are authored surface language, never normalized IDs.
    "opening_inventory": ("starting inventory", "inventory"),
    "scheduled_transfers": ("transfer schedule", "schedule"),
    "stock_log": ("stock count", "count"),
    "supply_ledger": ("supply tally", "tally"),
    "storage_updates": ("storage tally", "tally"),
    "shift_handover": ("handover inventory", "inventory"),
    "dispatch_record": ("dispatch tally", "tally"),
    "receiving_record": ("receiving tally", "tally"),
    "allocation_log": ("allocation count", "count"),
    "materials_register": ("materials count", "count"),
    "workshop_balance": ("workshop tally", "tally"),
    "collection_changes": ("collection tally", "tally"),
    "equipment_register": ("equipment count", "count"),
    "packing_inventory": ("packing count", "count"),
    "resource_movements": ("resource tally", "tally"),
    "cabinet_tally": ("cabinet inventory", "inventory"),
    "depot_balance": ("depot inventory", "inventory"),
    "closing_inventory": ("final inventory", "inventory"),
    # Rate and ratio frames.
    "steady_output": ("steady production", "production"),
    "timed_production": ("timed production", "production"),
    "interval_yield": ("interval output", "output"),
    "repeated_cycle": ("repeated operation", "operation"),
    "paired_collections": ("paired quantities", "quantities"),
    "matched_batches": ("matched quantities", "quantities"),
    "proportional_sets": ("proportional quantities", "quantities"),
    "scaled_groups": ("scaled quantities", "quantities"),
    "selected_share": ("selected portion", "portion"),
    "inspection_sample": ("inspection sample", "sample"),
    "allocated_portion": ("allocated portion", "portion"),
    "chosen_fraction": ("chosen portion", "portion"),
    "weighted_readings": ("weighted measurements", "measurements"),
    "grouped_measurements": ("grouped measurements", "measurements"),
    "combined_mean": ("combined average", "average"),
    "weighted_scores": ("weighted values", "values"),
    "parallel_channels": ("parallel operation", "operation"),
    "paired_streams": ("paired operation", "operation"),
    "joint_output": ("joint production", "production"),
    "simultaneous_processes": ("simultaneous operation", "operation"),
    # Discrete frames.
    "two_design_build": ("two-design allocation", "allocation"),
    "dual_recipe_plan": ("two-type allocation", "allocation"),
    "mixed_package_order": ("mixed-item allocation", "allocation"),
    "two_grade_allocation": ("two-grade allocation", "allocation"),
    "paired_type_schedule": ("paired-type allocation", "allocation"),
    "complete_container_count": ("complete-container task", "task"),
    "full_bundle_count": ("full-bundle task", "task"),
    "whole_package_count": ("whole-package task", "task"),
    "complete_set_count": ("complete-set task", "task"),
    "filled_crate_count": ("filled-container task", "task"),
    "equal_container_share": ("equal-share task", "task"),
    "balanced_location_split": ("balanced-allocation task", "task"),
    "even_station_allocation": ("even-allocation task", "task"),
    "uniform_group_share": ("uniform-share task", "task"),
    "equal_batch_distribution": ("equal-distribution task", "task"),
    "two_resource_capacity": ("two-resource limit", "limit"),
    "paired_supply_limit": ("paired-resource limit", "limit"),
    "dual_material_build": ("two-material limit", "limit"),
    "combined_constraint_capacity": ("combined-resource limit", "limit"),
    "two_component_limit": ("two-component limit", "limit"),
}


def _surface_lexeme(frame: str) -> SurfaceLexemeSpec:
    surface_key = frame.split(".")[-1]
    try:
        text, head = _SURFACE_PHRASES[surface_key]
    except KeyError as error:
        raise ValueError(f"no approved surface lexeme for frame {frame!r}") from error
    return SurfaceLexemeSpec(f"surface.{surface_key}", text, head)


def _plans(category: str) -> tuple[SentencePlanSpec, ...]:
    """Return four complete, meaningfully distinct grammatical constructions."""

    if category == "bookkeeping":
        return (
            SentencePlanSpec(
                "chronological_active",
                ("opening", "events", "question"),
                "{frame_lead}, {actor} records {initial_quantity} {item} in the {primary_location} at the {setting}.",
                "Next, {actor} transfers {event_quantity} {item} from the {event_origin} to the {event_destination}.",
                "After every recorded movement, {question_request}",
                "chronological",
                "active subject first",
            ),
            SentencePlanSpec(
                "ledger_passive",
                ("opening", "events", "question"),
                "{frame_lead}, the {primary_location} at the {setting} starts with {initial_quantity} {item}.",
                "A transfer of {event_quantity} {item} is recorded from the {event_origin} to the {event_destination}.",
                "Once the ledger is complete, {question_request}",
                "ledger sequence",
                "natural passive event",
            ),
            SentencePlanSpec(
                "movement_first",
                ("opening", "events", "question"),
                "Before the scheduled movements at the {setting}, {initial_quantity} {item} are stored in the {primary_location}.",
                "From the {event_origin}, {actor} moves {event_quantity} {item} to the {event_destination}.",
                "Using the resulting inventory, {question_request}",
                "before and after",
                "fronted source phrase",
            ),
            SentencePlanSpec(
                "register_summary",
                ("opening", "events", "question"),
                "The {frame_lead} register for the {setting} opens at {initial_quantity} {item} in the {primary_location}.",
                "The next entry moves {event_quantity} {item} from the {event_origin} to the {event_destination} under {actor}'s supervision.",
                "What should the closing register report when all entries are applied? {question_request}",
                "register summary",
                "record-centered construction",
            ),
        )
    if category == "rates":
        return (
            SentencePlanSpec(
                "direct_relation",
                ("facts", "question"),
                "At the {setting}, {fact_clause}",
                "{support_clause}",
                "{question_request}",
                "direct",
                "declarative facts then question",
            ),
            SentencePlanSpec(
                "operator_record",
                ("facts", "question"),
                "{actor}'s record at the {setting} states that {fact_clause}",
                "The same record notes that {support_clause}",
                "Based on those recorded quantities, {question_request}",
                "record based",
                "possessive subject",
            ),
            SentencePlanSpec(
                "condition_first",
                ("facts", "question"),
                "For this calculation at the {setting}, {support_clause}",
                "Under that condition, {fact_clause}",
                "{question_request}",
                "condition first",
                "fronted condition",
            ),
            SentencePlanSpec(
                "paired_summary",
                ("facts", "question"),
                "Two details govern the {frame_lead} situation at the {setting}: {fact_clause}",
                "In addition, {support_clause}",
                "Using both details, {question_request}",
                "paired summary",
                "colon introduction",
            ),
        )
    return (
        SentencePlanSpec(
            "constraint_sequence",
            ("facts", "constraints", "question"),
            "At the {setting}, {fact_clause}",
            "The governing condition is that {constraint_clause}",
            "{question_request}",
            "constraint sequence",
            "declarative condition",
        ),
        SentencePlanSpec(
            "planner_brief",
            ("facts", "constraints", "question"),
            "{actor} is preparing the {frame_lead} plan at the {setting}. {fact_clause}",
            "For the plan to be valid, {constraint_clause}",
            "Under these conditions, {question_request}",
            "planning brief",
            "actor-led construction",
        ),
        SentencePlanSpec(
            "condition_fronted",
            ("constraints", "facts", "question"),
            "Provided that {constraint_clause}, the {frame_lead} task can proceed at the {setting}.",
            "The available quantities are as follows: {fact_clause}",
            "{question_request}",
            "condition first",
            "fronted subordinate clause",
        ),
        SentencePlanSpec(
            "operations_note",
            ("facts", "constraints", "question"),
            "An operations note for the {setting} describes a {frame_lead} task. {fact_clause}",
            "It also specifies the following exact condition: {constraint_clause}",
            "Using only the stated quantities, {question_request}",
            "operations note",
            "document-centered construction",
        ),
    )


def _template(
    *,
    category: str,
    frame: str,
    index: int,
    targets: tuple[TargetKind, ...],
    event_types: tuple[str, ...],
) -> TemplateSpec:
    prefix = {"bookkeeping": "bk", "rates": "rt", "discrete": "ds"}[category]
    roles = {
        "bookkeeping": (
            "frame_lead",
            "actor",
            "initial_quantity",
            "item",
            "primary_location",
            "setting",
            "event_quantity",
            "event_origin",
            "event_destination",
            "question_request",
        ),
        "rates": (
            "setting",
            "actor",
            "frame_lead",
            "fact_clause",
            "support_clause",
            "question_request",
        ),
        "discrete": (
            "setting",
            "actor",
            "frame_lead",
            "fact_clause",
            "constraint_clause",
            "question_request",
        ),
    }[category]
    plans = _plans(category)
    return TemplateSpec(
        template_id=f"{prefix}.{index:02d}.{frame}",
        template_version="v1",
        reasoning_category={
            "bookkeeping": str(FailureCategory.MULTI_STEP_BOOKKEEPING),
            "rates": str(FailureCategory.RATE_RATIO_PERCENTAGE),
            "discrete": str(FailureCategory.CONSTRAINT_DISCRETE),
        }[category],
        semantic_frame=frame,
        surface_lexeme=_surface_lexeme(frame),
        compatible_target_types=targets,
        required_semantic_event_types=event_types,
        required_placeholder_roles=roles,
        allowed_units=("typed_count", "typed_rate", "typed_ratio", "typed_percentage"),
        allowed_object_families=("explicit_countable_object",),
        supported_difficulty_levels=_DIFFICULTIES,
        clause_order_plan=plans[0].clause_order,
        question_form="explicit typed target question",
        sentence_plan_variants=plans,
        optional_context_policy="omit_unless_source_ir_marks_safe_context",
        output_contract_compatible=True,
        provenance="original_hand_authored_foundry_v1",
        review_status="human_review_pending",
    )


def build_template_bank() -> tuple[TemplateSpec, ...]:
    """Build the immutable initial bank: 58 frames and 232 sentence plans."""

    templates: list[TemplateSpec] = []
    for index, frame in enumerate(_BOOKKEEPING_FRAMES, start=1):
        templates.append(
            _template(
                category="bookkeeping",
                frame=frame,
                index=index,
                targets=(TargetKind.REMAINING_QUANTITY, TargetKind.GROUP_COUNT),
                event_types=("initial_inventory", "typed_ledger_change", "target"),
            )
        )
    rate_targets = {
        "rate_total": TargetKind.TOTAL_QUANTITY,
        "ratio_scale": TargetKind.RATIO,
        "percentage": TargetKind.PERCENTAGE,
        "weighted_average": TargetKind.WEIGHTED_MEAN,
        "combined_rate": TargetKind.TOTAL_QUANTITY,
    }
    index = 0
    for relation, rate_frames in _RATE_FRAMES.items():
        for frame in rate_frames:
            index += 1
            templates.append(
                _template(
                    category="rates",
                    frame=f"{relation}.{frame}",
                    index=index,
                    targets=(rate_targets[relation],),
                    event_types=(relation, "target"),
                )
            )
    discrete_targets = {
        "two_type_allocation": TargetKind.COUNT,
        "complete_packages": TargetKind.GROUP_COUNT,
        "equal_distribution": TargetKind.COUNT,
        "dual_capacity": TargetKind.CAPACITY,
    }
    index = 0
    for relation, discrete_frames in _DISCRETE_FRAMES.items():
        for frame in discrete_frames:
            index += 1
            templates.append(
                _template(
                    category="discrete",
                    frame=f"{relation}.{frame}",
                    index=index,
                    targets=(discrete_targets[relation],),
                    event_types=(relation, "target"),
                )
            )
    if len(templates) != 58 or sum(len(item.sentence_plan_variants) for item in templates) != 232:
        raise AssertionError("initial template-bank capacity changed")
    if len({item.template_id for item in templates}) != len(templates):
        raise AssertionError("template IDs are not unique")
    return tuple(templates)
