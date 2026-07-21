import pytest

from foundry.training.common_scale_selection import build_final_validation, build_selection


def _matrix(scale: float, passed: bool) -> list[dict[str, object]]:
    result = []
    for adapter in ("a" * 64, "b" * 64):
        for suite in ("adjudication", "holdout"):
            item: dict[str, object] = {
                "adapter_sha256": adapter,
                "suite_id": suite,
                "suite_sha256": (adapter + suite).encode().hex()[:64].ljust(64, "0"),
                "gate_passed": passed,
                "summary_sha256": (suite + adapter).encode().hex()[:64].ljust(64, "0"),
            }
            if scale != 1.0:
                item["adapter_scale"] = scale
                item["state_restoration_verified"] = True
            result.append(item)
    return result


def test_selection_stops_at_first_common_pass() -> None:
    decision = build_selection(
        matrices=[(1.0, _matrix(1.0, False)), (0.75, _matrix(0.75, True))],
        scale_config_sha256="c" * 64,
    )
    assert decision["selected_common_scale"] == 0.75
    assert decision["selection_used_gsm1k"] is False


def test_selection_rejects_skipped_or_post_pass_scales() -> None:
    with pytest.raises(ValueError, match="descending order"):
        build_selection(
            matrices=[(1.0, _matrix(1.0, False)), (0.5, _matrix(0.5, True))],
            scale_config_sha256="c" * 64,
        )
    with pytest.raises(ValueError, match="after a scale passed"):
        build_selection(
            matrices=[
                (1.0, _matrix(1.0, False)),
                (0.75, _matrix(0.75, True)),
                (0.5, _matrix(0.5, True)),
            ],
            scale_config_sha256="c" * 64,
        )


def test_final_validation_requires_both_arms_at_selected_scale() -> None:
    selection = build_selection(
        matrices=[(1.0, _matrix(1.0, False)), (0.75, _matrix(0.75, True))],
        scale_config_sha256="c" * 64,
    )
    assessments = _matrix(0.75, True)[:2]
    assessments[1] = dict(assessments[1])
    assessments[1]["adapter_sha256"] = "b" * 64
    assessments[1]["suite_id"] = assessments[0]["suite_id"]
    result = build_final_validation(selection=selection, assessments=assessments)
    assert result["gsm1k_authorized"] is True
