from foundry.phase2.vetted_qlora import _schedule
from foundry.training.config import canonical_sha256


def test_schedule_requires_exact_hash_and_64_steps(tmp_path) -> None:
    value = [{"step": step, "loss_bearing_tokens": 1, "occurrences": []} for step in range(1, 65)]
    path = tmp_path / "schedule.json"
    import json

    path.write_text(json.dumps(value), encoding="utf-8")
    assert _schedule(path, canonical_sha256(value)) == value
