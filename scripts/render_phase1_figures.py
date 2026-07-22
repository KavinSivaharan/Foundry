"""Render deterministic, aggregate-only SVG figures for the Phase 1 release."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "results" / "phase1_figure_data.json"
OUTPUT_DIR = ROOT / "docs" / "assets" / "phase1"

NAVY = "#17324d"
BLUE = "#2563eb"
TEAL = "#0f766e"
GREEN = "#15803d"
AMBER = "#b45309"
RED = "#b91c1c"
INK = "#172033"
MUTED = "#586579"
GRID = "#d9e1ea"
LIGHT = "#f4f7fa"
WHITE = "#ffffff"


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value using Foundry's canonical representation."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_data() -> tuple[dict[str, Any], str]:
    """Load the figure source and verify its self-hash."""

    value: object = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("phase1 figure data must be an object")
    data = dict(value)
    recorded = data.pop("data_sha256", None)
    if not isinstance(recorded, str) or canonical_sha256(data) != recorded:
        raise ValueError("phase1 figure data self-hash differs")
    data["data_sha256"] = recorded
    return data, recorded


def esc(value: object) -> str:
    """Escape one SVG text value."""

    return html.escape(str(value), quote=True)


def text(
    x: float,
    y: float,
    value: object,
    css_class: str,
    *,
    anchor: str = "start",
) -> str:
    """Create one SVG text element."""

    return (
        f'<text x="{x:.1f}" y="{y:.1f}" class="{css_class}" '
        f'text-anchor="{anchor}">{esc(value)}</text>'
    )


def common_header(title: str, description: str, source_hash: str) -> list[str]:
    """Return an accessible SVG opening and shared publication styles."""

    return [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="700" '
        'viewBox="0 0 1200 700" role="img" aria-labelledby="title desc" '
        f'data-source-sha256="{source_hash}">',
        f'<title id="title">{esc(title)}</title>',
        f'<desc id="desc">{esc(description)}</desc>',
        "<style>",
        "text { font-family: 'Segoe UI', Arial, sans-serif; fill: #172033; }",
        ".title { font-size: 30px; font-weight: 600; }",
        ".subtitle { font-size: 16px; fill: #586579; }",
        ".label { font-size: 18px; font-weight: 600; }",
        ".value { font-size: 17px; font-weight: 600; }",
        ".small { font-size: 14px; fill: #586579; }",
        ".axis { font-size: 14px; fill: #586579; }",
        ".node-label { font-size: 17px; font-weight: 600; }",
        ".node-detail { font-size: 13px; fill: #586579; }",
        "</style>",
        f'<rect width="1200" height="700" fill="{WHITE}"/>',
        text(50, 52, title, "title"),
    ]


def close_svg(lines: list[str]) -> str:
    """Close and serialize an SVG document."""

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_accuracy(data: dict[str, Any], source_hash: str) -> str:
    """Render base, generic, and targeted GSM1K development accuracy."""

    rows = data["accuracy"]
    lines = common_header(
        "Frozen GSM1K development accuracy",
        "Bar chart comparing the untouched base with matched generic and targeted SFT. "
        "Targeted exceeds generic but both remain below the base.",
        source_hash,
    )
    lines.append(text(50, 82, "814 examples; common runtime LoRA scale = 0.50", "subtitle"))
    left = 290.0
    right = 920.0
    plot_width = right - left
    top = 145.0
    bottom = 545.0
    for tick in range(0, 71, 10):
        x = left + plot_width * tick / 70.0
        lines.append(
            f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{bottom:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        lines.append(text(x, bottom + 26, tick, "axis", anchor="middle"))
    lines.append(text((left + right) / 2, 610, "Accuracy (%)", "label", anchor="middle"))
    colors = [NAVY, "#7b8798", TEAL]
    for index, row in enumerate(rows):
        y = 180.0 + index * 125.0
        accuracy = float(row["accuracy_percent"])
        width = plot_width * accuracy / 70.0
        lines.append(text(left - 18, y + 31, row["label"], "label", anchor="end"))
        lines.append(
            f'<rect x="{left:.1f}" y="{y:.1f}" width="{width:.1f}" height="58" '
            f'rx="5" fill="{colors[index]}"/>'
        )
        value = f"{accuracy:.4f}%  ({row['correct']}/{row['total']})"
        lines.append(text(1120, y + 26, value, "value", anchor="end"))
        if index > 0:
            delta = float(row["delta_percentage_points_vs_base"])
            lines.append(text(1120, y + 49, f"{delta:.4f} points vs base", "small", anchor="end"))
    base_x = left + plot_width * float(rows[0]["accuracy_percent"]) / 70.0
    lines.append(
        f'<line x1="{base_x:.1f}" y1="{top - 10:.1f}" x2="{base_x:.1f}" '
        f'y2="{bottom:.1f}" stroke="{NAVY}" stroke-width="2" stroke-dasharray="6 5"/>'
    )
    lines.append(text(base_x - 8, 130, "Untouched-base reference", "small", anchor="end"))
    lines.append(
        text(
            50,
            665,
            "Provisional one-seed result; targeted did not beat the untouched base.",
            "subtitle",
        )
    )
    return close_svg(lines)


def render_paired_difference(data: dict[str, Any], source_hash: str) -> str:
    """Render the paired targeted-minus-generic estimate and confidence interval."""

    paired = data["paired_difference"]
    estimate = float(paired["estimate_percentage_points"])
    low, high = (float(value) for value in paired["interval_percentage_points"])
    lines = common_header(
        "Paired targeted-minus-generic result",
        "A paired bootstrap confidence interval for targeted minus generic accuracy. "
        "The entire interval is above zero.",
        source_hash,
    )
    lines.append(
        text(
            50,
            82,
            "10,000 paired bootstrap replicates; seed 20260720; 814 aligned examples",
            "subtitle",
        )
    )
    left = 130.0
    right = 1110.0
    y = 345.0
    minimum = -2.0
    maximum = 6.0

    def scale(value: float) -> float:
        return left + (value - minimum) / (maximum - minimum) * (right - left)

    for tick in range(-2, 7):
        x = scale(float(tick))
        lines.append(
            f'<line x1="{x:.1f}" y1="180" x2="{x:.1f}" y2="500" stroke="{GRID}" stroke-width="1"/>'
        )
        lines.append(text(x, 530, tick, "axis", anchor="middle"))
    zero_x = scale(0.0)
    lines.append(
        f'<line x1="{zero_x:.1f}" y1="165" x2="{zero_x:.1f}" y2="500" '
        f'stroke="{RED}" stroke-width="2"/>'
    )
    lines.append(text(zero_x, 150, "No difference", "small", anchor="middle"))
    low_x = scale(low)
    high_x = scale(high)
    estimate_x = scale(estimate)
    lines.append(
        f'<line x1="{low_x:.1f}" y1="{y:.1f}" x2="{high_x:.1f}" y2="{y:.1f}" '
        f'stroke="{TEAL}" stroke-width="12" stroke-linecap="round"/>'
    )
    for x in (low_x, high_x):
        lines.append(
            f'<line x1="{x:.1f}" y1="{y - 30:.1f}" x2="{x:.1f}" y2="{y + 30:.1f}" '
            f'stroke="{TEAL}" stroke-width="4"/>'
        )
    lines.append(
        f'<circle cx="{estimate_x:.1f}" cy="{y:.1f}" r="14" fill="{NAVY}" '
        f'stroke="{WHITE}" stroke-width="4"/>'
    )
    lines.append(text(estimate_x, y - 55, f"+{estimate:.4f} points", "label", anchor="middle"))
    lines.append(text(low_x, y + 58, f"+{low:.4f}", "value", anchor="middle"))
    lines.append(text(high_x, y + 58, f"+{high:.4f}", "value", anchor="middle"))
    lines.append(
        text(
            (left + right) / 2,
            590,
            "Targeted accuracy minus generic accuracy (percentage points)",
            "label",
            anchor="middle",
        )
    )
    lines.append(
        text(
            50,
            650,
            f"Targeted wins: {paired['targeted_wins']}; generic wins: "
            f"{paired['generic_wins']}; net: +{paired['targeted_net_wins']}.",
            "subtitle",
        )
    )
    return close_svg(lines)


def render_taxonomy(data: dict[str, Any], source_hash: str) -> str:
    """Render the reviewed base failure taxonomy distribution."""

    rows = data["taxonomy"]
    lines = common_header(
        "Untouched-base failure taxonomy",
        "Horizontal bars showing the primary categories assigned to all 293 reviewed "
        "development failures.",
        source_hash,
    )
    lines.append(text(50, 82, "Primary category; 293/293 failures manually reviewed", "subtitle"))
    left = 390.0
    right = 1030.0
    maximum = 70.0
    for index, row in enumerate(rows):
        y = 118.0 + index * 64.0
        count = int(row["count"])
        width = (right - left) * count / maximum
        lines.append(text(left - 18, y + 25, row["label"], "small", anchor="end"))
        lines.append(
            f'<rect x="{left:.1f}" y="{y:.1f}" width="{right - left:.1f}" '
            f'height="36" rx="4" fill="{LIGHT}"/>'
        )
        lines.append(
            f'<rect x="{left:.1f}" y="{y:.1f}" width="{width:.1f}" height="36" '
            f'rx="4" fill="{BLUE}"/>'
        )
        percent = 100.0 * count / int(data["taxonomy_total"])
        lines.append(text(left + width + 12, y + 25, f"{count}  ({percent:.1f}%)", "value"))
    lines.append(
        text(
            50,
            665,
            "Categories describe baseline failures; they are not independent benchmark strata.",
            "subtitle",
        )
    )
    return close_svg(lines)


def wrap_node_label(label: str) -> list[str]:
    """Return compact, deterministic line breaks for pipeline labels."""

    mapping = {
        "Evaluate": ["Evaluate"],
        "Classify failures": ["Classify", "failures"],
        "Generate verified data": ["Generate", "verified data"],
        "Train matched controls": ["Train matched", "controls"],
        "Retention gate": ["Retention", "gate"],
        "Benchmark comparison": ["Benchmark", "comparison"],
    }
    return mapping[label]


def wrap_node_detail(detail: str) -> list[str]:
    """Return compact line breaks for pipeline details."""

    mapping = {
        "814 frozen development examples": ["814 frozen", "development examples"],
        "293 failures reviewed": ["293 failures", "reviewed"],
        "500 targeted + 500 generic": ["500 targeted", "+ 500 generic"],
        "Token-matched Windows QLoRA": ["Token-matched", "Windows QLoRA"],
        "Common runtime scale 0.50": ["Common runtime", "scale 0.50"],
        "Targeted > generic; both < base": ["Targeted > generic", "Both < base"],
    }
    return mapping[detail]


def render_pipeline(data: dict[str, Any], source_hash: str) -> str:
    """Render the executed Phase 1 pipeline."""

    nodes = data["pipeline"]
    lines = common_header(
        "Foundry Phase 1 research pipeline",
        "Six connected stages from frozen evaluation through a matched benchmark comparison.",
        source_hash,
    )
    lines.append(
        text(
            50,
            82,
            "Executed path; benchmark comparison was directionally positive "
            "but absolutely negative",
            "subtitle",
        )
    )
    lines.extend(
        [
            "<defs>",
            '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/>',
            "</marker>",
            "</defs>",
        ]
    )
    start_x = 25.0
    width = 165.0
    gap = 28.0
    y = 225.0
    height = 210.0
    for index, node in enumerate(nodes):
        x = start_x + index * (width + gap)
        if index < len(nodes) - 1:
            arrow_start = x + width + 4
            arrow_end = x + width + gap - 6
            lines.append(
                f'<line x1="{arrow_start:.1f}" y1="{y + height / 2:.1f}" '
                f'x2="{arrow_end:.1f}" y2="{y + height / 2:.1f}" '
                f'stroke="{MUTED}" stroke-width="3" marker-end="url(#arrow)"/>'
            )
        negative = node["status"] == "negative_absolute"
        stroke = AMBER if negative else GREEN
        fill = "#fff8ed" if negative else "#f0fdf4"
        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
            f'rx="10" fill="{fill}" stroke="{stroke}" stroke-width="3"/>'
        )
        status = "MIXED RESULT" if negative else "PASSED"
        lines.append(text(x + width / 2, y + 32, status, "small", anchor="middle"))
        label_lines = wrap_node_label(str(node["label"]))
        for offset, label in enumerate(label_lines):
            lines.append(
                text(x + width / 2, y + 82 + offset * 24, label, "node-label", anchor="middle")
            )
        detail_lines = wrap_node_detail(str(node["detail"]))
        for offset, detail in enumerate(detail_lines):
            lines.append(
                text(
                    x + width / 2,
                    y + 150 + offset * 20,
                    detail,
                    "node-detail",
                    anchor="middle",
                )
            )
    lines.append(
        text(
            50,
            545,
            "Interpretation",
            "label",
        )
    )
    lines.append(
        text(
            50,
            578,
            "Failure-targeted data produced the stronger adaptation direction, but no tested arm "
            "surpassed the base.",
            "subtitle",
        )
    )
    lines.append(
        text(
            50,
            650,
            "Human language review remains pending; no sealed-final evaluation occurred.",
            "subtitle",
        )
    )
    return close_svg(lines)


def status_shape(x: float, y: float, status: str) -> str:
    """Render a status mark whose shape and color both encode outcome."""

    if status == "passed":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="{GREEN}"/>'
    if status == "failed":
        points = (
            f"{x:.1f},{y - 14:.1f} {x + 14:.1f},{y:.1f} {x:.1f},{y + 14:.1f} {x - 14:.1f},{y:.1f}"
        )
        return f'<polygon points="{points}" fill="{RED}"/>'
    return f'<rect x="{x - 12:.1f}" y="{y - 12:.1f}" width="24" height="24" rx="2" fill="{AMBER}"/>'


def timeline_label_lines(label: str) -> list[str]:
    """Wrap timeline labels at stable word boundaries."""

    words = label.split()
    if len(words) <= 2:
        return [label]
    midpoint = (len(words) + 1) // 2
    return [" ".join(words[:midpoint]), " ".join(words[midpoint:])]


def render_outcomes(data: dict[str, Any], source_hash: str) -> str:
    """Render the Phase 1 gate and branch outcome timeline."""

    events = data["outcome_timeline"]
    lines = common_header(
        "Phase 1 experiment outcomes",
        "A chronological two-row timeline showing passed gates, failed gates, "
        "and stopped branches.",
        source_hash,
    )
    lines.append(
        text(
            50,
            82,
            "Selected decision points from evaluator trust through GRPO closure",
            "subtitle",
        )
    )
    row_counts = (7, 6)
    start = 0
    y_values = (245.0, 515.0)
    for row_index, count in enumerate(row_counts):
        y = y_values[row_index]
        xs = [80.0 + index * (1040.0 / (count - 1)) for index in range(count)]
        lines.append(
            f'<line x1="{xs[0]:.1f}" y1="{y:.1f}" x2="{xs[-1]:.1f}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="5"/>'
        )
        for local_index, x in enumerate(xs):
            event = events[start + local_index]
            lines.append(status_shape(x, y, str(event["status"])))
            label_y = y - 68 if local_index % 2 == 0 else y + 62
            milestone_y = label_y - 22 if local_index % 2 == 0 else label_y + 38
            label_lines = timeline_label_lines(str(event["label"]))
            for offset, label in enumerate(label_lines):
                lines.append(text(x, label_y + offset * 18, label, "small", anchor="middle"))
            lines.append(text(x, milestone_y, f"M{event['milestone']}", "value", anchor="middle"))
        start += count
    lines.append(text(835, 655, "Passed", "small"))
    lines.append(status_shape(810, 650, "passed"))
    lines.append(text(955, 655, "Failed", "small"))
    lines.append(status_shape(930, 650, "failed"))
    lines.append(text(1085, 655, "Stopped", "small"))
    lines.append(status_shape(1060, 650, "stopped"))
    lines.append(
        text(
            50,
            655,
            "A stopped branch was not optimized or evaluated downstream.",
            "subtitle",
        )
    )
    return close_svg(lines)


def render_all(data: dict[str, Any], source_hash: str) -> dict[Path, str]:
    """Render every publication figure in stable path order."""

    return {
        OUTPUT_DIR / "gsm1k_accuracy.svg": render_accuracy(data, source_hash),
        OUTPUT_DIR / "paired_targeted_minus_generic.svg": render_paired_difference(
            data, source_hash
        ),
        OUTPUT_DIR / "base_failure_taxonomy.svg": render_taxonomy(data, source_hash),
        OUTPUT_DIR / "phase1_pipeline.svg": render_pipeline(data, source_hash),
        OUTPUT_DIR / "experiment_outcomes.svg": render_outcomes(data, source_hash),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify tracked SVGs instead of writing them",
    )
    return parser.parse_args()


def main() -> int:
    """Render or verify all Phase 1 figures."""

    args = parse_args()
    data, source_hash = load_data()
    figures = render_all(data, source_hash)
    if args.check:
        mismatches = [
            path
            for path, expected in figures.items()
            if not path.is_file() or path.read_text(encoding="utf-8") != expected
        ]
        if mismatches:
            for path in mismatches:
                print(f"figure differs: {path.relative_to(ROOT).as_posix()}")
            return 1
        print(f"verified {len(figures)} deterministic Phase 1 figures")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, content in figures.items():
        path.write_text(content, encoding="utf-8", newline="\n")
        print(f"wrote {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
