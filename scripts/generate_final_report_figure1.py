#!/usr/bin/env python3
"""Generate the main final-report comparison figure as an SVG.

The figure compares the four thesis-facing safe conditions used in the final
write-up:
1. Heuristic baseline
2. Generator best safe
3. One-shot selected
4. Iterative step 1

It intentionally uses only the Python standard library so it can run in the
project environment without plotting dependencies.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "final_condition_comparison.csv"
DEFAULT_OUT_DIR = ROOT / "docs" / "figures"
DEFAULT_SVG_PATH = DEFAULT_OUT_DIR / "final_report_condition_comparison.svg"


def round4(value: float) -> str:
    return f"{value:.4f}"


def extract_rows() -> List[Dict[str, float | str]]:
    metric_fields = {
        "final_objective",
        "fairness_score",
        "diversity_retention",
        "stability_score",
    }
    with RESULTS_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            key: float(value) if key in metric_fields else value
            for key, value in row.items()
        }
        for row in rows
    ]


def svg_text(x: float, y: float, text: str, size: int = 16, weight: str = "normal",
             fill: str = "#1f2933", anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Georgia, Times New Roman, serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{escape_xml(text)}</text>'
    )


def svg_rect(x: float, y: float, width: float, height: float, fill: str,
             rx: int = 0, stroke: str | None = None, stroke_width: int = 0) -> str:
    stroke_attr = ""
    if stroke is not None:
        stroke_attr = f' stroke="{stroke}" stroke-width="{stroke_width}"'
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{rx}" '
        f'fill="{fill}"{stroke_attr} />'
    )


def svg_line(x1: float, y1: float, x2: float, y2: float, stroke: str,
             stroke_width: int = 1, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{stroke_width}"{dash_attr} />'
    )


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_panel(rows: List[Dict[str, float | str]], metric_key: str, metric_title: str,
                x: int, y: int, width: int, height: int, colors: Dict[str, str]) -> str:
    parts: List[str] = []
    panel_fill = "#faf7f0"
    axis_color = "#5b6470"
    grid_color = "#d9d5ca"
    text_color = "#1f2933"

    parts.append(svg_rect(x, y, width, height, panel_fill, rx=14, stroke="#d1cab8", stroke_width=1))
    parts.append(svg_text(x + 24, y + 34, metric_title, size=22, weight="bold", fill=text_color))

    chart_left = x + 62
    chart_right = x + width - 24
    chart_top = y + 60
    chart_bottom = y + height - 74
    chart_height = chart_bottom - chart_top
    chart_width = chart_right - chart_left

    ticks = [0.0, 0.2, 0.4, 0.6]
    max_y = 0.6
    for tick in ticks:
        tick_y = chart_bottom - (tick / max_y) * chart_height
        parts.append(svg_line(chart_left, tick_y, chart_right, tick_y, grid_color, stroke_width=1, dash="4 6"))
        parts.append(svg_text(chart_left - 12, tick_y + 5, f"{tick:.1f}", size=12, fill=axis_color, anchor="end"))

    parts.append(svg_line(chart_left, chart_top, chart_left, chart_bottom, axis_color, stroke_width=2))
    parts.append(svg_line(chart_left, chart_bottom, chart_right, chart_bottom, axis_color, stroke_width=2))

    bar_area_width = chart_width / len(rows)
    bar_width = 56
    for index, row in enumerate(rows):
        value = float(row[metric_key])
        bar_left = chart_left + index * bar_area_width + (bar_area_width - bar_width) / 2
        bar_height = max(0.0, min(value / max_y, 1.0)) * chart_height
        bar_top = chart_bottom - bar_height
        color = colors[str(row["short_label"])]
        parts.append(svg_rect(bar_left, bar_top, bar_width, bar_height, color, rx=8))
        parts.append(svg_text(bar_left + bar_width / 2, bar_top - 10, round4(value), size=13, weight="bold",
                              fill=text_color, anchor="middle"))
        parts.append(svg_text(bar_left + bar_width / 2, chart_bottom + 24, str(row["short_label"]),
                              size=13, fill=text_color, anchor="middle"))

    return "\n".join(parts)


def build_svg(rows: List[Dict[str, float | str]]) -> str:
    width = 1100
    height = 700
    colors = {
        "Heuristic": "#7d8ca3",
        "Generator": "#c96b3b",
        "One-shot": "#3d8f73",
        "Iterative": "#5b7c99",
    }
    bg = "#f3efe6"
    title_color = "#18212b"
    note_color = "#4f5965"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_rect(0, 0, width, height, bg),
        svg_text(54, 58, "Figure 1. Final-Matrix Comparison of Safe Thesis-Facing Conditions", size=28,
                 weight="bold", fill=title_color),
        svg_text(
            54,
            88,
            "Bars compare the four conditions discussed in Sections 5.1-5.2 using the final objective and fairness metrics.",
            size=15,
            fill=note_color,
        ),
        svg_text(
            54,
            112,
            "All plotted conditions passed the thesis safety gate. Source: archived episode snapshot and verified result table.",
            size=15,
            fill=note_color,
        ),
        build_panel(rows, "final_objective", "Final Objective", x=48, y=150, width=490, height=470, colors=colors),
        build_panel(rows, "fairness_score", "Fairness Score", x=562, y=150, width=490, height=470, colors=colors),
        svg_text(54, 654, "Condition values used in this figure are stored in results/final_condition_comparison.csv.",
                 size=13, fill=note_color),
        "</svg>",
    ]
    return "\n".join(parts)


def main() -> None:
    rows = extract_rows()
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_SVG_PATH.write_text(build_svg(rows), encoding="utf-8")
    print(f"[OK] Wrote SVG: {DEFAULT_SVG_PATH}")


if __name__ == "__main__":
    main()
