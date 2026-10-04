from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


class ResultsAnalyzer:
    """Build lightweight summaries and SVG charts from persisted harness results."""

    def __init__(self, results_dir: str | Path = "results"):
        self.results_dir = Path(results_dir)
        self.run_dir: Path | None = None
        self.rows: list[dict[str, str]] = []
        self.summary: dict[str, Any] = {}

    def load_results(self, run_id: str | None = None) -> ResultsAnalyzer:
        if run_id:
            run_dir = self.results_dir / run_id
            if not run_dir.is_dir():
                raise FileNotFoundError(f"Results run not found: {run_dir}")
        else:
            candidates = sorted(
                path for path in self.results_dir.iterdir()
                if path.is_dir() and (path / "results.csv").exists()
            ) if self.results_dir.exists() else []
            if not candidates:
                raise ValueError(f"No completed result runs found in {self.results_dir}.")
            run_dir = candidates[-1]
        csv_path = run_dir / "results.csv"
        with csv_path.open(newline="", encoding="utf-8") as file:
            self.rows = list(csv.DictReader(file))
        summary_path = run_dir / "summary.json"
        self.summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        self.run_dir = run_dir
        return self

    def generate_comparison_charts(self, output_dir: str | Path = "reports") -> list[Path]:
        if not self.rows:
            raise ValueError("No run rows loaded. Call load_results() first.")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        groups: dict[str, list[dict[str, str]]] = {}
        for row in self.rows:
            name = f"{row['agent_type']} ({row.get('setting', 'off')})"
            groups.setdefault(name, []).append(row)
        rates = {
            name: mean(float(row["success"]) for row in rows)
            for name, rows in sorted(groups.items())
        }
        costs = {
            name: mean(float(row.get("cost_usd", 0)) for row in rows)
            for name, rows in sorted(groups.items())
        }
        latencies = {
            name: mean(float(row["latency_seconds"]) for row in rows)
            for name, rows in sorted(groups.items())
        }
        files = [
            _write_bar_svg(output / "success_rate.svg", rates, "Success rate", 0.0, 1.0),
            _write_scatter_svg(output / "success_vs_cost.svg", costs, rates),
            _write_latency_svg(output / "success_vs_latency.svg", latencies, rates),
        ]
        return files

    def generate_report(self, output_file: str | Path = "reports/run_summary.md") -> str:
        if not self.rows:
            raise ValueError("No run rows loaded. Call load_results() first.")
        sections = [
            "# Run Summary",
            "",
            f"Run directory: `{self.run_dir}`",
            "",
            "| Agent/setting | Runs | Success rate | 95% CI | Mean tokens/success | Mean latency | Mean cost/run |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        keys = sorted({(row["agent_type"], row.get("setting", "off")) for row in self.rows})
        for agent, setting in keys:
            rows = [r for r in self.rows if r["agent_type"] == agent and r.get("setting", "off") == setting]
            key = f"{agent}_{setting}"
            stats = self.summary.get(key, {})
            rate = mean(float(row["success"]) for row in rows)
            ci = stats.get("success_rate_ci95", [None, None])
            passed = [row for row in rows if row["success"].lower() in {"true", "1"}]
            tokens = [float(row["tokens_in"]) + float(row["tokens_out"]) for row in passed]
            latencies = [float(row["latency_seconds"]) for row in rows]
            costs = [float(row.get("cost_usd", 0)) for row in rows]
            sections.append(
                f"| {agent} / {setting} | {len(rows)} | {rate:.1%} | "
                f"{_format_ci(ci)} | {_format_number(mean(tokens) if tokens else None)} | "
                f"{mean(latencies):.3f}s | ${mean(costs):.5f} |"
            )
        sections.extend(["", "## Failure reasons", ""])
        reasons: dict[str, int] = {}
        for row in self.rows:
            reason = row.get("failure_reason") or "none"
            reasons[reason] = reasons.get(reason, 0) + 1
        sections.extend(f"- `{name}`: {count}" for name, count in sorted(reasons.items()))
        sections.extend(["", "## Cost and run metadata", ""])
        run_metadata = self.summary.get("run", {})
        sections.append(f"- Estimated run cost: ${run_metadata.get('estimated_cost_usd', 0):.5f}")
        sections.append(f"- Recorded model cost: ${run_metadata.get('actual_cost_usd', 0):.5f}")
        sections.append("- Cached model/tool calls are flagged in the JSONL trace; cached calls are not included in latency comparisons.")
        report = "\n".join(sections) + "\n"
        output = Path(output_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        return report


def _write_bar_svg(path: Path, values: dict[str, float], title: str, minimum: float, maximum: float) -> Path:
    width, row_height = 760, 38
    height = 70 + row_height * len(values)
    entries = []
    for index, (name, value) in enumerate(values.items()):
        y = 50 + index * row_height
        bar_width = max(0.0, min(480.0, 480.0 * (value - minimum) / (maximum - minimum)))
        entries.append(
            f'<text x="8" y="{y + 18}" font-size="13">{_escape(name)}</text>'
            f'<rect x="230" y="{y}" width="{bar_width:.1f}" height="22" fill="#4776a8"/>'
            f'<text x="{238 + bar_width:.1f}" y="{y + 17}" font-size="12">{value:.1%}</text>'
        )
    svg = _svg(width, height, title, "".join(entries))
    path.write_text(svg, encoding="utf-8")
    return path


def _write_scatter_svg(path: Path, costs: dict[str, float], success: dict[str, float]) -> Path:
    width, height = 760, 480
    max_cost = max(costs.values(), default=0.0) or 1.0
    points = []
    for name in sorted(costs):
        x = 80 + 620 * costs[name] / max_cost
        y = 390 - 320 * success[name]
        points.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#d87536"/>'
            f'<text x="{x + 10:.1f}" y="{y - 6:.1f}" font-size="12">{_escape(name)}</text>'
        )
    axes = (
        '<line x1="80" y1="390" x2="710" y2="390" stroke="#333"/>'
        '<line x1="80" y1="390" x2="80" y2="55" stroke="#333"/>'
        '<text x="350" y="445" font-size="14">Mean cost per run (USD; scaled)</text>'
        '<text x="8" y="48" font-size="14">Success rate</text>'
    )
    path.write_text(_svg(width, height, "Success versus cost", axes + "".join(points)), encoding="utf-8")
    return path


def _write_latency_svg(path: Path, latencies: dict[str, float], success: dict[str, float]) -> Path:
    width, height = 760, 480
    max_latency = max(latencies.values(), default=0.0) or 1.0
    points = []
    for name in sorted(latencies):
        x = 80 + 620 * latencies[name] / max_latency
        y = 390 - 320 * success[name]
        points.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#4776a8"/>'
            f'<text x="{x + 10:.1f}" y="{y - 6:.1f}" font-size="12">{_escape(name)}</text>'
        )
    axes = (
        '<line x1="80" y1="390" x2="710" y2="390" stroke="#333"/>'
        '<line x1="80" y1="390" x2="80" y2="55" stroke="#333"/>'
        '<text x="350" y="445" font-size="14">Mean latency per run (seconds; scaled)</text>'
        '<text x="8" y="48" font-size="14">Success rate</text>'
    )
    path.write_text(_svg(width, height, "Success versus latency", axes + "".join(points)), encoding="utf-8")
    return path


def _svg(width: int, height: int, title: str, content: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
        f'<text x="8" y="25" font-size="18" font-weight="bold">{_escape(title)}</text>{content}</svg>\n'
    )


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _format_ci(interval: list[Any]) -> str:
    if len(interval) < 2 or interval[0] is None:
        return "n/a"
    return f"{float(interval[0]):.1%}–{float(interval[1]):.1%}"


def _format_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize persisted evaluation outputs.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    analyzer = ResultsAnalyzer(args.results_dir).load_results(args.run)
    print(analyzer.generate_report(Path(args.output_dir) / "run_summary.md"))
    print("Charts:", ", ".join(str(path) for path in analyzer.generate_comparison_charts(args.output_dir)))
