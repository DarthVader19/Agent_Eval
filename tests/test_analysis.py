from analysis.aggregate_results import ResultsAnalyzer


def test_comparison_charts_include_success_vs_latency(tmp_path):
    analyzer = ResultsAnalyzer()
    analyzer.rows = [
        {
            "agent_type": "react",
            "setting": "off",
            "success": "1",
            "cost_usd": "0.01",
            "latency_seconds": "2.5",
        },
        {
            "agent_type": "direct",
            "setting": "off",
            "success": "0",
            "cost_usd": "0.02",
            "latency_seconds": "1.5",
        },
    ]

    files = analyzer.generate_comparison_charts(tmp_path)

    assert [path.name for path in files] == [
        "success_rate.svg",
        "success_vs_cost.svg",
        "success_vs_latency.svg",
    ]
    chart = files[-1].read_text(encoding="utf-8")
    assert "Success versus latency" in chart
    assert "Mean latency per run" in chart
