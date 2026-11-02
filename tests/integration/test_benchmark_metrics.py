from __future__ import annotations

import json
from pathlib import Path
import pytest

from querity.agent.service import AgentService
from querity.benchmarks.evaluator import BenchmarkEvaluator, compare_result_sets
from querity.benchmarks.reporter import BenchmarkReporter
from querity.benchmarks.runner import load_golden_queries, run_benchmark
from querity.cli.config import ConfigError, load_config


@pytest.fixture(scope="module")
def real_db_settings():
    try:
        config = load_config()
    except ConfigError:
        pytest.skip("No real PostgreSQL database configured in ~/.config/querity/config.json")

    url = config.database_url
    if not url or "user:pass@localhost" in url:
        pytest.skip("No real PostgreSQL database configured in config or environment")

    if not config.google_api_key:
        pytest.skip("No GOOGLE_API_KEY configured for live agent integration tests")

    try:
        settings = config.to_settings()
        service = AgentService(settings)
        try:
            service.check_connection()
        finally:
            service.close()
    except Exception as exc:
        pytest.skip(f"Live PostgreSQL database is not reachable: {exc}")

    return settings


@pytest.fixture(scope="module")
def golden_queries():
    return load_golden_queries()


def test_result_set_canonical_comparison():
    # Test identical sets
    r1 = [("Electronics", 1000.50), ("Apparel", 250.00)]
    r2 = [("Electronics", 1000.50), ("Apparel", 250.00)]
    rc_match, exact, diff = compare_result_sets(r1, r2)
    assert rc_match is True
    assert exact is True
    assert diff == 0

    # Test permutation invariance
    r3 = [("Apparel", 250.00), ("Electronics", 1000.50)]
    rc_match, exact, diff = compare_result_sets(r1, r3)
    assert rc_match is True
    assert exact is True

    # Test different counts
    r4 = [("Electronics", 1000.50)]
    rc_match, exact, diff = compare_result_sets(r1, r4)
    assert rc_match is False
    assert exact is False
    assert diff == 1


def test_entity_recall_and_value_parity_metrics():
    # Scenario: Golden has LIMIT 2 top suppliers; Generated has all 3 suppliers
    gen_cols = ["supplier_id", "company_name", "total_inventory_valuation", "gross_sales", "net_revenue"]
    gold_cols = ["supplier_id", "company_name", "country", "inventory_valuation_usd", "gross_sales_usd", "net_sales_usd"]

    gold_rows = [
        (15, "Supplier P15", "Taiwan", 10774719.4, 244096.0, 239996.0),
        (5, "Supplier F5", "Canada", 9432020.0, 243976.0, 239776.0),
    ]
    gen_rows = [
        (26, "Supplier A26", 9851435.0, 233476.0, 214482.34),
        (15, "Supplier P15", 10774719.4, 244096.0, 239996.0),
        (5, "Supplier F5", 9432020.0, 243976.0, 239776.0),
    ]

    res = compare_result_sets(gen_rows, gold_rows, gen_columns=gen_cols, gold_columns=gold_cols)

    # 1. Backwards compatible tuple unpacking
    rc_match, exact, diff = res
    assert rc_match is False
    assert exact is False
    assert diff == 1

    # 2. Rich entity recall and value parity metrics
    assert res.entity_recall == 1.0, f"Expected 100% recall, got {res.entity_recall}"
    assert round(res.entity_precision, 2) == 0.67
    assert res.soft_f1 > 0.75
    assert res.value_parity == 1.0, f"Expected 100% value parity, got {res.value_parity}"


def test_real_database_connectivity_and_catalog(real_db_settings):
    service = AgentService(real_db_settings)
    try:
        service.check_connection()
        status = service.ingestion_status()
        assert status["ingested"] is True
        assert status["tables"] >= 10
    finally:
        service.close()


def test_stage_latency_and_metrics_evaluator(real_db_settings, golden_queries, tmp_path):
    service = AgentService(real_db_settings)
    reporter = BenchmarkReporter(base_dir=tmp_path / "runs")
    evaluator = BenchmarkEvaluator(service)

    try:
        # Run evaluation on first 2 golden queries to thoroughly test all stages
        test_cases = golden_queries[:2]
        results = []
        for i, q in enumerate(test_cases, 1):
            res = evaluator.evaluate_query(i, q)
            results.append(res)

            # Stage Latency Assertions: every stage must record positive timings
            assert res.latencies.analyze_ms > 0, f"Analyze latency should be positive, got {res.latencies.analyze_ms}"
            assert res.latencies.retrieve_ms > 0, f"Retrieve latency should be positive, got {res.latencies.retrieve_ms}"
            assert res.latencies.generate_ms > 0, f"Generate latency should be positive, got {res.latencies.generate_ms}"
            assert res.latencies.guard_ms > 0, f"Guard latency should be positive, got {res.latencies.guard_ms}"
            assert res.latencies.total_turn_ms > 0

            # Guard verification
            assert res.accuracy.guard_passed is True, f"Guard failed: {res.guard_errors}"
            assert res.generated_sql is not None

            # Retrieval Recall: at least 70% of golden tables should be retrieved
            assert res.accuracy.table_recall >= 0.70, (
                f"Table recall too low ({res.accuracy.table_recall}): "
                f"expected {res.tables_expected}, retrieved {res.tables_retrieved}"
            )

            # Execution against real database
            assert res.accuracy.execution_success is True, f"Execution failed: {res.execution_error}"
            assert res.generated_rows_count >= 0
            assert res.latencies.execute_ms > 0

        # Reporter Artifacts Assertions
        summary = reporter.generate_summary(
            results,
            database_url=str(real_db_settings.source_url),
            source_id=real_db_settings.source_id,
            model=real_db_settings.llm_model,
            provider=real_db_settings.llm_provider,
        )
        run_dir = reporter.save_artifacts(summary, results)

        assert run_dir.exists()
        log_file = run_dir / "benchmark.log"
        assert log_file.exists()
        assert log_file.stat().st_size > 0

        metrics_file = run_dir / "benchmark_metrics.json"
        assert metrics_file.exists()
        with open(metrics_file) as f:
            metrics_data = json.load(f)
            assert metrics_data["total_queries"] == len(test_cases)
            assert metrics_data["successful_executions"] == len(test_cases)
            assert "stage_latencies" in metrics_data
            assert "analyze" in metrics_data["stage_latencies"]
            assert "retrieve" in metrics_data["stage_latencies"]
            assert "generate" in metrics_data["stage_latencies"]
            assert "guard" in metrics_data["stage_latencies"]
            assert "execute" in metrics_data["stage_latencies"]

        summary_md = run_dir / "summary.md"
        assert summary_md.exists()
        md_text = summary_md.read_text(encoding="utf-8")
        assert "Executive Metrics Scorecard" in md_text
        assert "Per-Stage Latency Breakdown" in md_text
        assert "Per-Query Results Detail" in md_text

        # Verify latest directory
        latest_dir = tmp_path / "runs" / "latest"
        assert latest_dir.exists()
        assert (latest_dir / "summary.md").exists()
    finally:
        reporter.close()
        service.close()

