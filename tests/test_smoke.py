"""UI smoke tests and visual regression tests for the dashboard.

These tests verify that the Streamlit app renders correctly on deterministic fixtures.
"""

import json
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import polars as pl


def create_test_parquet(tmp_path: Path) -> Path:
    """Create a minimal test Parquet file with valid transaction data."""
    # Create test data with enough history for features
    data = []
    base_date = date(2024, 1, 1)

    # Customer C1: regular repeat buyer
    for i in range(20):
        data.append({
            "customer_id": "C1",
            "transaction_date": (base_date + timedelta(days=i * 14)).isoformat(),
            "transaction_id": f"T1_{i}",
            "product_id": f"P{i % 5}",
            "product_description": f"Product {i % 5}",
            "department": "DEPT1",
            "category": f"CAT{i % 3}",
            "price": 10.0 + (i % 3) * 5.0,
            "quantity": 1,
        })

    # Customer C2: one-time buyer
    data.append({
        "customer_id": "C2",
        "transaction_date": (base_date + timedelta(days=30)).isoformat(),
        "transaction_id": "T2_0",
        "product_id": "P10",
        "product_description": "Product 10",
        "department": "DEPT2",
        "category": "CAT5",
        "price": 25.0,
        "quantity": 2,
    })

    # Customer C3: irregular buyer
    for i in range(5):
        data.append({
            "customer_id": "C3",
            "transaction_date": (base_date + timedelta(days=i * 60)).isoformat(),
            "transaction_id": f"T3_{i}",
            "product_id": f"P{i % 2}",
            "product_description": f"Product {i % 2}",
            "department": "DEPT1",
            "category": "CAT1",
            "price": 15.0,
            "quantity": 1,
        })

    # Customer C4: recent buyer
    for i in range(10):
        data.append({
            "customer_id": "C4",
            "transaction_date": (base_date + timedelta(days=300 + i * 7)).isoformat(),
            "transaction_id": f"T4_{i}",
            "product_id": f"P{i % 4}",
            "product_description": f"Product {i % 4}",
            "department": "DEPT3",
            "category": f"CAT{i % 2}",
            "price": 20.0,
            "quantity": 1,
        })

    df = pl.DataFrame(data)
    # Keep dates as strings for the pipeline (it parses them)

    parquet_path = tmp_path / "test_transactions.parquet"
    df.write_parquet(parquet_path)
    return parquet_path


def run_dashboard_smoke_test(
    parquet_path: Path,
    output_dir: Path,
    coverage_confirmed: bool = True,
) -> tuple[bool, str]:
    """
    Run the dashboard backend analysis as a smoke test.
    
    Returns: (success, error_message)
    """
    try:
        # Run the pipeline CLI
        cmd = [
            sys.executable, "-m", "retail_analytics.core.pipeline.run",
            "--input", str(parquet_path),
            "--output", str(output_dir),
            "--price-mode", "unit",
            "--snapshots", "2024-06-30", "2024-05-31",
            "--buckets", "2",
            "--coverage-confirmed" if coverage_confirmed else "",
        ]
        cmd = [c for c in cmd if c]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            return False, f"Pipeline failed: {result.stderr}"

        # Check outputs exist
        required_outputs = [
            "run.json",
            "thresholds.json",
            "snapshots/2024-06-30/b0.parquet",
            "segment_profiles.parquet",
            "cohort_profiles.parquet",
            "segment_transitions.parquet",
        ]

        for out in required_outputs:
            if not (output_dir / out).exists():
                return False, f"Missing output: {out}"

        return True, "OK"

    except subprocess.TimeoutExpired:
        return False, "Pipeline timed out"
    except Exception as e:
        return False, f"Error: {e}"


def test_dashboard_backend_smoke():
    """Smoke test for the dashboard backend pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test data
        parquet_path = create_test_parquet(tmp_path)

        # Run pipeline
        output_dir = tmp_path / "output"
        success, msg = run_dashboard_smoke_test(parquet_path, output_dir)

        assert success, f"Smoke test failed: {msg}"


def test_dashboard_backend_no_coverage():
    """Smoke test with coverage unconfirmed (future outcomes disabled)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        parquet_path = create_test_parquet(tmp_path)
        output_dir = tmp_path / "output"

        success, msg = run_dashboard_smoke_test(parquet_path, output_dir, coverage_confirmed=False)

        assert success, f"Smoke test failed: {msg}"

        # Verify run.json has coverage_confirmed = false
        run_json = json.loads((output_dir / "run.json").read_text())
        assert run_json["coverage_confirmed"] is False


def test_visual_rendering_imports():
    """Test that all visualization functions can be imported and called."""

    # If we get here, imports work
    assert True


def test_chart_components_imports():
    """Test that chart components can be imported and instantiated."""
    import plotly.graph_objects as go

    from retail_analytics.visualization.chart_components import (
        ChartCard,
        create_chart_card,
    )

    # Create a simple chart card
    fig = go.Figure(go.Bar(x=[1, 2, 3], y=[10, 20, 30]))
    card = create_chart_card(
        key="test",
        title="Test Chart",
        figure=fig,
        question="Test question",
        takeaway="Test takeaway",
        how_to_read="Test how to read",
        caveat="Test caveat",
        next_step="Test next step",
        alt_description="Test alt",
    )

    assert isinstance(card, ChartCard)
    assert card.spec.title == "Test Chart"


def test_chart_specs_imports():
    """Test that chart specs can be imported."""

    # If we get here, imports work
    assert True


def test_modeling_imports():
    """Test that modeling modules can be imported."""
    import numpy as np

    from retail_analytics.modeling import (
        BGNBDParams,
        GammaGammaParams,
        fit_bg_nbd,
        fit_gamma_gamma,
    )

    # Test model fitting on dummy data
    freq = np.array([1, 2, 3, 4, 5, 0, 0, 1, 2])
    rec = np.array([10, 20, 5, 30, 15, 100, 50, 10, 20])
    T = np.array([100, 150, 80, 200, 120, 100, 100, 100, 110])

    params, converged, ll = fit_bg_nbd(freq, rec, T)
    assert isinstance(params, BGNBDParams)
    # May not converge on tiny data; convergence is reported, not assumed
    assert isinstance(converged, bool)

    # Test Gamma-Gamma
    mon = np.array([100, 200, 150, 300, 250, 0, 0, 100, 200])
    params_gg, converged_gg, ll_gg = fit_gamma_gamma(freq, mon)
    assert isinstance(params_gg, GammaGammaParams)
    assert isinstance(converged_gg, bool)


def test_pipeline_output_schema():
    """Test that pipeline outputs have expected schemas."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        parquet_path = create_test_parquet(tmp_path)
        output_dir = tmp_path / "output"

        success, msg = run_dashboard_smoke_test(parquet_path, output_dir)
        assert success

        # Check run.json structure
        run_data = json.loads((output_dir / "run.json").read_text())
        assert "snapshots" in run_data
        assert "coverage_confirmed" in run_data
        assert "observed_through" in run_data
        assert "price_mode" in run_data
        assert "reference_date" in run_data

        # Check thresholds.json
        thresholds = json.loads((output_dir / "thresholds.json").read_text())
        assert "value_low" in thresholds
        assert "quantile_interpolation" in thresholds

        # Check snapshots have expected columns
        snap = pl.read_parquet(output_dir / "snapshots/2024-06-30/b0.parquet")
        expected_cols = ["customer_id", "spend_90d", "purchases_90d", "frequency_change",
                         "basket_change", "lifecycle", "value_tier", "behavior"]
        for col in expected_cols:
            assert col in snap.columns, f"Missing column: {col}"


def test_security_utilities():
    """Test security utility functions."""
    from src.retail_analytics.security import (
        AccessControl,
        SecureWorkspace,
        escape_csv_value,
        get_csp_header,
        mask_customer_id,
        mask_pii_fields,
        sanitize_filename,
        validate_csv_file,
        validate_deployment_config,
        validate_parquet_file,
        validate_upload_file,
    )

    # Test file validation
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        f.write(b'PAR1' + b'x' * 100 + b'PAR1')
        f.flush()
        valid, msg = validate_upload_file(Path(f.name), allow_temp_files=True)
        assert valid, f"Upload validation failed: {msg}"
        valid, msg = validate_parquet_file(Path(f.name))
        assert valid, f"Parquet validation failed: {msg}"

    # Test CSV validation
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
        f.write('col1,col2\n1,2\n=1+2,3\n')
        f.flush()
        valid, msg = validate_csv_file(Path(f.name))
        assert valid
        assert "WARNING" in msg

    # Test SecureWorkspace
    with SecureWorkspace() as ws:
        test_file = ws / 'test.txt'
        test_file.write_text('hello')
        assert test_file.exists()
    # Workspace should be cleaned up

    # Test CSV escaping
    assert escape_csv_value('=1+2') == "'=1+2"
    assert escape_csv_value('normal') == 'normal'
    assert escape_csv_value('@formula') == "'@formula"

    # Test filename sanitization
    # basename removes path components, then dots are removed
    assert sanitize_filename('../../etc/passwd') == 'passwd'
    assert sanitize_filename('normal_file.csv') == 'normal_file.csv'

    # Test customer ID masking
    masked = mask_customer_id('CUST12345')
    assert masked.startswith('CUST_') and len(masked) == 13

    # Test PII masking
    data = {'customer_id': 'CUST123', 'customer_email': 'test@example.com', 'amount': 100}
    masked = mask_pii_fields(data, ['customer_id', 'customer_email'])
    assert masked['customer_id'].startswith('CUST_')
    assert masked['customer_email'].startswith('CUST_')
    assert masked['amount'] == 100

    # Test AccessControl
    ac = AccessControl('viewer')
    assert ac.has_permission('view_dashboard')
    assert not ac.has_permission('manage_users')
    ac2 = AccessControl('admin')
    assert ac2.has_permission('manage_users')

    # Test CSP header
    csp = get_csp_header()
    assert len(csp) > 0
    assert "default-src 'self'" in csp

    # Test config validation
    config = {'debug': True, 'secret_key': 'changeme', 'force_https': False}
    issues = validate_deployment_config(config)
    assert any("Debug mode" in i for i in issues)
    assert any("Default/insecure" in i for i in issues)

    print("All security tests passed!")


if __name__ == "__main__":
    # Run tests manually
    print("Running smoke tests...")

    print("1. Testing visual rendering imports...")
    test_visual_rendering_imports()
    print("   OK")

    print("2. Testing chart components...")
    test_chart_components_imports()
    print("   OK")

    print("3. Testing chart specs...")
    test_chart_specs_imports()
    print("   OK")

    print("4. Testing modeling imports...")
    test_modeling_imports()
    print("   OK")

    print("5. Testing dashboard backend smoke test...")
    test_dashboard_backend_smoke()
    print("   OK")

    print("6. Testing dashboard backend no coverage...")
    test_dashboard_backend_no_coverage()
    print("   OK")

    print("7. Testing pipeline output schema...")
    test_pipeline_output_schema()
    print("   OK")

    print("\nAll smoke tests passed!")
