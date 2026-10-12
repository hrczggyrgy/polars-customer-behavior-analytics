"""Reproduction scripts for Phase 1 visualization defects in product_affinity.py"""

import polars as pl
from retail_analytics.visualization.product_affinity import (
    new_to_repeat_pathways,
    cross_sell_recommendations,
    basket_composition_by_segment,
    ProductAffinityConfig,
    product_affinity_analysis,
)
from retail_analytics.core.pipeline.utils import collect
from retail_analytics.core.constants import WINDOW_YEAR
import tempfile
import os


def create_test_data():
    """Create minimal test data for reproduction."""
    # Create validated transactions
    validated_data = pl.DataFrame({
        "customer_id": ["C1", "C1", "C1", "C2", "C2", "C3", "C3", "C3"],
        "transaction_id": ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"],
        "date": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-01-10", "2024-01-20", "2024-01-05", "2024-01-25", "2024-02-10"],
        "product_id": ["P1", "P2", "P3", "P1", "P2", "P1", "P3", "P2"],
        "category_key": ["CAT1", "CAT2", "CAT1", "CAT1", "CAT2", "CAT1", "CAT1", "CAT2"],
        "quantity": [1, 2, 1, 1, 1, 2, 1, 3],
        "value": [10.0, 20.0, 15.0, 10.0, 20.0, 20.0, 15.0, 30.0],
    }).with_columns(pl.col("date").str.to_date())
    
    # Create snapshots
    snapshots_data = pl.DataFrame({
        "customer_id": ["C1", "C2", "C3"],
        "behavior": ["loyal", "at_risk", "new"],
        "value_tier": ["high", "medium", "low"],
        "lifecycle": ["active", "active", "new"],
        "trajectory": ["growing", "stable", "new"],
        "evidence": ["strong", "weak", "minimal"],
        "spend_365d": [100.0, 50.0, 25.0],
    })
    
    return validated_data.lazy(), snapshots_data.lazy()


def test_defect_1_new_to_repeat_pathways():
    """Defect 1: Sort by removed column 'transaction_id'."""
    print("\n=== DEFECT 1: new_to_repeat_pathways - Sort by removed column ===")
    validated, snapshots = create_test_data()
    cutoff = "2024-02-28"
    
    try:
        result = new_to_repeat_pathways(validated, snapshots, cutoff)
        print(f"Result type: {type(result)}")
        print(f"Result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
        if isinstance(result, dict):
            for k, v in result.items():
                print(f"  {k}: {type(v)}, shape={v.shape if hasattr(v, 'shape') else 'N/A'}")
        print("UNEXPECTED: Function completed without error - defect may not reproduce or is handled")
    except Exception as e:
        print(f"REPRODUCED: {type(e).__name__}: {e}")
    
    # The defect: txns.select("customer_id", "date", "product_id", "category_key") 
    # but then .sort(["customer_id", "date", "transaction_id"]) - transaction_id not selected!


def test_defect_2_cross_sell_recommendations():
    """Defect 2: Eager frame passed to LazyFrame collector."""
    print("\n=== DEFECT 2: cross_sell_recommendations - Eager frame to collect() ===")
    
    # Create mock pairs DataFrame (eager)
    pairs = pl.DataFrame({
        "item_a": ["P1", "P1", "P2"],
        "item_b": ["P2", "P3", "P3"],
        "cooccurrence_count": [10, 5, 8],
        "support": [0.1, 0.05, 0.08],
        "confidence_a_to_b": [0.5, 0.3, 0.4],
        "confidence_b_to_a": [0.6, 0.2, 0.5],
        "lift": [1.5, 1.2, 1.8],
        "conviction_a_to_b": [2.0, 1.5, 2.5],
        "conviction_b_to_a": [2.5, 1.3, 3.0],
    })
    
    try:
        result = cross_sell_recommendations(pairs, top_n=10)
        print(f"Result type: {type(result)}")
        print(f"Keys: {result.keys()}")
        for k, v in result.items():
            print(f"  {k}: shape={v.shape}")
        print("UNEXPECTED: Function completed without error")
    except Exception as e:
        print(f"REPRODUCED: {type(e).__name__}: {e}")
    
    # The defect: _get_recs() creates eager DataFrames (recs_a_to_b, recs_b_to_a, all_recs)
    # then passes to collect() which does .collect(engine="streaming") - fails on eager frames


def test_defect_3_basket_composition_by_segment():
    """Defect 3: Self-referential alias in same with_columns()."""
    print("\n=== DEFECT 3: basket_composition_by_segment - Self-referential alias ===")
    validated, snapshots = create_test_data()
    cutoff = "2024-02-28"
    
    try:
        result = basket_composition_by_segment(snapshots, validated, cutoff, dimensions=["behavior"])
        print(f"Result type: {type(result)}")
        print(f"Keys: {result.keys()}")
        for dim, comp in result.items():
            print(f"  {dim}: product_comp shape={comp['product_composition'].shape}, category_comp shape={comp['category_composition'].shape}")
        print("UNEXPECTED: Function completed without error")
    except Exception as e:
        print(f"REPRODUCED: {type(e).__name__}: {e}")
    
    # The defect: Lines 701-704 and 716-719 create segment_spend and reference it in same with_columns()
    # .with_columns(
    #     pl.col("spend").sum().over(dim).alias("segment_spend"),
    #     (pl.col("spend") / pl.col("segment_spend")).alias("spend_share"),  # References segment_spend created above!
    # )


def test_defect_1_detailed():
    """Detailed investigation of Defect 1."""
    print("\n=== DETAILED DEFECT 1 INVESTIGATION ===")
    
    # Simulate the exact operations
    validated_data = pl.DataFrame({
        "customer_id": ["C1", "C1", "C1", "C2"],
        "transaction_id": ["T1", "T2", "T3", "T4"],
        "date": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-01-10"],
        "product_id": ["P1", "P2", "P3", "P1"],
        "category_key": ["CAT1", "CAT2", "CAT1", "CAT1"],
    }).with_columns(pl.col("date").str.to_date())
    
    txns = validated_data.select("customer_id", "date", "product_id", "category_key")
    print(f"Columns after select: {txns.columns}")
    
    try:
        # This is what the function tries to do
        ordered = txns.sort(["customer_id", "date", "transaction_id"])
        print("Sort succeeded (unexpected)")
    except Exception as e:
        print(f"Sort failed as expected: {type(e).__name__}: {e}")


def test_defect_3_detailed():
    """Detailed investigation of Defect 3."""
    print("\n=== DETAILED DEFECT 3 INVESTIGATION ===")
    
    df = pl.DataFrame({
        "dim": ["A", "A", "B", "B"],
        "product_id": ["P1", "P2", "P1", "P2"],
        "spend": [10.0, 20.0, 15.0, 25.0],
    })
    
    print("Testing self-referential with_columns...")
    try:
        result = df.with_columns(
            pl.col("spend").sum().over("dim").alias("segment_spend"),
            (pl.col("spend") / pl.col("segment_spend")).alias("spend_share"),
        )
        print("Succeeded (Polars may have fixed this)")
        print(result)
    except Exception as e:
        print(f"Failed as expected: {type(e).__name__}: {e}")
    
    # Correct approach - two with_columns
    print("\nCorrect approach (two with_columns):")
    result = df.with_columns(
        pl.col("spend").sum().over("dim").alias("segment_spend"),
    ).with_columns(
        (pl.col("spend") / pl.col("segment_spend")).alias("spend_share"),
    )
    print(result)


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 1 DEFECT REPRODUCTION")
    print("=" * 60)
    
    test_defect_1_new_to_repeat_pathways()
    test_defect_1_detailed()
    test_defect_2_cross_sell_recommendations()
    test_defect_3_basket_composition_by_segment()
    test_defect_3_detailed()
    
    print("\n" + "=" * 60)
    print("REPRODUCTION COMPLETE")
    print("=" * 60)