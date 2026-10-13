"""Tests for the dashboard route registry."""

from src.retail_analytics.dashboard.route_registry import (
    get_route_registry,
    get_core_views,
    get_advanced_views,
    validate_route_registry,
)


def test_route_registry_valid():
    """Route registry should pass validation."""
    errors = validate_route_registry()
    assert not errors, f"Route registry validation failed: {errors}"


def test_core_views_non_empty():
    """Core views should not be empty."""
    core = get_core_views()
    assert len(core) > 0, "No core views registered"


def test_advanced_views_non_empty():
    """Advanced views should not be empty."""
    advanced = get_advanced_views()
    assert len(advanced) > 0, "No advanced views registered"


def test_no_duplicate_titles():
    """All view titles should be unique."""
    registry = get_route_registry()
    titles = [info["title"] for info in registry.values()]
    assert len(titles) == len(set(titles)), "Duplicate titles found in route registry"


def test_all_views_have_renderer():
    """All views should have a renderer function name."""
    registry = get_route_registry()
    for view_id, info in registry.items():
        assert info["renderer"], f"View '{view_id}' has no renderer"


def test_all_views_have_category():
    """All views should have a valid category."""
    registry = get_route_registry()
    valid_categories = {"core", "advanced", "experimental"}
    for view_id, info in registry.items():
        assert info["category"] in valid_categories, f"View '{view_id}' has invalid category"


def test_coverage_flag_is_boolean():
    """All views should have a boolean requires_coverage flag."""
    registry = get_route_registry()
    for view_id, info in registry.items():
        assert isinstance(info["requires_coverage"], bool), f"View '{view_id}' requires_coverage not boolean"


if __name__ == "__main__":
    test_route_registry_valid()
    test_core_views_non_empty()
    test_advanced_views_non_empty()
    test_no_duplicate_titles()
    test_all_views_have_renderer()
    test_all_views_have_category()
    test_coverage_flag_is_boolean()
    print("All route registry tests passed!")