"""Tests for onboarding pure functions — context generation and completeness."""

from business_brain.action.onboarding import (
    _generate_context_text,
    compute_profile_completeness,
)


class _FakeProfile:
    """Minimal stand-in for CompanyProfile."""
    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "")
        self.industry = kwargs.get("industry", None)
        self.products = kwargs.get("products", None)
        self.departments = kwargs.get("departments", None)
        self.process_flow = kwargs.get("process_flow", None)
        self.systems = kwargs.get("systems", None)
        self.known_relationships = kwargs.get("known_relationships", None)


# ---------------------------------------------------------------------------
# _generate_context_text
# ---------------------------------------------------------------------------


class TestGenerateContextText:
    def test_empty_profile(self):
        p = _FakeProfile()
        text = _generate_context_text(p)
        assert text == ""

    def test_name_only(self):
        p = _FakeProfile(name="Acme Corp")
        text = _generate_context_text(p)
        assert "Company: Acme Corp" in text

    def test_industry(self):
        p = _FakeProfile(name="Steel Co", industry="Steel Manufacturing")
        text = _generate_context_text(p)
        assert "Industry: Steel Manufacturing" in text

    def test_products_list(self):
        p = _FakeProfile(name="X", products=["TMT Bars", "Wire Rods"])
        text = _generate_context_text(p)
        assert "TMT Bars" in text
        assert "Wire Rods" in text

    def test_products_string(self):
        p = _FakeProfile(name="X", products="Steel")
        text = _generate_context_text(p)
        assert "Steel" in text

    def test_departments_with_heads(self):
        p = _FakeProfile(departments=[
            {"name": "Finance", "head": "CFO"},
            {"name": "Operations"},
        ])
        text = _generate_context_text(p)
        assert "Finance" in text
        assert "CFO" in text
        assert "Operations" in text

    def test_departments_empty_list(self):
        p = _FakeProfile(departments=[])
        text = _generate_context_text(p)
        assert "Departments" not in text

    def test_process_flow(self):
        p = _FakeProfile(process_flow="Iron Ore → Smelting → Rolling → Finished Product")
        text = _generate_context_text(p)
        assert "Smelting" in text

    def test_systems(self):
        p = _FakeProfile(systems=[
            {"name": "SAP", "description": "ERP system"},
            {"name": "Excel"},
        ])
        text = _generate_context_text(p)
        assert "SAP: ERP system" in text
        assert "Excel" in text

    def test_systems_empty(self):
        p = _FakeProfile(systems=[])
        text = _generate_context_text(p)
        assert "Data systems" not in text

    def test_known_relationships(self):
        p = _FakeProfile(known_relationships="Supplier table links to Quality via batch_id")
        text = _generate_context_text(p)
        assert "batch_id" in text

    def test_full_profile(self):
        p = _FakeProfile(
            name="Steel Corp",
            industry="Manufacturing",
            products=["TMT Bars"],
            departments=[{"name": "QC", "head": "Manager"}],
            process_flow="A → B → C",
            systems=[{"name": "SAP", "description": "ERP"}],
            known_relationships="x→y",
        )
        text = _generate_context_text(p)
        assert "Company: Steel Corp" in text
        assert "Industry: Manufacturing" in text
        assert "TMT Bars" in text
        assert "QC" in text
        assert "SAP" in text
        assert "x→y" in text

    def test_departments_non_dict_items_skipped(self):
        """Non-dict items in departments should be ignored."""
        p = _FakeProfile(departments=["Finance", "HR"])
        text = _generate_context_text(p)
        # Non-dict items don't generate department lines
        assert "Departments" not in text


# ---------------------------------------------------------------------------
# compute_profile_completeness
# ---------------------------------------------------------------------------


class TestComputeProfileCompleteness:
    def test_empty_profile_zero(self):
        p = _FakeProfile()
        assert compute_profile_completeness(p) == 0

    def test_full_profile_100(self):
        p = _FakeProfile(
            name="X",
            industry="Y",
            products=["A"],
            departments=[{"name": "B"}],
            process_flow="C",
            systems=[{"name": "D"}],
        )
        assert compute_profile_completeness(p) == 100

    def test_half_profile(self):
        p = _FakeProfile(name="X", industry="Y", products=["A"])
        assert compute_profile_completeness(p) == 50

    def test_one_field(self):
        p = _FakeProfile(name="X")
        pct = compute_profile_completeness(p)
        assert 10 < pct < 20  # 1/6 ≈ 16.7%

    def test_known_relationships_not_counted(self):
        """known_relationships is not in the completeness fields list."""
        p1 = _FakeProfile(name="X")
        p2 = _FakeProfile(name="X", known_relationships="foo")
        assert compute_profile_completeness(p1) == compute_profile_completeness(p2)

    def test_empty_list_counts_as_truthy(self):
        """An empty list is falsy in Python."""
        p = _FakeProfile(name="X", products=[])
        pct = compute_profile_completeness(p)
        # products=[] is falsy, so only name counts
        assert pct == compute_profile_completeness(_FakeProfile(name="X"))

    def test_returns_int(self):
        p = _FakeProfile(name="X", industry="Y")
        result = compute_profile_completeness(p)
        assert isinstance(result, int)
