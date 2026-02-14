"""Tests for the onboarding module."""

from business_brain.action.onboarding import _generate_context_text, compute_profile_completeness
from business_brain.db.v3_models import CompanyProfile


class TestProfileCompleteness:
    """Test completeness calculation."""

    def _make_profile(self, **kwargs):
        p = CompanyProfile()
        p.name = kwargs.get("name", "")
        p.industry = kwargs.get("industry")
        p.products = kwargs.get("products")
        p.departments = kwargs.get("departments")
        p.process_flow = kwargs.get("process_flow")
        p.systems = kwargs.get("systems")
        return p

    def test_empty_profile(self):
        profile = self._make_profile()
        assert compute_profile_completeness(profile) == 0

    def test_full_profile(self):
        profile = self._make_profile(
            name="Steel Corp",
            industry="steel",
            products=["TMT bars"],
            departments=[{"name": "Production"}],
            process_flow="Scrap → Furnace → Rolling",
            systems=[{"name": "Tally"}],
        )
        assert compute_profile_completeness(profile) == 100

    def test_half_profile(self):
        profile = self._make_profile(
            name="Steel Corp",
            industry="steel",
            products=["TMT bars"],
        )
        assert compute_profile_completeness(profile) == 50

    def test_just_name(self):
        profile = self._make_profile(name="Corp")
        pct = compute_profile_completeness(profile)
        assert pct > 0
        assert pct < 50


class TestGenerateContextText:
    """Test natural language context generation from profiles."""

    def _make_profile(self, **kwargs):
        p = CompanyProfile()
        p.name = kwargs.get("name", "")
        p.industry = kwargs.get("industry")
        p.products = kwargs.get("products")
        p.departments = kwargs.get("departments")
        p.process_flow = kwargs.get("process_flow")
        p.systems = kwargs.get("systems")
        p.known_relationships = kwargs.get("known_relationships")
        return p

    def test_full_profile_context(self):
        profile = self._make_profile(
            name="Jindal Steel",
            industry="steel",
            products=["TMT bars", "billets"],
            departments=[
                {"name": "Production", "head": "Rahul"},
                {"name": "Quality", "head": "Suresh"},
            ],
            process_flow="Scrap iron → Induction furnace → Continuous casting → Rolling mill → TMT bars",
            systems=[
                {"name": "Tally", "description": "Financial accounting"},
                {"name": "SCADA", "description": "Electrical monitoring"},
            ],
        )
        text = _generate_context_text(profile)
        assert "Jindal Steel" in text
        assert "steel" in text.lower()
        assert "TMT bars" in text
        assert "Production" in text
        assert "Rahul" in text
        assert "Scrap iron" in text
        assert "Tally" in text
        assert "SCADA" in text

    def test_empty_profile_context(self):
        profile = self._make_profile()
        text = _generate_context_text(profile)
        assert text == ""

    def test_name_only_context(self):
        profile = self._make_profile(name="Test Corp")
        text = _generate_context_text(profile)
        assert "Test Corp" in text
        assert "Department" not in text

    def test_departments_without_heads(self):
        profile = self._make_profile(
            departments=[{"name": "HR"}, {"name": "Finance"}],
        )
        text = _generate_context_text(profile)
        assert "HR" in text
        assert "Finance" in text

    def test_products_as_string(self):
        """If products is a plain string instead of list, still works."""
        profile = self._make_profile(products="TMT bars")
        text = _generate_context_text(profile)
        assert "TMT bars" in text

    def test_systems_without_description(self):
        profile = self._make_profile(systems=[{"name": "Tally"}])
        text = _generate_context_text(profile)
        assert "Tally" in text
        assert ": " not in text.split("Tally")[1].split("\n")[0]

    def test_known_relationships_included(self):
        profile = self._make_profile(
            known_relationships="HRMS employee_id links to production operator_id",
        )
        text = _generate_context_text(profile)
        assert "Known data relationships" in text
        assert "employee_id" in text

    def test_departments_non_dict_ignored(self):
        """Non-dict department entries are silently skipped."""
        profile = self._make_profile(departments=["HR", "Finance"])
        text = _generate_context_text(profile)
        # Non-dict entries are skipped, so "Departments" section won't appear
        assert "Departments" not in text

    def test_systems_non_dict_ignored(self):
        """Non-dict system entries are silently skipped."""
        profile = self._make_profile(systems=["Tally", "SCADA"])
        text = _generate_context_text(profile)
        assert "Data systems" not in text

    def test_department_missing_name_uses_unknown(self):
        profile = self._make_profile(departments=[{"head": "Rahul"}])
        text = _generate_context_text(profile)
        assert "Unknown" in text
        assert "Rahul" in text

    def test_system_missing_name_uses_unknown(self):
        profile = self._make_profile(systems=[{"description": "ERP system"}])
        text = _generate_context_text(profile)
        assert "Unknown" in text
        assert "ERP system" in text

    def test_all_parts_separated_by_double_newline(self):
        profile = self._make_profile(
            name="Corp",
            industry="steel",
            process_flow="Melt → Cast",
        )
        text = _generate_context_text(profile)
        parts = text.split("\n\n")
        assert len(parts) == 3

    def test_empty_products_list(self):
        profile = self._make_profile(products=[])
        text = _generate_context_text(profile)
        # Empty list is falsy, so products section should be absent
        assert "Products" not in text


class TestProfileCompletenessEdgeCases:
    """Additional edge case tests for compute_profile_completeness."""

    def _make_profile(self, **kwargs):
        p = CompanyProfile()
        p.name = kwargs.get("name", "")
        p.industry = kwargs.get("industry")
        p.products = kwargs.get("products")
        p.departments = kwargs.get("departments")
        p.process_flow = kwargs.get("process_flow")
        p.systems = kwargs.get("systems")
        return p

    def test_empty_string_name_is_falsy(self):
        """Empty string name doesn't count as filled."""
        profile = self._make_profile(name="")
        assert compute_profile_completeness(profile) == 0

    def test_empty_list_fields_are_falsy(self):
        """Empty lists don't count as filled."""
        profile = self._make_profile(
            name="Corp",
            products=[],
            departments=[],
            systems=[],
        )
        # Only name counts (1/6)
        assert compute_profile_completeness(profile) == 16

    def test_none_fields_are_falsy(self):
        profile = self._make_profile(
            name="Corp",
            industry=None,
            products=None,
        )
        assert compute_profile_completeness(profile) == 16

    def test_all_fields_filled_returns_100(self):
        profile = self._make_profile(
            name="X",
            industry="Y",
            products=["Z"],
            departments=[{"name": "A"}],
            process_flow="B",
            systems=[{"name": "C"}],
        )
        assert compute_profile_completeness(profile) == 100

    def test_returns_int_not_float(self):
        profile = self._make_profile(name="X")
        result = compute_profile_completeness(profile)
        assert isinstance(result, int)
