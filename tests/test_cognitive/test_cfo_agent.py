"""Tests for the CFO Agent — python_analysis, JSON parsing edge cases."""

from unittest.mock import MagicMock, patch

from business_brain.cognitive.cfo_agent import CFOAgent


class TestCFOAgent:
    @patch("business_brain.cognitive.cfo_agent._get_client")
    def test_includes_python_analysis(self, mock_client):
        response = MagicMock()
        response.text = '{"approved": true, "reasoning": "Good ROI", "recommendations": ["Scale up"]}'
        mock_client.return_value.models.generate_content.return_value = response

        agent = CFOAgent()
        state = {
            "question": "Should we expand?",
            "analysis": {"findings": [{"type": "insight", "description": "Growth"}], "summary": "Growing"},
            "python_analysis": {
                "computations": [{"label": "ROI", "value": "15%"}],
                "narrative": "Strong return on investment.",
            },
        }
        result = agent.invoke(state)
        assert result["approved"] is True
        assert result["cfo_notes"] == "Good ROI"

        # Verify the prompt included python analysis
        call_kwargs = mock_client.return_value.models.generate_content.call_args
        assert call_kwargs is not None

    @patch("business_brain.cognitive.cfo_agent._get_client")
    def test_no_python_analysis(self, mock_client):
        response = MagicMock()
        response.text = '{"approved": false, "reasoning": "Risky", "recommendations": []}'
        mock_client.return_value.models.generate_content.return_value = response

        agent = CFOAgent()
        state = {
            "question": "test",
            "analysis": {"findings": [], "summary": "No data"},
        }
        result = agent.invoke(state)
        assert result["approved"] is False

    @patch("business_brain.cognitive.cfo_agent._get_client")
    def test_failure_defaults(self, mock_client):
        mock_client.return_value.models.generate_content.side_effect = RuntimeError("API down")

        agent = CFOAgent()
        state = {"question": "test", "analysis": {}}
        result = agent.invoke(state)
        assert result["approved"] is False
        assert "manual review" in result["cfo_notes"]

    # --- JSON parsing edge cases ---

    @patch("business_brain.cognitive.cfo_agent._get_client")
    def test_response_with_markdown_fence(self, mock_client):
        response = MagicMock()
        response.text = '```json\n{"approved": true, "reasoning": "OK", "recommendations": ["Go"]}\n```'
        mock_client.return_value.models.generate_content.return_value = response

        agent = CFOAgent()
        state = {"question": "test", "analysis": {"findings": [{"type": "insight", "description": "X"}], "summary": "Y"}}
        result = agent.invoke(state)
        assert result["approved"] is True
        assert result["cfo_notes"] == "OK"

    @patch("business_brain.cognitive.cfo_agent._get_client")
    def test_response_with_text_before_fence(self, mock_client):
        response = MagicMock()
        response.text = 'Based on my assessment:\n```json\n{"approved": false, "reasoning": "Too risky", "recommendations": ["Wait"]}\n```\n'
        mock_client.return_value.models.generate_content.return_value = response

        agent = CFOAgent()
        state = {"question": "test", "analysis": {"findings": [], "summary": "S"}}
        result = agent.invoke(state)
        assert result["approved"] is False
        assert result["cfo_notes"] == "Too risky"

    @patch("business_brain.cognitive.cfo_agent._get_client")
    def test_response_invalid_json(self, mock_client):
        response = MagicMock()
        response.text = '{"approved": true, BROKEN}'
        mock_client.return_value.models.generate_content.return_value = response

        agent = CFOAgent()
        state = {"question": "test", "analysis": {}}
        result = agent.invoke(state)
        assert result["approved"] is False
        assert "manual review" in result["cfo_notes"]

    @patch("business_brain.cognitive.cfo_agent._get_client")
    def test_response_empty_string(self, mock_client):
        response = MagicMock()
        response.text = ""
        mock_client.return_value.models.generate_content.return_value = response

        agent = CFOAgent()
        state = {"question": "test", "analysis": {}}
        result = agent.invoke(state)
        assert result["approved"] is False
        assert "manual review" in result["cfo_notes"]

    @patch("business_brain.cognitive.cfo_agent._get_client")
    def test_response_just_backticks(self, mock_client):
        response = MagicMock()
        response.text = "```\n```"
        mock_client.return_value.models.generate_content.return_value = response

        agent = CFOAgent()
        state = {"question": "test", "analysis": {}}
        result = agent.invoke(state)
        assert result["approved"] is False

    @patch("business_brain.cognitive.cfo_agent._get_client")
    def test_response_unclosed_fence(self, mock_client):
        response = MagicMock()
        response.text = '```json\n{"approved": true, "reasoning": "OK", "recommendations": []}'
        mock_client.return_value.models.generate_content.return_value = response

        agent = CFOAgent()
        state = {"question": "test", "analysis": {"findings": [{"type": "insight", "description": "X"}], "summary": "Y"}}
        result = agent.invoke(state)
        assert result["approved"] is True

    @patch("business_brain.cognitive.cfo_agent._get_client")
    def test_response_trailing_comma(self, mock_client):
        response = MagicMock()
        response.text = '{"approved": true, "reasoning": "OK", "recommendations": ["Go",],}'
        mock_client.return_value.models.generate_content.return_value = response

        agent = CFOAgent()
        state = {"question": "test", "analysis": {}}
        result = agent.invoke(state)
        # Invalid JSON — falls back
        assert result["approved"] is False

    @patch("business_brain.cognitive.cfo_agent._get_client")
    def test_response_missing_approved_key(self, mock_client):
        response = MagicMock()
        response.text = '{"reasoning": "OK", "recommendations": ["Go"]}'
        mock_client.return_value.models.generate_content.return_value = response

        agent = CFOAgent()
        state = {"question": "test", "analysis": {"findings": [{"type": "insight", "description": "X"}], "summary": "Y"}}
        result = agent.invoke(state)
        # result.get("approved", False) => False
        assert result["approved"] is False

    @patch("business_brain.cognitive.cfo_agent._get_client")
    def test_response_missing_reasoning_key(self, mock_client):
        response = MagicMock()
        response.text = '{"approved": true, "recommendations": ["Do it"]}'
        mock_client.return_value.models.generate_content.return_value = response

        agent = CFOAgent()
        state = {"question": "test", "analysis": {"findings": [{"type": "insight", "description": "X"}], "summary": "Y"}}
        result = agent.invoke(state)
        assert result["approved"] is True
        assert result["cfo_notes"] == ""  # reasoning defaults to ""

    @patch("business_brain.cognitive.cfo_agent._get_client")
    def test_python_analysis_in_prompt(self, mock_client):
        """Verify python_analysis metrics and narrative appear in the prompt."""
        response = MagicMock()
        response.text = '{"approved": true, "reasoning": "Metrics check out", "recommendations": []}'
        mock_client.return_value.models.generate_content.return_value = response

        agent = CFOAgent()
        state = {
            "question": "test",
            "analysis": {"findings": [], "summary": "S"},
            "python_analysis": {
                "computations": [{"label": "Margin", "value": "22%"}],
                "narrative": "Healthy margins.",
            },
        }
        result = agent.invoke(state)

        call_args = mock_client.return_value.models.generate_content.call_args
        prompt = call_args.kwargs.get("contents", call_args.args[0] if call_args.args else "")
        assert "Margin" in prompt
        assert "Healthy margins." in prompt

    @patch("business_brain.cognitive.cfo_agent._get_client")
    def test_empty_python_analysis(self, mock_client):
        """python_analysis exists but has empty computations and narrative."""
        response = MagicMock()
        response.text = '{"approved": true, "reasoning": "OK", "recommendations": []}'
        mock_client.return_value.models.generate_content.return_value = response

        agent = CFOAgent()
        state = {
            "question": "test",
            "analysis": {"findings": [{"type": "insight", "description": "X"}], "summary": "Y"},
            "python_analysis": {"computations": [], "narrative": ""},
        }
        result = agent.invoke(state)
        assert result["approved"] is True

        # "Computational Analysis" section should NOT appear since both are empty
        prompt = mock_client.return_value.models.generate_content.call_args.kwargs.get(
            "contents",
            mock_client.return_value.models.generate_content.call_args.args[0]
            if mock_client.return_value.models.generate_content.call_args.args
            else "",
        )
        assert "Computational Analysis" not in prompt

    @patch("business_brain.cognitive.cfo_agent._get_client")
    def test_no_analysis_key(self, mock_client):
        """State has no 'analysis' key."""
        response = MagicMock()
        response.text = '{"approved": false, "reasoning": "No data", "recommendations": []}'
        mock_client.return_value.models.generate_content.return_value = response

        agent = CFOAgent()
        state = {"question": "test"}
        result = agent.invoke(state)
        assert result["approved"] is False

    @patch("business_brain.cognitive.cfo_agent._get_client")
    def test_response_non_object_json(self, mock_client):
        """LLM returns a JSON array instead of object."""
        response = MagicMock()
        response.text = '[{"approved": true}]'
        mock_client.return_value.models.generate_content.return_value = response

        agent = CFOAgent()
        state = {"question": "test", "analysis": {}}
        result = agent.invoke(state)
        # list.get() raises AttributeError, caught by except
        assert result["approved"] is False
        assert "manual review" in result["cfo_notes"]

    @patch("business_brain.cognitive.cfo_agent._get_client")
    def test_response_with_newlines_in_fence(self, mock_client):
        response = MagicMock()
        response.text = '```json\n\n{"approved": true, "reasoning": "LGTM", "recommendations": []}\n\n```'
        mock_client.return_value.models.generate_content.return_value = response

        agent = CFOAgent()
        state = {"question": "test", "analysis": {"findings": [{"type": "insight", "description": "X"}], "summary": "Y"}}
        result = agent.invoke(state)
        assert result["approved"] is True
        assert result["cfo_notes"] == "LGTM"
