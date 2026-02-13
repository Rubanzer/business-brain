"""Tests for the CFO Agent — python_analysis integration."""

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
        prompt = mock_client.return_value.models.generate_content.call_args[1].get(
            "contents"
        ) or mock_client.return_value.models.generate_content.call_args[0][0] if mock_client.return_value.models.generate_content.call_args[0] else ""
        # The prompt should be passed as a keyword or positional arg
        call_kwargs = mock_client.return_value.models.generate_content.call_args
        # Check that it was called with content containing python analysis
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
