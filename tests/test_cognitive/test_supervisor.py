"""Tests for the Supervisor Agent — chat history context."""

from unittest.mock import MagicMock, patch

from business_brain.cognitive.supervisor import SupervisorAgent


class TestSupervisorAgent:
    @patch("business_brain.cognitive.supervisor._get_client")
    def test_basic_plan(self, mock_client):
        response = MagicMock()
        response.text = '[{"agent": "sql_agent", "task": "Get data"}, {"agent": "analyst_agent", "task": "Analyze"}]'
        mock_client.return_value.models.generate_content.return_value = response

        agent = SupervisorAgent()
        state = {"question": "What are our sales?"}
        result = agent.invoke(state)
        assert len(result["plan"]) == 2
        assert result["plan"][0]["agent"] == "sql_agent"

    @patch("business_brain.cognitive.supervisor._get_client")
    def test_chat_history_included_in_prompt(self, mock_client):
        response = MagicMock()
        response.text = '[{"agent": "sql_agent", "task": "Get follow-up data"}]'
        mock_client.return_value.models.generate_content.return_value = response

        agent = SupervisorAgent()
        state = {
            "question": "Break that down by region",
            "chat_history": [
                {"role": "user", "content": "Show total revenue"},
                {"role": "assistant", "content": "Total revenue is $1M"},
            ],
        }
        result = agent.invoke(state)

        # Verify the prompt contained chat history
        call_args = mock_client.return_value.models.generate_content.call_args
        prompt = call_args.kwargs.get("contents", call_args.args[0] if call_args.args else "")
        assert "Show total revenue" in prompt
        assert "Total revenue is $1M" in prompt

    @patch("business_brain.cognitive.supervisor._get_client")
    def test_fallback_plan_on_error(self, mock_client):
        mock_client.return_value.models.generate_content.side_effect = RuntimeError("fail")

        agent = SupervisorAgent()
        state = {"question": "test"}
        result = agent.invoke(state)
        assert len(result["plan"]) >= 2
        assert result["plan"][0]["agent"] == "sql_agent"
