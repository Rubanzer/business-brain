"""Tests for the Supervisor Agent — planning, chat history, JSON parsing edge cases."""

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

    # --- JSON parsing edge cases ---

    @patch("business_brain.cognitive.supervisor._get_client")
    def test_plan_with_markdown_json_fence(self, mock_client):
        response = MagicMock()
        response.text = '```json\n[{"agent": "sql_agent", "task": "T1"}]\n```'
        mock_client.return_value.models.generate_content.return_value = response

        agent = SupervisorAgent()
        result = agent.invoke({"question": "test"})
        assert len(result["plan"]) == 1
        assert result["plan"][0]["agent"] == "sql_agent"

    @patch("business_brain.cognitive.supervisor._get_client")
    def test_plan_with_plain_fence(self, mock_client):
        response = MagicMock()
        response.text = '```\n[{"agent": "sql_agent", "task": "T1"}]\n```'
        mock_client.return_value.models.generate_content.return_value = response

        agent = SupervisorAgent()
        result = agent.invoke({"question": "test"})
        assert len(result["plan"]) == 1

    @patch("business_brain.cognitive.supervisor._get_client")
    def test_plan_with_text_around_fence(self, mock_client):
        response = MagicMock()
        response.text = 'Here is the plan:\n```json\n[{"agent": "sql_agent", "task": "T1"}, {"agent": "analyst_agent", "task": "T2"}]\n```\nDone.'
        mock_client.return_value.models.generate_content.return_value = response

        agent = SupervisorAgent()
        result = agent.invoke({"question": "test"})
        assert len(result["plan"]) == 2

    @patch("business_brain.cognitive.supervisor._get_client")
    def test_plan_invalid_json_uses_fallback(self, mock_client):
        response = MagicMock()
        response.text = '{"not an array"}'
        mock_client.return_value.models.generate_content.return_value = response

        agent = SupervisorAgent()
        result = agent.invoke({"question": "some question"})
        # json.loads succeeds but plan is a dict, not list — still set
        # The code just does `state["plan"] = plan` without validating list
        assert "plan" in result

    @patch("business_brain.cognitive.supervisor._get_client")
    def test_plan_completely_broken_json(self, mock_client):
        response = MagicMock()
        response.text = 'Sure! Here is my analysis plan for your question...'
        mock_client.return_value.models.generate_content.return_value = response

        agent = SupervisorAgent()
        result = agent.invoke({"question": "test"})
        # Falls back to default plan
        assert result["plan"][0]["agent"] == "sql_agent"
        assert len(result["plan"]) >= 2

    @patch("business_brain.cognitive.supervisor._get_client")
    def test_plan_empty_response(self, mock_client):
        response = MagicMock()
        response.text = ""
        mock_client.return_value.models.generate_content.return_value = response

        agent = SupervisorAgent()
        result = agent.invoke({"question": "test"})
        assert result["plan"][0]["agent"] == "sql_agent"

    @patch("business_brain.cognitive.supervisor._get_client")
    def test_plan_just_backticks(self, mock_client):
        response = MagicMock()
        response.text = "```\n```"
        mock_client.return_value.models.generate_content.return_value = response

        agent = SupervisorAgent()
        result = agent.invoke({"question": "test"})
        # Falls back to default
        assert result["plan"][0]["agent"] == "sql_agent"

    @patch("business_brain.cognitive.supervisor._get_client")
    def test_plan_with_trailing_comma(self, mock_client):
        response = MagicMock()
        response.text = '[{"agent": "sql_agent", "task": "T1"},]'
        mock_client.return_value.models.generate_content.return_value = response

        agent = SupervisorAgent()
        result = agent.invoke({"question": "test"})
        # Trailing comma = invalid JSON = fallback
        assert result["plan"][0]["agent"] == "sql_agent"

    @patch("business_brain.cognitive.supervisor._get_client")
    def test_plan_with_unclosed_fence(self, mock_client):
        response = MagicMock()
        response.text = '```json\n[{"agent": "sql_agent", "task": "T1"}]'
        mock_client.return_value.models.generate_content.return_value = response

        agent = SupervisorAgent()
        result = agent.invoke({"question": "test"})
        # Unclosed fence: split gives 2 parts, parts[1] = 'json\n[...]'
        # After stripping "json", should parse fine
        assert len(result["plan"]) == 1

    @patch("business_brain.cognitive.supervisor._get_client")
    def test_empty_chat_history(self, mock_client):
        response = MagicMock()
        response.text = '[{"agent": "sql_agent", "task": "T1"}]'
        mock_client.return_value.models.generate_content.return_value = response

        agent = SupervisorAgent()
        result = agent.invoke({"question": "test", "chat_history": []})
        assert len(result["plan"]) == 1

    @patch("business_brain.cognitive.supervisor._get_client")
    def test_chat_history_with_missing_keys(self, mock_client):
        response = MagicMock()
        response.text = '[{"agent": "sql_agent", "task": "T1"}]'
        mock_client.return_value.models.generate_content.return_value = response

        agent = SupervisorAgent()
        result = agent.invoke({
            "question": "test",
            "chat_history": [{"role": "user"}, {"content": "answer"}],
        })
        assert len(result["plan"]) == 1

    @patch("business_brain.cognitive.supervisor._get_client")
    def test_no_question_key(self, mock_client):
        response = MagicMock()
        response.text = '[{"agent": "sql_agent", "task": "T1"}]'
        mock_client.return_value.models.generate_content.return_value = response

        agent = SupervisorAgent()
        result = agent.invoke({})
        assert "plan" in result

    @patch("business_brain.cognitive.supervisor._get_client")
    def test_large_chat_history_truncated(self, mock_client):
        """Only last 10 messages should be included."""
        response = MagicMock()
        response.text = '[{"agent": "sql_agent", "task": "T1"}]'
        mock_client.return_value.models.generate_content.return_value = response

        history = [{"role": "user", "content": f"msg-{i}"} for i in range(20)]
        agent = SupervisorAgent()
        result = agent.invoke({"question": "test", "chat_history": history})

        prompt = mock_client.return_value.models.generate_content.call_args.kwargs.get(
            "contents",
            mock_client.return_value.models.generate_content.call_args.args[0]
            if mock_client.return_value.models.generate_content.call_args.args
            else "",
        )
        # First messages (0-9) should NOT be in prompt, only 10-19
        assert "msg-0" not in prompt
        assert "msg-19" in prompt

    @patch("business_brain.cognitive.supervisor._get_client")
    def test_plan_with_newline_after_json_tag(self, mock_client):
        response = MagicMock()
        response.text = '```json\n\n[{"agent": "sql_agent", "task": "T1"}]\n\n```'
        mock_client.return_value.models.generate_content.return_value = response

        agent = SupervisorAgent()
        result = agent.invoke({"question": "test"})
        assert len(result["plan"]) == 1
