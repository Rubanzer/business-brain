"""CFO Filter agent — economic viability gate for proposed actions."""

from typing import Any

SYSTEM_PROMPT = """\
You are a CFO reviewing analysis findings. For each proposed action, assess:
1. Expected revenue impact
2. Implementation cost
3. Risk level
Only approve actions with positive expected ROI and acceptable risk.
"""


class CFOAgent:
    """Gates analysis outputs through an economic viability check."""

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Evaluate whether the analysis findings are economically viable.

        TODO: call LLM with SYSTEM_PROMPT + analysis findings.
        """
        analysis = state.get("analysis", {})
        print(f"[cfo_agent] Evaluating {len(analysis.get('findings', []))} findings")
        state["approved"] = True  # placeholder — always approve
        state["cfo_notes"] = "Placeholder: approved pending detailed review."
        return state
