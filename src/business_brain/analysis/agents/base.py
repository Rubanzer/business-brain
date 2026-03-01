"""Base class for analysis agents (Track 2)."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.analysis.models import AgentOutput, AnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Context passed to every agent."""

    session: AsyncSession
    result: AnalysisResult
    run_id: str
    time_scope: dict | None = None  # {column, window, compare_to}
    extra: dict | None = None  # agent-specific overrides


class AnalysisAgent(ABC):
    """Abstract base class for Track 2 agents."""

    agent_id: str = "base"

    @abstractmethod
    async def build_context(self, ctx: AgentContext) -> dict[str, Any]:
        """Gather context needed for analysis (SQL queries, RAG lookups, etc.)."""
        ...

    @abstractmethod
    async def analyze(self, ctx: AgentContext, context_data: dict[str, Any]) -> dict[str, Any]:
        """Run the analysis. Returns structured output."""
        ...

    async def run(self, ctx: AgentContext) -> AgentOutput:
        """Concrete runner with timing, error handling, and persistence."""
        start = time.monotonic()
        output_data: dict[str, Any] = {}
        confidence = 0.0
        error: str | None = None

        try:
            context_data = await self.build_context(ctx)
            output_data = await self.analyze(ctx, context_data)
            confidence = output_data.pop("_confidence", 0.5)
        except Exception as exc:
            error = str(exc)[:500]
            logger.warning("Agent %s failed on result %s: %s", self.agent_id, ctx.result.id, error)

        duration_ms = int((time.monotonic() - start) * 1000)

        agent_output = AgentOutput(
            run_id=ctx.run_id,
            result_id=ctx.result.id,
            agent_id=self.agent_id,
            output=output_data,
            confidence=confidence,
            duration_ms=duration_ms,
            error=error,
        )
        ctx.session.add(agent_output)
        return agent_output
