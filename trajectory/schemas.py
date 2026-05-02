"""Trajectory schemas: data classes for agent execution records."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    content: list[dict]
    stop_reason: str  # "end_turn" | "tool_use"
    usage: dict = field(default_factory=dict)
    model: str = ""

    @property
    def text(self) -> str:
        parts = []
        for b in self.content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
        return "\n".join(parts)

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        result = []
        for b in self.content:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                result.append(ToolUseBlock(
                    id=b["id"], name=b["name"], input=b["input"]
                ))
        return result


@dataclass
class AgentResult:
    messages: list[dict]
    final_response: str
    session_id: str
    stats: dict = field(default_factory=dict)

    @property
    def text_output(self) -> str:
        return self.final_response
