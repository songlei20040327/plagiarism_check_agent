"""
Memory tools: expose chunk hotspots + curated knowledge to the Agent.

Single read-only tool — the Agent queries memory to inform its
classification decisions but never writes to it. Writes happen
via the orchestrator (hotspot tracking) and CLI (human corrections).
"""

from ..memory.store import get_memory_summary


class MemoryTools:
    """Handler for memory read operations."""

    def memory_get_summary(self, max_tokens: int = 2000) -> str:
        """
        Return formatted memory summary: hotspots, common knowledge,
        and human-verified false positives.
        """
        return get_memory_summary(max_tokens=max_tokens)


# ── Register with ToolRegistry ─────────────────────────────────────────

from .registry import ToolRegistry  # noqa: E402

ToolRegistry.register(
    name="memory_get_summary",
    description=(
        "获取积累的领域知识摘要，包含三部分："
        "1) 高频匹配模式（在3篇以上不同论文中被命中的对照文献文本片段，"
        "很可能是领域通用表述，判重时应降级处理）；"
        "2) 领域常识（标准方法论、术语等不应标记为重复的内容）；"
        "3) 人工纠错记录（曾被误判但经人工确认无问题的文本模式）。"
        "在逐段查重分析前先调用此工具，避免将常识性内容和高频通用表述误判为抄袭。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "max_tokens": {
                "type": "integer",
                "description": "摘要的最大 token 数（默认 2000）",
            },
        },
    },
    handler_class=MemoryTools,
)
