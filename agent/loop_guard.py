"""
LoopGuard: detects agent loop anomalies and injects corrective prompts.

Generic mechanism — no business-specific knowledge. Detects:
- Repeated identical tool calls (same name + same args)
- Consecutive tool errors

Escalates: first anomaly → corrective message, too many → force STOP.

Also the MAX_ITERATIONS safety net: when the guard detects that the loop
is approaching the hard limit, it injects a countdown nudge.
"""

import json
from typing import Any


class LoopGuard:
    """Observable guard for the agent main loop."""

    MAX_CORRECTIONS = 2  # How many times we'll inject a correction before forcing STOP
    ITERATION_NUDGE = 5  # N remaining iterations → warn LLM to wrap up soon

    def __init__(self, max_iterations: int):
        self._max_iterations = max_iterations
        self._consecutive_errors = 0
        self._last_calls: list[tuple[str, str]] = []  # (tool_name, input_signature)
        self._correction_count = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def observe(
        self,
        tool_name: str,
        tool_input: dict,
        result: Any,
        iteration: int,
    ) -> str | None:
        """
        Observe a tool execution.

        Returns:
            None  — everything is fine, continue normally
            str   — corrective message to inject into conversation
            "STOP" — force end_turn (too many anomalies, giving up)
        """
        # 1. Consecutive errors
        result_msg = self._check_errors(result)
        if result_msg:
            return result_msg

        # 2. Repeated identical calls
        result_msg = self._check_repeats(tool_name, tool_input)
        if result_msg:
            return result_msg

        return None

    def nearing_limit(self, iteration: int) -> str | None:
        """Return a countdown nudge if within ITERATION_NUDGE of MAX."""
        remaining = self._max_iterations - iteration
        if remaining <= self.ITERATION_NUDGE and remaining > 0:
            return (
                f"[系统提示] 剩余 {remaining} 轮迭代机会，"
                f"请在剩余轮次内完成分析并调用 generate_report 结束。"
            )
        return None

    def reset(self):
        """Reset all internal state (e.g. after a successful progress step)."""
        self._consecutive_errors = 0
        self._last_calls.clear()
        self._correction_count = 0

    # ── Internals ───────────────────────────────────────────────────────────

    def _check_errors(self, result: Any) -> str | None:
        is_error = isinstance(result, dict) and "error" in result
        if not is_error:
            self._consecutive_errors = 0
            return None

        self._consecutive_errors += 1
        if self._consecutive_errors >= 3:
            return self._escalate(
                f"[系统提示] 连续 {self._consecutive_errors} 次工具调用返回错误。"
                f"请检查参数是否正确，尝试不同的搜索短语或调整相似度阈值。"
                f"如果确实没有更多内容需要分析，请调用 generate_report 结束。"
            )
        return None

    def _check_repeats(self, tool_name: str, tool_input: dict) -> str | None:
        sig = json.dumps(tool_input, sort_keys=True, ensure_ascii=False)
        self._last_calls.append((tool_name, sig))

        # Keep only last 4 entries for sliding-window comparison
        if len(self._last_calls) > 4:
            self._last_calls.pop(0)

        # Need at least 3 recent calls that are all identical
        if len(self._last_calls) >= 3:
            recent = self._last_calls[-3:]
            if all(c == recent[0] for c in recent):
                return self._escalate(
                    f"[系统提示] 检测到连续 3 次重复调用 `{tool_name}` 且参数不变，"
                    f"结果不会改变。请换一种方式处理（如不同的搜索短语），"
                    f"或如果任务已完成，直接调用 generate_report 生成报告并结束。"
                )

        return None

    def _escalate(self, message: str) -> str:
        self._correction_count += 1
        self._last_calls.clear()
        self._consecutive_errors = 0
        if self._correction_count > self.MAX_CORRECTIONS:
            return "STOP"
        return message
