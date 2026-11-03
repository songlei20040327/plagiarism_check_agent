"""
Tool router: maps tool names to Python functions with result truncation.

Decoupled from the tools layer — reads tool definitions and handlers
from ToolRegistry. Adding a new tool in tools/ requires no changes here.

Truncation strategy (v2): token-budget-based, not list-length-based.
Uses tiktoken to count actual tokens and greedily packs as much evidence
as possible into each tool result before hitting the waterline.
"""

import json
import math
from pathlib import Path
from jsonschema import Draft202012Validator, ValidationError
from typing import Any, Callable

import tiktoken

from ..tools.registry import ToolRegistry
from ..config import Config


class ToolRouter:
    """
    Dynamically dispatches tool calls based on ToolRegistry.

    No hardcoded imports of tool classes or TOOL_DEFINITIONS.
    All tool metadata comes from the registry.

    Result truncation uses a rigid token budget (MAX_TOOL_RESULT_TOKENS).
    When a tool returns large results — e.g. search_similar_chunks with many
    matches, or get_chunk_context with long text — the result is recursively
    walked and items/fields are greedily included until the token waterline
    is reached. This guarantees we never overflow the LLM context while
    maximizing evidence throughput.
    """

    MAX_TOOL_RESULT_TOKENS: int = Config.MAX_TOOL_RESULT_TOKENS
    MAX_RECURSION_DEPTH: int = 10

    # ── Init ────────────────────────────────────────────────────────────

    def __init__(self, dry_run: bool = False, document_tools=None):
        self._dry_run = dry_run
        self.session = None
        self.last_result = None

        # Ensure tool modules are imported (triggers registry registration)
        from .. import tools  # noqa: F401

        # Build dispatch table from registry
        from ..tools.document_tools import DocumentTools
        self.document_tools = document_tools or DocumentTools()
        instances = {DocumentTools: self.document_tools}
        self._dispatch: dict[str, Callable] = {}
        for d in ToolRegistry.get_definitions():
            name = d["name"]
            handler = ToolRegistry.get_handler(name, instances)
            if handler is not None:
                self._dispatch[name] = handler

        # Initialize tiktoken tokenizer (cl100k_base = GPT-4 / GPT-3.5)
        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    # ── Public API ──────────────────────────────────────────────────────

    def bind_task(self, thesis_document_id: str, output_dir: str):
        from ..tools.review_session import ReviewSession
        self.session = ReviewSession(self.document_tools, thesis_document_id, output_dir)

    def execute(self, tool_name: str, tool_input: dict) -> Any:
        result = self._execute_validated(tool_name, tool_input)
        self.last_result = result
        return self._truncate_result(result)

    def _execute_validated(self, tool_name: str, tool_input: dict) -> dict:
        if tool_name not in self._dispatch:
            return {"error": f"Unknown tool: {tool_name}", "error_code": "unknown_tool", "retryable": False}
        try:
            definition = next(d for d in self.tool_definitions if d["name"] == tool_name)
            Draft202012Validator(definition["input_schema"]).validate(tool_input)
            def check_finite(value):
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("Non-finite tool arguments are not allowed.")
                if isinstance(value, dict):
                    for v in value.values(): check_finite(v)
                if isinstance(value, list):
                    for v in value: check_finite(v)
            check_finite(tool_input)
            arguments = dict(tool_input)
            if self.session:
                allowed = {"search_similar_chunks", "get_chunk_context", "get_corpus_stats", "generate_report"}
                if tool_name not in allowed:
                    raise ValueError("This tool is unavailable during the review phase.")
                if tool_name == "search_similar_chunks":
                    arguments = self.session.search_arguments(arguments)
                elif tool_name == "get_chunk_context":
                    if arguments["chunk_id"] not in {k[1] for k in self.session.candidates}:
                        raise ValueError("Context may only be requested for retrieved evidence.")
                elif tool_name == "generate_report":
                    arguments = self.session.report_arguments(arguments)
            elif tool_name == "generate_report" and not self._dry_run:
                raise ValueError("Bind a review task before generating a report.")
            elif "thesis_chunk_id" in arguments:
                raise ValueError("thesis_chunk_id requires a bound review task.")
            if self._dry_run and ToolRegistry.is_write_tool(tool_name):
                result = {"dry_run": True, "tool": tool_name, "message": "Report write skipped."}
                if self.session:
                    self.session.report_result = result
                    self.session.report_findings = arguments.get("findings", [])
                return result
            result = self._dispatch[tool_name](**arguments)
            from ..tools.result_validation import validate_result
            validate_result(tool_name, result)
            if self.session and "error" not in result:
                if tool_name == "search_similar_chunks":
                    self.session.record_search(tool_input["thesis_chunk_id"], result)
                    result = {**result, "thesis_chunk_id": tool_input["thesis_chunk_id"]}
                elif tool_name == "generate_report":
                    self.session.report_result = result.copy()
                    self.session.report_findings = arguments["findings"]
            return result
        except (ValidationError, ValueError, TypeError) as exc:
            return {"error": str(exc), "tool": tool_name, "error_code": "validation_error", "retryable": False}
        except Exception as exc:
            return {"error": str(exc), "tool": tool_name, "error_code": "execution_error", "retryable": True}

    def completion_state(self) -> tuple[str, str | None]:
        if not self.session or not self.session.report_result:
            return "partial", "No validated report was generated."
        result = self.session.report_result
        if result.get("dry_run"):
            return "dry_run", None
        try:
            from ..tools.result_validation import validate_result
            validate_result("generate_report", result)
            data = json.loads(Path(result["json_data"]).read_text(encoding="utf-8"))
            if data.get("snapshot_id") != self.document_tools.snapshot_id or data.get("findings") != self.session.report_findings:
                raise ValueError("Report contents do not match this task's evidence.")
            return "completed", None
        except Exception as exc:
            return "failed", f"Report validation failed: {exc}"

    @property
    def tool_definitions(self) -> list[dict]:
        definitions = ToolRegistry.get_definitions()
        if self.session:
            definitions = [d for d in definitions if d["name"] in {
                "search_similar_chunks", "get_chunk_context", "get_corpus_stats", "generate_report"
            }]
        return definitions

    # ── Token counting ──────────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        """Count tokens in a string via tiktoken.

        Falls back to char-length as a conservative upper bound
        (safe for Chinese: 1 char ≈ 1-2 tokens; overestimates
        for English but that's the safe direction).
        """
        if self._enc is not None:
            return len(self._enc.encode(text))
        return len(text)

    def _count_tokens_scalar(self, value: Any) -> int:
        """Token cost of a scalar value as it would appear in JSON."""
        if isinstance(value, str):
            return self._count_tokens(value)
        if isinstance(value, bool):
            return 1  # "true" or "false"
        if value is None:
            return 1  # "null"
        # numbers
        return max(len(str(value)), 1)

    # ── Entry point ─────────────────────────────────────────────────────

    def _truncate_result(self, result: Any) -> Any:
        """Apply token-budget truncation to a tool result."""
        truncated, _ = self._truncate_value(
            result, self.MAX_TOOL_RESULT_TOKENS, depth=0
        )
        if self._count_tokens(json.dumps(truncated, ensure_ascii=False)) > self.MAX_TOOL_RESULT_TOKENS:
            truncated = {"error": "Result too large; request less context.", "retryable": True}
            if self._count_tokens(json.dumps(truncated, ensure_ascii=False)) > self.MAX_TOOL_RESULT_TOKENS:
                truncated = {}
        return truncated

    # ── Recursive dispatcher ────────────────────────────────────────────

    def _truncate_value(self, obj: Any, budget: int, depth: int) -> tuple[Any, int]:
        """Recursively truncate *obj* to fit within *budget* tokens.

        Returns (truncated_object, tokens_consumed).
        tokens_consumed is an estimate of the JSON-serialized form.
        """
        if budget <= 0:
            return self._budget_exhausted_sentinel(), 0

        if depth > self.MAX_RECURSION_DEPTH:
            return {"_error": "max recursion depth exceeded"}, 4

        if isinstance(obj, str):
            return self._truncate_string(obj, budget)
        elif isinstance(obj, list):
            return self._truncate_list(obj, budget, depth)
        elif isinstance(obj, dict):
            return self._truncate_dict(obj, budget, depth)
        else:
            return obj, self._count_tokens_scalar(obj)

    # ── Per-type truncation ─────────────────────────────────────────────

    def _truncate_string(self, text: str, budget: int) -> tuple[str, int]:
        """Truncate a string to fit within *budget* tokens."""
        if not text:
            return text, 0

        tokens = self._count_tokens(text)
        if tokens <= budget:
            return text, tokens

        # Reserve tokens for the trailing truncation notice
        notice = f"…[截断: 原文 {tokens} tokens → 保留 {budget} tokens]"
        notice_tokens = self._count_tokens(notice)
        available = max(budget - notice_tokens, 1)

        if self._enc is not None:
            raw_tokens = self._enc.encode(text)
            truncated_text = self._enc.decode(raw_tokens[:available])
        else:
            # Conservative: assume 1 char ≈ 1 token (safe for Chinese)
            truncated_text = text[:available]

        return truncated_text + notice, budget

    def _truncate_list(self, items: list, budget: int, depth: int) -> tuple[list, int]:
        """Greedily include list items until the token budget runs out.

        Each item gets its share of the remaining budget. When the
        budget is exhausted, remaining items are replaced by a
        single truncation-marker dict.
        """
        if not items:
            return items, 2  # "[]"

        result: list = []
        tokens_used = 0

        for item in items:
            # Structural overhead per list item in JSON: comma + whitespace ≈ 4 tokens
            overhead = 4
            remaining = budget - tokens_used - overhead
            if remaining <= 0:
                break

            truncated, item_tokens = self._truncate_value(item, remaining, depth + 1)
            result.append(truncated)
            tokens_used += item_tokens + overhead

        if len(result) < len(items):
            result.append({
                "_list_truncated": True,
                "_shown": len(result),
                "_total": len(items),
                "_message": (
                    f"列表已截断: 仅展示前 {len(result)}/{len(items)} 项 "
                    f"(token 预算 {self.MAX_TOOL_RESULT_TOKENS} 用尽)"
                ),
            })

        return result, min(tokens_used, budget)

    def _truncate_dict(self, obj: dict, budget: int, depth: int) -> tuple[dict, int]:
        """Process dict fields in insertion order until budget exhausted.

        Fields are visited in order (Python 3.7+ guarantees insertion
        order). Once the budget is spent, remaining keys are listed in
        a ``_fields_omitted`` sentinel so the LLM knows what was dropped.
        """
        if not obj:
            return obj, 2  # "{}"

        result: dict = {}
        tokens_used = 0
        skipped: list[str] = []

        for key, value in obj.items():
            # Estimate overhead for this key-value pair in JSON:
            #   "key": <value>  plus comma ≈ 10 tokens overhead
            key_overhead = self._count_tokens(json.dumps(key, ensure_ascii=False))
            entry_overhead = key_overhead + 8
            remaining = budget - tokens_used - entry_overhead
            if remaining <= 0:
                skipped.append(key)
                continue

            truncated, value_tokens = self._truncate_value(value, remaining, depth + 1)
            result[key] = truncated
            tokens_used += entry_overhead + value_tokens

        if skipped:
            result["_fields_omitted"] = {
                "_skipped_keys": skipped,
                "_reason": (
                    f"token 预算 {self.MAX_TOOL_RESULT_TOKENS} 用尽, "
                    f"以下 {len(skipped)} 个字段被省略"
                ),
            }

        return result, min(tokens_used, budget)

    # ── Sentinel ────────────────────────────────────────────────────────

    def _budget_exhausted_sentinel(self) -> dict:
        return {
            "_error": "truncation",
            "_message": (
                f"工具返回结果超出 token 预算 "
                f"({self.MAX_TOOL_RESULT_TOKENS} tokens), 已被完全截断"
            ),
        }
