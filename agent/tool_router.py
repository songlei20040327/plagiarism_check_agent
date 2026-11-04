"""Task-scoped tool contracts, source validation and evidence-preserving output limits."""

import json
import math
from pathlib import Path
from jsonschema import Draft202012Validator, ValidationError
from typing import Any, Callable

import tiktoken

from ..tools.registry import ToolRegistry
from ..config import Config


class ToolRouter:
    MAX_TOOL_RESULT_TOKENS: int = Config.MAX_TOOL_RESULT_TOKENS

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

        self._dispatch['submit_findings'] = lambda findings: self.session.submit_findings(findings)

        # Initialize tiktoken tokenizer (cl100k_base = GPT-4 / GPT-3.5)
        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    # ── Public API ──────────────────────────────────────────────────────

    def bind_task(self, thesis_document_id: str, output_dir: str):
        from ..tools.review_session import ReviewSession
        self.session = ReviewSession(self.document_tools, thesis_document_id, output_dir)

    def prepare_reviews(self):
        if self.session is None:
            raise ValueError('Bind a review task first.')
        if self.session.prepared:
            return self.session.preparation_stats()
        for chunk_id in self.session.chunks:
            result = self._execute_validated('search_similar_chunks', {'thesis_chunk_id': chunk_id})
            if 'error' in result:
                raise ValueError(f'Retrieval failed for {chunk_id}: {result["error"]}')
        self.session.finish_preparation()
        audit = {
            'snapshot_id': self.document_tools.snapshot_id,
            'retrieval_config': self.document_tools.retrieval_config(),
            'coverage': self.session.preparation_stats(),
            'pairs': [{'thesis_chunk_id': key[0], 'source_chunk_id': key[1],
                       'scores': self.session.candidates[key].get('scores', {}), **gate}
                      for key, gate in self.session.gate_audit.items()],
        }
        self.session.output_dir.mkdir(parents=True, exist_ok=True)
        (self.session.output_dir / 'retrieval_audit.json').write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
        return self.session.preparation_stats()

    def execute(self, tool_name: str, tool_input: dict) -> Any:
        result = self._execute_validated(tool_name, tool_input)
        observation = self._truncate_result(result)
        if self.session and tool_name == 'get_chunk_context' and 'error' not in observation:
            try:
                if observation.get('chunk_id') != tool_input['chunk_id']:
                    raise ValueError('Context returned a different chunk than requested.')
                self.session.record_context(observation)
            except (KeyError, ValueError, TypeError) as exc:
                result = observation = {'error': str(exc), 'error_code': 'validation_error', 'retryable': False}
        self.last_result = result
        return observation

    def _execute_validated(self, tool_name: str, tool_input: dict) -> dict:
        if tool_name not in self._dispatch:
            return {"error": f"Unknown tool: {tool_name}", "error_code": "unknown_tool", "retryable": False}
        try:
            definition = next((d for d in self.tool_definitions if d["name"] == tool_name), None)
            if definition is None:
                raise ValueError('This tool is unavailable in the current review phase.')
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
                allowed = {"search_similar_chunks", "get_chunk_context", "get_corpus_stats", "generate_report", "submit_findings"}
                if tool_name not in allowed:
                    raise ValueError("This tool is unavailable during the review phase.")
                if tool_name == "search_similar_chunks":
                    arguments = self.session.search_arguments(arguments)
                elif tool_name == "get_chunk_context":
                    if arguments["chunk_id"] not in {k[1] for k in self.session.context_candidates()}:
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
        if self.session and self.session.batch_mode and self.session.batch_result:
            return 'reviewed', None
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
            if self.session.batch_mode:
                from ..tools.review_session import SUBMIT_FINDINGS_TOOL
                return [d for d in definitions if d['name'] in {'get_chunk_context', 'get_corpus_stats'}] + [SUBMIT_FINDINGS_TOOL]
            definitions = [d for d in definitions if d["name"] in {
                "search_similar_chunks", "get_chunk_context", "get_corpus_stats", "generate_report"
            }]
            if self.session.prepared:
                definitions = [d for d in definitions if d['name'] != 'search_similar_chunks']
        return definitions

    def _count_tokens(self, text: str) -> int:
        if self._enc is not None:
            return len(self._enc.encode(text, disallowed_special=()))
        # UTF-8 byte length is a conservative fallback, including Chinese.
        return len(text.encode('utf-8'))

    def _truncate_result(self, result: Any) -> Any:
        """Keep source text and IDs intact; drop optional neighbors or return an error."""
        fits = lambda value: self._count_tokens(json.dumps(value, ensure_ascii=False)) <= self.MAX_TOOL_RESULT_TOKENS
        if fits(result):
            return result
        if isinstance(result, dict) and 'context_chunks' in result:
            matches = [c for c in result['context_chunks'] if c.get('is_match')]
            compact = {**result, 'context_chunks': matches, 'neighbors_omitted': True}
            if fits(compact):
                return compact
        error = {'error': 'Result exceeds budget; request context_size=0.', 'retryable': True,
                 'error_code': 'result_too_large'}
        if fits(error):
            return error
        error = {'error': 'result_too_large'}
        return error if fits(error) else {}
