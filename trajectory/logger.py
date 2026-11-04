"""Trajectory logger: records every agent step to JSONL for observability and training data."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schemas import LLMResponse, ToolUseBlock


class TrajectoryLogger:
    """Context-managed JSONL logger for agent execution traces."""

    def __init__(
        self,
        output_dir: Path,
        session_id: str,
        thesis_path: str = "",
        reference_count: int = 0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.file_path = self.output_dir / f"trace_{session_id}.jsonl"
        self._file = None
        self._start_time = None
        self._thesis_path = thesis_path
        self._reference_count = reference_count
        self._llm_calls = 0
        self._tool_calls = 0
        self._usage = {"input_tokens": 0, "output_tokens": 0}

    def __enter__(self):
        self._file = open(self.file_path, "w", encoding="utf-8")
        self._start_time = datetime.now(timezone.utc)
        self._write({
            "type": "session_start",
            "schema_version": 2,
            "session_id": self.session_id,
            "thesis": self._thesis_path,
            "reference_count": self._reference_count,
            "started_at": self._start_time.isoformat(),
        })
        return self

    def __exit__(self, *args):
        if self._file:
            self._file.close()

    def _write(self, record: dict):
        if self._file:
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._file.flush()

    def log_system_input(self, system: str, tools: list[dict]):
        self._write({"type": "system", "message": {"role": "system", "content": system},
                     "tools": tools})

    def log_error(self, error: str):
        self._write({"type": "run_error", "error": error})

    def log_script_step(self, name: str, result):
        self._write({'type': 'script_step', 'name': name, 'result': result})

    def log_user_input(self, task: str):
        self._write({
            "type": "user",
            "message": {"role": "user", "content": task},
        })

    def log_assistant(self, response: LLMResponse):
        self._llm_calls += 1
        for key in self._usage:
            self._usage[key] += response.usage.get(key, 0)
        self._write({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": response.content,
                "stop_reason": response.stop_reason,
                "usage": response.usage,
                "model": response.model,
            },
        })

    def log_tool_result(self, tool_use: ToolUseBlock, result, observation=None):
        self._tool_calls += 1
        serializable = result
        try:
            json.dumps(serializable, ensure_ascii=False)
        except (TypeError, ValueError):
            serializable = str(result)
        self._write({
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "tool_name": tool_use.name,
            "input": tool_use.input,
            "result": serializable,
            "observation": serializable if observation is None else observation,
        })

    def close(self, extra_stats: Optional[dict] = None) -> dict:
        duration_ms = int(
            (datetime.now(timezone.utc) - self._start_time).total_seconds() * 1000
        ) if self._start_time else 0
        stats = {
            "llm_calls": self._llm_calls,
            "tool_calls": self._tool_calls,
            "duration_ms": duration_ms,
            "usage": self._usage.copy(),
        }
        if extra_stats:
            stats.update(extra_stats)
        self._write({
            "type": "session_end",
            "session_id": self.session_id,
            "stats": stats,
        })
        return stats
