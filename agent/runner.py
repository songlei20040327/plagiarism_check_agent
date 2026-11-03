"""
AgentRunner: LLM + Tool-calling main loop.

Drives the conversation loop: LLM reasons → chooses tools → tools execute →
results feed back → repeat until end_turn. Records complete trajectory.

Skills are auto-discovered (no hardcoded sub-skill list).
Tool list is injected from ToolRegistry (SKILL.md doesn't name tools).
"""

import json
import uuid
from pathlib import Path
from typing import Optional

from .llm_client import LLMClient
from .tool_router import ToolRouter
from .skill_loader import SkillLoader
from .loop_guard import LoopGuard
from ..trajectory.logger import TrajectoryLogger
from ..trajectory.schemas import AgentResult, ToolUseBlock
from ..config import Config


class AgentRunner:
    MAX_ITERATIONS = Config.MAX_ITERATIONS

    def __init__(
        self,
        llm_client: LLMClient,
        tool_router: ToolRouter,
        skill_loader: SkillLoader,
        trajectories_dir: Path,
        skill_name: str = "plagiarism-check",
    ):
        self.llm_client = llm_client
        self.tool_router = tool_router
        self.skill_loader = skill_loader
        self.trajectories_dir = trajectories_dir
        self.skill_name = skill_name

        self.guard = LoopGuard(max_iterations=self.MAX_ITERATIONS)

        # Auto-discover sub-skills + inject tool list — no hardcoded names
        try:
            base_prompt = self.skill_loader.load_with_sub_skills(skill_name)
            base_prompt = SkillLoader.inject_memory_context(base_prompt)
            # Native tool definitions are task-scoped at request time.
            self.system_prompt = base_prompt
        except Exception:
            self.system_prompt = (
                "你是一个专业的学术论文查重助手。"
                "对比用户上传的论文与对照文献库，定位相似段落，"
                "溯源到具体论文，并给出改写建议。"
            )

    def run(
        self, task: str, thesis: str = "", extra_context: Optional[dict] = None,
    ) -> AgentResult:
        session_id = uuid.uuid4().hex[:12]
        self.guard.reset()
        system = self.system_prompt
        if extra_context:
            system += "\n\n## 任务上下文\n" + json.dumps(extra_context, ensure_ascii=False)
        messages = [{"role": "user", "content": task}]
        status, error, final_text = "failed", None, ""
        search_calls = context_calls = iteration = 0
        force_stop = False
        ref_count = (extra_context or {}).get("reference_count", 0)
        with TrajectoryLogger(
            self.trajectories_dir, session_id, thesis_path=thesis, reference_count=ref_count,
        ) as logger:
            logger.log_system_input(system, self.tool_router.tool_definitions)
            logger.log_user_input(task)
            while iteration < self.MAX_ITERATIONS and not force_stop:
                iteration += 1
                nudge = self.guard.nearing_limit(iteration)
                if nudge:
                    messages.append({"role": "user", "content": nudge})
                    logger.log_user_input(nudge)
                try:
                    response = self.llm_client.complete(
                        messages=messages, system=system, tools=self.tool_router.tool_definitions,
                    )
                except Exception as exc:
                    error = f"LLM request failed: {exc}"
                    logger.log_error(error)
                    break
                logger.log_assistant(response)
                messages.append({"role": "assistant", "content": response.content})
                if response.tool_uses:
                    results, warning, cancel_rest = [], None, False
                    for block in response.tool_uses:
                        if cancel_rest:
                            raw_result = {"error": "Cancelled after loop guard intervention.",
                                          "error_code": "cancelled", "retryable": True}
                            result = raw_result
                        else:
                            search_calls += int(block.name == "search_similar_chunks")
                            context_calls += int(block.name == "get_chunk_context")
                            result = self.tool_router.execute(block.name, block.input)
                            raw_result = self.tool_router.last_result
                            guard_message = self.guard.observe(block.name, block.input, raw_result, iteration)
                            if guard_message:
                                cancel_rest = True
                                warning = guard_message
                                if guard_message == "STOP":
                                    force_stop = True
                                    status, error = "failed", "Loop guard terminated the task."
                        logger.log_tool_result(block, raw_result, observation=result)
                        results.append({"type": "tool_result", "tool_use_id": block.id,
                                        "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]})
                    # Always complete the entire tool-call batch before another model request.
                    messages.append({"role": "user", "content": results})
                    if warning and warning != "STOP":
                        messages.append({"role": "user", "content": warning})
                        logger.log_user_input(warning)
                    if force_stop:
                        final_text = "[Agent 因循环保护而终止]"
                        break
                    continue
                if response.stop_reason == "end_turn":
                    final_text = response.text
                    status, error = self.tool_router.completion_state()
                    if status == "partial" and search_calls == 0:
                        status = "failed"
                    break
                continuation = "响应被截断或没有有效工具调用，请继续完成任务。"
                messages.append({"role": "user", "content": continuation})
                logger.log_user_input(continuation)
            else:
                status, error = "budget_exceeded", "Maximum agent iterations reached."
                final_text = f"[Agent 在 {self.MAX_ITERATIONS} 轮后停止]"
            session = self.tool_router.session
            report = session.report_result if session else None
            findings = session.report_findings if session else []
            dist = {name: sum(f.get("duplicate_type") == name for f in findings)
                    for name in ("verbatim", "near_paraphrase", "idea_overlap", "no_issue")}
            suggestions = [f.get("suggestion", "") for f in findings if f.get("suggestion")]
            stats = logger.close(extra_stats={
                "status": status, "error": error, "iterations": iteration,
                "findings_count": len(findings), "classification_distribution": dist,
                "suggestions_generated": len(suggestions),
                "avg_suggestion_length": int(sum(map(len, suggestions)) / len(suggestions)) if suggestions else 0,
                "has_detailed_analysis": any(f.get("analysis") for f in findings),
                "search_calls": search_calls, "context_calls": context_calls,
                "total_paragraphs": len(session.chunks) if session else 0,
                "searched_paragraphs": len(session.searched) if session else 0,
                "snapshot_id": self.tool_router.document_tools.snapshot_id,
            })
        return AgentResult(messages=messages, final_response=final_text, session_id=session_id,
                           stats=stats, status=status, error=error,
                           report_result=report or {}, findings=findings)
