"""
AgentRunner: LLM + Tool-calling main loop.

Drives the conversation loop: LLM reasons → chooses tools → tools execute →
results feed back → repeat until end_turn. Records complete trajectory.

Skills are auto-discovered (no hardcoded sub-skill list).
Tool list is injected from ToolRegistry (SKILL.md doesn't name tools).
"""

import json
import re as _re
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
            self.system_prompt = SkillLoader.inject_tools_summary(base_prompt)
        except Exception:
            self.system_prompt = (
                "你是一个专业的学术论文查重助手。"
                "对比用户上传的论文与对照文献库，定位相似段落，"
                "溯源到具体论文，并给出改写建议。"
            )

    def run(
        self,
        task: str,
        thesis: str = "",
        extra_context: Optional[dict] = None,
    ) -> AgentResult:
        session_id = uuid.uuid4().hex[:12]

        system = self.system_prompt
        if extra_context:
            ctx_json = json.dumps(extra_context, ensure_ascii=False, indent=2)
            system = f"{system}\n\n---\n\n## 额外上下文\n```json\n{ctx_json}\n```"

        messages: list[dict] = [{"role": "user", "content": task}]

        ref_count = extra_context.get("reference_count", 0) if extra_context else 0

        with TrajectoryLogger(
            output_dir=self.trajectories_dir,
            session_id=session_id,
            thesis_path=thesis,
            reference_count=ref_count,
        ) as logger:
            logger.log_user_input(task)

            iteration = 0
            final_text = ""
            search_calls = 0
            context_calls = 0
            force_stop = False

            while iteration < self.MAX_ITERATIONS and not force_stop:
                iteration += 1

                # Loop guard: warn LLM when approaching iteration limit
                limit_nudge = self.guard.nearing_limit(iteration)
                if limit_nudge:
                    messages.append({"role": "user", "content": limit_nudge})

                response = self.llm_client.complete(
                    messages=messages,
                    system=system,
                    tools=self.tool_router.tool_definitions,
                )
                logger.log_assistant(response)

                if response.stop_reason == "end_turn":
                    final_text = response.text
                    messages.append({
                        "role": "assistant",
                        "content": response.content,
                    })
                    break

                # Track tool call counts by name
                tool_results_content = []
                for block in response.tool_uses:
                    if block.name == "search_similar_chunks":
                        search_calls += 1
                    elif block.name == "get_chunk_context":
                        context_calls += 1

                    result = self.tool_router.execute(block.name, block.input)
                    logger.log_tool_result(block, result)

                    # Loop guard: detect repeated calls, consecutive errors
                    guard_msg = self.guard.observe(
                        block.name, block.input, result, iteration
                    )
                    if guard_msg == "STOP":
                        final_text = "[Agent 因检测到异常循环模式而提前终止]"
                        force_stop = True
                        break
                    elif guard_msg:
                        tool_results_content.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": [{"type": "text", "text": guard_msg}],
                        })
                        break

                    result_text = (
                        json.dumps(result, ensure_ascii=False)
                        if isinstance(result, (dict, list))
                        else str(result)
                    )
                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": [{"type": "text", "text": result_text}],
                    })

                messages.append({
                    "role": "assistant",
                    "content": response.content,
                })
                messages.append({
                    "role": "user",
                    "content": tool_results_content,
                })
            else:
                final_text = (
                    f"[Agent 在 {self.MAX_ITERATIONS} 轮后停止]"
                )

            # ── Extract rich stats from trajectory ────────────────────
            # Search both text blocks AND tool_use inputs (most findings are in tool calls)
            text_parts = []
            for m in messages:
                content = m.get("content")
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict):
                            if b.get("type") == "text":
                                text_parts.append(b.get("text", ""))
                            elif b.get("type") == "tool_use":
                                inp = b.get("input", {})
                                findings = inp.get("findings", [])
                                if isinstance(findings, list):
                                    for f in findings:
                                        if isinstance(f, dict):
                                            text_parts.append(json.dumps(f, ensure_ascii=False))
                                text_parts.append(json.dumps(inp, ensure_ascii=False))
            all_text = " ".join(text_parts)

            # Classification distribution
            verbatim_n = len(_re.findall(r'"duplicate_type"\s*:\s*"verbatim"', all_text))
            near_n = len(_re.findall(r'"duplicate_type"\s*:\s*"near_paraphrase"', all_text))
            idea_n = len(_re.findall(r'"duplicate_type"\s*:\s*"idea_overlap"', all_text))
            noissue_n = len(_re.findall(r'"duplicate_type"\s*:\s*"no_issue"', all_text))
            findings_count = verbatim_n + near_n + idea_n + noissue_n

            # Average suggestion length
            suggestions = _re.findall(r'"suggestion"\s*:\s*"([^"]*)"', all_text)
            if suggestions:
                avg_suggestion_length = int(
                    sum(len(s) for s in suggestions) / len(suggestions)
                )
            else:
                avg_suggestion_length = 0

            # Check for thinking blocks (analytical rigor indicator)
            has_thinking = any(
                isinstance(b, dict) and b.get("type") == "thinking"
                for m in messages if m.get("role") == "assistant"
                for b in (m.get("content") if isinstance(m.get("content"), list) else [])
            )

            total_paragraphs = extra_context.get("thesis_chunks", 0) if extra_context else 0

            extra_stats = {
                "iterations": iteration,
                "findings_count": findings_count,
                "suggestions_generated": len(suggestions),
                "classification_distribution": {
                    "verbatim": verbatim_n,
                    "near_paraphrase": near_n,
                    "idea_overlap": idea_n,
                    "no_issue": noissue_n,
                },
                "avg_suggestion_length": avg_suggestion_length,
                "has_detailed_analysis": has_thinking,
                "search_calls": search_calls,
                "context_calls": context_calls,
                "total_paragraphs": total_paragraphs,
            }
            stats = logger.close(extra_stats=extra_stats)

        return AgentResult(
            messages=messages,
            final_response=final_text,
            session_id=session_id,
            stats=stats,
        )
