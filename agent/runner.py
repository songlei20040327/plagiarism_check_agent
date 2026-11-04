"""Bounded LLM review conversations followed by deterministic report generation."""

import json
import uuid
from pathlib import Path
from typing import Optional

from .llm_client import LLMClient
from .tool_router import ToolRouter
from .skill_loader import SkillLoader
from .loop_guard import LoopGuard
from ..trajectory.logger import TrajectoryLogger
from ..trajectory.schemas import AgentResult
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

        self.system_prompt = self.skill_loader.load_with_sub_skills(skill_name)

    def run(self, task: str, thesis: str = "", extra_context: Optional[dict] = None) -> AgentResult:
        session = self.tool_router.session
        if not session or not session.prepared:
            return self._run_once(task, thesis, extra_context)
        prior, batch_ids = {}, []
        while session.unreviewed_candidates():
            try:
                payload = session.start_batch()
            except ValueError as exc:
                return AgentResult(messages=[], final_response='', session_id='', status='failed',
                                   error=str(exc), stats={**prior, 'batch_session_ids': batch_ids})
            context = {**(extra_context or {}), 'review_batch': len(batch_ids) + 1,
                       'batch_session_ids': batch_ids.copy()}
            result = self._run_once(task + '\n\n## 待精判候选\n' + json.dumps(payload, ensure_ascii=False),
                                    thesis, context)
            batch_ids.append(result.session_id)
            for key in ('llm_calls', 'tool_calls', 'duration_ms', 'iterations', 'context_calls'):
                prior[key] = prior.get(key, 0) + result.stats.get(key, 0)
            prior['usage'] = {key: prior.get('usage', {}).get(key, 0) + result.stats.get('usage', {}).get(key, 0)
                              for key in ('input_tokens', 'output_tokens')}
            if result.status != 'reviewed':
                result.stats.update(prior)
                result.stats['batch_session_ids'] = batch_ids
                return result
        session.batch_mode, session.batch_result = False, None
        context = {**(extra_context or {}), 'batch_session_ids': batch_ids}
        return self._run_once(task, thesis, context, prior_stats=prior)

    def _run_once(
        self, task: str, thesis: str = "", extra_context: Optional[dict] = None,
        prior_stats: Optional[dict] = None,
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
            session = self.tool_router.session
            script_only = bool(session and session.prepared and not session.batch_mode and not session.unreviewed_candidates())
            if script_only:
                self.tool_router.execute('generate_report', {
                    'thesis_title': (extra_context or {}).get('thesis_title', '查重报告'),
                    'output_dir': str(session.output_dir), 'findings': [],
                })
                logger.log_script_step('generate_report', self.tool_router.last_result)
                status, error = self.tool_router.completion_state()
                if status == 'completed':
                    final_text = '查重报告已由脚本生成并通过校验。'
                elif status == 'dry_run':
                    final_text = '试运行完成，报告写入已跳过。'
                else:
                    status = 'failed'
                    error = self.tool_router.last_result.get('error', error)
            while iteration < self.MAX_ITERATIONS and not force_stop and not script_only:
                iteration += 1
                nudge = self.guard.nearing_limit(iteration)
                if nudge:
                    messages.append({"role": "user", "content": nudge})
                    logger.log_user_input(nudge)
                payload = {'system': system, 'messages': messages, 'tools': self.tool_router.tool_definitions}
                if self.tool_router._count_tokens(json.dumps(payload, ensure_ascii=False)) > Config.MAX_INPUT_TOKENS:
                    error = 'Input budget exceeded; reduce REVIEW_BATCH_CHUNKS or context_size.'
                    logger.log_error(error)
                    break
                try:
                    contexts_seen = set(session.read_contexts) if session else set()
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
                            submitted = block.input.get('findings', []) if isinstance(block.input, dict) else []
                            if (session and block.name in {'submit_findings', 'generate_report'}
                                    and any(isinstance(f, dict) and f.get('source_chunk_id') not in contexts_seen for f in submitted)):
                                raw_result = result = {'error': 'Read source results in an earlier model turn before submitting judgments.',
                                                       'error_code': 'evidence_not_observed', 'retryable': False}
                            else:
                                result = self.tool_router.execute(block.name, block.input)
                                raw_result = self.tool_router.last_result
                            guard_message = self.guard.observe(block.name, block.input, result, iteration)
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
                    if self.tool_router.completion_state()[0] == 'reviewed':
                        status, error, final_text = 'reviewed', None, '本批候选判断已提交并通过证据校验。'
                        break
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
                if not script_only:
                    status, error = "budget_exceeded", "Maximum agent iterations reached."
                    final_text = f"[Agent 在 {self.MAX_ITERATIONS} 轮后停止]"
            # A verified artifact completes the task even if the optional final
            # model reply fails or there is no iteration left for that reply.
            artifact_status, _ = self.tool_router.completion_state()
            if artifact_status in {"completed", "dry_run", "reviewed"}:
                if status != artifact_status:
                    final_text = ("查重报告已生成并通过校验。" if artifact_status == "completed"
                                  else "试运行完成，报告写入已跳过。")
                status, error = artifact_status, None
            session = self.tool_router.session
            report = session.report_result if session else None
            findings = (session.batch_findings if status == 'reviewed' else session.report_findings) if session else []
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
                'script_search_calls': len(session.searched) if session and session.prepared else 0,
                'preparation': session.preparation_stats() if session else {},
                'batch_session_ids': (extra_context or {}).get('batch_session_ids', []),
            }, prior_stats=prior_stats)
        return AgentResult(messages=messages, final_response=final_text, session_id=session_id,
                           stats=stats, status=status, error=error,
                           report_result=report or {}, findings=findings)
