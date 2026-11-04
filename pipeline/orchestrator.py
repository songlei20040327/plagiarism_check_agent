"""
Pipeline orchestrator: coordinates the 4-phase plagiarism check pipeline.

Phase 1: Parse    — parse thesis PDF + all reference PDFs
Phase 2: Index    — chunk documents + build vector index
Phase 3: Analyze  — script retrieval/routing, with LLM review when needed
Phase 4: Report   — generate annotated HTML report
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..agent.runner import AgentRunner
from ..agent.llm_client import LLMClient
from ..agent.tool_router import ToolRouter
from ..agent.skill_loader import SkillLoader
from ..tools.document_tools import DocumentTools
from ..trajectory.schemas import AgentResult
from ..config import Config


@dataclass
class CheckResult:
    thesis_path: str
    session_id: str = ""
    total_findings: int = 0
    verbatim_count: int = 0
    near_paraphrase_count: int = 0
    idea_overlap_count: int = 0
    report_path: str = ""
    stats: dict = field(default_factory=dict)
    error: Optional[str] = None
    status: str = "failed"


class PlagiarismCheckOrchestrator:
    """
    Drives the full plagiarism check pipeline using AgentRunner.
    The LLM (via AgentRunner) handles analysis guided by the skill system prompt.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        base_url: str = "",
        trajectories_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        dry_run: bool = False,
    ):
        self.trajectories_dir = trajectories_dir or Config.TRAJECTORIES_DIR
        self.trajectories_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir or Config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        llm_client = LLMClient(
            api_key=api_key or None,
            model=model or None,
            base_url=base_url or None,
            max_tokens=Config.MAX_TOKENS,
        )
        self.doc_tools = DocumentTools()
        self.dry_run = dry_run
        tool_router = ToolRouter(dry_run=dry_run, document_tools=self.doc_tools)
        skill_loader = SkillLoader()

        self.runner = AgentRunner(
            llm_client=llm_client,
            tool_router=tool_router,
            skill_loader=skill_loader,
            trajectories_dir=self.trajectories_dir,
            skill_name="plagiarism-check",
        )

    def check(self, thesis_path: str, reference_dir: str, thesis_title: str = "") -> CheckResult:
        try:
            return self._check(thesis_path, reference_dir, thesis_title)
        except Exception as exc:
            return CheckResult(thesis_path=thesis_path, status='failed', error=f'查重执行失败: {exc}')
        finally:
            self.doc_tools.close()

    def _check(
        self,
        thesis_path: str,
        reference_dir: str,
        thesis_title: str = "",
    ) -> CheckResult:
        """
        Run a complete plagiarism check.

        Args:
            thesis_path: Path to the thesis PDF
            reference_dir: Directory containing reference PDFs
            thesis_title: Optional title override
        """
        self.doc_tools.close()
        self.doc_tools = DocumentTools()
        self.runner.tool_router = ToolRouter(
            dry_run=self.dry_run, document_tools=self.doc_tools
        )
        thesis_file = Path(thesis_path)
        ref_dir = Path(reference_dir)

        if not thesis_file.exists():
            return CheckResult(
                thesis_path=thesis_path,
                error=f"论文文件不存在: {thesis_path}",
            )
        if not ref_dir.is_dir():
            return CheckResult(
                thesis_path=thesis_path,
                error=f"对照文献目录不存在: {reference_dir}",
            )

        # Phase 1: Parse all documents
        print("\n" + "=" * 60)
        print("Phase 1/4: 解析文档")
        print("=" * 60)

        # 调用 用于解析文档：输出结果包含 doc_id、file_name、page_count、word_count 等元信息
        thesis_meta = self.doc_tools.parse_pdf(str(thesis_file.absolute()))
        if "error" in thesis_meta:
            return CheckResult(thesis_path=thesis_path, error=thesis_meta["error"])
        print(f"  论文: {thesis_meta['file_name']} ({thesis_meta['page_count']}页, "
              f"{thesis_meta['word_count']}词)")

        if not thesis_title:
            thesis_title = thesis_meta.get("title", thesis_file.stem)

        ref_files = sorted(p for p in ref_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
        if not ref_files:
            return CheckResult(
                thesis_path=thesis_path,
                error=f"对照文献目录中没有 PDF 文件: {reference_dir}",
            )

        ref_metas = []
        for rf in ref_files:
            meta = self.doc_tools.parse_pdf(str(rf.absolute()))
            if "error" in meta:
                return CheckResult(thesis_path=thesis_path, error=f"对照文献解析失败 {rf.name}: {meta['error']}")
            if meta["doc_id"] == thesis_meta["doc_id"]:
                print(f"  ⚠ 跳过待检论文的副本: {rf.name}")
                continue
            if any(m['doc_id'] == meta['doc_id'] for m in ref_metas):
                continue
            ref_metas.append(meta)
            print(f"  对照文献: {meta['file_name']} ({meta['page_count']}页)")

        if not ref_metas:
            return CheckResult(thesis_path=thesis_path, error="没有可用的独立对照文献")
        print(f"\n  共 {len(ref_metas)} 篇对照文献")

        # Phase 2: Chunk + Index
        print("\n" + "=" * 60)
        print("Phase 2/4: 文本分块 + 构建向量索引")
        print("=" * 60)

        all_doc_ids = [thesis_meta["doc_id"]] + [m["doc_id"] for m in ref_metas]

        # Chunk reference documents (skip thesis for search corpus)
        for doc_id in all_doc_ids[1:]:
            result = self.doc_tools.chunk_document(doc_id)
            if 'error' in result or not result.get('total_chunks'):
                return CheckResult(thesis_path=thesis_path, error=f"对照文献没有可检索文本: {doc_id}")
            print(f"  分块: {result.get('file_name', doc_id)} → "
                  f"{result.get('total_chunks', 0)} chunks")

        # Chunk the thesis for deterministic retrieval and selective LLM review.
        thesis_chunk_result = self.doc_tools.chunk_document(thesis_meta["doc_id"])
        if "error" in thesis_chunk_result or not thesis_chunk_result.get("total_chunks"):
            return CheckResult(thesis_path=thesis_path, error="待检论文未提取到可分析的文本")
        print(f"  论文章节: {thesis_chunk_result.get('total_chunks', 0)} 个段落")

        # Build index from reference documents only
        index_result = self.doc_tools.build_corpus_index(
            reference_document_ids=[m["doc_id"] for m in ref_metas],
            excluded_document_ids=[thesis_meta["doc_id"]],
            persist_dir=str(Config.VECTOR_DB_DIR),
        )
        if "error" in index_result:
            return CheckResult(thesis_path=thesis_path, error=index_result["error"])
        print(f"  索引: {index_result['total_chunks_indexed']} chunks, "
              f"{index_result['unique_documents']} documents")

        # Phase 3: deterministic retrieval and routing precede the LLM.
        print("\n" + "=" * 60)
        print("Phase 3/4 + 4/4: 脚本检索筛选 + 按需精判 + 生成报告")
        print("=" * 60)

        from uuid import uuid4
        run_output_dir = self.output_dir / uuid4().hex[:12]
        self.runner.tool_router.bind_task(thesis_meta["doc_id"], str(run_output_dir))
        try:
            preparation = self.runner.tool_router.prepare_reviews()
        except Exception as exc:
            self.doc_tools.close()
            return CheckResult(thesis_path=thesis_path, status='failed', error=f'检索准备失败: {exc}')
        print(f"  已检索 {preparation['searched']} 段；脚本处理 {preparation['automatic_pairs']} 对；"
              f"待LLM精判 {preparation['llm_pairs']} 对")
        task = (
            f"脚本已完成全部段落检索和自动分流。按当前批次的待精判清单审核。\n\n"
            f"论文标题：{thesis_title}\n"
            f"对照文献：{len(ref_metas)} 篇\n\n"
            f"1. 对清单中的来源读取 get_chunk_context，同一来源可复用已读上下文。\n"
            f"2. 根据完整证据和引用，判断 verbatim / near_paraphrase / idea_overlap / no_issue。\n"
            f"3. 调用 submit_findings，提交当前批次每个候选的分类和建议。"
            f"不能覆盖脚本结论，也不能提交其他批次的候选。\n"
            f"每项包含 thesis_chunk_id、source_chunk_id、duplicate_type、suggestion，"
            f"可包含 analysis。必须读取完整来源证据，检索分数不是重复概率。"
        )

        extra_context = {
            "thesis_doc_id": thesis_meta["doc_id"],
            "reference_doc_ids": [m["doc_id"] for m in ref_metas],
            "reference_count": len(ref_metas),
            "thesis_chunks": len(self.runner.tool_router.session.chunks),
            "selected_for_review": preparation['llm_chunks'],
            'prepared_review': True, 'preparation': preparation, 'thesis_title': thesis_title,
            "output_dir": str(run_output_dir.absolute()),
            "retrieval_config": self.doc_tools.retrieval_config(),
        }

        try:
            result: AgentResult = self.runner.run(
                task=task,
                thesis=thesis_path,
                extra_context=extra_context,
            )
        finally:
            self.doc_tools.close()

        report = result.report_result
        report_path = report.get("html_report", "")
        verbatim = report.get("verbatim", 0)
        near = report.get("near_paraphrase", 0)
        idea = report.get("idea_overlap", 0)
        total = report.get("total_findings", 0)
        if result.status != "completed":
            return CheckResult(
                thesis_path=thesis_path, session_id=result.session_id,
                status=result.status, error=result.error,
                report_path=report_path, stats=result.stats, total_findings=total,
            )
        return CheckResult(
            thesis_path=thesis_path,
            session_id=result.session_id,
            status=result.status,
            total_findings=total,
            verbatim_count=verbatim,
            near_paraphrase_count=near,
            idea_overlap_count=idea,
            report_path=report_path,
            stats=result.stats,
        )

    def batch_check(
        self,
        thesis_list: list[dict],
        max_workers: int = 1,
    ) -> list[CheckResult]:
        """Run checks for multiple theses (sequential by default)."""
        if max_workers <= 1:
            results = []
            for item in thesis_list:
                r = self.check(
                    thesis_path=item["thesis"],
                    reference_dir=item["references"],
                    thesis_title=item.get("title", ""),
                )
                results.append(r)
            return results

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for item in thesis_list:
                f = executor.submit(
                    self._isolated_check,
                    thesis_path=item["thesis"],
                    reference_dir=item["references"],
                    thesis_title=item.get("title", ""),
                )
                futures.append(f)
            return [f.result() for f in futures]

    def _isolated_check(self, **kwargs) -> CheckResult:
        llm = self.runner.llm_client
        worker = PlagiarismCheckOrchestrator.__new__(PlagiarismCheckOrchestrator)
        worker.trajectories_dir = self.trajectories_dir
        worker.output_dir = self.output_dir
        worker.dry_run = self.dry_run
        worker.doc_tools = DocumentTools()
        worker.runner = AgentRunner(
            llm_client=llm,
            tool_router=ToolRouter(dry_run=self.dry_run, document_tools=worker.doc_tools),
            skill_loader=SkillLoader(), trajectories_dir=self.trajectories_dir,
        )
        return worker.check(**kwargs)
