"""
Pipeline orchestrator: coordinates the 4-phase plagiarism check pipeline.

Phase 1: Parse    — parse thesis PDF + all reference PDFs
Phase 2: Index    — chunk documents + build vector index
Phase 3: Analyze  — for each thesis paragraph, search + LLM classify
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
from ..memory.store import record_chunk_matches
from ..memory.profile import analyze_check_result
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
    author: str = ""
    research_field: str = ""
    insights: list = field(default_factory=list)
    reading_suggestions: list = field(default_factory=list)


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
        tool_router = ToolRouter(dry_run=dry_run)
        skill_loader = SkillLoader()

        self.runner = AgentRunner(
            llm_client=llm_client,
            tool_router=tool_router,
            skill_loader=skill_loader,
            trajectories_dir=self.trajectories_dir,
            skill_name="plagiarism-check",
        )
        self.doc_tools = DocumentTools()

    def check(
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
        thesis_file = Path(thesis_path)
        ref_dir = Path(reference_dir)

        if not thesis_file.exists():
            return CheckResult(
                thesis_path=thesis_path,
                error=f"论文文件不存在: {thesis_path}",
            )
        if not ref_dir.exists():
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

        ref_files = sorted(ref_dir.glob("*.pdf"))
        if not ref_files:
            return CheckResult(
                thesis_path=thesis_path,
                error=f"对照文献目录中没有 PDF 文件: {reference_dir}",
            )

        ref_metas = []
        for rf in ref_files:
            meta = self.doc_tools.parse_pdf(str(rf.absolute()))
            if "error" in meta:
                print(f"  ⚠ 跳过 {rf.name}: {meta['error']}")
                continue
            ref_metas.append(meta)
            print(f"  对照文献: {meta['file_name']} ({meta['page_count']}页)")

        print(f"\n  共 {len(ref_metas)} 篇对照文献")

        # Phase 2: Chunk + Index
        print("\n" + "=" * 60)
        print("Phase 2/4: 文本分块 + 构建向量索引")
        print("=" * 60)

        all_doc_ids = [thesis_meta["doc_id"]] + [m["doc_id"] for m in ref_metas]

        # Chunk reference documents (skip thesis for search corpus)
        for doc_id in all_doc_ids[1:]:
            result = self.doc_tools.chunk_document(doc_id)
            print(f"  分块: {result.get('file_name', doc_id)} → "
                  f"{result.get('total_chunks', 0)} chunks")

        # Also chunk the thesis so we can send the text directly to the LLM
        thesis_chunk_result = self.doc_tools.chunk_document(thesis_meta["doc_id"])
        print(f"  论文章节: {thesis_chunk_result.get('total_chunks', 0)} 个段落")

        # Build index from reference documents only
        index_result = self.doc_tools.build_corpus_index(
            persist_dir=str(Config.VECTOR_DB_DIR)
        )
        if "error" in index_result:
            return CheckResult(thesis_path=thesis_path, error=index_result["error"])
        print(f"  索引: {index_result['total_chunks_indexed']} chunks, "
              f"{index_result['unique_documents']} documents")

        # Phase 3 & 4: Let the Agent handle analysis and report generation
        print("\n" + "=" * 60)
        print("Phase 3/4 + 4/4: LLM Agent 分析 + 生成报告")
        print("=" * 60)

        # Extract all thesis chunk texts for full-text analysis
        thesis_chunks = self.doc_tools.get_document_chunks(thesis_meta["doc_id"])

        chunks_text = "\n\n---\n\n".join(
            f"[段落 {c['index'] + 1}]\n{c['text'][:800]}"
            for c in thesis_chunks
        )

        task = (
            f"所有文档已经解析并构建好向量索引，你不需要再调用 parse_pdf 或 chunk_document。"
            f"直接开始查重分析。\n\n"
            f"论文标题：{thesis_title}\n"
            f"对照文献：{len(ref_metas)} 篇\n\n"
            f"以下是论文的全部段落（共 {len(thesis_chunks)} 段）：\n\n"
            f"{chunks_text}\n\n"
            f"请执行以下步骤（每步尽可能批量处理，减少调用次数）：\n"
            f"1. 对上述全部 {len(thesis_chunks)} 个段落，分批调用 search_similar_chunks "
            f"（每批 3-5 段，每次调用传一个段落文字作为 query_text）\n"
            f"2. 只对 similarity >= {Config.SIMILARITY_THRESHOLD} 的匹配，调用 get_chunk_context\n"
            f"3. 将高相似度匹配归类为 verbatim(>={Config.SIMILARITY_THRESHOLD + 0.15:.2f}) / "
            f"near_paraphrase({Config.SIMILARITY_THRESHOLD + 0.07:.2f}-{Config.SIMILARITY_THRESHOLD + 0.15:.2f}) / "
            f"idea_overlap({Config.SIMILARITY_THRESHOLD:.2f}-{Config.SIMILARITY_THRESHOLD + 0.07:.2f})\n"
            f"4. 针对所有相似度 >= {Config.SIMILARITY_THRESHOLD} 的发现，逐一精判并生成改写建议\n"
            f"5. 调用 generate_report，thesis_title=\"{thesis_title}\"，"
            f"output_dir=\"{self.output_dir.absolute()}\"，"
            f"findings 数组中每个元素包含 thesis_chunk、source_paper、similarity、"
            f"duplicate_type、overlapping_text、suggestion 字段\n\n"
            f"重要：逐段搜索全部 {len(thesis_chunks)} 个段落，不要跳过。"
            f"跳过相似度 < {Config.SIMILARITY_THRESHOLD} 的匹配结果。"
            f"尽量批量调用以减少轮数。"
        )

        extra_context = {
            "thesis_doc_id": thesis_meta["doc_id"],
            "reference_doc_ids": [m["doc_id"] for m in ref_metas],
            "reference_count": len(ref_metas),
            "thesis_chunks": len(thesis_chunks),
            "selected_for_review": len(thesis_chunks),
            "output_dir": str(self.output_dir.absolute()),
            "threshold": Config.SIMILARITY_THRESHOLD,
        }

        result: AgentResult = self.runner.run(
            task=task,
            thesis=thesis_path,
            extra_context=extra_context,
        )

        # Try to extract report path and counts from the final response
        all_text = " ".join(
            b.get("text", "")
            for m in result.messages if m.get("role") == "assistant"
            for b in (m.get("content") if isinstance(m.get("content"), list) else [])
            if isinstance(b, dict) and b.get("type") == "text"
        )

        report_path = ""
        verbatim = near = idea = total = 0

        # Try to parse the last generate_report result from tool messages
        for m in result.messages:
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for block in m["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        inner = block.get("content", [])
                        text = ""
                        if isinstance(inner, list):
                            text = "\n".join(
                                c.get("text", "") for c in inner
                                if isinstance(c, dict)
                            )
                        else:
                            text = str(inner)
                        try:
                            data = json.loads(text)
                            if isinstance(data, dict) and "html_report" in data:
                                report_path = data.get("html_report", "")
                                verbatim = data.get("verbatim", 0)
                                near = data.get("near_paraphrase", 0)
                                idea = data.get("idea_overlap", 0)
                                total = data.get("total_findings", 0)
                        except (json.JSONDecodeError, ValueError):
                            pass

        # ── Record matched chunks for hotspot tracking ─────────────
        # Extract matched texts from search_similar_chunks tool results
        matched_chunks = []
        all_findings = []  # full findings from generate_report
        for m in result.messages:
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for block in m["content"]:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    inner = block.get("content", [])
                    text = ""
                    if isinstance(inner, list):
                        text = "\n".join(
                            c.get("text", "") for c in inner if isinstance(c, dict)
                        )
                    else:
                        text = str(inner)
                    try:
                        data = json.loads(text)
                        # search_similar_chunks result
                        if isinstance(data, dict) and "matches" in data:
                            for match in data["matches"]:
                                if match.get("similarity", 0) >= 0.75:
                                    matched_chunks.append({
                                        "text": match.get("matched_text", ""),
                                        "source_paper": match.get("source_paper", "unknown"),
                                        "similarity": match.get("similarity", 0),
                                    })
                        # generate_report result: extract findings array
                        if isinstance(data, dict) and "total_findings" in data:
                            # The report result has counts but not individual findings.
                            # Find the generate_report *call* to get the findings array.
                            pass
                    except (json.JSONDecodeError, ValueError):
                        pass

        # Extract findings from the generate_report tool *call* (assistant message)
        for m in result.messages:
            if m.get("role") == "assistant" and isinstance(m.get("content"), list):
                for block in m["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        if block.get("name") == "generate_report":
                            inp = block.get("input", {})
                            findings = inp.get("findings", [])
                            if isinstance(findings, list):
                                all_findings = findings

        if matched_chunks:
            hotspot_result = record_chunk_matches(
                thesis_title=thesis_title,
                matches=matched_chunks,
            )
            if hotspot_result.get("new_hotspots", 0) > 0:
                print(f"  [Memory] {hotspot_result['new_hotspots']} 个新的高频匹配模式被检测到")

        # ── Author profile analysis ────────────────────────────────
        author = thesis_meta.get("author", "")
        if author == "unknown":
            author = thesis_file.stem  # fallback to filename

        # Extract thesis abstract (first 500 chars) for field detection
        thesis_text = ""
        for chunk in thesis_chunks[:3]:  # first 3 paragraphs usually contain intro
            thesis_text += chunk.get("text", "")[:300]

        profile_result = analyze_check_result(
            author=author,
            thesis_title=thesis_title,
            thesis_abstract=thesis_text[:800],
            findings=all_findings,
            reference_papers=[m["file_name"] for m in ref_metas],
            thesis_paragraphs=[
                {"index": c["index"], "text": c["text"]}
                for c in thesis_chunks
            ],
        )

        # ── Print profile insights ────────────────────────────────
        if profile_result["insights"] or profile_result["reading_suggestions"]:
            print("\n" + "=" * 60)
            print("📊 用户画像分析")
            print("=" * 60)
            if profile_result["field"] and profile_result["field"] != "未知领域":
                topics_str = "、".join(profile_result.get("topics", [])[:3])
                print(f"  研究领域: {profile_result['field']}")
                if topics_str:
                    print(f"  关键主题: {topics_str}")

            for insight in profile_result["insights"]:
                sev_icon = {"warning": "⚠️", "info": "ℹ️", "tip": "💡", "positive": "✅"}
                icon = sev_icon.get(insight.get("severity", ""), "•")
                print(f"\n  {icon} {insight['title']}")
                print(f"     {insight['finding']}")
                print(f"     👉 {insight['suggestion']}")

            if profile_result["reading_suggestions"]:
                print(f"\n  📚 文献阅读建议")
                for s in profile_result["reading_suggestions"]:
                    print(f"     [{s.get('priority', 'medium')}] {s.get('topic', '')}")
                    print(f"     {s.get('reason', '')}")
                    # Print Semantic Scholar results if available
                    papers = s.get("papers", [])
                    if papers:
                        for p in papers[:3]:
                            authors = ", ".join(p.get("authors", [])[:2])
                            year = p.get("year", "")
                            citations = p.get("citations", 0)
                            print(f"       • {p['title']} ({authors}, {year}) — 引用 {citations}次")
                            if p.get("url"):
                                print(f"         {p['url']}")
            print()

        return CheckResult(
            thesis_path=thesis_path,
            session_id=result.session_id,
            total_findings=total,
            verbatim_count=verbatim,
            near_paraphrase_count=near,
            idea_overlap_count=idea,
            report_path=report_path,
            stats=result.stats,
            author=author,
            research_field=profile_result.get("field", ""),
            insights=profile_result.get("insights", []),
            reading_suggestions=profile_result.get("reading_suggestions", []),
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
                    self.check,
                    thesis_path=item["thesis"],
                    reference_dir=item["references"],
                    thesis_title=item.get("title", ""),
                )
                futures.append(f)
            return [f.result() for f in futures]
