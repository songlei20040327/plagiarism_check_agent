"""
Report generation tools: produce annotated plagiarism reports in HTML and DOCX formats.

The HTML report highlights suspicious passages in red with source annotations.
The DOCX report preserves formatting with tracked-change-style comments.
"""

import json
import math
from html import escape
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..config import Config

# ── Tool definitions ─────────────────────────────────────────────────────────
# Tools are registered via ToolRegistry at module level (see bottom of file).

# HTML template for the report
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>查重报告 — {thesis_title}</title>
<style>
  :root {{ --critical: #dc3545; --high: #fd7e14; --medium: #ffc107; --low: #28a745; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; line-height: 1.8; color: #333; max-width: 1100px; margin: 0 auto; padding: 20px; }}
  h1 {{ border-bottom: 3px solid #2563eb; padding-bottom: 12px; margin-bottom: 8px; }}
  .meta {{ color: #666; font-size: 14px; margin-bottom: 30px; }}
  .summary {{ display: flex; gap: 16px; margin-bottom: 32px; flex-wrap: wrap; }}
  .stat {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 24px; text-align: center; min-width: 100px; }}
  .stat .num {{ font-size: 28px; font-weight: 700; }}
  .stat .label {{ font-size: 13px; color: #666; }}
  .stat.critical .num {{ color: #dc3545; }}
  .stat.high .num {{ color: #fd7e14; }}
  .stat.medium .num {{ color: #ffc107; }}
  .stat.low .num {{ color: #28a745; }}
  .finding {{ border: 1px solid #e2e8f0; border-radius: 10px; margin-bottom: 24px; overflow: hidden; }}
  .finding-header {{ padding: 14px 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }}
  .finding-header.verbatim {{ background: #fff5f5; border-bottom: 1px solid #fecaca; }}
  .finding-header.near_paraphrase {{ background: #fff7ed; border-bottom: 1px solid #fed7aa; }}
  .finding-header.idea_overlap {{ background: #fffbea; border-bottom: 1px solid #fef08a; }}
  .finding-header.no_issue {{ background: #f0fdf4; border-bottom: 1px solid #bbf7d0; }}
  .dup-type {{ font-weight: 600; font-size: 14px; padding: 3px 10px; border-radius: 4px; }}
  .verbatim .dup-type {{ background: #dc3545; color: #fff; }}
  .near_paraphrase .dup-type {{ background: #fd7e14; color: #fff; }}
  .idea_overlap .dup-type {{ background: #ffc107; color: #333; }}
  .no_issue .dup-type {{ background: #28a745; color: #fff; }}
  .source {{ font-size: 14px; color: #555; }}
  .similarity {{ font-size: 13px; color: #888; }}
  .finding-body {{ padding: 20px; }}
  .section-label {{ font-weight: 600; font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin: 12px 0 6px; }}
  .section-label:first-child {{ margin-top: 0; }}
  .overlap-box {{ background: #fff5f5; border-left: 4px solid #dc3545; padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 6px 0 12px; white-space: pre-wrap; font-size: 14px; }}
  .suggestion-box {{ background: #f0fdf4; border-left: 4px solid #28a745; padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 6px 0 12px; font-size: 14px; }}
  .no-match {{ background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 40px; text-align: center; color: #16a34a; }}
</style>
</head>
<body>
<h1>📋 论文查重报告</h1>
<div class="meta">
  <strong>论文</strong>: {thesis_title}<br>
  <strong>生成时间</strong>: {timestamp}<br>
  <strong>查重方式</strong>: 参考文献检索 + 原文证据判定 (对照文献库范围)
</div>

<div class="summary">
  <div class="stat critical"><div class="num">{count_verbatim}</div><div class="label">🔴 逐字重复</div></div>
  <div class="stat high"><div class="num">{count_near}</div><div class="label">🟠 近义改写</div></div>
  <div class="stat medium"><div class="num">{count_idea}</div><div class="label">🟡 观点重复</div></div>
  <div class="stat low"><div class="num">{count_ok}</div><div class="label">🟢 无问题</div></div>
</div>

{findings_html}

<div style="margin-top: 40px; padding: 16px; background: #f8fafc; border-radius: 8px; font-size: 13px; color: #888;">
  <strong>⚠️ 免责声明</strong>: 本报告仅在用户提供的 {ref_count} 篇对照文献范围内进行语义检索和文本对比。
  检索分数用于筛选候选，重复类型由 LLM 基于原文证据判断，不接入任何第三方查重系统。
  标红段落为系统识别的疑似重复内容，仅供参考，最终判定以人工复核为准。
</div>
</body>
</html>"""


def _classify_verbatim(s: float) -> str:
    if s >= 0.90:
        return "verbatim"
    elif s >= 0.82:
        return "near_paraphrase"
    elif s >= 0.75:
        return "idea_overlap"
    return "no_issue"


def _type_label(t: str) -> str:
    labels = {
        "verbatim": "逐字重复",
        "near_paraphrase": "近义改写",
        "idea_overlap": "观点重复",
        "no_issue": "无问题",
    }
    return labels.get(t, t)


class ReportTools:
    """Handler for report generation operations."""

    def generate_report(
        self,
        thesis_title: str,
        findings: list[dict],
        output_dir: str,
        snapshot_id: str = "",
        reference_count: Optional[int] = None,
        coverage: Optional[dict] = None,
        retrieval_config: Optional[dict] = None,
    ) -> dict:
        """Generate an annotated HTML report from LLM findings."""
        findings = [dict(f) for f in findings]
        for f in findings:
            score = f.get("similarity")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not -1 <= score <= 1:
                raise ValueError("similarity must be a finite number between -1 and 1")
            if f.get("duplicate_type") and f["duplicate_type"] not in {"verbatim", "near_paraphrase", "idea_overlap", "no_issue"}:
                raise ValueError("Invalid duplicate_type")
            for field in ("thesis_chunk", "source_paper", "overlapping_text", "suggestion"):
                if not isinstance(f.get(field), str):
                    raise ValueError(f"{field} must be a string")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        ref_count = reference_count if reference_count is not None else len(set(
            f.get("source_paper", "") for f in findings if f.get("source_paper")
        ))

        # Legacy standalone fallback; agent reports require an explicit judgment.
        for f in findings:
            if "duplicate_type" not in f or not f["duplicate_type"]:
                f["duplicate_type"] = _classify_verbatim(f.get("similarity", 0))
            if "suggestion" not in f:
                f["suggestion"] = ""
            if "overlapping_text" not in f:
                f["overlapping_text"] = ""

        # Count by type
        count_verbatim = sum(1 for f in findings if f["duplicate_type"] == "verbatim")
        count_near = sum(1 for f in findings if f["duplicate_type"] == "near_paraphrase")
        count_idea = sum(1 for f in findings if f["duplicate_type"] == "idea_overlap")
        count_ok = sum(1 for f in findings if f["duplicate_type"] == "no_issue")

        # Build findings HTML
        findings_parts = []
        for i, f in enumerate(findings):
            dtype = f["duplicate_type"]
            channel_names = {"dense": "语义检索", "bm25": "词项匹配", "literal": "字面重合"}
            channels = "、".join(channel_names.get(str(name), str(name))
                                for name in f.get("retrieval_channels", ["dense"]))
            findings_parts.append(f"""<div class="finding">
  <div class="finding-header {dtype}">
    <span class="dup-type">{_type_label(dtype)}</span>
    <span class="source">📄 来源: {escape(f.get('source_paper', 'unknown'))}</span>
    <span class="similarity">向量相似度: {f.get('similarity', 0):.2%}（非重复概率）</span>
    <span class="source">召回方式: {escape(channels)}</span>
  </div>
  <div class="finding-body">
    <div class="section-label">📍 论文章节 (疑似重复)</div>
    <div class="overlap-box">{escape(f.get('thesis_chunk', ''))}</div>
    <div class="section-label">🔗 对照文献原文</div>
    <div class="overlap-box">{escape(f.get('overlapping_text', ''))}</div>
    <div class="section-label">💡 修改建议</div>
    <div class="suggestion-box">{escape(f.get('suggestion', '（无建议）'))}</div>
  </div>
</div>""")

        findings_html = "\n".join(findings_parts) if findings_parts else (
            '<div class="no-match">✅ 未发现明显重复段落</div>'
        )

        html = HTML_TEMPLATE.format(
            thesis_title=escape(thesis_title),
            timestamp=timestamp,
            ref_count=ref_count,
            count_verbatim=count_verbatim,
            count_near=count_near,
            count_idea=count_idea,
            count_ok=count_ok,
            findings_html=findings_html,
        )

        html_path = output_path / f"plagiarism_report_{_safe_filename(thesis_title)}.html"
        html_path.write_text(html, encoding="utf-8")

        # Also save a JSON version for further processing
        json_path = output_path / f"plagiarism_data_{_safe_filename(thesis_title)}.json"
        json_path.write_text(
            json.dumps({
                "thesis_title": thesis_title,
                "timestamp": timestamp,
                "reference_count": ref_count,
                "snapshot_id": snapshot_id,
                "coverage": coverage or {},
                "retrieval_config": retrieval_config or {},
                "summary": {
                    "verbatim": count_verbatim,
                    "near_paraphrase": count_near,
                    "idea_overlap": count_idea,
                    "no_issue": count_ok,
                },
                "findings": findings,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "status": "ok",
            "snapshot_id": snapshot_id,
            "html_report": str(html_path.absolute()),
            "json_data": str(json_path.absolute()),
            "total_findings": len(findings),
            "verbatim": count_verbatim,
            "near_paraphrase": count_near,
            "idea_overlap": count_idea,
            "no_issue": count_ok,
        }


def _safe_filename(title: str) -> str:
    """Sanitize title for use in filename."""
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in title)
    return safe[:50] if safe else "report"


# ── Register with ToolRegistry ───────────────────────────────────────────────

from .registry import ToolRegistry  # noqa: E402

ToolRegistry.register(
    name="generate_report",
    description=(
        "Generate a final plagiarism check report. Produces an HTML file "
        "with red-highlighted suspicious passages, source paper citations, "
        "AND rewrite suggestions for each flagged section."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "thesis_title": {"type": "string"},
            "findings": {
                "type": "array",
                "description": "Array of finding objects from the LLM analysis",
                "items": {
                    "type": "object",
                    "properties": {
                        "thesis_chunk": {"type": "string"},
                        "source_paper": {"type": "string"},
                        "similarity": {"type": "number", "minimum": -1, "maximum": 1},
                        "thesis_chunk_id": {"type": "string"},
                        "source_chunk_id": {"type": "string"},
                        "analysis": {"type": "string"},
                        "duplicate_type": {
                            "type": "string",
                            "enum": [
                                "verbatim", "near_paraphrase",
                                "idea_overlap", "no_issue"
                            ],
                        },
                        "overlapping_text": {"type": "string"},
                        "suggestion": {"type": "string"},
                    },
                    "required": ["thesis_chunk_id", "source_chunk_id", "duplicate_type", "suggestion"],
                },
            },
            "output_dir": {
                "type": "string",
                "description": "Directory to write report files",
            },
        },
        "required": ["thesis_title", "findings", "output_dir"],
    },
    handler_class=ReportTools,
    is_write_tool=True,
)
