#!/usr/bin/env python3
"""
Plagiarism Check Agent CLI.

Usage:
  python cli.py check thesis.pdf --refs ./reference_papers/
  python cli.py check thesis.pdf --refs ./refs/ --model gpt-4o
  python cli.py inspect ./trajectories/trace_abc123.jsonl
  python cli.py export --input ./trajectories --format sft

Environment:
  OPENAI_API_KEY   LLM API key (required)
  OPENAI_MODEL     Model name (default: deepseek-chat)
  OPENAI_BASE_URL  API base URL (default: DeepSeek)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from plagiarism_check_agent.config import Config
from plagiarism_check_agent.pipeline.orchestrator import PlagiarismCheckOrchestrator
from plagiarism_check_agent.trajectory.exporter import export_directory


def cmd_check(args) -> int:
    """Run a plagiarism check on a thesis against reference papers."""
    thesis = Path(args.thesis)
    refs = Path(args.refs)

    if not thesis.exists():
        print(f"错误: 论文文件不存在 — {thesis}")
        return 1
    if not refs.exists() or not refs.is_dir():
        print(f"错误: 对照文献目录不存在 — {refs}")
        return 1

    orchestrator = PlagiarismCheckOrchestrator(
        api_key=args.api_key or Config.OPENAI_API_KEY,
        model=args.model or Config.OPENAI_MODEL,
        base_url=args.base_url or Config.OPENAI_BASE_URL,
        trajectories_dir=Path(args.trajectories_dir),
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
    )

    print(f"论文: {thesis}")
    print(f"对照文献目录: {refs}")
    print(f"模型: {args.model or Config.OPENAI_MODEL}\n")

    result = orchestrator.check(
        thesis_path=str(thesis.absolute()),
        reference_dir=str(refs.absolute()),
        thesis_title=args.title or thesis.stem,
    )

    if result.error:
        print(f"\n错误: {result.error}")
        return 1

    print(f"\n{'=' * 60}")
    print("查重完成")
    print(f"{'=' * 60}")
    print(f"Session:  {result.session_id}")
    print(f"发现总数: {result.total_findings}")
    print(f"  🔴 逐字重复: {result.verbatim_count}")
    print(f"  🟠 近义改写: {result.near_paraphrase_count}")
    print(f"  🟡 观点重复: {result.idea_overlap_count}")

    if result.report_path:
        print(f"\n📋 报告: {result.report_path}")
    print(f"📊 轨迹: {args.trajectories_dir}/trace_{result.session_id}.jsonl")

    return 0


def cmd_inspect(args) -> int:
    """Inspect a trajectory JSONL file."""
    trace_file = Path(args.trace_file)
    if not trace_file.exists():
        print(f"错误: 轨迹文件不存在 — {trace_file}")
        return 1

    records = []
    with open(trace_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"轨迹文件: {trace_file}")
    print(f"记录数: {len(records)}\n")

    for i, record in enumerate(records):
        rtype = record.get("type", "?")
        if rtype == "session_start":
            print(f"[{i}] SESSION_START  thesis={record.get('thesis')}  "
                  f"refs={record.get('reference_count')}  "
                  f"id={record.get('session_id')}")
        elif rtype == "session_end":
            stats = record.get("stats", {})
            print(f"[{i}] SESSION_END    llm_calls={stats.get('llm_calls')}  "
                  f"tool_calls={stats.get('tool_calls')}  "
                  f"duration={stats.get('duration_ms')}ms  "
                  f"iterations={stats.get('iterations')}")
        elif rtype == "user":
            content = record.get("message", {}).get("content", "")
            if isinstance(content, str):
                preview = content[:100].replace("\n", " ")
                print(f"[{i}] USER          {preview!r}")
            elif isinstance(content, list):
                tool_ids = [
                    b.get("tool_use_id", "?") for b in content
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                ]
                if tool_ids:
                    print(f"[{i}] TOOL_RESULT   ids={tool_ids}")
        elif rtype == "assistant":
            msg = record.get("message", {})
            content = msg.get("content", [])
            stop = msg.get("stop_reason", "?")
            usage = msg.get("usage", {})
            blocks = []
            for b in content:
                if isinstance(b, dict):
                    bt = b.get("type", "?")
                    if bt == "text":
                        blocks.append(f"text({len(b.get('text',''))}chars)")
                    elif bt == "tool_use":
                        blocks.append(f"tool_use({b.get('name')})")
            print(f"[{i}] ASSISTANT     stop={stop}  "
                  f"in={usage.get('input_tokens')}  "
                  f"out={usage.get('output_tokens')}  "
                  f"blocks=[{', '.join(blocks)}]")
    return 0


def cmd_export(args) -> int:
    """Export trajectories to training data formats."""
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.exists():
        print(f"错误: {input_dir} 不存在")
        return 1

    formats = args.format if args.format else ["sft", "tool-supervision"]

    system_prompt = None
    if "sft" in formats:
        try:
            from plagiarism_check_agent.agent.skill_loader import SkillLoader
            system_prompt = SkillLoader().load_skill("plagiarism-check")
        except Exception:
            pass

    print(f"导出: {input_dir} → {output_dir}")
    counts = export_directory(
        traces_dir=input_dir,
        output_dir=output_dir,
        system_prompt=system_prompt,
        formats=formats,
    )
    print("\n导出完成:")
    for fmt, count in counts.items():
        print(f"  {fmt}: {count} samples")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plagiarism-check-agent",
        description="学术论文查重助手 — 基于向量检索 + LLM精判的论文相似度检测系统",
    )

    subparsers = parser.add_subparsers(dest="command")

    # check
    p_check = subparsers.add_parser("check", help="运行论文查重")
    p_check.add_argument("thesis", help="论文 PDF 文件路径")
    p_check.add_argument("--refs", required=True, help="对照文献 PDF 目录")
    p_check.add_argument("--title", default="", help="论文标题（默认使用文件名）")
    p_check.add_argument("--model", default="", help="LLM 模型名称")
    p_check.add_argument("--api-key", default="", help="API key")
    p_check.add_argument("--base-url", default="", help="API base URL")
    p_check.add_argument("--trajectories-dir", default="./trajectories",
                          help="轨迹输出目录")
    p_check.add_argument("--output-dir", default="./output",
                          help="报告输出目录")
    p_check.add_argument("--dry-run", action="store_true",
                          help="跳过报告写入")
    p_check.set_defaults(func=cmd_check)

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="查看轨迹文件")
    p_inspect.add_argument("trace_file", help="trace_*.jsonl 文件路径")
    p_inspect.set_defaults(func=cmd_inspect)

    # export
    p_export = subparsers.add_parser("export", help="导出训练数据")
    p_export.add_argument("--input", required=True, help="轨迹目录")
    p_export.add_argument("--output", default="./exports", help="输出目录")
    p_export.add_argument("--format", nargs="+",
                           choices=["sft", "tool-supervision"],
                           help="导出格式")
    p_export.set_defaults(func=cmd_export)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
