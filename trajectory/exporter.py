"""
Trajectory exporter: raw JSONL trace → SFT / tool-supervision / preference-pairs.

SFT format: OpenAI native tool_calls structure (not text-only).
  - assistant tool_use → {"role":"assistant","tool_calls":[...]}
  - tool_result       → {"role":"tool","tool_call_id":...,"content":...}
  - thinking          → wrapped in <thinking>...</thinking> XML

Tool supervision: standalone (tool_name, tool_input, tool_result) triples.

Preference pairs (DPO): pairs of (chosen, rejected) trajectories ranked by
a plagiarism-specific quality_score that rewards classification coverage,
suggestion depth, analytical reasoning, and search efficiency.
"""

import json
from pathlib import Path
from typing import Optional


# ── Helpers ────────────────────────────────────────────────────────────

def _load_trace(trace_file: Path) -> list[dict]:
    records = []
    with open(trace_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _get_session_meta(records: list[dict]) -> dict:
    for r in records:
        if r.get("type") == "session_start":
            return r
    return {}


def _get_session_stats(records: list[dict]) -> dict:
    for r in records:
        if r.get("type") == "session_end":
            return r.get("stats", {})
    return {}


# ── Quality scoring (PCA-specific) ─────────────────────────────────────

def quality_score(stats: dict, _records: Optional[list[dict]] = None) -> float:
    """
    Score a trajectory for DPO pairing. Higher = better analysis.

    Dimensions:
      1. Classification coverage — found diverse duplicate types (max +3.0)
      2. Suggestion depth    — average suggestion length (max +2.0)
      3. Analytical rigor    — has detailed reasoning text (max +1.5)
      4. Search efficiency   — reasonable batch coverage (max +1.0)
    """
    score = 0.0

    # 1. Classification coverage
    dist = stats.get("classification_distribution", {})
    types_found = sum(
        1 for t in ["verbatim", "near_paraphrase", "idea_overlap"]
        if dist.get(t, 0) > 0
    )
    score += types_found * 1.0

    # 2. Suggestion depth
    avg_sug = stats.get("avg_suggestion_length", 0)
    if avg_sug > 100:
        score += 2.0
    elif avg_sug > 50:
        score += 1.0

    # 3. Analytical rigor
    if stats.get("has_detailed_analysis", False):
        score += 1.5

    # 4. Search efficiency (ideal: ~1 batch per 3 paragraphs)
    search_calls = stats.get("search_calls", 0)
    total_paragraphs = stats.get("total_paragraphs", 1)
    expected = max(total_paragraphs / 3, 1)
    coverage_ratio = search_calls / expected
    if 0.5 <= coverage_ratio <= 1.5:
        score += 1.0

    return score


# ── SFT export (native tool_calls format) ──────────────────────────────

def _build_sft_messages_native(
    records: list[dict],
    system_prompt: Optional[str] = None,
) -> list[dict]:
    """
    Build SFT messages in OpenAI native tool_calls format.

    Mirrors CRA exporter behavior:
      - tool_use blocks → {"role":"assistant","tool_calls":[...]}
      - tool_result blocks → {"role":"tool","tool_call_id":...,"content":...}
      - thinking blocks → "<thinking>\n...\n</thinking>" in assistant content
      - text blocks → {"role":"assistant","content":"..."}
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for record in records:
        rtype = record.get("type")
        if rtype not in ("user", "assistant"):
            continue

        msg = record.get("message", {})
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            # Detect tool_result blocks (PCA logger wraps them inside user messages)
            if isinstance(content, list) and all(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in content
            ):
                for block in content:
                    tool_call_id = block.get("tool_use_id", "")
                    inner = block.get("content", [])
                    if isinstance(inner, list):
                        result_text = "\n".join(
                            c.get("text", "")
                            for c in inner
                            if isinstance(c, dict)
                        )
                    elif isinstance(inner, str):
                        result_text = inner
                    else:
                        result_text = str(inner)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result_text,
                    })
                continue

            # Real user message
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "\n".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                text = str(content)

            if text.strip():
                messages.append({"role": "user", "content": text})

        elif role == "assistant":
            if not isinstance(content, list):
                text = str(content)
                if text.strip():
                    messages.append({"role": "assistant", "content": text})
                continue

            text_parts = []
            tool_calls = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "thinking":
                    text_parts.append(
                        f"<thinking>\n{block.get('thinking', '')}\n</thinking>"
                    )
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(
                                block.get("input", {}), ensure_ascii=False
                            ),
                        },
                    })

            if tool_calls:
                assistant_msg = {"role": "assistant", "tool_calls": tool_calls}
                combined_text = "\n\n".join(p for p in text_parts if p.strip())
                if combined_text.strip():
                    assistant_msg["content"] = combined_text
                messages.append(assistant_msg)
            elif text_parts:
                combined_text = "\n\n".join(p for p in text_parts if p.strip())
                if combined_text.strip():
                    messages.append({
                        "role": "assistant", "content": combined_text
                    })

    return messages


def export_sft(
    trace_file: Path,
    system_prompt: Optional[str] = None,
) -> Optional[dict]:
    """
    Convert a single trace to OpenAI SFT format with native tool_calls.

    Returns {"messages": [...], "metadata": {...}} or None if trace is invalid.
    """
    records = _load_trace(trace_file)
    meta = _get_session_meta(records)
    stats = _get_session_stats(records)

    if not meta:
        return None

    messages = _build_sft_messages_native(records, system_prompt=system_prompt)

    if len(messages) < 2:
        return None

    return {
        "messages": messages,
        "metadata": {
            "source": "plagiarism_check_agent",
            "session_id": meta.get("session_id", ""),
            "thesis": meta.get("thesis", ""),
            "reference_count": meta.get("reference_count", 0),
            "findings_count": stats.get("findings_count", 0),
            "classification_distribution": stats.get(
                "classification_distribution", {}
            ),
            "avg_suggestion_length": stats.get("avg_suggestion_length", 0),
            "has_detailed_analysis": stats.get("has_detailed_analysis", False),
            "llm_calls": stats.get("llm_calls", 0),
            "tool_calls": stats.get("tool_calls", 0),
            "iterations": stats.get("iterations", 0),
            "quality_score": quality_score(stats, records),
        },
    }


# ── Tool supervision export ────────────────────────────────────────────

def export_tool_supervision(trace_file: Path) -> list[dict]:
    """
    Extract (tool_use → tool_result) pairs as standalone training samples.
    """
    records = _load_trace(trace_file)
    meta = _get_session_meta(records)
    samples = []

    # Build tool_use_id → result text map
    tool_results: dict[str, str] = {}
    for record in records:
        if record.get("type") != "user":
            continue
        content = record.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tid = block.get("tool_use_id", "")
                inner = block.get("content", [])
                text = ""
                if isinstance(inner, list):
                    text = "\n".join(
                        c.get("text", "")
                        for c in inner
                        if isinstance(c, dict)
                    )
                elif isinstance(inner, str):
                    text = inner
                tool_results[tid] = text

    # Extract tool_use blocks from assistant messages
    for record in records:
        if record.get("type") != "assistant":
            continue
        content = record.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tid = block.get("id", "")
            result = tool_results.get(tid, "")
            if result:
                samples.append({
                    "tool_name": block.get("name", ""),
                    "tool_input": block.get("input", {}),
                    "tool_result": result,
                    "metadata": {
                        "source": "plagiarism_check_agent",
                        "session_id": meta.get("session_id", ""),
                        "thesis": meta.get("thesis", ""),
                    },
                })

    return samples


# ── Preference pairs export (DPO) ──────────────────────────────────────

def export_preference_pairs(trace_files: list[Path]) -> list[dict]:
    """
    Build DPO preference pairs from ranked trajectories.

    Each trace is scored by quality_score(). Traces are sorted
    high-to-low and adjacent pairs are formed where score(chosen) > score(rejected).
    """
    candidates = []
    for tf in trace_files:
        sft = export_sft(tf)
        if sft is None:
            continue
        stats = _get_session_stats(_load_trace(tf))
        candidates.append({
            "sft": sft,
            "stats": stats,
            "session_id": sft["metadata"].get("session_id", ""),
            "thesis": sft["metadata"].get("thesis", ""),
            "score": sft["metadata"].get("quality_score", 0.0),
        })

    if len(candidates) < 2:
        return []

    ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)

    pairs = []
    for i in range(len(ranked) - 1):
        chosen = ranked[i]
        rejected = ranked[i + 1]
        if chosen["score"] > rejected["score"]:
            pairs.append({
                "chosen": chosen["sft"]["messages"],
                "rejected": rejected["sft"]["messages"],
                "metadata": {
                    "source": "plagiarism_check_agent",
                    "chosen_session": chosen["session_id"],
                    "rejected_session": rejected["session_id"],
                    "chosen_thesis": chosen["thesis"],
                    "rejected_thesis": rejected["thesis"],
                    "chosen_score": chosen["score"],
                    "rejected_score": rejected["score"],
                },
            })

    return pairs


# ── Directory export (main entry point) ────────────────────────────────

def export_directory(
    traces_dir: Path,
    output_dir: Path,
    system_prompt: Optional[str] = None,
    formats: Optional[list[str]] = None,
) -> dict[str, int]:
    """
    Export all traces in a directory to the specified training formats.

    formats: list of "sft", "tool-supervision", "preference-pairs"
    Default: all three.
    Returns {"sft": N, "tool-supervision": N, "preference-pairs": N}
    """
    if formats is None:
        formats = ["sft", "tool-supervision", "preference-pairs"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_files = sorted(traces_dir.glob("trace_*.jsonl"))

    counts: dict[str, int] = {}

    if "sft" in formats:
        sft_path = output_dir / "sft_train.jsonl"
        count = 0
        with open(sft_path, "w", encoding="utf-8") as f:
            for tf in trace_files:
                sample = export_sft(tf, system_prompt=system_prompt)
                if sample:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    count += 1
        counts["sft"] = count
        print(f"SFT: {count} samples → {sft_path}")

    if "tool-supervision" in formats:
        ts_path = output_dir / "tool_supervision.jsonl"
        count = 0
        with open(ts_path, "w", encoding="utf-8") as f:
            for tf in trace_files:
                samples = export_tool_supervision(tf)
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
                    count += 1
        counts["tool-supervision"] = count
        print(f"Tool supervision: {count} samples → {ts_path}")

    if "preference-pairs" in formats:
        pp_path = output_dir / "preference_pairs.jsonl"
        pairs = export_preference_pairs(trace_files)
        with open(pp_path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        counts["preference-pairs"] = len(pairs)
        print(f"Preference pairs: {len(pairs)} pairs → {pp_path}")

    return counts
