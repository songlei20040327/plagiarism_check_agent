"""
Trajectory exporter: raw JSONL trace → SFT / tool-supervision / preference-pairs.

SFT format: OpenAI native tool_calls structure (not text-only).
  - assistant tool_use → {"role":"assistant","tool_calls":[...]}
  - tool_result       → {"role":"tool","tool_call_id":...,"content":...}
  - thinking          → wrapped in <thinking>...</thinking> XML

Tool supervision: standalone (tool_name, tool_input, tool_result) triples.

Preference pairs require an explicit verified evaluation score and identical
prompt, tool definitions and corpus snapshot. No heuristic output-length reward.
"""

import json
import math
import hashlib
from collections import defaultdict
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
    """Use only an explicit verified evaluation; ungraded traces score zero."""
    evaluation = stats.get("evaluation", {})
    if not isinstance(evaluation, dict):
        return 0.0
    score = evaluation.get("score") if isinstance(evaluation, dict) else None
    if (evaluation.get("verified") is not True or isinstance(score, bool)
            or not isinstance(score, (int, float)) or not math.isfinite(score)
            or not 0 <= score <= 1):
        return 0.0
    return float(score)


def _normalize_records(records: list[dict]) -> list[dict]:
    """Accept current standalone events and legacy user-wrapped results."""
    normalized = []
    for record in records:
        if record.get("type") == "tool_result":
            result = record.get("observation", record.get("result"))
            text = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
            normalized.append({"type": "user", "message": {"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": record["tool_use_id"],
                "content": [{"type": "text", "text": text}],
            }]}})
        else:
            normalized.append(record)
    return normalized


def _valid_protocol(messages: list[dict]) -> bool:
    pending, used = set(), set()
    for message in messages:
        role = message["role"]
        if pending and role != "tool":
            return False
        if role == "tool":
            tid = message.get("tool_call_id")
            if tid not in pending:
                return False
            pending.remove(tid)
        elif role == "assistant":
            for call in message.get("tool_calls", []):
                tid = call.get("id")
                if not tid or tid in used:
                    return False
                try:
                    arguments = json.loads(call["function"]["arguments"])
                except (ValueError, KeyError, TypeError):
                    return False
                if not isinstance(arguments, dict):
                    return False
                used.add(tid)
                pending.add(tid)
    return not pending


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

    for record in _normalize_records(records):
        rtype = record.get("type")
        if rtype == "system":
            if not system_prompt:
                messages.append(record["message"])
            continue
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
    include_failed: bool = False,
) -> Optional[dict]:
    """
    Convert a single trace to OpenAI SFT format with native tool_calls.

    Returns {"messages": [...], "metadata": {...}} or None if trace is invalid.
    """
    records = _load_trace(trace_file)
    meta = _get_session_meta(records)
    stats = _get_session_stats(records)

    if not meta or not stats:
        return None
    if not include_failed and stats.get("status") not in (None, "completed"):
        return None

    recorded_system = any(r.get("type") == "system" for r in records)
    messages = _build_sft_messages_native(records, system_prompt=None if recorded_system else system_prompt)

    if len(messages) < 2 or not _valid_protocol(messages) or messages[-1]["role"] != "assistant":
        return None
    tool_definitions = next((r["tools"] for r in records if r.get("type") == "system" and "tools" in r), [])
    tools = [{"type": "function", "function": {
        "name": d["name"], "description": d.get("description", ""), "parameters": d["input_schema"],
    }} for d in tool_definitions]
    return {
        "messages": messages,
        "tools": tools,
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
            "status": stats.get("status", "unverified_legacy"),
            "snapshot_id": stats.get("snapshot_id", ""),
        },
    }


# ── Tool supervision export ────────────────────────────────────────────

def export_tool_supervision(trace_file: Path) -> list[dict]:
    """Export complete tool-call observations from both trace schema versions."""
    records = _load_trace(trace_file)
    meta = _get_session_meta(records)
    messages = _build_sft_messages_native(records)
    if not _valid_protocol(messages):
        return []
    results = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
    samples = []
    for message in messages:
        for call in message.get("tool_calls", []):
            if call["id"] in results:
                samples.append({
                    "tool_name": call["function"]["name"],
                    "tool_input": json.loads(call["function"]["arguments"]),
                    "tool_result": results[call["id"]],
                    "metadata": {"source": "plagiarism_check_agent",
                                 "session_id": meta.get("session_id", ""),
                                 "thesis": meta.get("thesis", "")},
                })
    return samples


def export_preference_pairs(trace_files: list[Path]) -> list[dict]:
    """Pair explicitly graded trajectories only within identical task contexts."""
    groups = defaultdict(list)
    for trace in trace_files:
        records = _load_trace(trace)
        stats = _get_session_stats(records)
        evaluation = stats.get("evaluation", {})
        if not isinstance(evaluation, dict) or evaluation.get("verified") is not True:
            continue
        score = evaluation.get("score")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1):
            continue
        sample = export_sft(trace, include_failed=True)
        if not sample:
            continue
        messages = sample["messages"]
        split = next((i for i, m in enumerate(messages) if m["role"] == "assistant"), None)
        if split is None or not any(m["role"] == "user" for m in messages[:split]):
            continue
        context = {"prompt": messages[:split], "tools": sample["tools"],
                   "snapshot_id": stats.get("snapshot_id", "")}
        # Missing snapshots cannot establish that the environment was identical.
        if not context["snapshot_id"]:
            continue
        key = hashlib.sha256(json.dumps(context, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        groups[key].append({"sample": sample, "score": float(score), "context": context,
                            "completion": messages[split:]})
    pairs = []
    for task_key, candidates in groups.items():
        ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
        for chosen, rejected in zip(ranked, ranked[1:]):
            if chosen["score"] <= rejected["score"]:
                continue
            pairs.append({
                "prompt": chosen["context"]["prompt"], "tools": chosen["context"]["tools"],
                "chosen": chosen["completion"], "rejected": rejected["completion"],
                "metadata": {"source": "plagiarism_check_agent", "task_key": task_key,
                             "snapshot_id": chosen["context"]["snapshot_id"],
                             "chosen_session": chosen["sample"]["metadata"]["session_id"],
                             "rejected_session": rejected["sample"]["metadata"]["session_id"],
                             "chosen_score": chosen["score"], "rejected_score": rejected["score"]},
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
