"""
Plagiarism memory store: chunk hotspot tracking + curated domain knowledge.

Four record types stored in separate files under MEMORY_DATA_DIR:

  1. common_knowledge.jsonl  — static, human-curated domain-standard descriptions
     that should NOT be flagged. Read-only by the Agent, written manually.

  2. false_positives.jsonl   — human corrections: text patterns the Agent previously
     misclassified but that are actually legitimate parameters or standard configs.

  3. chunk_hotspots.jsonl    — auto-tracked: reference chunks matched across
     multiple *different* theses. When hit >= 3 times, it's likely a domain-standard
     description that should trigger leniency.

  4. author_profiles.jsonl   — per-author profile: research field, check history,
     accumulated insights, reading suggestions.

Legacy data from ~/.claude/plagiarism_memory.jsonl is auto-migrated on first access.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import Config

# ── File paths ──────────────────────────────────────────────────────────

MEMORY_DIR = Config.MEMORY_DATA_DIR
_LEGACY_FILE = Path.home() / ".claude" / "plagiarism_memory.jsonl"

_TYPE_FILES = {
    "common_knowledge": MEMORY_DIR / "common_knowledge.jsonl",
    "false_positive":    MEMORY_DIR / "false_positives.jsonl",
    "chunk_hotspot":     MEMORY_DIR / "chunk_hotspots.jsonl",
    "author_profile":    MEMORY_DIR / "author_profiles.jsonl",
}

HOTSPOT_MIN_HITS = 3
HOTSPOT_MIN_SIMILARITY = 0.75

# ── Built-in common knowledge ──────────────────────────────────────────

BUILTIN_COMMON_KNOWLEDGE = [
    {
        "type": "common_knowledge", "id": "builtin-ck-1",
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
        "content": {
            "name": "深度学习优化算法描述",
            "description": "梯度下降、反向传播、Adam优化器等是深度学习通用基础知识",
            "field": "深度学习",
            "indicators": [
                "梯度下降", "反向传播", "损失函数", "随机梯度下降", "SGD",
                "Adam优化器", "学习率衰减", "mini-batch",
            ],
        },
    },
    {
        "type": "common_knowledge", "id": "builtin-ck-2",
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
        "content": {
            "name": "Transformer 架构描述",
            "description": "自注意力机制、多头注意力等是 NLP 领域的通用架构描述",
            "field": "自然语言处理",
            "indicators": [
                "自注意力机制", "多头注意力", "位置编码", "编码器-解码器",
                "缩放点积注意力", "前馈神经网络", "层归一化",
            ],
        },
    },
    {
        "type": "common_knowledge", "id": "builtin-ck-3",
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
        "content": {
            "name": "统计显著性检验标准表述",
            "description": "p值、置信区间等统计学术语的标准用法不应判为重复",
            "field": "统计学",
            "indicators": [
                "p < 0.05", "p < 0.01", "置信区间", "t检验", "方差分析",
                "ANOVA", "卡方检验", "显著性水平",
            ],
        },
    },
    {
        "type": "common_knowledge", "id": "builtin-ck-4",
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
        "content": {
            "name": "数据集划分与评估标准表述",
            "description": "训练/验证/测试集划分、交叉验证等是机器学习通用方法论",
            "field": "机器学习",
            "indicators": [
                "训练集", "验证集", "测试集", "k折交叉验证",
                "留出法", "数据增强", "准确率", "F1分数",
            ],
        },
    },
    {
        "type": "common_knowledge", "id": "builtin-ck-5",
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
        "content": {
            "name": "实验环境与超参数描述",
            "description": "GPU型号、框架版本、学习率、batch size等实验配置描述是标准化的",
            "field": "计算机科学",
            "indicators": [
                "GPU", "NVIDIA", "CUDA", "PyTorch", "TensorFlow",
                "Python 3", "Ubuntu", "学习率", "batch size", "epoch",
            ],
        },
    },
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_id(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d")
    rand = uuid.uuid4().hex[:4]
    return f"{prefix}-{ts}-{rand}"


def _chunk_hash(text: str) -> str:
    normalized = text.strip()[:300]
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


# ── Low-level file I/O ──────────────────────────────────────────────────

def _migrate_from_legacy():
    """One-time migration from old single-file to new type-split files."""
    if not _LEGACY_FILE.exists():
        return
    # Check if any new file already exists (partial migration = skip)
    if any(f.exists() for f in _TYPE_FILES.values()):
        return

    records_by_type: dict[str, list[dict]] = {t: [] for t in _TYPE_FILES}
    with open(_LEGACY_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rtype = rec.get("type", "")
                if rtype in records_by_type:
                    records_by_type[rtype].append(rec)
            except json.JSONDecodeError:
                pass

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    for rtype, records in records_by_type.items():
        filepath = _TYPE_FILES[rtype]
        with open(filepath, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Rename legacy file so we don't re-migrate
    _LEGACY_FILE.rename(_LEGACY_FILE.with_suffix(".jsonl.bak"))


def _load_records(record_type: str) -> list[dict]:
    """Load all records of a given type from its dedicated file."""
    _migrate_from_legacy()

    filepath = _TYPE_FILES.get(record_type)
    if filepath is None:
        return []

    if not filepath.exists():
        # Auto-init common_knowledge with builtins
        if record_type == "common_knowledge":
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            _write_records("common_knowledge", list(BUILTIN_COMMON_KNOWLEDGE))
            return list(BUILTIN_COMMON_KNOWLEDGE)
        return []

    records = []
    with open(filepath, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if record_type == "common_knowledge" and not records:
        _write_records("common_knowledge", list(BUILTIN_COMMON_KNOWLEDGE))
        return list(BUILTIN_COMMON_KNOWLEDGE)

    return records


def _write_records(record_type: str, records: list[dict]) -> None:
    """Overwrite the type-specific file with a list of records."""
    filepath = _TYPE_FILES[record_type]
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _append_record(record_type: str, record: dict) -> None:
    """Append a single record to the type-specific file."""
    filepath = _TYPE_FILES[record_type]
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Chunk hotspot tracking ─────────────────────────────────────────────

def record_chunk_matches(
    thesis_title: str,
    matches: list[dict],
) -> dict:
    """Record matched reference chunks for cross-thesis hotspot detection."""
    if not matches:
        return {"status": "ok", "new_hotspots": 0, "total_matches": 0}

    records = _load_records("chunk_hotspot")
    now = _now_iso()
    new_hotspots = 0

    hotspot_index: dict[str, int] = {}
    thesis_sets: dict[str, set] = {}
    for i, r in enumerate(records):
        h = r["content"].get("chunk_hash", "")
        if h:
            hotspot_index[h] = i
            thesis_sets[h] = set(r["content"].get("thesis_list", []))

    for match in matches:
        similarity = match.get("similarity", 0)
        if similarity < HOTSPOT_MIN_SIMILARITY:
            continue

        text = match.get("text", "") or match.get("matched_text", "")
        if not text or len(text.strip()) < 20:
            continue

        h = _chunk_hash(text)
        source = match.get("source_paper", "unknown")

        if h in hotspot_index:
            idx = hotspot_index[h]
            content = records[idx]["content"]
            thesis_set = thesis_sets.get(h, set())
            thesis_set.add(thesis_title)
            content["hit_count"] = content.get("hit_count", 0) + 1
            content["thesis_list"] = sorted(thesis_set)
            content["thesis_count"] = len(thesis_set)
            content["last_seen"] = now
            content["is_hotspot"] = len(thesis_set) >= HOTSPOT_MIN_HITS
            if content["is_hotspot"] and content.get("hit_count") == HOTSPOT_MIN_HITS:
                new_hotspots += 1
            records[idx]["updated_at"] = now
        else:
            record = {
                "type": "chunk_hotspot",
                "id": _new_id("chunk_hotspot"),
                "created_at": now,
                "updated_at": now,
                "content": {
                    "chunk_hash": h,
                    "text_sample": text.strip()[:300],
                    "source_paper": source,
                    "similarity": round(similarity, 4),
                    "hit_count": 1,
                    "thesis_list": [thesis_title],
                    "thesis_count": 1,
                    "first_seen": now,
                    "last_seen": now,
                    "is_hotspot": False,
                },
            }
            records.append(record)
            hotspot_index[h] = len(records) - 1
            thesis_sets[h] = {thesis_title}

    _write_records("chunk_hotspot", records)
    return {
        "status": "ok",
        "new_hotspots": new_hotspots,
        "total_matches": len(matches),
        "total_hotspots": sum(
            1 for r in records
            if r["content"].get("is_hotspot")
        ),
    }


def get_hotspots(min_thesis_count: int = HOTSPOT_MIN_HITS) -> list[dict]:
    """Return all chunks that have reached hotspot status."""
    return [
        r for r in _load_records("chunk_hotspot")
        if r["content"].get("thesis_count", 0) >= min_thesis_count
    ]


# ── Human-correction false positives ───────────────────────────────────

def add_false_positive(
    pattern: str,
    reason: str,
    text_sample: str = "",
    field: str = "",
) -> dict:
    record = {
        "type": "false_positive",
        "id": _new_id("false_positive"),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "content": {
            "pattern": pattern,
            "reason": reason,
            "text_sample": text_sample[:500] if text_sample else "",
            "field": field,
        },
    }
    _append_record("false_positive", record)
    return record


def get_false_positives() -> list[dict]:
    return _load_records("false_positive")


# ── Author profile ─────────────────────────────────────────────────────

def get_author_profile(author: str) -> Optional[dict]:
    for r in _load_records("author_profile"):
        if r.get("content", {}).get("author") == author:
            return r
    return None


def update_author_profile(
    author: str,
    thesis_title: str = "",
    field: str = "",
    topics: Optional[list[str]] = None,
    findings_summary: Optional[dict] = None,
    insights: Optional[list[dict]] = None,
    reading_suggestions: Optional[list[dict]] = None,
) -> dict:
    records = _load_records("author_profile")
    now = _now_iso()

    existing_idx = None
    for i, r in enumerate(records):
        if r.get("content", {}).get("author") == author:
            existing_idx = i
            break

    if existing_idx is not None:
        profile = records[existing_idx]
        content = profile["content"]
    else:
        content = {
            "author": author,
            "field": field,
            "topics": topics or [],
            "check_history": [],
            "insight_history": [],
            "total_checks": 0,
            "first_seen": now,
            "last_updated": now,
        }
        profile = {
            "type": "author_profile",
            "id": _new_id("author_profile"),
            "created_at": now,
            "updated_at": now,
            "content": content,
        }

    content["total_checks"] = content.get("total_checks", 0) + 1
    content["last_updated"] = now

    if field and field != "未知领域":
        content["field"] = field
    if topics:
        existing_topics = set(content.get("topics", []))
        existing_topics.update(topics)
        content["topics"] = list(existing_topics)[:10]

    check_entry = {
        "thesis": thesis_title,
        "date": now,
        "findings": findings_summary or {},
    }
    content.setdefault("check_history", []).append(check_entry)
    content["check_history"] = content["check_history"][-10:]

    if insights:
        existing_types = {
            h.get("type") for h in content.get("insight_history", [])
        }
        content.setdefault("insight_history", [])
        for ins in insights:
            if ins.get("type") not in existing_types:
                content["insight_history"].append(ins)
                existing_types.add(ins["type"])

    if reading_suggestions:
        content["latest_reading_suggestions"] = reading_suggestions

    profile["updated_at"] = now
    profile["content"] = content

    if existing_idx is not None:
        records[existing_idx] = profile
        _write_records("author_profile", records)
    else:
        _append_record("author_profile", profile)

    return profile


# ── Summary for prompt injection ───────────────────────────────────────

def get_memory_summary(max_tokens: int = 2000) -> str:
    """
    Generate a formatted summary for system prompt injection.
    Reads from all type-specific files.
    """
    max_chars = max_tokens * 2
    sections = []

    # 1. Hotspots (most actionable)
    hotspots = [
        r for r in _load_records("chunk_hotspot")
        if r["content"].get("is_hotspot")
    ]
    if hotspots:
        hotspots.sort(key=lambda r: r["content"].get("thesis_count", 0), reverse=True)
        hotspots = hotspots[:15]

        lines = ["## 高频匹配模式（跨论文热点，可能是领域通用表述）", ""]
        lines.append(
            "以下文本片段在 ≥3 篇不同论文中被匹配到，很可能是该领域的标准方法论描述。"
            "遇到高度相似的段落时，优先考虑降级为 idea_overlap 或 no_issue，"
            "除非存在逐字照搬的证据。"
        )
        lines.append("")
        for i, r in enumerate(hotspots, 1):
            c = r["content"]
            text_preview = c.get("text_sample", "")[:120].replace("\n", " ")
            lines.append(
                f"{i}. [命中{c['thesis_count']}次, 来源: {c.get('source_paper', '?')}] "
                f"「{text_preview}...」"
            )
        sections.append("\n".join(lines))

    # 2. Common knowledge
    ck_list = _load_records("common_knowledge")
    if ck_list:
        lines = ["## 领域常识（不应标记为重复）"]
        for i, r in enumerate(ck_list, 1):
            c = r.get("content", {})
            field_tag = f"[{c.get('field', '')}] " if c.get("field") else ""
            indicators = "、".join(c.get("indicators", [])[:6])
            lines.append(
                f"{i}. {field_tag}{c.get('name', '')} — "
                f"关键词: {indicators}"
            )
        sections.append("\n".join(lines))

    # 3. False positives (human corrections)
    fp_list = _load_records("false_positive")
    if fp_list:
        lines = ["## 人工纠错记录（以下模式曾被误判，请慎重）"]
        for i, r in enumerate(fp_list, 1):
            c = r.get("content", {})
            sample = c.get("text_sample", "")[:80]
            lines.append(
                f"{i}. {c.get('pattern', '')} — 原因: {c.get('reason', '')}"
            )
            if sample:
                lines.append(f"   样例: 「{sample}...」")
        sections.append("\n".join(lines))

    if not sections:
        return ""

    result = "\n\n".join(sections)
    if len(result) > max_chars:
        total_records = sum(
            len(_load_records(t)) for t in _TYPE_FILES
        )
        result = result[:max_chars] + (
            f"\n\n[已截断，共 {total_records} 条记忆记录]"
        )

    return result
