"""
Author profile analysis engine.

Runs AFTER each plagiarism check to produce actionable insights from
cross-paragraph pattern analysis. Every insight answers "so what?" —
if it doesn't tell the user something they don't already know, it's not
worth generating.

Five analysis dimensions:
  1. Source concentration   → literature reading advice
  2. Section diagnosis      → chapter-level writing feedback
  3. Writing pattern        → specific rewriting skill tips
  4. Citation hygiene       → citation norm compliance
  5. Progress tracking      → cross-version improvement (if prior check exists)

Plus: field-aware paper recommendations (Semantic Scholar optional).
"""

import json
import re
import time
import urllib.request
import urllib.parse
import urllib.error
from collections import Counter
from pathlib import Path
from typing import Optional

from ..config import Config

# ── Section classifier (keyword-based) ─────────────────────────────────

SECTION_PATTERNS = [
    ("引言/背景", ["引言", "绪论", "背景", "研究意义", "研究现状", "研究动机",
                   "本文的主要贡献", "本文的工作", "本文提出"]),
    ("文献综述/相关工作", ["文献综述", "相关工作", "related work", "近年来",
                          "已有研究", "前人工作", "国内外研究", "现有方法"]),
    ("方法/模型", ["方法", "模型", "算法", "架构", "框架", "流程", "网络结构",
                   "本文方法", "提出方法", "设计", "公式", "推导", "训练",
                   "优化", "损失函数", "实验设置", "数据集", "参数设置"]),
    ("实验/结果", ["实验", "结果", "性能", "准确率", "精度", "召回", "F1",
                   "对比", "比较", "优于", "baseline", "消融", "可视化",
                   "评估", "验证", "测试"]),
    ("讨论/结论", ["讨论", "分析", "结论", "总结", "贡献", "未来工作",
                   "展望", "不足之处", "局限性", "启示"]),
]

# Common citation markers in Chinese academic writing
CITATION_PATTERNS = [
    r"[\[（(][\d,\s\-;]+[\]）)]",           # [1], (2024), 等
    r"等人?\s*[\[(（]",                        # 张三等(
    r"指出|认为|提出|发现|证明|表明|报道",     # citation verbs
    r"根据|按照|参见|参考|详见",               # reference intro words
    r"[A-Z][a-z]+\s+et\s+al\.?",             # English et al.
]


def classify_section(text: str) -> str:
    """Classify a paragraph into a section type based on keyword matching."""
    scores = Counter()
    for section_name, keywords in SECTION_PATTERNS:
        for kw in keywords:
            if kw.lower() in text.lower():
                scores[section_name] += 1
    if scores:
        return scores.most_common(1)[0][0]
    return "其他"


# ── Field extraction from thesis metadata ──────────────────────────────

FIELD_KEYWORDS = {
    "自然语言处理": ["自然语言", "文本", "机器翻译", "对话", "语义", "BERT", "GPT",
                    "Transformer", "大语言模型", "LLM", "情感分析", "命名实体"],
    "计算机视觉": ["图像", "视觉", "目标检测", "分割", "识别", "CNN", "ResNet",
                   "视频", "GAN", "扩散模型", "ViT", "姿态估计"],
    "机器学习": ["机器学习", "深度学习", "强化学习", "迁移学习", "对比学习",
                 "联邦学习", "自监督", "元学习", "神经架构搜索"],
    "数据挖掘": ["推荐系统", "聚类", "分类", "回归", "异常检测", "时间序列",
                 "特征工程", "图神经网络"],
    "生物医学": ["医学", "临床", "病理", "药物", "基因", "蛋白", "细胞",
                 "诊断", "影像组学", "生存分析"],
    "机器人/自动驾驶": ["机器人", "自动驾驶", "规划", "控制", "SLAM", "路径",
                       "感知", "决策", "激光雷达", "轨迹"],
    "系统/网络": ["分布式", "云计算", "边缘计算", "调度", "负载均衡",
                  "网络", "区块链", "安全", "隐私"],
    "多模态": ["多模态", "视觉语言", "图文", "音频", "视频理解", "跨模态",
               "CLIP", "视觉问答"],
}


def extract_field(title: str, abstract: str = "") -> str:
    """Extract research field from thesis title and abstract keywords."""
    combined = (title + " " + abstract).lower()
    scores = Counter()
    for field, keywords in FIELD_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in combined:
                scores[field] += 1
    if scores and scores.most_common(1)[0][1] >= 2:
        return scores.most_common(1)[0][0]
    # Fallback: grab top 2
    top = [f for f, _ in scores.most_common(2)]
    return " & ".join(top) if top else "未知领域"


def extract_topics(text: str, max_topics: int = 5) -> list[str]:
    """Extract key research topics from text using keyword frequency."""
    # Remove common words and extract meaningful noun phrases
    cleaned = re.sub(r'[，。；：“”‘’！？、\n（）()[\]]', ' ', text)
    words = [w.strip() for w in cleaned.split() if len(w.strip()) >= 2]
    # Count significant words (filter out stop words)
    stop_words = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
        '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
        '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那', '些',
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
    }
    meaningful = [w for w in words if w.lower() not in stop_words and len(w) >= 2]
    counter = Counter(meaningful)
    # Return top topics (words appearing 3+ times)
    return [w for w, c in counter.most_common(max_topics * 2)
            if c >= 3 and not w.isdigit()][:max_topics]


# ── Source concentration analysis ──────────────────────────────────────

def _analyze_source_concentration(
    findings: list[dict],
    reference_papers: list[str],
) -> Optional[dict]:
    """
    Check if matches concentrate on a small subset of reference papers.
    If 60%+ of matches come from <= 2 papers → reading scope too narrow.
    """
    if not findings:
        return None

    source_counter = Counter()
    for f in findings:
        src = f.get("source_paper", "unknown")
        if src and src != "unknown":
            source_counter[src] += 1

    if not source_counter:
        return None

    total = sum(source_counter.values())
    top_sources = source_counter.most_common(3)
    top_ratio = sum(c for _, c in top_sources[:2]) / total if total > 0 else 0
    concentration = len(source_counter)

    if top_ratio >= 0.6 and len(reference_papers) >= 3:
        top_names = "、".join(f"「{s}」" for s, _ in top_sources[:2])
        return {
            "type": "source_concentration",
            "severity": "warning",
            "title": "文献来源过于集中",
            "finding": (
                f"你的 {total} 处高相似匹配中，{top_ratio:.0%} 集中在 {top_names} "
                f"这 {min(2, concentration)} 篇文献上。"
            ),
            "suggestion": (
                f"当前仅深度引用了 {concentration}/{len(reference_papers)} 篇对照文献。"
                "建议将阅读范围扩大到 10-15 篇文献，在更广的文献基础上形成自己的论述框架。"
                "具体可以：1) 追踪这 2 篇核心文献的引用链找到更多相关工作；"
                "2) 在 Google Scholar 上搜索相邻关键词；"
                "3) 阅读相关综述论文了解领域全貌。"
            ),
            "data": {
                "concentration_ratio": round(top_ratio, 2),
                "top_sources": dict(top_sources[:3]),
                "total_sources_matched": concentration,
                "total_references": len(reference_papers),
            },
        }

    return None


# ── Section diagnosis ──────────────────────────────────────────────────

def _diagnose_sections(
    findings: list[dict],
    thesis_paragraphs: Optional[list[dict]] = None,
) -> Optional[dict]:
    """
    Classify each finding's paragraph into a section type and identify
    patterns: which sections have what kinds of issues.
    """
    if not findings or len(findings) < 2:
        return None

    # Classify each finding by section
    section_issues: dict[str, dict] = {}
    for f in findings:
        text = f.get("thesis_chunk", "")
        section = classify_section(text)
        dtype = f.get("duplicate_type", "no_issue")
        if section not in section_issues:
            section_issues[section] = {
                "verbatim": 0, "near_paraphrase": 0,
                "idea_overlap": 0, "total": 0,
            }
        section_issues[section][dtype] = section_issues[section].get(dtype, 0) + 1
        section_issues[section]["total"] += 1

    # Generate diagnosis for the section with most issues
    if not section_issues:
        return None

    most_problematic = max(section_issues.items(), key=lambda x: x[1]["total"])

    section_advice = {
        "引言/背景": (
            "引言或背景章节的高相似度通常意味着过度依赖少数综述文章的结构。"
            "建议用自己的研究动机来组织引言，而非复制现有综述的叙述逻辑。"
        ),
        "文献综述/相关工作": (
            "文献综述章节应避免逐篇描述'某人用了某方法得到某结果'。"
            "更有效的方法是先提炼 3-5 个核心观点或技术路线，再以观点为线索串联文献。"
        ),
        "方法/模型": (
            "方法学章节的相似度高是常见现象（标准算法描述、公开数据集等），"
            "但建议补充你的模型与标准实现的差异点，强调你的独特设计。"
        ),
        "实验/结果": (
            "实验章节出现重复可能是评价指标的表述过于模板化。"
            "尝试在每个实验结果后加入你自己的分析解读，而不仅是呈现数字。"
        ),
        "讨论/结论": (
            "讨论章节的高相似度需要特别注意——这是最体现原创性的部分。"
            "建议将与你工作最相关的 2-3 篇文献作为锚点，深入讨论你的发现与它们的异同。"
        ),
    }

    section_name = most_problematic[0]
    counts = most_problematic[1]
    advice = section_advice.get(section_name, "针对该章节类型，建议重新审视文章组织方式。")

    # Only diagnose if the section has significant issues
    if counts.get("verbatim", 0) + counts.get("near_paraphrase", 0) >= 2:
        return {
            "type": "section_diagnosis",
            "severity": "warning" if counts.get("verbatim", 0) > 0 else "info",
            "title": f"「{section_name}」章节需重点改进",
            "finding": (
                f"{section_name}章节共检测到 {counts['total']} 处相似匹配，"
                f"其中逐字重复 {counts.get('verbatim', 0)} 处、"
                f"近义改写 {counts.get('near_paraphrase', 0)} 处。"
            ),
            "suggestion": advice,
            "data": dict(section_issues),
        }

    return None


# ── Writing pattern analysis ───────────────────────────────────────────

def _analyze_writing_patterns(findings: list[dict]) -> Optional[dict]:
    """
    Analyze near_paraphrase findings to detect common rewriting mistakes.
    """
    near_paraphrases = [
        f for f in findings
        if f.get("duplicate_type") == "near_paraphrase"
        and f.get("suggestion", "").strip()
    ]
    if len(near_paraphrases) < 2:
        return None

    # Check if suggestions contain common patterns
    suggestion_keywords = {
        "同义词替换": ["同义词", "替换", "换词", "换一个词"],
        "语序": ["语序", "顺序", "调整句子"],
        "句子结构": ["句子结构", "句式", "句型", "结构相似"],
        "拆分合并": ["拆分", "合并", "长句", "短句"],
        "论证顺序": ["论证", "论点", "逻辑", "组织"],
    }

    pattern_hits = Counter()
    for f in near_paraphrases:
        sug = f.get("suggestion", "")
        for pattern, keywords in suggestion_keywords.items():
            if any(kw in sug for kw in keywords):
                pattern_hits[pattern] += 1

    if pattern_hits:
        top_pattern = pattern_hits.most_common(1)[0]

        pattern_advice = {
            "同义词替换": (
                "你倾向于用同义词替换来改写，但保留了原文的句子骨架。"
                "更有效的方法是：读一段原文→合上书→用自己的话写出核心意思→"
                "再对照检查是否无意中复述了原文结构。"
            ),
            "语序": (
                "调整语序是好的开始，但如果只调语序不改变论证结构，"
                "仍然可能被判为相似。尝试打乱论点的呈现顺序，"
                "或者从结论倒推来重新组织段落。"
            ),
            "句子结构": (
                "你的改写改变了句子结构，这是好迹象。"
                "下一步可以尝试将多个来源的观点融合成一个新的论述，"
                "而不是逐篇改写。"
            ),
            "拆分合并": (
                "拆分合并句子是有效的改写技巧。"
                "更进一步：将原文的论证逻辑反转——比如原文是"
                "'因为A所以B'，你可以写成'要达到B不一定需要A，还可以通过C'。"
            ),
            "论证顺序": (
                "改变论证顺序是最彻底的改写方式。"
                "你已经在这方面做了尝试，继续用这个策略覆盖更多段落。"
            ),
        }

        return {
            "type": "writing_pattern",
            "severity": "tip",
            "title": "改写风格分析",
            "finding": (
                f"你的 {len(near_paraphrases)} 处近义改写中，"
                f"最常出现「{top_pattern[0]}」的模式。"
            ),
            "suggestion": pattern_advice.get(
                top_pattern[0],
                "建议在改写时更多关注论证结构的重组，而非仅仅改变表面措辞。"
            ),
            "data": {
                "near_paraphrase_count": len(near_paraphrases),
                "pattern_distribution": dict(pattern_hits),
            },
        }

    return None


# ── Citation hygiene check ─────────────────────────────────────────────

def _check_citation_hygiene(findings: list[dict]) -> Optional[dict]:
    """
    Check if high-similarity paragraphs contain citation markers.
    High similarity + no citation marker = potential academic integrity issue.
    """
    high_sim = [
        f for f in findings
        if f.get("similarity", 0) >= 0.80
        and f.get("duplicate_type") not in ("no_issue",)
    ]
    if not high_sim:
        return None

    missing_citations = []
    for f in high_sim:
        text = f.get("thesis_chunk", "")
        has_citation = any(
            re.search(pat, text) for pat in CITATION_PATTERNS
        )
        if not has_citation:
            missing_citations.append(f)

    if len(missing_citations) >= 2:
        return {
            "type": "citation_hygiene",
            "severity": "warning",
            "title": "高相似段落缺少引用标记",
            "finding": (
                f"在 {len(high_sim)} 处高相似度匹配中，"
                f"有 {len(missing_citations)} 处未检测到引用标记"
                f"（如'XXX指出''参见XXX'等）。"
            ),
            "suggestion": (
                "即使内容已改写，明确的引用标记仍是学术规范的要求。"
                "检查这些段落，确保标注了相应的文献来源。"
                "对于 verbatim 级别的相似度，直接引用+引号是最好选择。"
            ),
            "data": {
                "high_similarity_count": len(high_sim),
                "missing_citation_count": len(missing_citations),
                "affected_paragraphs": [
                    f.get("thesis_chunk", "")[:100] + "..."
                    for f in missing_citations[:3]
                ],
            },
        }

    return None


# ── Progress tracking ──────────────────────────────────────────────────

def _track_progress(
    author: str,
    current_stats: dict,
) -> Optional[dict]:
    """
    Compare current check against previous check for the same author.
    """
    from .store import get_author_profile

    profile = get_author_profile(author)
    if not profile:
        return None

    history = profile.get("content", {}).get("check_history", [])
    # Need at least 1 prior check to compare against (current check not yet saved)
    if len(history) < 1:
        return None

    prev = history[-1]["findings"]
    curr = current_stats

    prev_verbatim = prev.get("verbatim", 0)
    curr_verbatim = curr.get("verbatim", 0)
    prev_near = prev.get("near_paraphrase", 0)
    curr_near = curr.get("near_paraphrase", 0)

    insights = []

    # verbatim improvement
    if prev_verbatim > 0 and curr_verbatim < prev_verbatim:
        reduction = prev_verbatim - curr_verbatim
        insights.append(
            f"逐字重复从 {prev_verbatim} 处降至 {curr_verbatim} 处 "
            f"（减少 {reduction} 处），进步显著！"
        )
    elif curr_verbatim > 0 and prev_verbatim > 0:
        insights.append(
            f"逐字重复仍然存在（{curr_verbatim} 处），"
            "建议优先处理这些段落。"
        )

    # near_paraphrase pattern
    if curr_verbatim < prev_verbatim and curr_near >= prev_near:
        insights.append(
            "逐字重复减少了，但近义改写增加了——这说明你在努力改写，"
            "只是改写策略还比较单一。试试重组论证结构而非仅仅替换词汇。"
        )
    elif curr_near < prev_near and curr_verbatim == 0:
        insights.append(
            "近义改写也在持续减少，写作的独立性在稳步提升。继续保持这个方向！"
        )

    if not insights:
        return None

    return {
        "type": "progress_tracking",
        "severity": "positive",
        "title": "进步追踪",
        "finding": "与上次查重对比：" + "；".join(insights),
        "suggestion": (
            "每次改写后再次查重是很好的习惯。建议记录每次的改进策略，"
            "形成自己的写作方法论。"
        ),
        "data": {
            "previous": {
                "verbatim": prev_verbatim,
                "near_paraphrase": prev_near,
                "thesis": history[-1].get("thesis", ""),
            },
            "current": {
                "verbatim": curr_verbatim,
                "near_paraphrase": curr_near,
            },
        },
    }


# ── Paper recommendation ───────────────────────────────────────────────

def _generate_search_query(field: str, topics: list[str]) -> str:
    """Build a Semantic Scholar search query from field + topics."""
    if topics:
        return " ".join(topics[:3])
    return field


def _search_semantic_scholar(query: str, limit: int = 5) -> list[dict]:
    """
    Search Semantic Scholar API for relevant papers.
    Uses API key if configured, otherwise free tier (rate-limited).
    API docs: https://api.semanticscholar.org/api-docs/graph
    """
    api_key = Config.SEMANTIC_SCHOLAR_API_KEY
    encoded_query = urllib.parse.quote(query)
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={encoded_query}&limit={limit}"
        "&fields=title,authors,year,venue,citationCount,url,abstract"
    )

    headers = {"Accept": "application/json", "User-Agent": "PlagiarismCheckAgent/1.0"}
    if api_key:
        headers["x-api-key"] = api_key

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            papers = []
            for p in data.get("data", []):
                authors = [
                    a.get("name", "")
                    for a in p.get("authors", [])[:3]
                ]
                papers.append({
                    "title": p.get("title", ""),
                    "authors": authors,
                    "year": p.get("year", ""),
                    "venue": p.get("venue", ""),
                    "citations": p.get("citationCount", 0),
                    "url": p.get("url", ""),
                    "abstract": (p.get("abstract", "") or "")[:200],
                })
            return papers
    except (urllib.error.URLError, urllib.error.HTTPError,
            json.JSONDecodeError, OSError) as e:
        return []


def _suggest_readings(
    field: str,
    topics: list[str],
    reference_titles: list[str],
    findings: list[dict],
) -> list[dict]:
    """
    Generate reading suggestions based on field gaps and topic analysis.
    Returns structured suggestions with optional Semantic Scholar results.
    """
    suggestions = []

    # 1. Topic gap analysis: what does the thesis discuss that references don't cover?
    ref_keywords = set()
    for title in reference_titles:
        for word in re.findall(r'[\w一-鿿]{2,}', title.lower()):
            ref_keywords.add(word)

    thesis_topics_lower = {t.lower() for t in topics}
    gap_topics = thesis_topics_lower - ref_keywords
    if gap_topics and len(gap_topics) >= 2:
        gap_list = "、".join(list(gap_topics)[:3])
        suggestions.append({
            "topic": f"补充文献: {gap_list} 方向",
            "reason": (
                f"你的论文讨论了「{gap_list}」相关内容，"
                "但对照文献中没有覆盖这些主题的论文。建议补充该方向的文献。"
            ),
            "priority": "high",
        })

    # 2. Source concentration → suggest broader reading
    source_counter = Counter(
        f.get("source_paper", "") for f in findings
        if f.get("source_paper") and f.get("source_paper") != "unknown"
    )
    if len(source_counter) <= 2 and len(reference_titles) >= 4:
        unmatched = [
            t for t in reference_titles
            if t not in source_counter
        ]
        if len(unmatched) >= 2:
            suggestions.append({
                "topic": "利用已有文献",
                "reason": (
                    f"你有 {len(unmatched)} 篇对照文献在查重中未被匹配到"
                    "——它们可能是不同方向的文献，阅读它们可以拓宽你的论述视野。"
                    f"如：{'、'.join(unmatched[:2])}"
                ),
                "priority": "medium",
            })

    # 3. Semantic Scholar search (if available or free tier)
    if field != "未知领域" and topics:
        query = _generate_search_query(field, topics)
        papers = _search_semantic_scholar(query, limit=5)
        if papers:
            suggestions.append({
                "topic": f"Semantic Scholar 推荐: {query}",
                "reason": f"在 Semantic Scholar 上搜索「{query}」找到 {len(papers)} 篇高引论文：",
                "papers": papers,
                "priority": "medium",
            })

    return suggestions


# ── Main analysis entry point ──────────────────────────────────────────

def analyze_check_result(
    author: str = "",
    thesis_title: str = "",
    thesis_abstract: str = "",
    findings: Optional[list[dict]] = None,
    reference_papers: Optional[list[str]] = None,
    thesis_paragraphs: Optional[list[dict]] = None,
) -> dict:
    """
    Run all profile analyses on a completed plagiarism check.

    Args:
        author: Author name (for profile tracking)
        thesis_title: Thesis title
        thesis_abstract: Thesis abstract text (optional)
        findings: List of finding dicts from the Agent's analysis
        reference_papers: List of reference paper filenames/titles
        thesis_paragraphs: List of {index, text} dicts for all thesis paragraphs

    Returns:
        Dict with insights list, reading_suggestions list, and updated profile
    """
    findings = findings or []
    reference_papers = reference_papers or []

    # Extract field and topics
    field = extract_field(thesis_title, thesis_abstract)
    topics = extract_topics(thesis_title + " " + thesis_abstract)

    # Build stats
    current_stats = {
        "verbatim": sum(1 for f in findings if f.get("duplicate_type") == "verbatim"),
        "near_paraphrase": sum(1 for f in findings if f.get("duplicate_type") == "near_paraphrase"),
        "idea_overlap": sum(1 for f in findings if f.get("duplicate_type") == "idea_overlap"),
        "no_issue": sum(1 for f in findings if f.get("duplicate_type") == "no_issue"),
    }

    # Run all analyses
    insights = []
    analyses = [
        _analyze_source_concentration(findings, reference_papers),
        _diagnose_sections(findings, thesis_paragraphs),
        _analyze_writing_patterns(findings),
        _check_citation_hygiene(findings),
        _track_progress(author, current_stats) if author else None,
    ]

    for result in analyses:
        if result:
            insights.append(result)

    # Reading suggestions
    reference_titles = reference_papers or []
    reading_suggestions = _suggest_readings(
        field, topics, reference_titles, findings,
    )

    # Update author profile in memory store
    from .store import update_author_profile

    profile = update_author_profile(
        author=author or "未知作者",
        thesis_title=thesis_title,
        field=field,
        topics=topics,
        findings_summary=current_stats,
        insights=insights,
        reading_suggestions=reading_suggestions,
    )

    return {
        "field": field,
        "topics": topics,
        "insights": insights,
        "reading_suggestions": reading_suggestions,
        "profile": profile,
    }
