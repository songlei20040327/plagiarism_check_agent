---
name: plagiarism-check-analyze
description: |
  查重分析阶段：逐段检索对照文献，批量向量搜索 + LLM 精判分类。
  定义相似度分级标准和精判输出格式。
---

# Plagiarism Check — Analyze

查重分析阶段。对论文每个段落进行向量检索和 LLM 精判，输出结构化的查重发现。

## 相似度分级标准

| 分数范围 | 分类 | 含义 |
|----------|------|------|
| ≥ 0.90 | **verbatim**（逐字重复） | 文字几乎完全相同，或仅有极微小的措辞调整 |
| 0.82 – 0.90 | **near_paraphrase**（近义改写） | 结构高度相似，使用了同义词替换或语序调整但核心表述雷同 |
| 0.75 – 0.82 | **idea_overlap**（观点重复） | 观点或论证思路接近，但表述方式有实质差异 |
| < 0.75 | 跳过 | 低于阈值，不提交精判 |

## 批量搜索策略

- 每批传入 3-5 个段落到搜索工具，减少调用轮数
- 仅对 similarity ≥ 0.75 的匹配结果获取上下文并进行 LLM 精判
- 如果一个段落匹配到多篇对照文献，分别分析

## 精判输出格式

对每个 similarity ≥ 0.75 的候选，须输出：

```json
{
  "thesis_chunk": "论文段落原文",
  "source_paper": "对照文献名称",
  "similarity": 0.89,
  "duplicate_type": "verbatim | near_paraphrase | idea_overlap | no_issue",
  "overlapping_text": "对照文献中的匹配文本",
  "suggestion": "具体的改写建议",
  "analysis": "判定理由（1-2句话）"
}
```

## 精判要点

- **verbatim**：看文字重合度，而非观点相似度。即使两段讨论同一话题，只要表述方式不同就不算 verbatim。
- **near_paraphrase**：看句子结构和论证顺序。同义词替换+语序微调=近义改写。
- **idea_overlap**：看表述的独立性。即使讨论同一个研究方向，只要用自己的语言重新组织就算独立表述。
- **no_issue**：常识性内容、领域通用知识、或完全独立的表述。
