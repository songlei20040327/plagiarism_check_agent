---
name: plagiarism-check-report
description: |
  报告生成：整合所有精判发现，生成结构化 HTML 查重报告。
  定义报告内容规范和各重复类型的改写建议撰写指南。
---

# Plagiarism Check — Report

报告生成阶段。整合所有分析发现，生成包含摘要、逐条详情和改写建议的 HTML 报告。

## 报告内容规范

最终报告须包含：

1. **总体摘要**：各类型计数（verbatim / near_paraphrase / idea_overlap / no_issue）
2. **逐条详情**：每个发现的以下字段：
   - 论文原文（疑似重复段落）
   - 对照文献名称及匹配文本
   - 相似度分数
   - 重复类型
   - 改写建议
3. **免责声明**：标注查重范围仅限于用户上传的对照文献库

## findings 数组字段规范

调用报告生成工具时，findings 数组中每个元素须包含：

| 字段 | 必填 | 说明 |
|------|------|------|
| `thesis_chunk` | 是 | 论文中疑似重复的段落原文 |
| `source_paper` | 是 | 对照文献文件名 |
| `similarity` | 是 | 向量检索相似度分数（0-1） |
| `duplicate_type` | 是 | verbatim / near_paraphrase / idea_overlap / no_issue |
| `overlapping_text` | 是 | 对照文献中的匹配文本 |
| `suggestion` | 是 | 改写建议（no_issue 时可为空） |

## 改写建议撰写指南

针对每种重复类型的建议侧重点不同：

- **verbatim**：建议完全重写。给出具体的句式改写方向（如被动改主动、拆分长句、改变论证顺序），帮作者摆脱原文结构。避免只说"请重写"。
- **near_paraphrase**：建议调整句子结构和表达方式（如重新组织论点顺序、合并或拆分句子、更换关键术语），保留核心观点的同时改变表述。
- **idea_overlap**：建议加入自己的分析、批判或扩展，或明确引用来源。目的是让作者表达独立观点，而不只是复述。
- **no_issue**：无需建议。若相似度低且内容为常识性描述或独立表述，可标记为 no_issue。
