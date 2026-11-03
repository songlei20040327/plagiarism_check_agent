---
name: plagiarism-check-report
description: |
  整合基于真实检索证据的判断，生成结构化查重报告。
---

# Plagiarism Check — Report

完成所有段落检索和全部候选判断后，调用 `generate_report`：

- `thesis_title`：论文标题。
- `output_dir`：必须与任务上下文中给出的输出目录一致。
- `findings`：每个元素包含 `thesis_chunk_id`、`source_chunk_id`、`duplicate_type`、`suggestion`，可以提供 `analysis`。
- 一个论文块对应多个候选时，分别提交；同一候选不能重复提交。
- 所有候选必须有判断，包含 `no_issue`；全部段落检索均无匹配时可提交空数组。

来源原文、名称、向量分数由服务端从本次任务的检索记录填入，不能凭空填写。工具会验证覆盖率、证据、输出目录和生成文件。工具返回错误时修正后重试，不能宣称已经生成报告。只有工具明确成功后才能向用户报告完成。

建议聚焦独立表达、论证组织和规范引用，避免仅做机械同义替换。`no_issue` 的建议可以为空。
