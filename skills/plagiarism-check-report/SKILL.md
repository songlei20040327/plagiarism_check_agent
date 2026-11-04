---
name: plagiarism-check-report
description: 提交当前批次基于完整原文证据的结构化判断。
---

# 当前批次判断提交

准备模式下，读取来源上下文后，在下一轮调用 `submit_findings`。

- `findings` 必须覆盖当前批次的每一个候选，包括 `no_issue`。
- 每项填写 `thesis_chunk_id`、`source_chunk_id`、`duplicate_type`、`suggestion`，可以提供 `analysis`。
- 不提交其他批次或脚本自动处理的候选，不编造或修改原文、来源与分数。
- 工具报错时，根据具体错误修正；工具成功只表示本批判断被接受。
- 脚本会合并各批判断及自动结论，统一生成最终HTML和JSON报告。
- 不能宣称本批提交已经完成整篇论文的报告。

建议聚焦独立表达、论证组织和规范引用。`no_issue` 的建议可以为空。
