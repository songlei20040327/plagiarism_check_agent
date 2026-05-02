"""
SkillLoader: reads SKILL.md files and builds system prompts dynamically.

Supports auto-discovery of sub-skills and dynamic tool list injection,
so SKILL.md files don't need to hardcode tool names.

## Skill 架构（三层隔离的 Skills 层）

三个技能各自独立维护，加载时动态拼接为一个完整的 system prompt：

1. **plagiarism-check**（主技能 — 轻量编排器）
   - 定义整体使命、核心原则、pipeline 概览
   - 告知 Agent 文档已解析+索引已构建（直接开始分析）
   - 不包含具体的分析分类标准或报告格式规范

2. **plagiarism-check-analyze**（分析子技能 — 自包含能力模块）
   - 定义相似度分级标准（verbatim/near_paraphrase/idea_overlap 的阈值和含义）
   - 定义批量搜索策略和 LLM 精判输出格式
   - 所有"怎么分析"的内容只在这里定义一次

3. **plagiarism-check-report**（报告子技能 — 自包含能力模块）
   - 定义报告内容规范和 findings 数组字段 schema
   - 定义各重复类型的改写建议撰写指南
   - 所有"怎么生成报告"的内容只在这里定义一次

## 动态提示词构建流程

```
SkillLoader.load_with_sub_skills("plagiarism-check")
  ├── 扫描 skills/ 目录，发现 plagiarism-check-* 子目录
  ├── load_skill("plagiarism-check")           → 主技能内容
  ├── load_skill("plagiarism-check-analyze")   → 分析子技能内容
  ├── load_skill("plagiarism-check-report")    → 报告子技能内容
  └── SkillLoader.inject_tools_summary(...)    → 注入 ToolRegistry 工具列表

最终 system_prompt = 三个 SKILL.md 拼接 + 动态工具列表
```

添加新子技能只需在 skills/ 下创建 `{主技能名}-{子技能名}/SKILL.md`，
无需修改 runner.py 或 skill_loader.py 中的任何代码。
"""

from pathlib import Path
from typing import Optional


_PROJECT_SKILLS_DIR = Path(__file__).parent.parent / "skills"
SKILLS_DIR = (
    _PROJECT_SKILLS_DIR
    if _PROJECT_SKILLS_DIR.exists()
    else Path.home() / ".claude" / "skills"
)


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter (--- ... ---) from skill markdown."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].lstrip("\n")
    return text


class SkillLoader:
    def __init__(self, skills_dir: Optional[Path] = None):
        self.skills_dir = skills_dir or SKILLS_DIR

    # ── Basic loading ─────────────────────────────────────────────────────

    def load_skill(self, skill_name: str, include_references: bool = True) -> str:
        skill_dir = self.skills_dir / skill_name
        skill_md_path = skill_dir / "SKILL.md"
        if not skill_md_path.exists():
            raise FileNotFoundError(f"Skill not found: {skill_md_path}")
        body = _strip_frontmatter(skill_md_path.read_text(encoding="utf-8"))
        if not include_references:
            return body
        refs_dir = skill_dir / "references"
        ref_sections = []
        if refs_dir.exists():
            for ref_file in sorted(refs_dir.glob("*.md")):
                ref_content = ref_file.read_text(encoding="utf-8")
                ref_sections.append(
                    f"===== Reference: {ref_file.name} =====\n\n{ref_content}"
                )
        if ref_sections:
            return body + "\n\n" + "\n\n".join(ref_sections)
        return body

    def load_combined(self, *skill_names: str) -> str:
        parts = []
        for name in skill_names:
            try:
                content = self.load_skill(name, include_references=True)
                parts.append(f"===== {name} =====\n\n{content}")
            except FileNotFoundError:
                pass
        return "\n\n".join(parts)

    # ── Auto-discovery ────────────────────────────────────────────────────

    def discover_sub_skills(self, main_skill_name: str) -> list[str]:
        """
        Auto-discover sub-skills by scanning for directories matching
        '{main_skill_name}-*' under the skills directory.

        Returns sorted list of sub-skill names (deterministic order).
        """
        sub_skills = []
        prefix = f"{main_skill_name}-"
        if self.skills_dir.exists():
            for d in sorted(self.skills_dir.iterdir()):
                if d.is_dir() and d.name.startswith(prefix):
                    # Verify it has a SKILL.md
                    if (d / "SKILL.md").exists():
                        sub_skills.append(d.name)
        return sub_skills

    def load_with_sub_skills(self, main_skill_name: str) -> str:
        """
        Load main skill + all auto-discovered sub-skills.
        No hardcoded list — sub-skills are discovered by directory scanning.

        Adding a new sub-skill = creating skills/{main}-{newsub}/SKILL.md.
        No code changes needed.
        """
        sub_skills = self.discover_sub_skills(main_skill_name)
        all_skills = [main_skill_name] + sub_skills
        return self.load_combined(*all_skills)

    # ── Tool list injection ───────────────────────────────────────────────

    @staticmethod
    def inject_memory_context(system_prompt: str, max_tokens: int = 2000) -> str:
        """
        Inject accumulated domain knowledge (common knowledge, false positives,
        field baselines) from plagiarism memory into the system prompt.
        """
        try:
            from ..memory.store import get_memory_summary

            memory_text = get_memory_summary(max_tokens)
            if memory_text:
                return (
                    f"{system_prompt}\n\n---\n"
                    f"## 积累的领域知识\n{memory_text}"
                )
        except Exception:
            pass
        return system_prompt

    @staticmethod
    def inject_tools_summary(system_prompt: str) -> str:
        """
        Append the current tool list (from ToolRegistry) to the system prompt.
        This way SKILL.md never needs to mention specific tool names —
        the LLM learns about available tools from the injected list.

        The tool descriptions are always in sync with the actual registered tools.
        """
        from ..tools.registry import ToolRegistry

        tools_summary = ToolRegistry.get_tools_summary()
        return f"{system_prompt}\n\n---\n\n{tools_summary}"
