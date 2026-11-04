"""Load repository-owned Markdown instructions and their references."""

from pathlib import Path
from typing import Optional


_PROJECT_SKILLS_DIR = Path(__file__).parent.parent / "skills"
SKILLS_DIR = _PROJECT_SKILLS_DIR


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
            content = self.load_skill(name, include_references=True)
            parts.append(f"===== {name} =====\n\n{content}")
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
