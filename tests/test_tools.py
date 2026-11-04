"""Basic unit tests for document and report tools."""

import json
import tempfile
from pathlib import Path

import pytest

# These tests don't require a real PDF or LLM — they test the pure logic


class TestToolRouter:
    def test_unknown_tool(self):
        from plagiarism_check_agent.agent.tool_router import ToolRouter
        router = ToolRouter(dry_run=True)
        result = router.execute("nonexistent_tool", {})
        assert "error" in result

    def test_dry_run_blocks_write(self):
        from plagiarism_check_agent.agent.tool_router import ToolRouter
        router = ToolRouter(dry_run=True)
        result = router.execute("generate_report", {
            "thesis_title": "test",
            "findings": [],
            "output_dir": "/tmp",
        })
        assert result.get("dry_run") is True


class TestConfig:
    def test_defaults(self):
        from plagiarism_check_agent.config import Config
        assert Config.SIMILARITY_THRESHOLD == 0.75
        assert Config.CHUNK_SIZE == 512
        assert Config.TOP_K_RESULTS == 5


class TestDocumentToolsErrorHandling:
    def test_parse_nonexistent_pdf(self):
        from plagiarism_check_agent.tools.document_tools import DocumentTools
        dt = DocumentTools()
        result = dt.parse_pdf("/nonexistent/file.pdf")
        assert "error" in result

    def test_chunk_unknown_document(self):
        from plagiarism_check_agent.tools.document_tools import DocumentTools
        dt = DocumentTools()
        result = dt.chunk_document("nonexistent_id")
        assert "error" in result

    def test_corpus_stats_empty(self):
        from plagiarism_check_agent.tools.document_tools import DocumentTools
        dt = DocumentTools()
        result = dt.get_corpus_stats()
        assert result["documents_parsed"] == 0
        assert result["total_chunks"] == 0


class TestReportTools:
    def test_generate_report_basic(self):
        from plagiarism_check_agent.tools.report_tools import ReportTools
        rt = ReportTools()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = rt.generate_report(
                thesis_title="测试论文",
                findings=[
                    {
                        "thesis_chunk": "这是一段测试文本",
                        "source_paper": "参考论文A",
                        "similarity": 0.92,
                        "duplicate_type": "verbatim",
                        "overlapping_text": "这是一段相似的文本",
                        "suggestion": "建议改写为...",
                    },
                    {
                        "thesis_chunk": "另一段独立文本",
                        "source_paper": "参考论文B",
                        "similarity": 0.60,
                        "duplicate_type": "no_issue",
                        "overlapping_text": "",
                        "suggestion": "",
                    },
                ],
                output_dir=tmpdir,
            )

            assert result["status"] == "ok"
            assert result["total_findings"] == 2
            assert result["verbatim"] == 1
            assert result["no_issue"] == 1
            assert Path(result["html_report"]).exists()
            assert Path(result["json_data"]).exists()

            # Verify HTML content
            html = Path(result["html_report"]).read_text(encoding="utf-8")
            assert "测试论文" in html
            assert "逐字重复" in html
            assert "参考论文A" in html

    @pytest.mark.parametrize('label', ['', None, 'made_up'])
    def test_report_never_classifies_from_similarity(self, tmp_path, label):
        from plagiarism_check_agent.tools.report_tools import ReportTools
        with pytest.raises(ValueError, match='explicit valid duplicate_type'):
            ReportTools().generate_report('Explicit judgments required', [{
                'thesis_chunk': 'Copied wording', 'source_paper': 'A', 'similarity': .99,
                'duplicate_type': label, 'overlapping_text': 'Copied wording', 'suggestion': '',
            }], str(tmp_path))


class TestSkillLoader:
    def test_load_plagiarism_check_skill(self):
        from plagiarism_check_agent.agent.skill_loader import SkillLoader
        loader = SkillLoader()
        prompt = loader.load_skill("plagiarism-check")
        assert len(prompt) > 100
        assert "查重" in prompt

    def test_load_combined_skills(self):
        from plagiarism_check_agent.agent.skill_loader import SkillLoader
        loader = SkillLoader()
        prompt = loader.load_combined(
            "plagiarism-check",
            "plagiarism-check-analyze",
            "plagiarism-check-report",
        )
        assert len(prompt) > 500
