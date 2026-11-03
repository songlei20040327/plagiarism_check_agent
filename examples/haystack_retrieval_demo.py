"""Print each retrieval stage using real local models and synthetic papers.

Run from the repository with:
    python examples/haystack_retrieval_demo.py

This example performs no LLM API calls and makes no plagiarism verdict. Model
weights may download on first use. The temporary PDFs/index are deleted on exit.
"""
import sys
import tempfile
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plagiarism_check_agent.config import Config
from plagiarism_check_agent.tools.document_tools import DocumentTools


def main():
    if Config.RETRIEVAL_MODE != "hybrid_rerank":
        raise ValueError("Use RETRIEVAL_MODE=hybrid_rerank for this example.")
    query = "通过调节风机转速，保持呼吸机输出气道压力稳定。"
    papers = {
        "ref_pressure": ("压力控制参考文献", query),
        "ref_paraphrase": ("通气系统参考文献", "控制风机的旋转速度能够维持呼吸机的气道压力稳定。"),
        "ref_marine": ("海洋生态参考文献", "海洋浮游生物的季节变化影响生态系统结构。"),
    }
    tools = DocumentTools()
    try:
        with tempfile.TemporaryDirectory(prefix="haystack-retrieval-demo-") as directory:
            root = Path(directory)

            def add(name, title, text):
                path = root / (name + ".pdf")
                with fitz.open() as pdf:
                    pdf.set_metadata({"title": title})
                    page = pdf.new_page()
                    page.insert_textbox(fitz.Rect(40, 40, 555, 780), text,
                                        fontname="china-s", fontsize=11)
                    pdf.save(path)
                meta = tools.parse_pdf(str(path))
                if "error" in meta:
                    raise ValueError(meta["error"])
                tools.chunk_document(meta["doc_id"])
                return meta["doc_id"]

            thesis = add("thesis", "待检论文", query)
            references = [add(name, title, text) for name, (title, text) in papers.items()]
            built = tools.build_corpus_index(references, str(root / "index"), [thesis])
            if "error" in built:
                raise ValueError(built["error"])
            text = tools.get_document_chunks(thesis)[0]["text"]
            result = tools.search_similar_chunks(text)
            if "error" in result:
                raise ValueError(result["error"])
            diagnostic = result["retrieval_diagnostics"]

            def titles(ids):
                return [tools._indexed_chunks[key]["title"] for key in ids]

            print("\n查询文字：", text)
            print("向量候选：", titles(diagnostic["dense_ids"]))
            print("BM25 候选：", titles(diagnostic["bm25_ids"]))
            print("去重后候选数：", len(diagnostic["union_ids"]))
            print("Cross-Encoder 重排结果：")
            for position, match in enumerate(result["matches"], 1):
                print(f"  {position}. {match['source_paper']} | "
                      f"向量={match['scores']['dense']:.4f} | "
                      f"BM25={match['scores']['bm25']:.4f} | "
                      f"重排={match['scores']['reranker']:.4f}")
            print("以上均为检索相关性分数；重复类型仍需结合原文和引用判断。")
    finally:
        tools.close()


if __name__ == "__main__":
    main()
