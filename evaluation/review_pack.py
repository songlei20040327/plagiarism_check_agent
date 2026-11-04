"""Collect real retrieval candidates and assemble a private human-review packet.

Run with the same Python environment as the application. No chat-model API is
called. Generated PDFs' excerpts and model caches must remain outside Git.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plagiarism_check_agent.config import Config
from plagiarism_check_agent.tools.document_tools import DocumentTools


LABELS = {"no_issue", "verbatim", "near_paraphrase", "idea_overlap", "uncertain"}
ATTRIBUTIONS = {"adequate", "missing_or_unclear", "not_applicable", "uncertain"}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def normalize(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_corpus(thesis, refs):
    tools = DocumentTools()
    meta = tools.parse_pdf(str(thesis))
    if "error" in meta:
        raise ValueError(meta["error"])
    tools.chunk_document(meta["doc_id"])
    references = []
    for path in sorted(refs.glob("*.pdf")):
        ref = tools.parse_pdf(str(path))
        if "error" in ref:
            raise ValueError(ref["error"])
        tools.chunk_document(ref["doc_id"])
        references.append(ref["doc_id"])
    return tools, meta["doc_id"], references


def location(tools, chunk, output):
    doc = tools._document_store[chunk["document_id"]]
    parts = [normalize(page["text"]) for page in doc["pages"]]
    full = "".join(parts)
    needle = normalize(chunk["text"])
    start = full.find(needle) if needle else -1
    if start < 0:
        raise ValueError(f"Cannot locate exact normalized chunk in source PDF: {chunk['chunk_id']}")
    end = start + len(needle)
    offset, pages = 0, []
    for number, text in enumerate(parts, 1):
        if offset < end and offset + len(text) > start:
            pages.append(number)
        offset += len(text)
    siblings = tools.get_document_chunks(chunk["document_id"])
    index = chunk["index"]
    metadata = doc["metadata"]
    return {
        "kind": "pdf", "text": chunk["text"], "chunk_id": chunk["chunk_id"],
        "chunk_index": index, "document_id": chunk["document_id"],
        "file_name": metadata["file_name"], "title": metadata.get("title") or metadata["file_name"],
        "pdf_sha256": metadata["sha256"], "pdf_pages": pages,
        "pdf_relative_path": os.path.relpath(metadata["file_path"], output),
        "location_method": "NFKC+whitespace removal; contiguous exact match; physical PDF pages",
        "context_before": siblings[index - 1]["text"] if index else "",
        "context_after": siblings[index + 1]["text"] if index + 1 < len(siblings) else "",
    }


def collect(args):
    if Config.RETRIEVAL_MODE != "hybrid_rerank":
        raise ValueError("Use RETRIEVAL_MODE=hybrid_rerank.")
    import torch
    torch.set_num_threads(args.threads)
    out = args.output.resolve()
    tools, thesis_id, refs = parse_corpus(args.thesis, args.refs)
    try:
        built = tools.build_corpus_index(refs, str(out / "vector_index"), [thesis_id])
        if "error" in built:
            raise ValueError(built["error"])
        print(f"Reference snapshot ready: {built['total_chunks_indexed']} chunks", flush=True)
        chunks = tools.get_document_chunks(thesis_id)
        plan = json.loads(args.plan.read_text())
        queries = []
        for spec in plan["queries"]:
            chunk = chunks[spec["index"]]
            result = tools.search_similar_chunks(chunk["text"], top_k=Config.TOP_K_RESULTS)
            matches = []
            for rank, match in enumerate(result["matches"], 1):
                source = tools._chunk_store[match["chunk_id"]]
                matches.append({"rank": rank, "source": location(tools, source, out),
                                "scores": match["scores"], "channels": match["retrieval_channels"]})
            queries.append({"query": location(tools, chunk, out), "purpose": spec["purpose"],
                            "take_ranks": spec["take_ranks"], "matches": matches,
                            "retrieval_diagnostics": result["retrieval_diagnostics"]})
            print(f"Collected thesis chunk {spec['index']}: {len(matches)} candidates", flush=True)
        dump(out / "pool.json", {
            "schema_version": 1, "snapshot_id": tools.snapshot_id,
            "retrieval_config": tools.retrieval_config(), "queries": queries,
            "all_reference_chunks": {key: location(tools, c, out) for key, c in tools._indexed_chunks.items()},
            "intended_use": "development review only; selected examples, not an unbiased benchmark",
        })
    finally:
        tools.close()


def validate_annotations(pack, submission):
    if submission.get("batch_id") != pack["batch_id"]:
        raise ValueError("Annotations belong to a different review batch.")
    if not isinstance(submission.get("annotations"), list):
        raise ValueError("Annotations must be a list.")
    cases = {case["id"]: case for case in pack["cases"]}
    seen = set()
    for row in submission.get("annotations", []):
        case_id = row.get("case_id")
        if case_id not in cases or case_id in seen or row.get("case_hash") != cases[case_id]["case_hash"]:
            raise ValueError("Unknown, duplicate or changed review case.")
        seen.add(case_id)
        if not isinstance(row.get("confirmed"), bool):
            raise ValueError("Each annotation needs an explicit confirmed boolean.")
        if row.get("decision", "") not in LABELS | {""} or row.get("attribution", "") not in ATTRIBUTIONS | {""}:
            raise ValueError("Unknown annotation label.")
        if row["confirmed"] and (not row.get("decision") or not row.get("attribution")):
            raise ValueError("Confirmed annotations require both labels.")
        if not isinstance(row.get("note", ""), str):
            raise ValueError("Annotation notes must be text.")
    return sum(row["confirmed"] for row in submission["annotations"])


def render_pages(cases, view):
    """Render only the selected evidence pages, never expose the corpus root."""
    import fitz
    images = view / "pages"
    images.mkdir(parents=True, exist_ok=True)
    rendered, verified = set(), set()
    for case in cases:
        for side in ("query", "source"):
            item = case[side]
            if item.get("kind") != "pdf":
                continue
            path = (view / item["pdf_relative_path"]).resolve()
            if path not in verified:
                if hashlib.sha256(path.read_bytes()).hexdigest() != item["pdf_sha256"]:
                    raise ValueError("PDF changed after evidence collection; collect a new batch.")
                verified.add(path)
            item["page_images"] = []
            for page in item["pdf_pages"]:
                relative = f"pages/{item['pdf_sha256'][:16]}-p{page:03d}.png"
                item["page_images"].append({"page": page, "path": relative})
                if relative in rendered:
                    continue
                with fitz.open(path) as document:
                    document[page - 1].get_pixmap(matrix=fitz.Matrix(1.3, 1.3), alpha=False).save(view / relative)
                rendered.add(relative)
    return len(rendered)


def assemble(args):
    out = args.output.resolve()
    view = out / "view"
    view.mkdir(parents=True, exist_ok=True)
    pool = json.loads((out / "pool.json").read_text())
    spec = json.loads(args.spec.read_text())
    cases = []
    for query in pool["queries"]:
        for match in query["matches"]:
            if match["rank"] in query["take_ranks"]:
                cases.append({"origin": "natural", "purpose": f"{query['purpose']} · 候选 {match['rank']}",
                              "query": query["query"], **match})
    cases.extend(spec["controls"])
    for index, case in enumerate(cases, 1):
        case["id"] = f"R{index:02d}"
        case.update(spec.get("overrides", {}).get(case["id"], {}))
        for side in ("query", "source"):
            case[side] = dict(case[side])
            if case[side].get("kind") == "pdf":
                case[side]["pdf_relative_path"] = "../" + case[side]["pdf_relative_path"]
        case["advice"] = spec["advice"][case["id"]]
    if getattr(args, "render_pages", False):
        print(f"Rendered evidence pages: {render_pages(cases, view)}", flush=True)
    for case in cases:
        case["case_hash"] = digest({k: v for k, v in case.items() if k not in {"advice", "case_hash"}})
    pack = {
        "schema_version": 1, "batch_id": "review-" + digest([c["case_hash"] for c in cases])[:16],
        "created_at": datetime.now(timezone.utc).isoformat(), "cases": cases,
        "snapshot_id": pool["snapshot_id"], "retrieval_config": pool["retrieval_config"],
        "intended_use": pool["intended_use"], "policy": spec["policy"],
    }
    dump(out / "review_pack.json", pack)
    template = Path(__file__).with_name("review_ui.html").read_text(encoding="utf-8")
    payload = json.dumps(pack, ensure_ascii=False).replace("<", "\\u003c")
    (view / "review.html").write_text(template.replace("__PACK_JSON__", payload), encoding="utf-8")
    dump(out / "annotations_blank.json", {
        "schema_version": 1, "batch_id": pack["batch_id"], "reviewer": "",
        "annotations": [{"case_id": c["id"], "case_hash": c["case_hash"], "decision": "",
                         "attribution": "", "note": "", "confirmed": False} for c in cases],
    })
    print(json.dumps({"batch_id": pack["batch_id"], "cases": len(cases), "html": str(view / "review.html")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--thesis", type=Path, required=True)
    collect_parser.add_argument("--refs", type=Path, required=True)
    collect_parser.add_argument("--plan", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--threads", type=int, default=4)
    collect_parser.set_defaults(run=collect)
    assemble_parser = commands.add_parser("assemble")
    assemble_parser.add_argument("--output", type=Path, required=True)
    assemble_parser.add_argument("--spec", type=Path, required=True)
    assemble_parser.add_argument("--render-pages", action="store_true")
    assemble_parser.set_defaults(run=assemble)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--pack", type=Path, required=True)
    validate_parser.add_argument("--annotations", type=Path, required=True)
    validate_parser.set_defaults(run=lambda a: print(f"Confirmed: {validate_annotations(json.loads(a.pack.read_text()), json.loads(a.annotations.read_text()))}"))
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
