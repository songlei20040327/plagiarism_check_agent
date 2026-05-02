#!/usr/bin/env python3
"""
Plagiarism Check Agent — 自动化评估脚本

对比 3 种方案的查重分类效果：
  A. 纯 Embedding   — 只用余弦相似度 + 阈值分类
  B. 纯 LLM         — 把两段文字直接给 LLM 判断
  C. Hybrid (Agent) — Embedding 检索 + LLM 精判（你的系统）

用法:
  python eval.py                          # 运行全部 3 个方案
  python eval.py --method embedding       # 只跑 embedding
  python eval.py --method llm             # 只跑 LLM
  python eval.py --method hybrid          # 只跑 hybrid

环境变量:
  OPENAI_API_KEY   LLM API key (方案 B, C 需要)
  OPENAI_MODEL     模型名 (默认 deepseek-chat)
  OPENAI_BASE_URL  API 地址
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

LABEL_MAP = {
    "verbatim": 0,
    "near_paraphrase": 1,
    "idea_overlap": 2,
    "no_issue": 3,
}
LABEL_NAMES = ["verbatim", "near_paraphrase", "idea_overlap", "no_issue"]


def load_dataset(dataset_path: str = "tests/eval_dataset.json") -> dict:
    with open(dataset_path, encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Compute per-class precision, recall, f1 + macro/weighted averages."""
    from collections import defaultdict

    metrics = {}
    n_classes = len(LABEL_NAMES)

    for cls_idx, cls_name in enumerate(LABEL_NAMES):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls_idx and p == cls_idx)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls_idx and p == cls_idx)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls_idx and p != cls_idx)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[cls_name] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": sum(1 for t in y_true if t == cls_idx),
        }

    # Macro average
    metrics["macro_avg"] = {
        "precision": round(np.mean([m["precision"] for m in metrics.values() if isinstance(m, dict) and "precision" in m]), 3),
        "recall": round(np.mean([m["recall"] for m in metrics.values() if isinstance(m, dict) and "recall" in m]), 3),
        "f1": round(np.mean([m["f1"] for m in metrics.values() if isinstance(m, dict) and "f1" in m]), 3),
    }

    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    metrics["accuracy"] = round(correct / len(y_true), 3) if y_true else 0.0

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 方案 A: 纯 Embedding（只用余弦相似度 + 阈值分类）
# ═══════════════════════════════════════════════════════════════════════════════

def eval_embedding_only(dataset: dict) -> tuple[list[int], list[int], float]:
    """
    使用数据集内置的最优阈值（由网格搜索在标注集上校准）。
    若数据集未提供，回退到默认阈值。
    """
    from sentence_transformers import SentenceTransformer

    print("\n" + "=" * 60)
    print("方案 A: 纯 Embedding (余弦相似度 + 校准阈值)")
    print("=" * 60)

    # Use calibrated thresholds from dataset, or defaults
    opt = dataset.get("optimal_thresholds_embedding_only", {})
    t_verb = opt.get("verbatim", 0.88)
    t_near = opt.get("near_paraphrase", 0.80)
    t_idea = opt.get("idea_overlap", 0.72)
    print(f"阈值: verbatim>={t_verb}, near_paraphrase>={t_near}, idea_overlap>={t_idea}")

    model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
    print(f"Embedding 模型: BAAI/bge-large-zh-v1.5")

    pairs = dataset["pairs"]
    thesis_texts = [p["thesis_chunk"] for p in pairs]
    ref_texts = [p["ref_chunk"] for p in pairs]
    y_true = [LABEL_MAP[p["ground_truth"]] for p in pairs]

    thesis_embs = model.encode(thesis_texts, normalize_embeddings=True,
                                show_progress_bar=True)
    ref_embs = model.encode(ref_texts, normalize_embeddings=True,
                             show_progress_bar=True)

    y_pred = []
    start_time = time.time()
    for t_emb, r_emb in zip(thesis_embs, ref_embs):
        sim = float(np.dot(t_emb, r_emb))
        if sim >= t_verb:
            y_pred.append(LABEL_MAP["verbatim"])
        elif sim >= t_near:
            y_pred.append(LABEL_MAP["near_paraphrase"])
        elif sim >= t_idea:
            y_pred.append(LABEL_MAP["idea_overlap"])
        else:
            y_pred.append(LABEL_MAP["no_issue"])
    elapsed = time.time() - start_time

    return y_true, y_pred, elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# 方案 B: 纯 LLM
# ═══════════════════════════════════════════════════════════════════════════════

LLM_JUDGE_PROMPT = """你是一个学术论文查重专家。判断以下论文段落与对照文献段落是否存在重复。

## 分类标准
- verbatim（逐字重复）：连续 30+ 字几乎完全相同
- near_paraphrase（近义改写）：核心内容高度相似，用词略有替换但句子结构相似
- idea_overlap（观点重复）：同一观点但表述方式有差异
- no_issue（无问题）：内容实质不同，或属于公共知识/正常引用

## 输入
【论文章节】
{thesis_chunk}

【对照文献段落】
{ref_chunk}

## 输出格式
只输出一个 JSON 对象，不要其他内容：
{{"label": "verbatim|near_paraphrase|idea_overlap|no_issue", "reason": "20字以内判定理由"}}
"""


def eval_llm_only(dataset: dict) -> tuple[list[int], list[int], float]:
    from openai import OpenAI
    import os

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    model = os.environ.get("OPENAI_MODEL", "deepseek-chat")

    if not api_key:
        print("错误: 请设置 OPENAI_API_KEY 环境变量")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"方案 B: 纯 LLM ({model})")
    print("=" * 60)

    client = OpenAI(api_key=api_key, base_url=base_url)
    pairs = dataset["pairs"]
    y_true = [LABEL_MAP[p["ground_truth"]] for p in pairs]

    y_pred = []
    start_time = time.time()

    for i, pair in enumerate(pairs):
        prompt = LLM_JUDGE_PROMPT.format(
            thesis_chunk=pair["thesis_chunk"][:600],
            ref_chunk=pair["ref_chunk"][:600],
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            # Parse JSON from response (handle markdown code blocks)
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            label = data.get("label", "no_issue")
            if label not in LABEL_MAP:
                label = "no_issue"
            y_pred.append(LABEL_MAP[label])
        except Exception as e:
            print(f"  [{i+1}/{len(pairs)}] LLM 调用失败: {e}, 默认 no_issue")
            y_pred.append(LABEL_MAP["no_issue"])

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(pairs)}] 已完成 (预测: {LABEL_NAMES[y_pred[-1]]})")

    elapsed = time.time() - start_time
    return y_true, y_pred, elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# 方案 C: Hybrid — Embedding 初筛 + LLM 精判（你的系统）
# ═══════════════════════════════════════════════════════════════════════════════

HYBRID_PROMPT = """你是一个学术论文查重专家。Embedding 检索已标记以下文本对的相似度为 {similarity}。
请精判最终分类。

## 分类标准
- verbatim（逐字重复）：连续 30+ 字几乎完全相同
- near_paraphrase（近义改写）：核心内容高度相似，用词略有替换，句子结构相似
- idea_overlap（观点重复）：同一观点但表述方式不同
- no_issue（无问题）：内容实质不同

## 文本对

【论文章节】
{thesis_chunk}

【对照文献段落】
{ref_chunk}

【Embedding 初步判定】similarity={similarity}

## 输出
只输出一个 JSON 对象：
{{"label": "verbatim|near_paraphrase|idea_overlap|no_issue", "reason": "20字以内判定理由"}}
"""


def eval_hybrid(dataset: dict) -> tuple[list[int], list[int], float, int]:
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer
    import os

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    model = os.environ.get("OPENAI_MODEL", "deepseek-chat")

    if not api_key:
        print("错误: 请设置 OPENAI_API_KEY 环境变量")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"方案 C: Hybrid — Embedding 初筛 + LLM 精判 ({model})")
    print("=" * 60)

    # Step 1: Embedding
    emb_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
    print("Step 1/2: Embedding 检索...")
    client = OpenAI(api_key=api_key, base_url=base_url)
    pairs = dataset["pairs"]
    y_true = [LABEL_MAP[p["ground_truth"]] for p in pairs]

    thesis_texts = [p["thesis_chunk"] for p in pairs]
    ref_texts = [p["ref_chunk"] for p in pairs]
    thesis_embs = emb_model.encode(thesis_texts, normalize_embeddings=True,
                                    show_progress_bar=True)
    ref_embs = emb_model.encode(ref_texts, normalize_embeddings=True,
                                 show_progress_bar=True)

    y_pred = []
    llm_calls = 0
    start_time = time.time()

    print(f"Step 2/2: LLM 精判 ({len(pairs)} 对)...")
    for i, (pair, t_emb, r_emb) in enumerate(zip(pairs, thesis_embs, ref_embs)):
        sim = float(np.dot(t_emb, r_emb))

        # Only call LLM for borderline/suspicious cases
        # Clear verbatim (sim > 0.95) and clear no_issue (sim < 0.50) skip LLM
        if sim >= 0.95:
            y_pred.append(LABEL_MAP["verbatim"])
            label_str = "verbatim (skip LLM)"
        elif sim < 0.50:
            y_pred.append(LABEL_MAP["no_issue"])
            label_str = "no_issue (skip LLM)"
        else:
            llm_calls += 1
            prompt = HYBRID_PROMPT.format(
                similarity=round(sim, 3),
                thesis_chunk=pair["thesis_chunk"][:600],
                ref_chunk=pair["ref_chunk"][:600],
            )
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.0,
                )
                text = resp.choices[0].message.content.strip()
                if "```" in text:
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                data = json.loads(text)
                label = data.get("label", "no_issue")
                if label not in LABEL_MAP:
                    label = "no_issue"
                y_pred.append(LABEL_MAP[label])
                label_str = f"{label}"
            except Exception as e:
                # Fallback to calibrated embedding thresholds
                opt = dataset.get("optimal_thresholds_embedding_only", {})
                tv = opt.get("verbatim", 0.88)
                tn = opt.get("near_paraphrase", 0.80)
                ti = opt.get("idea_overlap", 0.72)
                if sim >= tv:
                    y_pred.append(LABEL_MAP["verbatim"])
                elif sim >= tn:
                    y_pred.append(LABEL_MAP["near_paraphrase"])
                elif sim >= ti:
                    y_pred.append(LABEL_MAP["idea_overlap"])
                else:
                    y_pred.append(LABEL_MAP["no_issue"])
                label_str = f"fallback (LLM error: {e})"

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(pairs)}] sim={sim:.3f} → {label_str}")

    elapsed = time.time() - start_time
    return y_true, y_pred, elapsed, llm_calls


# ═══════════════════════════════════════════════════════════════════════════════
# 结果输出
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(method_name: str, y_true: list[int], y_pred: list[int],
                  elapsed: float):
    metrics = compute_metrics(y_true, y_pred)

    print(f"\n{'─' * 60}")
    print(f"📊 {method_name}")
    print(f"{'─' * 60}")
    print(f"耗时: {elapsed:.1f}s")
    print(f"准确率: {metrics['accuracy']:.1%}")
    print()
    print(f"{'类别':<20} {'精确率':>8} {'召回率':>8} {'F1':>8} {'样本数':>6}")
    print(f"{'─' * 52}")

    for cls_name in LABEL_NAMES:
        m = metrics[cls_name]
        print(f"{cls_name:<20} {m['precision']:>8.3f} {m['recall']:>8.3f} "
              f"{m['f1']:>8.3f} {m['support']:>6}")

    print(f"{'─' * 52}")
    ma = metrics["macro_avg"]
    print(f"{'Macro Avg':<20} {ma['precision']:>8.3f} {ma['recall']:>8.3f} "
          f"{ma['f1']:>8.3f}")

    return metrics


def print_comparison(results: dict):
    """Print a side-by-side comparison table."""
    print("\n" + "=" * 70)
    print("🏆 方案对比")
    print("=" * 70)

    methods = list(results.keys())
    header = f"{'指标':<15}"
    for m in methods:
        header += f" {m:>15}"
    print(header)
    print(f"{'─' * 15}{'─' * (16 * len(methods))}")

    for metric_name in ["accuracy"] + LABEL_NAMES + ["macro_avg"]:
        row = f"{metric_name:<15}"
        for m in methods:
            if metric_name == "accuracy":
                val = results[m].get("accuracy", 0)
            elif metric_name == "macro_avg":
                val = results[m]["macro_avg"]["f1"]
            else:
                val = results[m][metric_name]["f1"]
            # Highlight best
            row += f" {val:>15.3f}"
        print(row)

    print(f"\n💰 成本估算 (DeepSeek-V3, ~¥1/百万tokens):")
    for m in methods:
        n_pairs = results[m].get("_n_pairs", 0)
        if m == "Embedding":
            print(f"  {m}: 免费 (本地)")
        elif m == "LLM":
            tokens_per = 600  # ~2400 chars input + 100 output
            cost = n_pairs * tokens_per * 1 / 1_000_000
            print(f"  {m}: ~¥{cost:.4f} ({n_pairs} 对 × ~{tokens_per} tokens)")
        elif m == "Hybrid":
            tokens_per = 600
            llm_calls = results[m].get("_llm_calls", n_pairs)
            cost = llm_calls * tokens_per * 1 / 1_000_000
            print(f"  {m}: ~¥{cost:.4f} ({llm_calls}/{n_pairs} 对需LLM, "
                  f"其余被embedding预筛)")


def score_trajectory_quality(trace_path: str) -> float:
    """
    Score a trajectory file for DPO pairing purposes.

    Uses the same quality_score function as trajectory/exporter.py.
    Returns a float score (higher = better).
    """
    from plagiarism_check_agent.trajectory.exporter import (
        _load_trace, _get_session_stats, quality_score,
    )

    records = _load_trace(Path(trace_path))
    stats = _get_session_stats(records)
    return quality_score(stats, records)


def main():
    parser = argparse.ArgumentParser(description="Plagiarism Check Agent 评估")
    parser.add_argument("--method", choices=["embedding", "llm", "hybrid", "all"],
                        default="all", help="评估方案 (默认: all)")
    parser.add_argument("--dataset", default="tests/eval_dataset.json",
                        help="数据集路径")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    print(f"数据集: {args.dataset}")
    print(f"样本数: {dataset['total_pairs']}")
    print(f"分布: {dataset['distribution']}")

    results = {}

    if args.method in ("embedding", "all"):
        y_true, y_pred, elapsed = eval_embedding_only(dataset)
        results["Embedding"] = print_results("方案 A: 纯 Embedding",
                                              y_true, y_pred, elapsed)
        results["Embedding"]["_n_pairs"] = len(y_true)

    if args.method in ("llm", "all"):
        y_true, y_pred, elapsed = eval_llm_only(dataset)
        results["LLM"] = print_results("方案 B: 纯 LLM",
                                        y_true, y_pred, elapsed)
        results["LLM"]["_n_pairs"] = len(y_true)

    if args.method in ("hybrid", "all"):
        y_true, y_pred, elapsed, llm_calls = eval_hybrid(dataset)
        results["Hybrid"] = print_results("方案 C: Hybrid (本系统)",
                                           y_true, y_pred, elapsed)
        results["Hybrid"]["_n_pairs"] = len(y_true)
        results["Hybrid"]["_llm_calls"] = llm_calls

    if len(results) >= 2:
        print_comparison(results)


if __name__ == "__main__":
    main()
