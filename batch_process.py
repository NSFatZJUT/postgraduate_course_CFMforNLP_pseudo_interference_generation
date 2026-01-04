"""
batch_process.py

面向 10 万行以上文本的批量伪干扰生成与导出（支持断点续传）。

输入：CSV（至少 text 列，可选 label 列）
输出：JSONL（每行包含 original / perturbed / label / scores / trace）
断点：在输出目录写入 checkpoint.json，记录已处理行号与配置快照。

用法示例：
  python batch_process.py --input data/big.csv --text-col text --label-col label --out out/run1.jsonl --resume
"""

import argparse
import csv
import json
import os
from dataclasses import asdict

from tqdm import tqdm

from nlp_pyramidal_flow.config import PIGConfig
from nlp_pyramidal_flow.directed import DirectedConfig, compute_directed_spans
from nlp_pyramidal_flow.pyramidal_flow import PyramidalFlowPerturber
from nlp_pyramidal_flow.quality import GPT2FluencyScorer, score_quality


def load_checkpoint(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--style", default="通用")
    ap.add_argument("--directed", default="通用", choices=["通用", "NER-保护实体", "匹配-保留关键词", "QA-改写问题"])
    ap.add_argument("--keyword-topk", type=int, default=6)
    ap.add_argument("--max-ops", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--use-gpt2-fluency", action="store_true")
    ap.add_argument("--gpt2-model", default="distilgpt2")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    ckpt_path = os.path.join(out_dir, "checkpoint.json")

    cfg = PIGConfig(seed=args.seed, max_ops=args.max_ops)
    perturber = PyramidalFlowPerturber(cfg)
    dcfg = DirectedConfig(mode=args.directed, keyword_topk=args.keyword_topk)

    flu = None
    if args.use_gpt2_fluency:
        flu = GPT2FluencyScorer(model_name=args.gpt2_model, device=args.device)

    start = 0
    if args.resume:
        ck = load_checkpoint(ckpt_path)
        if ck and ck.get("out") == os.path.abspath(args.out):
            start = int(ck.get("processed", 0))

    mode = "a" if start > 0 else "w"
    os.makedirs(out_dir, exist_ok=True)

    with open(args.input, "r", encoding="utf-8", newline="") as f_in, open(args.out, mode, encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        it = enumerate(reader)
        # skip processed
        for _ in range(start):
            try:
                next(it)
            except StopIteration:
                break

        for i, r in tqdm(it, desc="processing", initial=start):
            x = str(r.get(args.text_col, "")).strip()
            if not x:
                continue
            y = str(r.get(args.label_col, "")) if args.label_col in r else ""

            spans = compute_directed_spans(x, dcfg)
            pr = perturber.perturb(x, style=args.style, extra_protected_spans=spans)

            qs = score_quality(x, pr.text, classifier=None, fluency_scorer=flu)
            rec = {
                "original": x,
                "perturbed": pr.text,
                "label": y,
                "scores": qs.to_dict(),
                "trace": pr.trace.to_dict(),
                "config": {"pig": asdict(cfg), "style": args.style, "directed": asdict(dcfg)},
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if (i + 1) % 200 == 0:
                save_checkpoint(
                    ckpt_path,
                    {"out": os.path.abspath(args.out), "processed": i + 1, "pig": asdict(cfg), "directed": asdict(dcfg)},
                )
        # final checkpoint: processed indicates last seen line index + 1
        save_checkpoint(
            ckpt_path,
            {"out": os.path.abspath(args.out), "processed": i + 1 if "i" in locals() else start, "pig": asdict(cfg), "directed": asdict(dcfg)},
        )


if __name__ == "__main__":
    main()


