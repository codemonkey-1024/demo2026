#!/usr/bin/env python
"""Evaluate the released checkpoint on prepared validation datasets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def evaluate_one(model, name, root, text_cfg, cfg, device, batch_size):
    from queryflow.datasets import QADataset
    from queryflow.doc_rankers import SoftplusIDFWeightedRanker
    from queryflow.ultra import query_utils
    from queryflow.utils import batch_evaluate, document_recall_metrics, evaluate, get_entities_weight
    from queryflow.workflow.stage2_qa_finetune import _prepare_graph

    qa = QADataset(str(root), name, text_cfg)
    if len(qa._data) < 2 or len(qa._data[1]) == 0:
        raise RuntimeError(f"{name}: no prepared validation rows; run prepare_data.py first")
    loader = DataLoader(qa._data[1], batch_size=batch_size, shuffle=False)
    graph = _prepare_graph(qa.kg, device)
    ent2docs = qa.ent2docs.to(device)
    ranker = SoftplusIDFWeightedRanker(ent2docs)
    entities_weight = get_entities_weight(ent2docs) if cfg.train.init_entities_weight else None
    ent_preds, ent_targets, doc_preds, doc_targets = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = query_utils.cuda(batch, device=device)
            ent_score = model(graph, batch, entities_weight=entities_weight)
            doc_score = ranker(ent_score)
            target_ent = batch["supporting_entities_masks"].bool()
            target_doc = batch["supporting_docs_masks"].bool()
            ent_ranking, ent_target_ranking = batch_evaluate(ent_score, target_ent)
            doc_ranking, doc_target_ranking = batch_evaluate(doc_score, target_doc)
            ent_preds.append((ent_ranking, (torch.sigmoid(ent_score) > 0.5).sum(-1)))
            ent_targets.append((ent_target_ranking, target_ent.sum(-1)))
            doc_preds.append((doc_ranking, (torch.sigmoid(doc_score) > 0.5).sum(-1)))
            doc_targets.append((doc_target_ranking, target_doc.sum(-1)))
    # Rankings are already (ranking, target_ranking); concatenate variable rows.
    def cat(items):
        return (torch.cat([x[0] for x in items]), torch.cat([x[1] for x in items]))
    ent_pred = cat(ent_preds)
    ent_target = cat(ent_targets)
    doc_pred = cat(doc_preds)
    doc_target = cat(doc_targets)
    ent_metrics = evaluate(ent_pred, ent_target, cfg.task.metric)
    doc_metrics = evaluate(doc_pred, doc_target, cfg.task.metric)
    r10, fr10 = document_recall_metrics(doc_target[0], doc_target[1], k=10)
    r30, fr30 = document_recall_metrics(doc_target[0], doc_target[1], k=30)
    return {
        "num_questions": len(qa._data[1]),
        "num_raw_questions": len(qa.raw_test_data),
        "num_filtered_questions": len(qa.raw_test_data) - len(qa._data[1]),
        "r10": r10, "fr10": fr10, "r30": r30, "fr30": fr30,
        "entity_mrr": ent_metrics["mrr"], "doc_mrr": doc_metrics["mrr"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/model_best.pth")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/raw/eval")
    parser.add_argument("--datasets", nargs="+", default=["wiki", "hotpotqa", "musique", "CQA_val"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cfg = OmegaConf.load(ROOT / "configs/default.yaml")
    model = instantiate(cfg.model, rel_emb_dim=int(cfg.datasets.feat_dim))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    results = {}
    for name in args.datasets:
        print(f"Evaluating {name}", flush=True)
        results[name] = evaluate_one(model, name, args.dataset_root, cfg.datasets.cfgs.text_emb_model_cfgs, cfg, device, args.batch_size)
    keys = ("r10", "fr10", "r30", "fr30")
    results["macro"] = {key: float(np.mean([row[key] for name, row in results.items() if name != "macro"])) for key in keys}
    payload = {"checkpoint": str(args.checkpoint), "results": results}
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
