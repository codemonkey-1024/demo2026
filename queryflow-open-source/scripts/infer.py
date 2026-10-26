#!/usr/bin/env python
"""Run entity/document retrieval from a prepared Stage-2 dataset.

This loads validation samples from QADataset and never creates an optimizer.
Questions are aligned through sample_id after the original Stage-2 filtering.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/raw/eval")
    parser.add_argument("--dataset", default="wiki")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/model_best.pth")
    parser.add_argument("--max-questions", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from queryflow.datasets import QADataset
    from queryflow.doc_rankers import SoftplusIDFWeightedRanker
    from queryflow.workflow.stage2_qa_finetune import _prepare_graph
    from queryflow.utils.qa_utils import DocumentRetriever
    from queryflow.utils import get_entities_weight

    cfg = OmegaConf.load(ROOT / "configs/default.yaml")
    text_cfg = cfg.datasets.cfgs.text_emb_model_cfgs
    dataset = QADataset(str(args.dataset_root), args.dataset, text_cfg)
    model = instantiate(cfg.model, rel_emb_dim=int(cfg.datasets.feat_dim))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    graph = _prepare_graph(dataset.kg, device)
    ent2docs = dataset.ent2docs.to(device)
    ranker = SoftplusIDFWeightedRanker(ent2docs)
    entities_weight = get_entities_weight(ent2docs) if cfg.train.init_entities_weight else None
    retriever = DocumentRetriever(dataset.doc, dataset.id2doc)
    rows = []
    valid_data = dataset._data[1]
    for i in range(min(args.max_questions, len(valid_data))):
        sample = valid_data[i]
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if torch.is_tensor(v)}
        with torch.no_grad():
            ent_scores = model(graph, batch, entities_weight=entities_weight)
            doc_scores = ranker(ent_scores)[0]
        docs = retriever(doc_scores.cpu(), top_k=min(args.top_k, len(dataset.id2doc)))
        sample_id = int(sample["sample_id"])
        raw = dataset.raw_test_data[sample_id]
        rows.append({"index": i, "sample_id": sample_id, "id": raw.get("id"), "question": raw.get("question", ""), "documents": docs})
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
