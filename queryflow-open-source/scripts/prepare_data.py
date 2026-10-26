#!/usr/bin/env python
"""Build Stage-1 graphs and Stage-2 tensors for the public package.

Only raw corpora are shipped. This command runs the same KG/QA constructors and
QADataset processor used by the research code. Stage-1 extraction needs an LLM
endpoint; set the provider credentials expected by LangChain before running.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf
from hydra.utils import instantiate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAIN_NAMES = [
    "2wikimultihopqa_train_train_00000",
    "hotpotqa_train_train_00000",
    "musique_train_train_00019",
]
EVAL_NAMES = ["wiki", "hotpotqa", "musique", "CQA_val"]


def _constructor_cfg(args: argparse.Namespace):
    model_name = args.embedding_model
    llm = {
        "_target_": "queryflow.kg_construction.openie_model.LLMOPENIEModel",
        "llm_api": args.llm_api,
        "model_name": args.llm_model,
        "max_ner_tokens": 300,
        "max_triples_tokens": 4096,
    }
    ner = {
        "_target_": "queryflow.kg_construction.ner_model.LLMNERModel",
        "llm_api": args.llm_api,
        "model_name": args.llm_model,
        "max_tokens": 300,
    }
    el = {
        "_target_": "queryflow.kg_construction.entity_linking_model.DPRELModel",
        "model_name": model_name,
        "root": str(ROOT / "data" / "tmp"),
        "use_cache": True,
        "normalize": True,
        "batch_size": 32,
        "device": args.embedding_device,
    }
    return OmegaConf.create(
        {
            "root": str(ROOT / "data" / "tmp" / "kg_construction"),
            "open_ie_model": llm,
            "el_model": el,
            "num_processes": args.workers,
            "cosine_sim_edges": True,
            "threshold": 0.8,
            "max_sim_neighbors": 100,
            "add_title": True,
            "force": args.force,
        }
    ), OmegaConf.create(
        {
            "root": str(ROOT / "data" / "tmp" / "qa_construction"),
            "ner_model": ner,
            "el_model": el,
            "num_processes": args.workers,
            "force": args.force,
        }
    )


def build_stage1(root: Path, names: list[str], args: argparse.Namespace) -> None:
    from queryflow.kg_construction import KGConstructor, QAConstructor
    from queryflow.kg_indexer import KGIndexer

    kg_cfg, qa_cfg = _constructor_cfg(args)
    kg_indexer = KGIndexer(KGConstructor.from_config(kg_cfg), QAConstructor.from_config(qa_cfg))
    for name in names:
        print(f"[stage1] {root / name}", flush=True)
        kg_indexer.index_data(OmegaConf.create({"root": str(root), "data_name": name}))


def build_stage2(root: Path, names: list[str], args: argparse.Namespace) -> None:
    from queryflow.datasets import QADataset

    text_cfg = OmegaConf.create(
        {
            "_target_": "queryflow.text_emb_models.BaseTextEmbModel",
            "text_emb_model_name": args.embedding_model,
            "normalize": False,
            "batch_size": 16,
            "query_instruct": None,
            "passage_instruct": None,
            "model_kwargs": None,
        }
    )
    for name in names:
        print(f"[stage2] {root / name}", flush=True)
        dataset = QADataset(
            root=str(root),
            data_name=name,
            text_emb_model_cfgs=text_cfg,
            force_rebuild=args.force,
        )
        print(
            f"  train={len(dataset._data[0])} valid={len(dataset._data[1])} "
            f"nodes={dataset.kg.num_nodes} docs={len(dataset.doc2entities)}",
            flush=True,
        )


def build_labels(root: Path, names: list[str], args: argparse.Namespace) -> None:
    from queryflow.datasets import QADataset
    from queryflow.evidence_supervision import evidence_entity_record, file_sha256

    sources = json.loads((root / "native_annotations.json").read_text())["questions"]
    cfg = OmegaConf.load(ROOT / "configs/default.yaml")
    cfg.datasets.cfgs.text_emb_model_cfgs.text_emb_model_name = args.embedding_model
    output_dir = ROOT / "data/evidence_labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        qa = QADataset(str(root), name, cfg.datasets.cfgs.text_emb_model_cfgs)
        records = {}
        for sample_id, question in enumerate(qa.raw_train_data):
            try:
                record = evidence_entity_record(
                    question, sources[question["id"]], qa.doc, qa.doc2entities,
                    qa.ent2id, sample_id=sample_id,
                )
            except ValueError as error:
                if not str(error).startswith(("Source document mismatch:", "Invalid support sentence index:")):
                    raise
                record = {"id": question["id"], "sample_id": sample_id,
                          "positive_weights": [], "groups": [], "alignment_error": str(error),
                          "coarse_positive_count": len(set(question["supporting_entities"]) & qa.ent2id.keys())}
            records[str(sample_id)] = record
        payload = {"status": "complete", "schema": "native-entity-evidence-v1",
                   "shard": name, "num_entities": qa.kg.num_nodes,
                   "stage1_sha256": file_sha256(Path(qa.raw_dir) / "train.json"),
                   "ent2id_sha256": file_sha256(Path(qa.processed_dir) / "ent2id.json"),
                   "corpus_sha256": file_sha256(root / name / "raw/dataset_corpus.json"),
                   "records": records}
        (output_dir / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False) + "\n")
        print(f"[labels] {name}: {len(records)} records", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--stage", choices=["stage1", "stage2", "labels", "all"], default="all")
    parser.add_argument("--names", nargs="*", help="Override the default names")
    parser.add_argument("--llm-api", default=os.getenv("QUERYFLOW_LLM_API", "openai"))
    parser.add_argument("--llm-model", default=os.getenv("QUERYFLOW_LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("QUERYFLOW_TEXT_MODEL", "sentence-transformers/all-mpnet-base-v2"),
    )
    parser.add_argument("--embedding-device", default=os.getenv("QUERYFLOW_EMBEDDING_DEVICE", "auto"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    os.environ["QUERYFLOW_EMBEDDING_DEVICE"] = args.embedding_device

    if args.scope == "train":
        groups = [(ROOT / "data" / "raw" / "train", TRAIN_NAMES)]
    elif args.scope == "eval":
        groups = [(ROOT / "data" / "raw" / "eval", EVAL_NAMES)]
    else:
        groups = [
            (ROOT / "data" / "raw" / "train", TRAIN_NAMES),
            (ROOT / "data" / "raw" / "eval", EVAL_NAMES),
        ]
    if args.names:
        allowed = set(args.names)
        unknown = allowed - {name for _, names in groups for name in names}
        if unknown:
            parser.error(f"Unknown datasets in selected scope: {sorted(unknown)}")
        groups = [(root, [name for name in names if name in allowed]) for root, names in groups]

    if args.stage in {"stage1", "all"}:
        for root, names in groups:
            build_stage1(root, names, args)
    if args.stage in {"stage2", "all"}:
        for root, names in groups:
            build_stage2(root, names, args)
    if args.stage in {"labels", "all"}:
        for root, names in groups:
            if root.name == "train":
                build_labels(root, names, args)


if __name__ == "__main__":
    main()
