"""Native-evidence entity targets over the existing, unchanged KG vocabulary."""

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset


def unescape_unicode(text):
    return re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m[1], 16)), str(text))


def normalize_entity(text):
    return " ".join(re.sub("[^a-z0-9 ]", " ", unescape_unicode(text).lower()).split())


def text_key(text):
    return " ".join(unicodedata.normalize("NFKC", unescape_unicode(text)).casefold().split())


def document_key(text):
    # Canonical corpora detokenize forms such as "8 mm" -> "8mm".
    return "".join(re.findall(r"[^\W_]+", text_key(text)))


def contains_entity(entity, text):
    return bool(entity) and f" {entity} " in f" {text} "


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_sentence_documents(context, facts, endpoints):
    # Case is part of Wikipedia document identity: "Shore Leave" != "Shore leave".
    context = {unescape_unicode(title).strip(): (title, list(sentences))
               for title, sentences in context}
    grouped = defaultdict(list)
    for title, index in facts:
        grouped[unescape_unicode(title).strip()].append(int(index))
    documents = []
    for title_key, indices in grouped.items():
        title, sentences = context[title_key]
        documents.append({
            "title": title, "sentences": sentences, "sentence_ids": sorted(set(indices)),
            "strong_entities": endpoints, "kind": "native_sentence",
        })
    return documents


def evidence_entity_record(question, source, corpus, doc2entities, ent2id, *, sample_id):
    if text_key(question["question"]) != text_key(source["question"]):
        raise ValueError(f"Source question mismatch: {question['id']}")
    support_ids = set(question["supporting_facts"])
    groups = []
    resolved = set()
    weights = defaultdict(float)
    for evidence in source["documents"]:
        title = evidence["title"]
        sentences = evidence["sentences"]
        explicit_doc = evidence.get("doc_id")
        if explicit_doc is not None:
            if explicit_doc not in support_ids or explicit_doc not in corpus:
                raise ValueError(f"Source document mismatch: {question['id']}: {title}")
            doc = explicit_doc
        else:
            source_text = document_key("\n".join([title, *sentences]))
            matches = [doc for doc in support_ids if doc in corpus and document_key(corpus[doc]) == source_text]
            if len(matches) != 1:
                raise ValueError(f"Source document mismatch: {question['id']}: {title}")
            doc = matches[0]
        resolved.add(doc)
        sentence_ids = evidence["sentence_ids"]
        if any(index < 0 or index >= len(sentences) for index in sentence_ids):
            raise ValueError(f"Invalid support sentence index: {question['id']}: {title}")
        text = normalize_entity(" ".join(sentences[index] for index in sentence_ids))
        normalized_title = normalize_entity(title)
        strong = {normalize_entity(entity) for entity in evidence["strong_entities"]}
        # Wikipedia disambiguation suffixes are often absent from KG entity names.
        aliases = {normalize_entity(re.sub(r"\s*\([^()]*\)\s*$", "", unescape_unicode(entity)))
                   for entity in evidence["strong_entities"]}
        candidates = {entity: normalize_entity(entity) for entity in doc2entities.get(doc, []) if entity in ent2id}
        positive = {}
        strong_count = 0
        for entity, normalized in candidates.items():
            if not normalized:
                continue
            if evidence.get("kind") == "document_only_fallback":
                continue
            if normalized in strong:
                positive[ent2id[entity]] = 1.0
                strong_count += 1
            elif normalized in aliases:
                positive[ent2id[entity]] = 0.7
                strong_count += 1
            elif contains_entity(normalized, text) or normalized == normalized_title:
                positive[ent2id[entity]] = 0.25
        native_count = len(positive)
        fallback = native_count == 0 and bool(candidates)
        if fallback:
            positive = {ent2id[entity]: 0.1 for entity in candidates}
        total = sum(positive.values())
        group_scale = 0.1 if fallback else 1.0
        if total:
            for entity_id, confidence in positive.items():
                weights[entity_id] += group_scale * confidence / total
        groups.append({
            "doc_id": doc, "title": title, "kind": evidence["kind"],
            "step_id": evidence.get("step_id"), "sentence_ids": sentence_ids,
            "annotated_entities": evidence["strong_entities"],
            "sentence_selection": evidence.get("sentence_selection", "native"),
            "native_positive_count": native_count, "strong_positive_count": strong_count,
            "weak_document_fallback": fallback, "positive_entity_count": len(positive),
        })
    if resolved != support_ids:
        raise ValueError(f"Source supporting document set mismatch: {question['id']}")
    group_count = max(len(groups), 1)
    return {
        "id": question["id"], "sample_id": sample_id,
        "positive_weights": [[entity_id, weight / group_count] for entity_id, weight in sorted(weights.items())],
        "coarse_positive_count": len({ent2id[e] for doc in support_ids
                                      for e in doc2entities.get(doc, []) if e in ent2id}),
        "groups": groups,
    }


class EvidenceLabelDataset(Dataset):
    """Add training-only targets while preserving cached model inputs and metrics."""

    def __init__(self, dataset, records, question_ids):
        self.dataset = dataset
        self.records = records
        self.question_ids = question_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        sample_id = int(item["sample_id"])
        record = self.records[str(sample_id)]
        if record["id"] != self.question_ids[sample_id]:
            raise ValueError(f"Evidence label question ID mismatch at sample_id={sample_id}")
        targets = torch.zeros_like(item["supporting_entities_masks"], dtype=torch.float32)
        unknown = item["supporting_entities_masks"].bool() | item["question_entities_masks"].bool()
        targets[unknown] = -1
        for entity_id, weight in record["positive_weights"]:
            if not 0 <= entity_id < len(targets) or not 0 < weight <= 1:
                raise ValueError(f"Invalid evidence entity target: {record['id']}")
            targets[entity_id] = weight
        for entity_id in record.get("verified_negative_entity_ids", []):
            if (not 0 <= entity_id < len(targets)
                    or targets[entity_id] > 0
                    or item["question_entities_masks"][entity_id]
                    or not item["supporting_entities_masks"][entity_id]):
                raise ValueError(f"Invalid or conflicting verified negative: {record['id']}")
            targets[entity_id] = 0
        item["evidence_entity_targets"] = targets
        return item


def load_evidence_dataset(dataset, qa_data, labels_dir):
    path = Path(labels_dir) / f"{qa_data.name}.json"
    payload = json.loads(path.read_text())
    if payload["status"] != "complete" or payload["num_entities"] != qa_data.kg.num_nodes:
        raise ValueError(f"Incomplete or incompatible evidence labels: {path}")
    checks = {
        "stage1_sha256": Path(qa_data.raw_dir) / "train.json",
        "ent2id_sha256": Path(qa_data.processed_dir) / "ent2id.json",
    }
    for key, source in checks.items():
        if payload[key] != file_sha256(source):
            raise ValueError(f"Stale evidence labels ({key}): {path}")
    ids = {index: row["id"] for index, row in enumerate(qa_data.raw_train_data)}
    return EvidenceLabelDataset(dataset, payload["records"], ids)
