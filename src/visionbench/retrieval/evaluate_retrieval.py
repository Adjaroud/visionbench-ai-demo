"""Mesure la recherche documentaire et l'abstention sur les cas J3."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from document_search import DEFAULT_CORPUS, TOP_K, DocumentIndex, answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--cases", type=Path, default=Path("eval/j3_dev.json"))
    args = parser.parse_args()
    index = DocumentIndex(args.corpus)
    raw_cases = args.cases.read_bytes()
    cases = json.loads(raw_cases)
    valid_ids = {document["id"] for document in index.documents}
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Identifiants de cas dupliqués.")

    details = []
    for case in cases:
        expected = set(case["expected_ids"])
        if not expected <= valid_ids:
            raise ValueError(f"Passage attendu absent du corpus pour {case['id']}.")
        result = answer(index, case["question"])
        retrieved = result["retrieved_ids"]
        cited = {source["id"] for source in result["sources"]}
        found = expected.intersection(retrieved)
        details.append({**case, "retrieved_ids": retrieved, "cited_ids": sorted(cited),
                        "status": result["status"], "hit_at_5": bool(found) if expected else None,
                        "recall_at_5": len(found) / len(expected) if expected else None,
                        "expected_cited": expected <= cited if expected else None})

    positives = [row for row in details if row["expected_ids"]]
    negatives = [row for row in details if not row["expected_ids"]]
    if not positives or not negatives:
        raise ValueError("Le jeu doit contenir des cas répondables et hors périmètre.")
    print(json.dumps({
        "kind": "j3_retrieval_dev", "mode": "extractive", "top_k": TOP_K,
        "corpus_revision": index.revision, "corpus_sha256": index.corpus_sha256,
        "cases_sha256": hashlib.sha256(raw_cases).hexdigest(),
        "answerable": len(positives), "out_of_scope": len(negatives),
        "hit_at_5": mean(row["hit_at_5"] for row in positives),
        "mean_recall_at_5": mean(row["recall_at_5"] for row in positives),
        "all_expected_sources_cited": mean(row["expected_cited"] for row in positives),
        "abstained_on_out_of_scope": sum(row["status"] == "insufficient_evidence" for row in negatives),
        "details": details,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
