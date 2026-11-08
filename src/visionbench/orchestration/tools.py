"""Outils documentaires et expérimentaux utilisés par les deux workflows."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from visionbench.retrieval.document_search import DocumentIndex, enough_lexical_evidence
from visionbench.storage.benchmark_store import check_comparability as compare_rows, query


METRICS = ("ap", "ap50", "ap75", "pipeline_median", "pipeline_p95", "peak_pytorch_allocated")


@dataclass(frozen=True)
class Question:
    text: str
    metrics: tuple[str, ...]
    gpu: str | None = None
    p95_limit_ms: float | None = None
    memory_limit_mib: float | None = None
    ap_min: float | None = None
    p95_strict: bool = False
    memory_strict: bool = False
    ap_strict: bool = False


def gpu_matches(requested, actual):
    # Conserver les distinctions comme Ti et Laptop dans le filtre matériel.
    generic = {"nvidia", "geforce", "amd", "radeon", "gpu"}

    def normalize(value):
        value = re.sub(r"\b(rtx|gtx|rx)\s*(\d{4})\b", r"\1 \2", value.casefold())
        return [word for word in re.findall(r"[a-z0-9]+", value) if word not in generic]

    return normalize(requested) == normalize(actual)


class VisionTools:
    def __init__(self, corpus: Path, database: Path):
        self.index = DocumentIndex(corpus)
        self.database = database

    def search_documents(self, request):
        passages = self.index.search(request.text)
        if not enough_lexical_evidence(request.text, passages):
            return []
        return [{key: passage[key] for key in ("id", "title", "section", "url", "text")}
                for passage in passages[:3]]

    def query_benchmarks(self, request):
        measurements = []
        reports = {}
        available_gpus = set()
        for metric in request.metrics:
            if metric not in METRICS:
                raise ValueError(f"Métrique non autorisée ({metric}).")
            for row in query(self.database, metric):
                source = (self.database.parent / row["source_path"]).resolve()
                if not source.is_relative_to(self.database.parent.resolve()):
                    raise ValueError("Le rapport source doit rester dans le dossier de la base.")
                if source not in reports:
                    data = source.read_bytes()
                    reports[source] = hashlib.sha256(data).hexdigest(), json.loads(data)
                digest, report = reports[source]
                if digest != row["source_sha256"]:
                    raise ValueError("Le rapport source a changé depuis la construction du catalogue.")
                gpu = report["environment"]["gpu"]
                available_gpus.add(gpu)
                if request.gpu and not gpu_matches(request.gpu, gpu):
                    continue
                measurements.append({**row, "gpu": gpu})
        return {"measurements": measurements, "available_gpus": sorted(available_gpus)}

    def check_comparability(self, request, benchmarks):
        groups = {}
        for metric in request.metrics:
            rows = [row for row in benchmarks["measurements"] if row["metric"] == metric]
            if len(rows) == 2 and rows[0]["model"] != rows[1]["model"]:
                groups[metric] = compare_rows(*rows)
            else:
                groups[metric] = {"comparable": False, "reasons": ["Deux mesures vérifiées sont nécessaires."],
                                  "scope": None}
        return groups

    def propose_experiment(self, request, comparisons):
        missing = [metric for metric in request.metrics if not comparisons.get(metric, {}).get("comparable")]
        if not missing:
            return None
        return {"gpu_requested": request.gpu, "metrics": missing,
                "protocol": "Tester les deux détecteurs sur les mêmes images, en FP32 et batch 1. "
                "Documenter poids, réglages natifs et versions. Pour la latence, utiliser le même chronomètre "
                "avec échauffement et alternance. Pour l’AP, utiliser le même jeu annoté et COCOeval."}


def select_under_limit(request, benchmarks, comparisons):
    if all(value is None for value in (request.p95_limit_ms, request.memory_limit_mib, request.ap_min)):
        return None
    required = ["ap"]
    if request.p95_limit_ms is not None:
        required.append("pipeline_p95")
    if request.memory_limit_mib is not None:
        required.append("peak_pytorch_allocated")
    if not all(comparisons.get(metric, {}).get("comparable") for metric in required):
        return {"status": "insufficient_evidence", "candidate": None}
    groups = {metric: {row["model"]: row for row in benchmarks["measurements"] if row["metric"] == metric}
              for metric in required}
    ap = groups["ap"]
    for rows in groups.values():
        if ap.keys() != rows.keys() or any(ap[name]["weights_sha256"] != rows[name]["weights_sha256"]
                                         or ap[name]["gpu"] != rows[name]["gpu"] for name in ap):
            return {"status": "insufficient_evidence", "candidate": None}

    candidates = []
    for name in ap:
        if request.p95_limit_ms is not None:
            value = groups["pipeline_p95"][name]["value"]
            if value > request.p95_limit_ms or (request.p95_strict and value == request.p95_limit_ms):
                continue
        if request.memory_limit_mib is not None:
            value = groups["peak_pytorch_allocated"][name]["value"]
            if value > request.memory_limit_mib or (request.memory_strict and value == request.memory_limit_mib):
                continue
        if request.ap_min is not None:
            value = ap[name]["value"]
            if value < request.ap_min or (request.ap_strict and value == request.ap_min):
                continue
        candidates.append(name)
    return {"status": "pilot_candidate" if candidates else "no_candidate",
            "candidate": max(candidates, key=lambda name: ap[name]["value"]) if candidates else None,
            "eligible": candidates, "p95_limit_ms": request.p95_limit_ms,
            "memory_limit_mib": request.memory_limit_mib, "ap_min": request.ap_min,
            "scope": "AP sur COCO et latence sur le pilote. Confirmer la latence sur la charge cible."}
