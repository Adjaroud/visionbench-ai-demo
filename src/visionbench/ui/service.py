"""Adaptation du moteur évalué pour la démo CPU, sans dépendance à Streamlit."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


MODEL_NAMES = {
    "ssdlite320_mobilenet_v3_large": "SSDLite · MobileNetV3",
    "fasterrcnn_mobilenet_v3_large_320_fpn": "Faster R-CNN · MobileNetV3",
}
METRIC_NAMES = {
    "ap": "AP [0,50:0,95]", "ap50": "AP50", "ap75": "AP75",
    "pipeline_median": "Latence pipeline médiane", "pipeline_p95": "Latence pipeline p95",
    "peak_pytorch_allocated": "Pic de mémoire PyTorch allouée",
}


def artifact(root, relative, expected=None):
    """Ne lit que des fichiers du projet ; vérifie l'empreinte si elle est fournie."""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Le fichier demandé doit rester dans le projet.")
    raw = path.read_bytes()
    if expected and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Empreinte incorrecte : {relative}. Restaurer le fichier avant la démo.")
    return raw


def load_manifest(root):
    manifest = json.loads(artifact(root, "data/demo/manifest.json"))
    for relative, expected in manifest["files_sha256"].items():
        artifact(root, relative, expected)
    return manifest


def load_recording(root, manifest, scenario):
    relative = scenario["recording"]
    return json.loads(artifact(root, relative, manifest["files_sha256"][relative]))


def load_comparison(root):
    manifest = json.loads(artifact(root, "data/demo/comparison/manifest.json"))
    for relative, expected in manifest["files_sha256"].items():
        artifact(root, relative, expected)
    return manifest


def load_comparison_recording(root, manifest, mode):
    relative = manifest["recordings"][mode]
    return json.loads(artifact(root, relative, manifest["files_sha256"][relative]))["result"]


def catalogue_path(root, catalogue):
    if catalogue == "pilot":
        return "reports/visionbench.sqlite"
    if catalogue != "resources_100":
        raise ValueError("Catalogue inconnu.")
    manifest = json.loads(artifact(root, "reports/resources_100.manifest.json"))
    database = "reports/resources_100.sqlite"
    artifact(root, database, manifest["database_sha256"])
    for relative in ("resources_100/raw.json", "j2_coco_eval/metrics.json"):
        artifact(root, "reports/" + relative, manifest["source_sha256"][relative])
    return database


def _run_cli(root, arguments, timeout):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-X", "utf8", "-B", "-m", *arguments]
    try:
        return subprocess.run(command, cwd=root, env=environment, capture_output=True,
                              encoding="utf-8", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"L'exécution a dépassé {timeout} secondes.") from exc


def _cli_error(result):
    detail = result.stderr.strip().splitlines()
    return ValueError(detail[-1].split("error: ")[-1] if detail else "Le moteur a interrompu la demande.")


def run_fixed_report(root, question, *, gpu=None, metrics=(), p95=None, memory=None, ap=None, catalogue="pilot"):
    """Appelle exactement l'entrée CLI évaluée ; aucun mode agent n'est exposé ici."""
    root = Path(root).resolve()
    command = ["visionbench.orchestration.workflow", question, "--mode", "fixed",
               "--db", catalogue_path(root, catalogue)]
    if gpu:
        command.extend(["--gpu", gpu])
    for metric in metrics:
        command.extend(["--metric", metric])
    for flag, value in (("--p95-max-ms", p95), ("--memory-max-mib", memory), ("--ap-min", ap)):
        if value is not None:
            command.extend([flag, str(value)])
    result = _run_cli(root, command, timeout=10)
    if result.returncode:
        # argparse place la précision utile à la fin ; on n'affiche pas sa liste d'options.
        raise _cli_error(result)
    return json.loads(result.stdout)


def run_live_answer(root, mode, question, *, gpu="RTX 3050 Laptop", p95=120, memory=50):
    """Réutilise les entrées CLI et leurs contrôles pour une exécution locale."""
    root = Path(root).resolve()
    if mode == "rag":
        command = ["visionbench.retrieval.verified_rag", "--", question]
    elif mode in {"fixed", "agent"}:
        command = ["visionbench.synthesis.agent_demo", "--mode", mode, "--question=" + question,
                   "--gpu=" + gpu, "--p95-max-ms", str(p95), "--memory-max-mib", str(memory),
                   "--db", catalogue_path(root, "resources_100")]
    else:
        raise ValueError("Parcours inconnu.")
    with TemporaryDirectory(prefix="visionbench-live-") as directory:
        output = Path(directory) / "result"
        # Les options RAG précèdent le séparateur qui protège la question libre.
        if mode == "rag":
            command[1:1] = ["--out", str(output)]
        else:
            command.extend(["--out", str(output)])
        process = _run_cli(root, command, timeout=200)
        path = output / "run.json"
        if not path.is_file():
            raise _cli_error(process)
        result = json.loads(path.read_text(encoding="utf-8"))
        return {**result["answer"], "protocol": result["protocol"]} if mode == "rag" else result


def measurement_rows(report):
    return [{"Modèle": MODEL_NAMES.get(row["model"], row["model"]),
             "Métrique": METRIC_NAMES.get(row["metric"], row["metric"]),
             "Valeur": round(row["value"], 5), "Unité": row["unit"],
             "Images": row["image_count"], "Source": row["source_path"]}
            for row in report.get("measurements", [])]


def verified_sources(root, report):
    sources = {}
    for row in report.get("measurements", []):
        relative = "reports/" + row["source_path"]
        raw = artifact(root, relative, row["source_sha256"])
        sources[relative] = {"sha256": row["source_sha256"], "raw": raw}
    return sources
