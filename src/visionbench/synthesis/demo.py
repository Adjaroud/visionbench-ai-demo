"""Premier scénario mixte : décision numérique et explication documentaire avec LLM local."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter

from visionbench.orchestration.constraints import bind_constraints
from visionbench.orchestration.tools import Question, VisionTools, gpu_matches
from visionbench.orchestration.workflow import build_report, run_fixed
from visionbench.synthesis.grounded import DEFAULT_MODEL, KEEP_ALIVE, MODEL_NAMES, MODEL_OPTIONS, PHASE_PROMPTS, PHASE_TOKENS, SYSTEM_PROMPT, model_metadata, retrieve_documents, synthesize


DEFAULT_QUESTION = "Explique le choix du détecteur et compare son AP à celle de l'autre modèle. Définis la mémoire allouée et précise le périmètre des mesures."


def markdown(run):
    report, synthesis = run["workflow_report"], run["synthesis"]
    decision = report.get("recommendation") or {}
    lines = ["# Réponse mixte — scénario de développement", "", report["request"]["text"], "",
             "## Décision calculée par Python", "",
             f"Statut : **{decision.get('status')}**. Candidat : **{MODEL_NAMES.get(decision.get('candidate'), decision.get('candidate'))}**.", "",
             ]
    for claim in synthesis["claims"]:
        if claim.get("origin") == "python":
            lines.append(claim["text"] + " " + " ".join(f"[^{c}]" for c in claim["citations"]) + "\n")
    lines += ["## Explication documentaire générée par le LLM", "",
             f"Modèle : `{synthesis['model']}`. Requêtes LLM tentées : {synthesis['model_calls']}. Réponse reçue : {synthesis['response_received']}. Statut : `{synthesis['status']}`.", ""]
    for claim in synthesis["claims"]:
        if claim.get("origin") == "python":
            continue
        lines.append(claim["text"] + " " + " ".join(f"[^{c}]" for c in claim["citations"]) + "\n")
    if synthesis["error"]:
        lines.append("Explication indisponible : " + synthesis["error"] + "\n")
    if synthesis.get("verified_scope"):
        lines += ["## Périmètre vérifié par Python", "", synthesis["verified_scope"]["text"] + " [^S1]", ""]
    lines += ["## Preuves fournies", ""]
    for item in synthesis["context"]["evidence"]:
        source = item["source"]
        origin = f"[{source['title']}]({source['url']}) — {source['section']}" if item["kind"] == "document" else (
            f"Rapport `{source['source_path']}`, SHA-256 `{source['source_sha256']}`" if item["kind"] == "measurement" else (
                "Contrôle Python à partir de " + ", ".join(f"[^{name}]" for name in source["inputs"]) if source.get("inputs") else "Demande ou contrôle Python"))
        lines.append(f"[^{item['id']}]: {origin}. {item['text']}\n")
    lines += ["## Portée de cette exécution", "",
              "Python rédige la sélection, les comparaisons et le périmètre à partir des preuves. Les contrôles de structure, de citations et de nombres s’appliquent à la définition générée. Une revue du sens de cette définition reste nécessaire.", "",
              "Scénario de développement connu ; aucune nouvelle performance générale du RAG ou de l'agent n'est revendiquée.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--gpu", default="RTX 3050 Laptop")
    parser.add_argument("--p95-max-ms", type=float, default=80)
    parser.add_argument("--memory-max-mib", type=float, default=50)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/vision_docs.json"))
    parser.add_argument("--db", type=Path, default=Path("reports/visionbench.sqlite"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.question.strip() or len(args.question) > 700 or not args.gpu.strip():
        parser.error("Précise une question de 1 à 700 caractères et un GPU.")
    try:
        bounds = bind_constraints(args.question, p95=args.p95_max_ms, memory=args.memory_max_mib)
    except ValueError as exc:
        parser.error(str(exc))
    if args.out.exists():
        parser.error("Le dossier de sortie existe déjà ; choisis un nouveau chemin.")
    mentioned = re.findall(r"\b(?:RTX|GTX|RX)\s*\d{4}(?:\s+Ti)?(?:\s+Laptop)?\b", args.question, re.IGNORECASE)
    if any(not gpu_matches(gpu, args.gpu) for gpu in mentioned):
        parser.error("Le GPU du texte et le filtre --gpu diffèrent.")
    paths = [args.corpus, args.db, *sorted(Path("src/visionbench").rglob("*.py"))]
    fingerprints = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    request = Question(args.question, ("ap", "pipeline_p95", "peak_pytorch_allocated"), args.gpu, **bounds)
    tools = VisionTools(args.corpus, args.db)
    started = perf_counter()
    state = run_fixed(tools, request)
    report = build_report(request, "fixed", None, state, perf_counter() - started)
    documents, retrieval = retrieve_documents(tools.index, request)
    report["documents"] = documents
    provider = None
    if documents and state["stop_reason"] == "completed":
        try:
            provider = model_metadata(args.model)
        except (OSError, ValueError, KeyError) as exc:
            provider = {"metadata_unavailable": str(exc)}
    synthesis = synthesize(report, documents, args.model)
    if fingerprints != {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}:
        synthesis.update(status="generation_rejected", claims=[], error="Une entrée a changé pendant l'exécution.")
    for row in report["measurements"]:
        source = args.db.parent / row["source_path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["source_sha256"]:
            synthesis.update(status="generation_rejected", claims=[], error="Un rapport source a changé pendant l'exécution.")
    run = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "kind": "mixed_answer_development_demo",
           "workflow_report": report, "retrieval": retrieval, "synthesis": synthesis,
           "total_model_calls": report["model_calls"] + synthesis["model_calls"],
           "protocol": {"request": asdict(request), "model_options": MODEL_OPTIONS, "think": False, "keep_alive": KEEP_ALIVE,
                        "timeout_s": 90, "system_prompt": SYSTEM_PROMPT, "phase_prompts": PHASE_PROMPTS, "phase_tokens": PHASE_TOKENS,
                        "provider": provider, "source_sha256": fingerprints,
                        "api_charge_eur": 0, "electricity_cost": "not_measured"}}
    args.out.mkdir(parents=True)
    (args.out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "answer.md").write_text(markdown(run), encoding="utf-8")
    print(markdown(run).split("## Preuves fournies")[0], flush=True)
    print(f"Rapport : {args.out / 'run.json'}")
    if synthesis["status"] in {"generation_rejected", "generation_unavailable", "incomplete_workflow"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
