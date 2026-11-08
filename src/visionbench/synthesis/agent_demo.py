"""Scénario mixte avec collecte LangGraph ou workflow fixe et synthèse commune."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from time import monotonic

from visionbench.orchestration.constraints import bind_constraints
from visionbench.orchestration.evidence_agent import (
    COLLECTION_TIMEOUT_S, MAX_MODEL_CALLS, MAX_SEARCHES, MAX_TOOL_CALLS,
    PLANNER_OPTIONS, PLANNER_PROMPT, evidence_report, ollama_evidence_planner,
    run_evidence_agent, run_evidence_fixed,
)
from visionbench.orchestration.tools import Question, VisionTools, gpu_matches
from visionbench.synthesis.demo import DEFAULT_QUESTION, markdown
from visionbench.synthesis.grounded import (
    DEFAULT_MODEL, KEEP_ALIVE, MODEL_OPTIONS, PHASE_PROMPTS, PHASE_TOKENS, SYSTEM_PROMPT,
    call_ollama, model_metadata, synthesize,
)


TOTAL_TIMEOUT_S = 180


def run_session(tools, request, mode="agent", model=DEFAULT_MODEL, *, planner=None,
                generate=call_ollama, timeout_s=TOTAL_TIMEOUT_S):
    """Point d'entrée réutilisable : collecte puis même synthèse dans les deux modes."""
    if mode not in {"fixed", "agent"}:
        raise ValueError("Mode inconnu.")
    started = monotonic()
    calls = []
    if mode == "agent":
        state = run_evidence_agent(tools, request, planner or ollama_evidence_planner(model, calls, generate),
                                   timeout_s=min(COLLECTION_TIMEOUT_S, timeout_s), numeric_first=planner is None)
    else:
        state = run_evidence_fixed(tools, request)
    report = evidence_report(request, mode, model, state, monotonic() - started)
    remaining = max(0, min(90, timeout_s - (monotonic() - started)))
    synthesis = synthesize(report, report["documents"], model, timeout_s=remaining, generate=generate)
    usage = [*calls, *synthesis["llm_calls"]]
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "agent_mixed_answer_development_demo", "workflow_report": report,
        "planning_calls": calls, "synthesis": synthesis,
        "total_model_calls": report["model_calls"] + synthesis["model_calls"],
        "total_usage": {key: sum(call.get(key) or 0 for call in usage)
                        for key in ("prompt_tokens", "generated_tokens", "total_duration_ns")},
        "duration_s": round(monotonic() - started, 3)}


def render(run):
    report = run["workflow_report"]
    lines = ["## Collecte des preuves", "",
        f"Mode : **{report['mode']}** ; arrêt : `{report['stop_reason']}` ; "
        f"appels LLM de planification : {report['model_calls']} ; "
        f"appels LLM au total : {run['total_model_calls']}.", ""]
    for step in report["decisions"]:
        lines.append(f"- {step['origin']} : `{step['action']}` — {step['reason']}")
    lines += ["", "Recherches effectivement exécutées :", ""]
    for search in report["searches"]:
        lines.append(f"- `{search['query']}` → {', '.join(search['accepted_ids']) or 'aucun passage retenu'}")
    if not report["documentary_coverage"]["memory_definition_found"]:
        lines += ["", "**Limite documentaire : le passage définissant la mémoire allouée n'a pas été trouvé.**"]
    if report.get("proposed_experiment"):
        lines += ["", "Proposition Python, sans exécution : " + report["proposed_experiment"]["protocol"]]
    if report.get("planner_error_detail"):
        lines += ["", "Erreur de collecte : " + report["planner_error_detail"]]
    lines += ["", f"Durée totale : {run['duration_s']:.3f} s. Tokens générés reçus : {run['total_usage']['generated_tokens']}.", ""]
    text = markdown(run)
    title, rest = text.split("\n", 1)
    return title + "\n\n" + "\n".join(lines) + rest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixed", "agent"), default="agent")
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
    if args.out.exists():
        parser.error("Le dossier de sortie existe déjà ; choisis un nouveau chemin.")
    try:
        bounds = bind_constraints(args.question, p95=args.p95_max_ms, memory=args.memory_max_mib)
    except ValueError as exc:
        parser.error(str(exc))
    mentioned = re.findall(r"\b(?:RTX|GTX|RX)\s*\d{4}(?:\s+Ti)?(?:\s+Laptop)?\b", args.question, re.IGNORECASE)
    if any(not gpu_matches(gpu, args.gpu) for gpu in mentioned):
        parser.error("Le GPU du texte et le filtre --gpu diffèrent.")
    request = Question(args.question, ("ap", "pipeline_p95", "peak_pytorch_allocated"), args.gpu, **bounds)
    paths = [args.corpus, args.db, *sorted(Path("src/visionbench").rglob("*.py"))]
    fingerprints = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    try:
        provider = model_metadata(args.model)
    except (OSError, ValueError, KeyError) as exc:
        provider = {"metadata_unavailable": str(exc)}
    run = run_session(VisionTools(args.corpus, args.db), request, args.mode, args.model)
    unchanged = fingerprints == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    for row in run["workflow_report"]["measurements"]:
        unchanged &= hashlib.sha256((args.db.parent / row["source_path"]).read_bytes()).hexdigest() == row["source_sha256"]
    if not unchanged:
        run["synthesis"].update(status="generation_rejected", claims=[], error="Une entrée a changé pendant l'exécution.")
    run["protocol"] = {"request": asdict(request), "provider": provider, "source_sha256": fingerprints,
        "collection_timeout_s": COLLECTION_TIMEOUT_S, "total_timeout_s": TOTAL_TIMEOUT_S,
        "synthesis_timeout_s": 90, "max_planning_calls": MAX_MODEL_CALLS, "max_searches": MAX_SEARCHES,
        "max_tool_calls": MAX_TOOL_CALLS, "planner_options": PLANNER_OPTIONS, "planner_prompt": PLANNER_PROMPT,
        "numeric_prerequisites": "python_before_live_planning",
        "model_options": MODEL_OPTIONS, "system_prompt": SYSTEM_PROMPT, "phase_prompts": PHASE_PROMPTS,
        "phase_tokens": PHASE_TOKENS, "think": False, "keep_alive": KEEP_ALIVE,
        "api_charge_eur": 0, "electricity_cost": "not_measured"}
    args.out.mkdir(parents=True)
    (args.out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "answer.md").write_text(render(run), encoding="utf-8")
    print(render(run).split("## Preuves fournies")[0], flush=True)
    print(f"Rapport : {args.out / 'run.json'}")
    if run["workflow_report"]["stop_reason"] != "completed" or run["synthesis"]["status"] in {
            "generation_rejected", "generation_unavailable", "incomplete_workflow"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
