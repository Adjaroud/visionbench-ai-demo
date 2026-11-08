"""J6 : interface légère autour du moteur et des traces évaluées."""

import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from visionbench.ui.service import (METRIC_NAMES, MODEL_NAMES, artifact, load_manifest,
                                   load_comparison, load_recording, measurement_rows, run_fixed_report, verified_sources)
from visionbench.ui.comparison import render_evaluation, render_panel
from visionbench.ui.guided import render_guided


def render_report(envelope):
    report = envelope["report"]
    if envelope["kind"] == "recorded":
        st.info(f"Enregistrement · {envelope['model']} · campagne du {envelope['campaign_date']} · "
                "aucun appel au modèle pendant cette consultation.")
    else:
        st.caption("Calcul à la demande sur le catalogue existant · CPU · aucun appel LLM.")
    st.write("**Question traitée**")
    st.write(report["request"]["text"])
    recommendation = report.get("recommendation") or {}
    if report["status"] == "partial":
        st.error("Exécution interrompue : les preuves sont incomplètes.")
    elif recommendation.get("status") == "pilot_candidate":
        st.success("Candidat sur les mesures disponibles · " + MODEL_NAMES.get(recommendation["candidate"], recommendation["candidate"]))
    elif recommendation.get("status") == "no_candidate":
        st.warning("Aucun des deux modèles mesurés ne satisfait toutes les contraintes.")
    elif report["status"] == "insufficient_evidence":
        st.warning("Preuves insuffisantes : aucune recommandation chiffrée n'est disponible pour cette demande.")
    elif report["status"] == "passages_found":
        st.info("Passages documentaires retrouvés. Ils sont présentés comme extraits, sans synthèse générée.")
    else:
        st.success("Mesures retrouvées et contrôles disponibles ci-dessous.")

    tabs = st.tabs(["Mesures", "Documents", "Contraintes et contrôles", "Trace"])
    with tabs[0]:
        rows = measurement_rows(report)
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
            st.caption("L'AP est une fraction entre 0 et 1. La colonne Images indique le périmètre de chaque mesure.")
            with st.expander("Provenance et fichiers de mesures"):
                for relative, source in verified_sources(ROOT, report).items():
                    st.code(relative + "\nSHA-256 : " + source["sha256"], language=None)
                    st.download_button("Télécharger " + Path(relative).name, source["raw"],
                                       file_name=Path(relative).name, mime="application/json", key=relative)
        else:
            st.write("Aucune mesure dans ce résultat.")
        if report.get("proposed_experiment"):
            st.write("**Expérience proposée**")
            st.write(report["proposed_experiment"]["protocol"])
            st.json(report["proposed_experiment"])
    with tabs[1]:
        if not report.get("documents"):
            st.write("Aucun passage suffisamment pertinent n'a été retenu.")
        for document in report.get("documents", []):
            st.markdown(f"**[{document['title']}]({document['url']})**")
            st.caption(document["section"])
            st.write(document["text"])
        st.caption("La recherche est lexicale. Une source retrouvée ne garantit pas qu'elle suffise à répondre.")
    with tabs[2]:
        st.json(report["request"])
        for metric, verdict in report.get("comparisons", {}).items():
            st.write("**" + METRIC_NAMES.get(metric, metric) + "**")
            st.write("Comparable dans cette campagne" if verdict["comparable"] else "Comparaison insuffisante ou incompatible")
            for reason in verdict["reasons"]:
                st.write(reason)
            if verdict.get("scope"):
                st.caption("Périmètre : " + verdict["scope"])
        st.caption("Comparer les pipelines complets ne permet pas d'isoler l'effet de leur architecture.")
    with tabs[3]:
        st.dataframe(report["trace"], hide_index=True, use_container_width=True)
        st.caption(f"Arrêt : {report['stop_reason']} · appels LLM : {report['model_calls']} · "
                   f"durée de l'exécution {'enregistrée' if envelope['kind'] == 'recorded' else 'courante'} : {report['duration_s']:.3f} s")
        with st.expander("Rapport JSON complet"):
            st.json(envelope)
    st.download_button("Exporter ce résultat en JSON", json.dumps(envelope, ensure_ascii=False, indent=2),
                       file_name="visionbench_resultat.json", mime="application/json")


def evaluation(manifest):
    summary = json.loads(artifact(ROOT, "reports/model_comparison/summary.json"))
    st.subheader("Comparaison historique des outils")
    st.write("20 cas communs, trois passages par cas et par variante : 180 exécutions. "
             "Les répétitions ne sont pas 180 problèmes indépendants.")
    rows = []
    for key, label in (("fixed", "Workflow fixe"), ("llama3.1:8b", "Agent Llama 3.1 8B"), ("qwen3:4b", "Agent Qwen3 4B")):
        values = summary[key]
        rows.append({"Variante": label, "Exécutions réussies": f"{values['successes']}/{values['runs']}",
                     "Cas réussis aux 3 passages": f"{values['cases_passing_all_repeats']}/20",
                     "Médiane globale (s)": round(values["median_s"], 3),
                     "Requêtes avec LLM": values["model_used_runs"],
                     "Médiane avec LLM (s)": round(values["model_used_median_s"], 3) if values["model_used_median_s"] else None})
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption("Mesures historiques sur CPU, échauffements exclus. Les outils calculent les recommandations en Python. "
               "Une même réussite ne montre pas un bénéfice de l'agent sur ces cas.")
    with st.expander("Limites et protocole"):
        st.write("Cas construits en connaissant l'architecture ; domaine restreint. "
                 "Qwen est plus rapide sur les requêtes utilisant le modèle dans cet essai. "
                 "Les synthèses libres sont évaluées séparément et restent imparfaites, notamment sur l'abstention.")
        st.json(manifest["evaluation"])
    st.download_button("Télécharger le résumé de comparaison", artifact(ROOT, "reports/model_comparison/summary.json"),
                       file_name="comparaison_modeles.json", mime="application/json")


def render_explorer(manifest, comparison):
    columns = st.columns(3)
    columns[0].metric("Détecteurs mesurés", "2")
    columns[1].metric("Images pour l’AP", "500")
    columns[2].metric("GPU des mesures", "RTX 3050 Laptop")
    st.caption("Les calculs consultent des résultats préexistants. Deux campagnes de ressources sont disponibles.")
    with st.sidebar:
        st.header("Explorer VisionBench")
        mode = st.radio("Mode", ["Calcul à la demande", "Agent enregistré"], key="mode")
        catalogue = st.selectbox("Catalogue de mesures", ["pilot", "resources_100"],
                                 format_func={"pilot": "Pilote sur 3 images", "resources_100": "Campagne sur 100 images"}.__getitem__,
                                 key="catalogue", disabled=mode == "Agent enregistré")
        scenario_id = st.selectbox("Scénario", list(manifest["scenarios"]),
                                   format_func=lambda key: manifest["scenarios"][key]["label"], key="scenario")
        st.caption("Sans clé API ni GPU pour cette démonstration.")
    scenario = manifest["scenarios"][scenario_id]
    identity = (mode, scenario_id, catalogue)
    if st.session_state.get("identity") != identity:
        st.session_state["identity"] = identity
        st.session_state.pop("result", None)
        st.session_state.pop("input_error", None)

    explore, llm, results = st.tabs(["Explorer les preuves", "Parcours LLM", "Évaluation"])
    with explore:
        st.subheader(scenario["label"])
        if mode == "Agent enregistré":
            st.write("Consulte une exécution réelle de Qwen3 4B, conservée lors de la comparaison. "
                     "La question et les contraintes sont celles de cet enregistrement.")
            if st.button("Afficher l’enregistrement", type="primary"):
                st.session_state["result"] = load_recording(ROOT, manifest, scenario)
        else:
            defaults = scenario["live"]
            prefix = scenario_id + "_"
            with st.form("request_" + scenario_id):
                question = st.text_area("Question", value=defaults["question"], max_chars=700, key=prefix + "question")
                quantitative = defaults["kind"] != "documents"
                gpu = None
                p95 = memory = ap = None
                if quantitative:
                    gpu = st.text_input("GPU auquel les mesures doivent correspondre", defaults["gpu"], key=prefix + "gpu")
                if defaults["kind"] == "selection":
                    left, center, right = st.columns(3)
                    with left:
                        use_p95 = st.checkbox("Limiter le p95", value=defaults.get("p95") is not None, key=prefix + "use_p95")
                        value_p95 = st.number_input("p95 maximal (ms)", min_value=0.001, value=float(defaults.get("p95") or 80), key=prefix + "p95")
                        p95 = value_p95 if use_p95 else None
                    with center:
                        use_memory = st.checkbox("Limiter la mémoire allouée", value=defaults.get("memory") is not None, key=prefix + "use_memory")
                        value_memory = st.number_input("Mémoire maximale (Mio)", min_value=0.001, value=float(defaults.get("memory") or 50), key=prefix + "memory")
                        memory = value_memory if use_memory else None
                    with right:
                        use_ap = st.checkbox("Exiger une AP minimale", value=defaults.get("ap") is not None, key=prefix + "use_ap")
                        value_ap = st.number_input("AP minimale (%)", min_value=0.0, max_value=100.0, value=float(defaults.get("ap", 0.25) * 100), key=prefix + "ap")
                        ap = value_ap / 100 if use_ap else None
                    st.caption("Les limites activées s'appliquent ensemble. Si tu inscris aussi une valeur dans la question, elle doit correspondre.")
                submitted = st.form_submit_button("Calculer sur les mesures", type="primary")
            if submitted:
                st.session_state.pop("result", None)
                st.session_state.pop("input_error", None)
                try:
                    if quantitative and not gpu.strip():
                        raise ValueError("Précise le GPU demandé.")
                    if defaults["kind"] == "selection" and all(v is None for v in (p95, memory, ap)):
                        raise ValueError("Active au moins une contrainte pour sélectionner un détecteur.")
                    with st.spinner("Consultation du catalogue et vérification des contraintes…"):
                        report = run_fixed_report(ROOT, question, gpu=gpu, metrics=defaults.get("metrics", ()),
                                                  p95=p95, memory=memory, ap=ap, catalogue=catalogue)
                    st.session_state["result"] = {"kind": "calculated", "created_at_utc": datetime.now(timezone.utc).isoformat(), "report": report}
                except (OSError, ValueError) as exc:
                    st.session_state["input_error"] = str(exc)
        if st.session_state.get("input_error"):
            st.error(st.session_state["input_error"])
        if st.session_state.get("result"):
            st.divider()
            render_report(st.session_state["result"])
        else:
            st.info("Choisis un scénario puis lance le calcul ou affiche son enregistrement.")
    with llm:
        render_panel(ROOT, comparison)
    with results:
        render_evaluation(ROOT, comparison)
        evaluation(manifest)


def main():
    st.set_page_config(page_title="VisionBench AI", page_icon="🔎", layout="wide")
    st.caption("DÉMONSTRATION INTERACTIVE")
    st.title("VisionBench AI")
    try:
        manifest = load_manifest(ROOT)
        comparison = load_comparison(ROOT)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Démonstration indisponible : {exc}")
        st.stop()

    view = st.radio("Choisir un parcours", ["Découverte guidée", "Explorer librement"],
                    horizontal=True, label_visibility="collapsed", key="demo_view")
    if view == "Découverte guidée":
        st.session_state.pop("result", None)
        st.session_state.pop("input_error", None)
        st.session_state.pop("comparison_result", None)
        render_guided(ROOT, comparison)
    else:
        render_explorer(manifest, comparison)


if __name__ == "__main__":
    main()
