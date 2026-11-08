"""Lecture des traces de comparaison et exécutions locales explicites."""

import json
import os

import streamlit as st

from visionbench.ui.service import MODEL_NAMES, artifact, load_comparison_recording, measurement_rows, run_live_answer, verified_sources


MODES = {"rag": "RAG documentaire", "fixed": "Workflow fixe avec LLM", "agent": "Agent LangGraph"}
SAMPLE_QUESTION = "Quel détecteur respecte ces limites ? Compare les AP et définis la mémoire PyTorch allouée."


def _result_summary(mode, result, archived):
    if mode == "rag":
        claims = result["claims"]
        calls = result["model_calls"]
        tokens = result["usage"]["generated_tokens"]
        status = result["status"]
        model = result["model"]
        passages = {doc["id"]: doc for doc in result["retrieved_passages"]}
    else:
        synthesis = result["synthesis"]
        claims = synthesis["claims"]
        calls = result["total_model_calls"]
        tokens = result["total_usage"]["generated_tokens"]
        status = synthesis["status"]
        model = synthesis["model"]
        passages = {item["id"]: item["source"] for item in synthesis["context"]["evidence"] if item["kind"] == "document"}
    columns = st.columns(3)
    columns[0].metric("Appels LLM", calls)
    columns[1].metric("Tokens générés", tokens)
    columns[2].metric("Durée enregistrée" if archived else "Durée", f"{result['duration_s']:.2f} s")
    st.caption(f"Statut de la réponse · {status}")
    if status == "reviewed_answer":
        st.caption("Réponse relue · extrait du corpus")
    else:
        st.caption(f"Modèle · {model}")
    for claim in claims:
        origin = {"python": "Calcul Python", "llm": "Définition documentaire"}.get(claim.get("origin"), "Réponse documentaire")
        if claim.get("rendering") == "exact_quote":
            origin = "Extrait exact du corpus"
        st.caption(origin)
        st.write(claim["text"])
        labels = []
        for identifier in claim["citations"]:
            passage = passages.get(identifier)
            labels.append(f"[{passage['title']}]({passage['url']})" if passage else f"`{identifier}`")
        st.markdown("Sources · " + ", ".join(labels))
        if claim.get("support_quote") and claim.get("rendering") != "exact_quote":
            with st.expander("Lire l’extrait exact"):
                st.write(claim["support_quote"])
    if mode == "rag":
        for item in result["missing_information"]:
            st.warning(item)
        if result["error"]:
            st.error(result["error"])
        return
    synthesis = result["synthesis"]
    if synthesis["error"]:
        st.warning(synthesis["error"])
    if synthesis.get("verified_scope"):
        st.caption(synthesis["verified_scope"]["text"])


def render_panel(root, manifest):
    st.subheader("Parcours documentaires et mixtes")
    labels = ["Exécution enregistrée"]
    if os.environ.get("VISIONBENCH_LIVE") == "1":
        labels.append("Appels Ollama sur ce PC")
    access = st.radio("Type d’exécution", labels, key="comparison_access")
    mode = st.selectbox("Parcours", list(MODES), format_func=MODES.__getitem__, key="comparison_mode")
    identity = (access, mode)
    if st.session_state.get("comparison_identity") != identity:
        st.session_state["comparison_identity"] = identity
        st.session_state.pop("comparison_result", None)
    archived = access == "Exécution enregistrée"
    if archived:
        st.info("Cette trace provient de la campagne locale. La lecture ne déclenche aucun appel LLM.")
        result = load_comparison_recording(root, manifest, mode)
    else:
        st.info("Ollama utilise le CPU de ce PC. Les parcours mixtes consultent la campagne sur 100 images.")
        with st.form("llm_request_" + mode):
            question = st.text_area("Question pour le LLM", value=manifest["rag_question"] if mode == "rag" else SAMPLE_QUESTION, max_chars=700)
            gpu, p95, memory = "RTX 3050 Laptop", 120.0, 50.0
            if mode != "rag":
                gpu = st.text_input("GPU mesuré", gpu)
                p95 = st.number_input("p95 maximal en ms", min_value=0.001, value=p95)
                memory = st.number_input("Mémoire allouée maximale en Mio", min_value=0.001, value=memory)
            submitted = st.form_submit_button("Exécuter le parcours", type="primary")
        if submitted:
            st.session_state.pop("comparison_result", None)
            try:
                with st.spinner("Recherche des preuves et appels au modèle local…"):
                    st.session_state["comparison_result"] = run_live_answer(root, mode, question, gpu=gpu, p95=p95, memory=memory)
            except (OSError, ValueError) as exc:
                st.error(str(exc))
        result = st.session_state.get("comparison_result")
    if result is None:
        return

    if mode == "rag":
        st.write(result["question"])
    else:
        report = result["workflow_report"]
        st.write(report["request"]["text"])
        recommendation = report.get("recommendation") or {}
        candidate = recommendation.get("candidate")
        if candidate:
            st.success(MODEL_NAMES.get(candidate, candidate))
        with st.expander("Contraintes, mesures et traces"):
            verified_sources(root, report)
            st.json(report["request"])
            st.dataframe(measurement_rows(report), hide_index=True)
            st.json(report.get("decisions") or report.get("trace"))
            if report.get("proposed_experiment"):
                st.write(report["proposed_experiment"]["protocol"])
    _result_summary(mode, result, archived)
    with st.expander("Rapport JSON du parcours"):
        st.json(result)


def render_evaluation(root, manifest):
    summary_path = manifest["summary"]
    summary = json.loads(artifact(root, summary_path, manifest["files_sha256"][summary_path]))
    st.subheader("Comparaison sur les mesures à 100 images")
    st.write("Quatre cas appariés pour le fixe et l’agent. Quatre questions documentaires distinctes pour le RAG. Trois passages par cas.")
    rows = []
    for mode, label in MODES.items():
        entry = summary["modes"][mode]
        rows.append({"Parcours": label, "Contrôles satisfaits": f"{entry['automatic_passes']}/{entry['runs']}",
                     "Durée médiane (s)": round(entry["median_s"], 2), "Appels LLM": entry["model_calls"],
                     "Tokens générés": entry["generated_tokens"]})
    st.dataframe(rows, hide_index=True)
    st.caption("La comparaison directe porte sur les deux parcours mixtes. Les contrôles de texte du RAG sont complétés par une relecture.")
    with st.expander("Résultats appariés par cas"):
        st.dataframe([{"Cas": row["id"], "Fixe": f"{row['fixed']}/{row['repetitions']}",
                       "Agent": f"{row['agent']}/{row['repetitions']}"}
                      for row in summary["paired_cases"]], hide_index=True)
    st.download_button("Télécharger la comparaison récente", artifact(root, summary_path, manifest["files_sha256"][summary_path]),
                       file_name="comparaison_visionbench.json", mime="application/json")

