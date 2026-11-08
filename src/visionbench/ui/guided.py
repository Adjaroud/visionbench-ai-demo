"""Parcours guidé pour découvrir la sélection et ses preuves."""

import streamlit as st

from visionbench.ui.service import MODEL_NAMES, load_comparison_recording, run_fixed_report, verified_sources


GUIDED_LATENCY_MS = 120
GUIDED_MEMORY_MIB = (50, 120)
GUIDED_GPU = "RTX 3050 Laptop"
QUESTION = "Quel détecteur respecte les contraintes indiquées avec la meilleure AP ?"

def _number(value, digits=1):
    return f"{value:.{digits}f}".replace(".", ",")


def _measurements_by_model(report):
    by_model = {}
    for row in report["measurements"]:
        by_model.setdefault(row["model"], {})[row["metric"]] = row["value"]
    return by_model


def _render_selection(root, report, next_memory):
    recommendation = report.get("recommendation") or {}
    candidate = recommendation.get("candidate")
    if candidate is None:
        st.warning("Les mesures disponibles ne permettent pas de recommander un détecteur pour ces contraintes.")
        return

    eligible = recommendation["eligible"]
    memory_limit = report["request"]["memory_limit_mib"]
    models = _measurements_by_model(report)
    st.success(f"Détecteur recommandé · {MODEL_NAMES[candidate]}")
    if len(eligible) == 1:
        for model, values in models.items():
            memory = values["peak_pytorch_allocated"]
            if model not in eligible and memory > memory_limit:
                st.write(f"**{MODEL_NAMES[model]}** dépasse le plafond mémoire "
                         f"avec **{_number(memory)} Mio**. {MODEL_NAMES[candidate]} respecte les deux limites.")
    else:
        st.write(f"Les deux détecteurs respectent les limites. {MODEL_NAMES[candidate]} a le score de "
                 f"précision (AP) le plus élevé, **{_number(100 * models[candidate]['ap'], 2)} %**.")

    st.button(f"Passer à {next_memory} Mio", type="primary", on_click=st.session_state.update,
              args=({"guided_memory": next_memory},))
    st.caption("Le bouton change uniquement le budget mémoire. Les mesures du catalogue pilote restent les mêmes.")

    with st.expander("Voir les preuves et les sources"):
        rows = [{
            "Détecteur": MODEL_NAMES[model],
            "Latence p95": f"{_number(values['pipeline_p95'])} ms",
            "Mémoire allouée": f"{_number(values['peak_pytorch_allocated'])} Mio",
            "Précision (AP)": f"{_number(100 * values['ap'], 2)} %",
            "Décision": "Recommandé" if model == candidate else "Admissible" if model in eligible else "Hors limites",
        } for model, values in models.items()]
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption("Le p95 est le temps sous lequel se situent environ 95 % des observations. "
                   "Latence et mémoire mesurées sur le pilote de 3 images. AP évaluée sur 500 images COCO.")
        st.write("**Rapports de mesure**")
        for relative, source in verified_sources(root, report).items():
            st.download_button(f"Télécharger {relative.rsplit('/', 1)[-1]}", source["raw"],
                               file_name=relative.rsplit("/", 1)[-1], mime="application/json", key=f"guided_{relative}")


def _render_recordings(root, comparison):
    st.header("2. Suivre les preuves")
    st.write("Ouvre les exemples enregistrés pour voir la réponse du RAG et comment l’agent adapte sa recherche.")
    rag_mode = "guided_rag" if "guided_rag" in comparison["recordings"] else "rag"
    rag = load_comparison_recording(root, comparison, rag_mode)
    rag_question = "Que mesure la mémoire GPU allouée ?" if rag_mode == "guided_rag" else rag["question"]
    with st.expander(f"RAG · {rag_question}"):
        passages = {passage["id"]: passage for passage in rag["retrieved_passages"]}
        st.write("Le RAG retrouve une définition dans la documentation et affiche l’extrait qui l’étaye.")
        for claim in rag["claims"]:
            st.write(claim["text"])
            for source in claim["citations"]:
                passage = passages[source]
                st.markdown(f"Source · [{passage['title']}]({passage['url']})")
        st.caption(f"Exécution enregistrée · {rag['model_calls']} appels au modèle.")

    if "guided_agent" in comparison["recordings"]:
        _render_agent_recovery(load_comparison_recording(root, comparison, "guided_agent"))


def _render_agent_recovery(agent):
    report = agent["workflow_report"]
    first, second = report["searches"]
    passages = {passage["id"]: passage for passage in report["documents"]}
    distractor = passages[first["accepted_ids"][0]]
    definition = passages["memory-allocated"]

    with st.expander("Agent · Que fait-il quand la première recherche échoue ?"):
        st.write("L’agent doit expliquer la mémoire GPU allouée. Pour tester sa réaction, "
                 "le corpus contient des fiches synthétiques qui reprennent les bons mots sans donner la définition.")

        st.markdown("**1. La première recherche passe à côté de la définition**")
        st.markdown(f"> {first['query']}")
        st.write(f"La recherche renvoie {len(first['accepted_ids'])} fiches de contexte synthétiques. "
                 "Voici un extrait de l’une d’elles.")
        st.markdown(f"> {distractor['text']}")

        st.markdown("**2. L’agent choisit une requête plus ciblée**")
        st.markdown(f"> {second['query']}")
        st.write("Il cible le nom de la fonction PyTorch. L’outil de recherche écarte les fiches déjà remontées.")

        st.markdown("**3. La fiche qui contient la définition est retrouvée**")
        st.markdown(f"> {definition['text']}")
        st.markdown(f"Documentation de référence · [{definition['title']}]({definition['url']})")

        st.caption(f"Exécution enregistrée · {len(report['searches'])} recherches · "
                   f"{agent['total_model_calls']} appels au modèle, synthèse comprise.")
        st.write("Le modèle choisit comment chercher la preuve. Python vérifie les mesures et calcule "
                 "la recommandation. Dans ce test, le workflow fixe a aussi trouvé la fiche avec "
                 "ses requêtes prévues à l’avance.")


def render_guided(root, comparison):
    st.header("1. Choisir un détecteur")
    st.write("Tu disposes de peu de mémoire GPU et tu veux garder la meilleure précision possible. "
             "Quel détecteur choisir sur cette machine ?")
    memory = st.session_state.get("guided_memory", GUIDED_MEMORY_MIB[0])
    next_memory = GUIDED_MEMORY_MIB[1] if memory == GUIDED_MEMORY_MIB[0] else GUIDED_MEMORY_MIB[0]
    st.caption(f"{GUIDED_GPU} · Latence p95 maximale de {GUIDED_LATENCY_MS} ms · Budget mémoire de {memory} Mio")
    try:
        report = run_fixed_report(root, QUESTION, gpu=GUIDED_GPU, p95=GUIDED_LATENCY_MS, memory=memory)
        _render_selection(root, report, next_memory)
    except (OSError, ValueError) as exc:
        st.error(f"Impossible de calculer la recommandation. {exc}")

    _render_recordings(root, comparison)
    st.caption("Ouvre « Explorer librement » en haut de la page pour choisir d’autres contraintes, "
               "changer de catalogue ou consulter les résultats d’évaluation.")
