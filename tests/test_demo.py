import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from visionbench.ui.service import artifact, load_manifest, load_recording, verified_sources


ROOT = Path(__file__).resolve().parents[1]


class DemoTest(unittest.TestCase):
    def app(self, *, explore=True):
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=15).run()
        if explore:
            app.radio(key="demo_view").set_value("Explorer librement").run()
        self.assertEqual(len(app.exception), 0)
        return app

    def submit(self, app):
        next(button for button in app.button if button.label == "Calculer sur les mesures").click().run()
        self.assertEqual(len(app.exception), 0)
        return app.session_state["result"]["report"] if "result" in app.session_state else None

    def test_guided_budget_changes_eligibility_and_keeps_the_same_measurements(self):
        app = self.app(explore=False)
        self.assertNotIn("scenario", {item.key for item in app.selectbox})
        self.assertIn("SSDLite", app.success[0].value)
        self.assertTrue(any("110,4 Mio" in item.value for item in app.markdown))
        before = app.dataframe[0].value
        next(button for button in app.button if button.label == "Passer à 120 Mio").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("Faster R-CNN", app.success[0].value)
        after = app.dataframe[0].value
        self.assertTrue(before.drop(columns="Décision").equals(after.drop(columns="Décision")))
        self.assertEqual(set(before["Décision"]), {"Recommandé", "Hors limites"})
        self.assertEqual(set(after["Décision"]), {"Recommandé", "Admissible"})
        next(button for button in app.button if button.label == "Passer à 50 Mio").click().run()
        self.assertIn("SSDLite", app.success[0].value)

    def test_guided_examples_show_document_and_agent_trace_without_live_generation(self):
        with patch("visionbench.ui.comparison.run_live_answer", side_effect=AssertionError("Aucun appel LLM")):
            app = self.app(explore=False)
        self.assertTrue(any("memory_allocated.html" in item.value for item in app.markdown))
        agent = next(item for item in app.expander if item.label.startswith("Agent ·"))
        content = "\n".join(item.value for item in agent.markdown)
        for fragment in ("première recherche passe à côté", "3 fiches de contexte synthétiques",
                         "définition de la mémoire allouée par PyTorch dans les pipelines de détection",
                         "définition de max_memory_allocated dans PyTorch pour les détecteurs de vision",
                         "maximum d’octets CUDA occupés par les tenseurs actifs", "workflow fixe a aussi trouvé la fiche"):
            self.assertIn(fragment, content)
        self.assertTrue(any("2 recherches · 3 appels" in item.value for item in agent.caption))
        self.assertEqual(sum("Exécution enregistrée" in item.value for item in app.caption), 2)

    def test_selection_widgets_change_the_actual_candidate(self):
        app = self.app()
        self.assertEqual(self.submit(app)["recommendation"]["candidate"], "ssdlite320_mobilenet_v3_large")
        app.number_input(key="memory_memory").set_value(120.0)
        app.number_input(key="memory_p95").set_value(25.0)
        app.checkbox(key="memory_use_ap").set_value(True)
        app.number_input(key="memory_ap").set_value(25.0)
        report = self.submit(app)
        self.assertEqual(report["request"]["ap_min"], 0.25)
        self.assertEqual(report["recommendation"]["candidate"], "fasterrcnn_mobilenet_v3_large_320_fpn")

    def test_no_candidate_and_missing_hardware_are_not_recommendations(self):
        app = self.app()
        app.selectbox(key="scenario").select("none").run()
        self.assertEqual(self.submit(app)["recommendation"]["status"], "no_candidate")
        app.selectbox(key="scenario").select("missing").run()
        report = self.submit(app)
        self.assertEqual(report["status"], "insufficient_evidence")
        self.assertEqual(report["measurements"], [])
        self.assertIsNotNone(report["proposed_experiment"])

    def test_resource_catalogue_changes_the_candidate_at_the_same_limits(self):
        app = self.app()
        pilot = self.submit(app)
        self.assertEqual(pilot["recommendation"]["candidate"], "ssdlite320_mobilenet_v3_large")
        app.selectbox(key="catalogue").select("resources_100").run()
        self.assertNotIn("result", app.session_state)
        campaign = self.submit(app)
        self.assertEqual(campaign["recommendation"]["status"], "no_candidate")
        self.assertIn("resources_100/raw.json", {row["source_path"] for row in campaign["measurements"]})

    def test_new_recorded_paths_are_readable_without_a_model_or_subprocess(self):
        app = self.app()
        with patch("visionbench.ui.service.subprocess.run", side_effect=AssertionError("La lecture est hors ligne.")):
            for mode in ("fixed", "agent", "rag"):
                app.selectbox(key="comparison_mode").select(mode).run()
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any(item.label == "Durée enregistrée" for item in app.metric))

    def test_recorded_mode_never_calls_the_engine(self):
        app = self.app()
        self.submit(app)
        with patch("visionbench.ui.service.subprocess.run", side_effect=AssertionError("Pas de calcul pour la lecture")):
            app.radio(key="mode").set_value("Agent enregistré").run()
            self.assertNotIn("result", app.session_state)
            next(button for button in app.button if button.label == "Afficher l’enregistrement").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["result"]["kind"], "recorded")
        self.assertTrue(any("aucun appel au modèle" in item.value for item in app.info))

    def test_conflicting_question_clears_old_result(self):
        app = self.app()
        self.submit(app)
        app.text_area(key="memory_question").set_value("Choisis un modèle avec une AP minimale de 90 %.")
        app.checkbox(key="memory_use_ap").set_value(True)
        self.assertIsNone(self.submit(app))
        self.assertTrue(any("diffère" in error.value for error in app.error))

    def test_documentary_and_comparison_scenarios_render(self):
        app = self.app()
        for scenario in ("documents", "compare", "latency"):
            app.selectbox(key="scenario").select(scenario).run()
            report = self.submit(app)
            self.assertEqual(report["stop_reason"], "completed")
            json.dumps(app.session_state["result"], allow_nan=False)

    def test_recordings_have_verified_sources_and_tampering_is_rejected(self):
        manifest = load_manifest(ROOT)
        for scenario in manifest["scenarios"].values():
            recording = load_recording(ROOT, manifest, scenario)
            verified_sources(ROOT, recording["report"])
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            for relative in ("data/demo/manifest.json", *manifest["files_sha256"]):
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
            (temporary / "reports/j2_latency_pilot.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Empreinte incorrecte"):
                load_manifest(temporary)
            with self.assertRaisesRegex(ValueError, "rester dans le projet"):
                artifact(temporary, "../hors_projet.txt")


if __name__ == "__main__":
    unittest.main()
