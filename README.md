# VisionBench AI · Démonstration

VisionBench AI aide à choisir un détecteur d’objets sous contraintes de latence, de mémoire GPU et de précision. La démonstration compare SSDLite et Faster R-CNN, tous deux avec MobileNetV3. Les mesures proviennent d’une RTX 3050 Laptop et l’AP a été évaluée sur 500 images COCO.

## Essayer la démo

```sh
docker build -t visionbench-ai:demo .
docker run --rm -p 127.0.0.1:8502:8501 visionbench-ai:demo
```

Ouvrir <http://127.0.0.1:8502>.

En **Découverte guidée**, passe de 50 à 120 Mio pour voir la recommandation changer. Ouvre les preuves pour retrouver les mesures, la citation du RAG et les deux requêtes de l’agent dans un test avec des fiches distractrices. Choisis **Explorer librement** pour accéder aux réglages et aux comparaisons.

- **Explorer les preuves** calcule une recommandation à partir de l’un des deux catalogues de mesures. Modifier les limites change les candidats admissibles.
- **Parcours LLM** affiche des exécutions locales enregistrées du RAG, du workflow fixe et de l’agent LangGraph, avec leurs sources et leurs décisions.
- **Évaluation** présente la comparaison associée à ces enregistrements, puis la comparaison historique des choix d’outils.

Le conteneur tourne sur CPU. Il lit les mesures et les traces fournies. Le calcul sur les catalogues s’exécute à la demande.

## Lire le code

- [RAG documentaire](src/visionbench/retrieval/verified_rag.py) · recherche de passages, vérification des affirmations et affichage des extraits cités.
- [Workflow fixe](src/visionbench/orchestration/workflow.py) et [agent LangGraph](src/visionbench/orchestration/evidence_agent.py) · collecte des preuves et recommandation sous contraintes.
- [Synthèse sourcée](src/visionbench/synthesis/grounded.py) · calcul de la sélection et contrôle de la définition documentaire.
- [Tests de la démo](tests/test_demo.py) et [guide Docker](docs/J6_DOCKER_ET_DEPLOIEMENT.md) · vérification des données, des parcours et de la santé HTTP.

## Rejouer les parcours LLM en local

Avec Python 3.12 dans un environnement isolé et Ollama démarré, installer les dépendances et télécharger le modèle local.

```sh
python -m pip install -r requirements-demo.txt
python -m pip install --target .visionbench-j4 -r requirements-j4.txt
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

Sous PowerShell, activer les appels locaux et démarrer Streamlit.

```powershell
$env:VISIONBENCH_LIVE = "1"
$env:PYTHONPATH = "$PWD\src;$PWD\.visionbench-j4"
python -X utf8 -B -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8502
```

Le dossier séparé contient les dépendances de l’agent. Dans **Explorer librement → Parcours LLM**, choisir **Appels Ollama sur ce PC**. La version hébergée est prévue pour la lecture des traces.

## Vérifier ou héberger la démo

```sh
python scripts/validate_container.py --image visionbench-ai:demo --out reports/validation-locale.json
```

Le fichier `render.yaml` décrit un service Docker Render Free. Il reste à connecter le dépôt dans Render, puis à vérifier l’URL attribuée après le premier déploiement. [Guide de déploiement Docker](https://render.com/docs/docker)
