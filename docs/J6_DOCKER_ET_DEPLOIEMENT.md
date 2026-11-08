# J6.2 — Conteneur validé et déploiement préparé

## Validation initiale

La démo fonctionne dans un conteneur Linux amd64 avec Python 3.12.16, sans installation de PyTorch ou de LangGraph. Le conteneur utilise l'utilisateur `demo`, UID 10001. Les six tests d'interface réussissent avec 512 Mio de RAM, deux CPU, aucun réseau, une racine en lecture seule et un répertoire temporaire en mémoire. Ils vérifient aussi la consultation des traces sans appel LLM.

Une instance distincte du serveur a ensuite répondu `200 / ok` à `/_stcore/health` et obtenu le statut Docker `healthy`. Le calcul et l'affichage sont vérifiés dans le navigateur. Ce sont des essais locaux, sans test de charge ni validation d'un hébergement distant.

L'image Python est identifiée par son digest dans `Dockerfile`. `requirements-demo.lock.txt` fixe les 40 versions de paquets résolues sous Linux ; il sert de contraintes à l'installation des dépendances directes. Une modification future des dépendances ou de la base demande une nouvelle validation. Le verrou ne contient pas les empreintes des distributions Python et ne constitue pas une garantie de reconstruction bit à bit.

## Refaire la validation

Depuis le dossier du projet :

```powershell
docker build -t visionbench-ai:comparison-local .
python -X utf8 -B scripts/validate_container.py --image visionbench-ai:comparison-local --port 8502 --out reports/nouvelle_validation_docker.json --keep-running
```

Le rapport conserve l'identifiant de l'image, les versions, les contrôles d'intégrité, les sorties des tests, la santé du serveur et un relevé de ressources. Le script refuse d'écraser un rapport. Il choisit un nom de conteneur unique et ne touche pas aux autres conteneurs. Sans `--keep-running`, il arrête son propre serveur après les contrôles. Avec cette option, son nom figure dans le rapport et il reste disponible sur <http://127.0.0.1:8502>.

Pour un lancement simple :

```powershell
docker run --rm --name visionbench-demo -p 127.0.0.1:8502:8501 visionbench-ai:comparison-local
```

Le port **8501 dans le conteneur** est publié sur **8502 sur le PC**. Le serveur Python historique peut continuer à utiliser le port 8501 du PC. Un autre conteneur utilisant déjà le port ou le nom choisi doit être identifié avant de relancer cette commande.

## Comprendre Docker dans ce projet

| Élément | Rôle |
|---|---|
| `Dockerfile` | Recette : base Python, dépendances, copie des données, utilisateur et commande de démarrage |
| Image | Résultat de la construction ; contient le nécessaire pour lancer la démo |
| Conteneur | Processus lancé depuis cette image, isolé du Python Anaconda du PC |
| `.dockerignore` | Limite les fichiers accessibles à la construction, avant les instructions `COPY` |
| `HEALTHCHECK` | Vérifie que le serveur répond ; les tests d'interface vérifient séparément les fonctionnalités |
| `-p 127.0.0.1:8502:8501` | Rend le serveur accessible uniquement sur ce PC, sur le port 8502 |

Le conteneur ne transforme pas les mesures GPU en nouvelles mesures CPU. L'application lit toujours les résultats de J2 et leurs empreintes. Changer une limite modifie les candidats admissibles, pas les valeurs du tableau.

## Option publique préparée : Render

`render.yaml` décrit un service web Docker, plan `free`, région `frankfurt`, port 8501 et vérification `/_stcore/health`. Sa structure est validée localement avec le [schéma officiel](https://render.com/schema/render.yaml.json). Cela ne crée pas de service et ne remplace pas les essais sur le fournisseur. Render permet de construire depuis le Dockerfile d'un dépôt Git. [Déploiement Docker chez Render](https://render.com/docs/docker)

Le fichier désactive les déploiements automatiques à chaque commit. Le workflow GitHub Actions `.github/workflows/demo.yml` prépare la construction et le même validateur de conteneur. Le workflow a été préparé et sa commande testée localement ; aucune exécution GitHub Actions distante n'est revendiquée. [Référence des Blueprints Render](https://render.com/docs/blueprint-spec)

Pour publier cette copie :

1. Placer le contenu de cette copie à la racine d'un dépôt GitHub destiné à la démo.
2. Vérifier le contenu et les empreintes, puis publier ce dépôt sur le compte choisi.
3. Depuis le compte Render choisi, créer un Blueprint à partir de ce dépôt et vérifier le plan **Free** avant la création.
4. Laisser la construction se terminer, puis relever l'URL réellement attribuée. Aucun nom d'URL n'est présumé réservé.
5. Tester sur cette URL le calcul SSDLite, le changement de candidat, l'absence de candidat, les preuves manquantes, les traces enregistrées et le téléchargement JSON.

Le plan Free se met en veille après 15 minutes sans trafic et son redémarrage peut prendre environ une minute. Son disque est éphémère : la copie SQLite fournie dans l'image convient à la consultation, pas à la conservation de nouvelles expériences. Des quotas d'heures, de bande passante et de construction s'appliquent. Aucun service payant ni moyen de paiement n'est configuré par les fichiers du projet. [Limites du plan Free](https://render.com/docs/free)

**Statut :** Docker validé localement ; archive, workflow CI et configuration publique préparés. Aucun dépôt distant n'a été créé, aucun compte d'hébergement n'a été connecté et aucune URL publique n'est encore déployée.

## Petit exercice de lecture

Lire les instructions `FROM`, `COPY`, `USER` et `CMD` du Dockerfile. Puis retrouver dans `validate_container.py` les deux preuves distinctes : les tests d'interface et la santé HTTP. Une page qui répond n'est pas, à elle seule, une preuve que ses recommandations sont correctes.
