# semantic-segmentation-api

Cette API fournit un service de segmentation sémantique d'image en s'appuyant sur un modèle TensorFlow hébergé sur Hugging Face.

## Déploiement

Le projet est déployé automatiquement sur Heroku via une pipeline GitHub Actions.

## Structure du projet
├── api/ │ └── main.py # Code principal de l'API (FastAPI) ├── tests/ │ └── test_api.py # Tests unitaires ├── Dockerfile ├── Procfile ├── pyproject.toml ├── poetry.lock ├── runtime.txt ├── .gitignore ├── .slugignore ├── .github/ │ └── workflows/ │ └── deploy.yml # Déploiement continu vers Heroku
## Modèle

Le modèle de segmentation sémantique est stocké sur Hugging Face à l'adresse suivante :

https://huggingface.co/JulienDataScientist64/semantic-segmentation-model

Il est automatiquement téléchargé au démarrage de l'API à l'aide de `snapshot_download`.

## Endpoint principal

- `POST /predict` : envoie une image (format PNG ou JPEG) et retourne un masque segmenté en sortie (format PNG).