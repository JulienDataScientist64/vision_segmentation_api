# ========================
# Dockerfile FastAPI
# ========================

# Utiliser une image officielle de Python (ici 3.10-slim)
FROM python:3.10-slim

# Variables d'environnement utiles pour éviter l'écriture de bytecode et avoir un output non bufferisé
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires (ici build-essential, souvent utile pour compiler certaines dépendances)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Mettre à jour pip et installer Poetry (ici la version 1.6.1, comme dans l'exemple)
RUN pip install --upgrade pip && pip install poetry==1.6.1

# Configurer Poetry pour qu'il n'utilise pas d'environnement virtuel (ainsi, les dépendances seront installées directement dans l'image)
RUN poetry config virtualenvs.create false

# Copier uniquement les fichiers de configuration de Poetry pour profiter du cache Docker
COPY pyproject.toml poetry.lock /app/

# Installer les dépendances du projet sans installer le package lui-même (option --no-root évite de devoir installer un README.md et la configuration du package)
RUN poetry install --no-interaction --no-ansi --no-root

# Copier le reste du code de l'application (y compris api/, model/, etc.)
COPY . /app/

# Exposer le port 8000 (pour uvicorn)
EXPOSE 8000

# Lancer l'application FastAPI via uvicorn. Ici, on utilise $PORT s'il est défini (utile pour Heroku), sinon on utilise 8000 par défaut.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
