# ========================
#   Dockerfile FastAPI
# ========================

# 1. Base officielle avec Python
FROM python:3.10-slim

# 2. Définir le répertoire de travail
WORKDIR /app

# 3. Copier les fichiers nécessaires
COPY pyproject.toml poetry.lock ./
COPY api/ ./api
COPY model/ ./model
COPY tests/ ./tests

# 4. Installer Poetry
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root

# 5. Installer les dépendances du projet
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# 6. Exposer le port utilisé par Uvicorn
EXPOSE 8000

# 7. Lancer l'application FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
