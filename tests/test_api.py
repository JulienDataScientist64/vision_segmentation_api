import pytest
from fastapi.testclient import TestClient
from api.main import app

import sys
from pathlib import Path

# Ceci indique clairement à Python où trouver le dossier api
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Instancier un client de test pour ton API
client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is running"}

def test_predict_endpoint():
    
    image_path = "tests/frankfurt_000000_001236_leftImg8bit.png"

    
    with open(image_path, "rb") as img:
        files = {"image_file": ("test_image.png", img, "image/png")}
        response = client.post("/predict", files=files)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
