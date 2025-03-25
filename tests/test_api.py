import pytest
from fastapi.testclient import TestClient
from api.main import app
from pathlib import Path
import sys

# S'assurer que le dossier api/ est dans le path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is running"}

def test_predict_endpoint():
    # Chemin vers une image test (tu dois l’avoir dans le dossier tests/)
    image_path = Path("tests/frankfurt_000000_001236_leftImg8bit.png")

    assert image_path.exists(), f"L'image de test est introuvable à : {image_path}"

    with open(image_path, "rb") as img:
        files = {"image_file": ("test_image.png", img, "image/png")}
        response = client.post("/predict", files=files)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
