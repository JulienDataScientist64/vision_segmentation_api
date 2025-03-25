import pytest
from fastapi.testclient import TestClient
from api.main import app
from pathlib import Path

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is running"}

@pytest.mark.skipif(
    not Path("tests/frankfurt_000000_001236_leftImg8bit.png").exists(),
    reason="Image de test absente."
)
def test_predict_endpoint():
    # Vérifie d'abord si le modèle est bien présent (pour du debug rapide)
    model_file = Path("./hf_cache/saved_model.pb")
    assert model_file.exists(), f"Le modèle n'existe pas à l'emplacement : {model_file.resolve()}"

    image_path = Path("tests/frankfurt_000000_001236_leftImg8bit.png")
    with open(image_path, "rb") as img:
        files = {"image_file": ("test_image.png", img, "image/png")}
        response = client.post("/predict", files=files)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
