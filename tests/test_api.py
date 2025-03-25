import pytest
from fastapi.testclient import TestClient
from huggingface_hub import snapshot_download
from pathlib import Path

# ⚠️ On force le téléchargement du modèle pour éviter l'erreur de chargement
snapshot_download(repo_id="cantalapiedra/semantic-segmentation-model", local_dir_use_symlinks=False)

from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is running"}

@pytest.mark.skipif(
    not Path("tests/frankfurt_000000_001236_leftImg8bit.png").is_file(),
    reason="Image de test introuvable, test ignoré."
)
def test_predict_endpoint():
    image_path = Path("tests/frankfurt_000000_001236_leftImg8bit.png")

    with open(image_path, "rb") as img_file:
        files = {"image_file": ("test_image.png", img_file, "image/png")}
        response = client.post("/predict", files=files)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
