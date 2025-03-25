from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import patch
import pytest
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
@patch('api.main.predict_mask')
def test_predict_endpoint(mock_predict_mask):
    # Simuler une prédiction du modèle sans le charger
    mock_predict_mask.return_value = (np.zeros((256, 512), dtype=np.uint8))

    image_path = Path("tests/frankfurt_000000_001236_leftImg8bit.png")
    with open(image_path, "rb") as img:
        files = {"image_file": ("test_image.png", img, "image/png")}
        response = client.post("/predict", files=files)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
