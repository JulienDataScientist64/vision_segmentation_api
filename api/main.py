import os
import logging
import numpy as np
import tensorflow as tf
from huggingface_hub import snapshot_download
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="API Segmentation Sémantique")

CLASS_TO_CATEGORY = {
    7: 0, 8: 0, 9: 0, 10: 0,
    24: 1, 25: 1,
    26: 2, 27: 2, 28: 2, 29: 2, 30: 2, 31: 2, 32: 2, 33: 2,
    11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3,
    17: 4, 18: 4, 19: 4, 20: 4,
    21: 5, 22: 5,
    23: 6,
    0: 7, 1: 7, 2: 7, 3: 7, 4: 7, 5: 7, 255: 7
}

# Palette 8 classes (exemple style Cityscapes)
CITYSCAPES_8_PALETTE = [
    (128, 64, 128),   # classe 0
    (244, 35, 232),   # classe 1
    (70, 70, 70),     # classe 2
    (102, 102, 156),  # classe 3
    (190, 153, 153),  # classe 4
    (153, 153, 153),  # classe 5
    (250, 170, 30),   # classe 6
    (220, 220, 0),    # classe 7
]

model = None
infer = None

def load_model():
    global model, infer
    if model is None or infer is None:
        logging.info("Chargement du modèle depuis Hugging Face...")

        hf_token = os.environ.get("HF_TOKEN")
        model_path = snapshot_download(
            repo_id="cantalapiedra/semantic-segmentation-model",
            revision="master",
            use_auth_token=hf_token,
            local_dir="/tmp/hf_cache",
            local_dir_use_symlinks=False
        )
        model = tf.saved_model.load(model_path)
        infer = model.signatures["serving_default"]
        logging.info("Modèle chargé avec succès.")

def preprocess_image(img: Image.Image) -> tf.Tensor:
    img_resized = img.resize((512, 256))
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    tensor = tf.convert_to_tensor(arr)
    return tf.expand_dims(tensor, axis=0)

def predict_mask(img_tensor: tf.Tensor) -> np.ndarray:
    load_model()
    pred = infer(img_tensor)
    pred_key = list(pred.keys())[0]
    # shape (256,512), valeurs 0..7
    mask = tf.argmax(pred[pred_key], axis=-1)[0].numpy()
    return mask

def colorize_mask_8(mask_array: np.ndarray, palette: list) -> Image.Image:
    """
    Convertit un masque (H,W) indices [0..7] en image RGB avec la palette.
    """
    h, w = mask_array.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(palette):
        color_mask[mask_array == cls_idx] = color
    return Image.fromarray(color_mask, mode="RGB")

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/predict")
async def predict(image_file: UploadFile = File(...)):
    logging.info(f"Requête reçue avec content-type: {image_file.content_type}")

    if image_file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(status_code=400, detail="Image doit être PNG ou JPEG.")

    try:
        contents = await image_file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        logging.info(f"Image chargée : taille={img.size}, mode={img.mode}")
    except Exception as e:
        logging.error(f"Erreur lecture image : {e}")
        raise HTTPException(status_code=400, detail="Impossible de lire l'image.")

    try:
        img_tensor = preprocess_image(img)
        pred_mask = predict_mask(img_tensor)  # shape (256,512)

        # Colorisation du masque
        colored_mask = colorize_mask_8(pred_mask, CITYSCAPES_8_PALETTE)

        buffer = io.BytesIO()
        colored_mask.save(buffer, format="PNG")
        buffer.seek(0)

        logging.info("Masque coloré généré avec succès.")
        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        logging.error(f"Erreur inférence : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur serveur pendant l'inférence : {e}")
