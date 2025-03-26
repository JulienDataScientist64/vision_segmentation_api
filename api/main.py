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

CLASS_TO_CATEGORY = { ... }  # inchangé
MAPPING_TABLE = tf.constant(
    [CLASS_TO_CATEGORY.get(i, 7) for i in range(256)],
    dtype=tf.uint8
)

model = None
infer = None

def load_model():
    global model, infer
    if model is None or infer is None:
        logging.info("Chargement du modèle depuis Hugging Face...")

        HF_TOKEN = os.environ.get("HF_TOKEN")  # Récupère le token HF
        model_path = snapshot_download(
            repo_id="cantalapiedra/semantic-segmentation-model",
            local_dir="/tmp/hf_cache",
            local_dir_use_symlinks=False,
            use_auth_token=HF_TOKEN,
            revision="master"  # si ton dépôt n’a pas de branche main
        )

        # Ton saved_model.pb est à la racine
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
    mask = tf.argmax(pred[pred_key], axis=-1)[0].numpy()
    return mask

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
        pred_mask = predict_mask(img_tensor)

        mask_image = Image.fromarray(pred_mask.astype(np.uint8), mode="L")
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        buffer.seek(0)

        logging.info("Masque généré avec succès.")
        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        logging.error(f"Erreur inférence : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur serveur pendant l'inférence : {e}")
