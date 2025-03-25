# api/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from huggingface_hub import snapshot_download
from pathlib import Path

app = FastAPI(
    title="API Segmentation Sémantique",
    version="1.0.0",
    description="Inférence segmentation sémantique utilisant TensorFlow."
)

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

MAPPING_TABLE = tf.constant(
    [CLASS_TO_CATEGORY.get(i, 7) for i in range(256)],
    dtype=tf.uint8
)

# Charger le modèle uniquement à la première requête
model = None
infer = None

def load_model():
    global model, infer
    if model is None or infer is None:
        model_path = snapshot_download("cantalapiedra/semantic-segmentation-model")
        model = tf.saved_model.load(model_path)
        infer = model.signatures["serving_default"]

def preprocess_image(img: Image.Image):
    img_resized = img.resize((512, 256))
    img_tensor = tf.convert_to_tensor(np.array(img_resized), dtype=tf.float32) / 255.0
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    return img_tensor

def predict_mask(img_tensor):
    load_model()
    pred = infer(img_tensor)
    pred_key = list(pred.keys())[0]
    pred_mask = tf.argmax(pred[pred_key], axis=-1)[0].numpy()
    return pred_mask

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/predict")
async def predict(image_file: UploadFile = File(...)):
    if image_file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(400, "Image doit être PNG ou JPEG.")

    contents = await image_file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Impossible de lire l'image.")

    img_tensor = preprocess_image(img)
    pred_mask = predict_mask(img_tensor)

    mask_image = Image.fromarray(pred_mask.astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")
