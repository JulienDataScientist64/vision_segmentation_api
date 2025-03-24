from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib

# Création de l'application FastAPI avec quelques métadonnées
app = FastAPI(
    title="API Inference - Segmentation Sémantique",
    version="1.0.0",
    description="API pour l'inférence en segmentation sémantique utilisant TensorFlow."
)

# ---------------------------------------------------
# 1) Mapping Cityscapes → 8 classes (identique à l'entraînement)
# ---------------------------------------------------
CLASS_TO_CATEGORY = {
    7: 0, 8: 0, 9: 0, 10: 0,          # route
    24: 1, 25: 1,                     # véhicules
    26: 2, 27: 2, 28: 2, 29: 2, 30: 2, 31: 2, 32: 2, 33: 2,  # bâtiments
    11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3,  # piétons
    17: 4, 18: 4, 19: 4, 20: 4,       # poteaux
    21: 5, 22: 5,                     # panneaux
    23: 6,                            # autre
    0: 7, 1: 7, 2: 7, 3: 7, 4: 7, 5: 7,
    255: 7
}
max_label = 256
mapping_table_np = np.arange(max_label, dtype=np.uint8)
for old_c, new_c in CLASS_TO_CATEGORY.items():
    mapping_table_np[old_c] = new_c
MAPPING_TABLE = tf.constant(mapping_table_np, dtype=tf.uint8)
NUM_CLASSES = 8

# ---------------------------------------------------
# 2) Chargement du modèle d'inférence
# ---------------------------------------------------
# Remplace MODEL_SAVE_PATH par le chemin local de ton modèle
MODEL_SAVE_PATH = r"C:\Users\julien\vision_segmentation\model\saved_model_vgg16_unet"
loaded_model = tf.saved_model.load(MODEL_SAVE_PATH)
infer = loaded_model.signatures['serving_default']

# ---------------------------------------------------
# 3) Fonctions utilitaires
# ---------------------------------------------------
def preprocess_image(img: Image.Image):
    """
    Prétraite l'image pour l'inférence : redimensionnement et normalisation.
    Le modèle attend une image de dimensions 256x512 (hauteur x largeur).
    """
    # Redimensionnement avec PIL : (width, height)
    image = img.resize((512, 256))
    img_np = np.array(image)
    img_tensor = tf.convert_to_tensor(img_np, dtype=tf.float32) / 255.0
    # On redimensionne de nouveau pour s'assurer des dimensions attendues par le modèle
    img_tensor = tf.image.resize(img_tensor, (256, 512), method='bilinear')
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    return img_tensor

def predict_image(img_tensor):
    """
    Exécute l'inférence sur l'image (tenseur 4D) et renvoie le masque prédit.
    """
    pred = infer(img_tensor)
    pred_key = list(pred.keys())[0]
    pred_mask = tf.argmax(pred[pred_key], axis=-1)[0].numpy()
    return pred_mask

def preprocess_mask(mask_bytes: bytes):
    """
    Prétraite le masque réel (fichier PNG) en appliquant le mapping et le redimensionnement.
    Retourne un np.array uint8 de dimensions (256,512).
    """
    msk_tensor = tf.io.decode_png(mask_bytes, channels=1)
    msk_tensor = tf.cast(msk_tensor, tf.int32)
    msk_tensor = tf.gather(MAPPING_TABLE, msk_tensor)
    msk_tensor = tf.image.resize(msk_tensor, (256, 512), method='nearest')
    msk_tensor = tf.squeeze(msk_tensor, axis=-1)
    msk_tensor = tf.cast(msk_tensor, tf.uint8)
    return msk_tensor.numpy()

def calculate_iou(y_true, y_pred):
    """
    Calcule l'IoU entre deux masques booléens.
    """
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return 0 if union == 0 else intersection / union

# ---------------------------------------------------
# 4) Endpoints FastAPI
# ---------------------------------------------------
@app.get("/")
def read_root():
    """
    Endpoint racine pour vérifier que l'API est opérationnelle.
    """
    return {"message": "API is running"}

@app.post("/predict")
async def predict_endpoint(image_file: UploadFile = File(...),
                           mask_file: UploadFile = File(...)):
    """
    Ce endpoint reçoit une image et son masque réel,
    exécute l'inférence et retourne le mIoU (moyenne des IoU par classe)
    ainsi que les IoU par classe dans un format JSON.
    """
    if image_file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(status_code=400, detail="Le fichier image doit être PNG ou JPEG.")
    if mask_file.content_type != "image/png":
        raise HTTPException(status_code=400, detail="Le fichier masque doit être un PNG.")

    # Lecture et préparation de l'image
    image_contents = await image_file.read()
    try:
        img = Image.open(io.BytesIO(image_contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Impossible de lire l'image.") from e
    img_tensor = preprocess_image(img)

    # Lecture et prétraitement du masque réel
    mask_contents = await mask_file.read()
    gt_mask = preprocess_mask(mask_contents)

    # Exécuter l'inférence
    pred_mask = predict_image(img_tensor)

    # Calculer l'IoU pour chaque classe et le mIoU
    class_ious = []
    for c in range(NUM_CLASSES):
        iou = calculate_iou(gt_mask == c, pred_mask == c)
        class_ious.append(iou)
    mIoU = float(np.mean(class_ious))

    return JSONResponse(content={
        "mIoU": mIoU,
        "IoU_per_class": class_ious
    })
