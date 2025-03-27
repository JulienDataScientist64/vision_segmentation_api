# main.py (corrigé)
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

# Ce mapping est correct (correspond au notebook), mais n'est pas utilisé directement ici.
# Sert de documentation pour la signification des indices 0-7 que le modèle sort.
CLASS_TO_CATEGORY = {
    7: 0, 8: 0, 9: 0, 10: 0,          # indice 0: route
    24: 1, 25: 1,                     # indice 1: véhicules
    26: 2, 27: 2, 28: 2, 29: 2, 30: 2, 31: 2, 32: 2, 33: 2,  # indice 2: bâtiments
    11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3,  # indice 3: piétons/cyclistes
    17: 4, 18: 4, 19: 4, 20: 4,       # indice 4: poteaux/signalisation
    21: 5, 22: 5,                     # indice 5: panneaux -> CORRECTION: végétation/terrain
    23: 6,                           # indice 6: autre -> CORRECTION: ciel
    0: 7, 1: 7, 2: 7, 3: 7, 4: 7, 5: 7,  # indice 7: fond/surface plate
    255: 7                           # indice 7: ignoré
}

# --- CORRECTION DE LA PALETTE ---
# Palette 8 classes SÉMANTIQUEMENT CORRECTE (inspirée de Cityscapes)
# Doit correspondre à la signification des indices 0-7 ci-dessus.
CITYSCAPES_8_PALETTE = [
    (128, 64, 128),    # 0: Route (Violet)
    (0, 0, 142),       # 1: Véhicule (Bleu Voiture)
    (70, 70, 70),      # 2: Bâtiment (Gris Foncé)
    (220, 20, 60),     # 3: Personne/Cycliste (Rouge Vif)
    (153, 153, 153),   # 4: Poteau/Signalisation (Gris Clair)
    (107, 142, 35),    # 5: Végétation/Terrain (Vert Olive)
    (70, 130, 180),    # 6: Ciel (Bleu Ciel)
    (0, 0, 0)          # 7: Fond/Ignoré (Noir)
]
# --- FIN CORRECTION PALETTE ---

model = None
infer = None

# --- Fonctions load_model, preprocess_image, predict_mask, colorize_mask_8 ---
# --- INCHANGÉES ---
def load_model():
    global model, infer
    if model is None or infer is None:
        logging.info("Chargement du modèle depuis Hugging Face...")

        hf_token = os.environ.get("HF_TOKEN")
        # Utiliser un chemin de cache plus robuste potentiellement
        cache_dir = os.environ.get("HF_HOME", "/tmp/hf_cache")
        os.makedirs(cache_dir, exist_ok=True)
        logging.info(f"Utilisation du cache Hugging Face dans : {cache_dir}")

        try:
            model_path = snapshot_download(
                repo_id="cantalapiedra/semantic-segmentation-model",
                revision="main", # Utiliser 'main' est généralement préférable à 'master'
                use_auth_token=hf_token,
                cache_dir=cache_dir, # Spécifier cache_dir
                local_dir_use_symlinks=False # Garder False, c'est la norme
            )
            logging.info(f"Modèle téléchargé/chargé depuis : {model_path}")
            model = tf.saved_model.load(model_path)
            infer = model.signatures["serving_default"]
            logging.info("Modèle chargé avec succès dans TensorFlow.")
        except Exception as e:
            logging.error(f"ERREUR CRITIQUE lors du chargement du modèle: {e}", exc_info=True)
            # Empêcher l'application de démarrer correctement si le modèle ne charge pas
            raise RuntimeError(f"Impossible de charger le modèle: {e}")


def preprocess_image(img: Image.Image) -> tf.Tensor:
    # Redimensionne en (Hauteur, Largeur) pour TF, mais PIL prend (Largeur, Hauteur)
    target_height, target_width = 256, 512
    if img.size != (target_width, target_height):
        img_resized = img.resize((target_width, target_height), Image.BILINEAR)
    else:
        img_resized = img
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    tensor = tf.convert_to_tensor(arr)
    # S'assurer qu'il y a bien 3 canaux (au cas où une image en N&B serait envoyée)
    if len(tensor.shape) == 2:
        tensor = tf.stack([tensor]*3, axis=-1)
    elif tensor.shape[2] == 1:
         tensor = tf.image.grayscale_to_rgb(tensor)
    elif tensor.shape[2] == 4: # Enlever canal Alpha si présent
         tensor = tensor[:,:,:3]

    return tf.expand_dims(tensor, axis=0)

def predict_mask(img_tensor: tf.Tensor) -> np.ndarray:
    # Assurer que le modèle est chargé (important surtout si load_model échoue au démarrage)
    if model is None or infer is None:
         load_model() # Tentative de chargement si pas déjà fait
         if model is None or infer is None: # Vérifier à nouveau
             raise HTTPException(status_code=503, detail="Modèle non disponible, chargement échoué.")

    try:
        pred = infer(img_tensor) # img_tensor doit être [1, 256, 512, 3]
        pred_key = list(pred.keys())[0]
        # Sortie attendue: [1, 256, 512, 8]
        mask = tf.argmax(pred[pred_key], axis=-1)[0].numpy()  # -> shape [256, 512]
        return mask
    except Exception as e:
        logging.error(f"Erreur pendant l'inférence TensorFlow: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur serveur pendant l'inférence: {e}")


def colorize_mask_8(mask_array: np.ndarray, palette: list) -> Image.Image:
    """
    Convertit un masque (H,W) indices [0..7] en image RGB avec la palette.
    """
    h, w = mask_array.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    # Assurer que la palette a au moins autant d'éléments que la classe max + 1
    num_classes = len(palette)
    if np.max(mask_array) >= num_classes:
        logging.warning(f"Indice de classe ({np.max(mask_array)}) hors limites de la palette (taille {num_classes}). Utilisation de la couleur 0.")
        # Option : Mettre une couleur spécifique pour les indices invalides
        # color_mask.fill(0) # Remplir de noir par défaut

    for cls_idx, color in enumerate(palette):
         # Appliquer la couleur uniquement pour les indices valides
         valid_pixels = (mask_array == cls_idx)
         color_mask[valid_pixels] = color

    return Image.fromarray(color_mask, mode="RGB")

# --- Endpoints FastAPI (@app.get, @app.post) ---
# --- LÉGÈREMENT AJUSTÉS POUR ROBUSTESSE ---
@app.on_event("startup")
async def startup_event():
    # Essayer de charger le modèle au démarrage pour détecter les erreurs tôt
    # et potentiellement améliorer le temps de réponse de la 1ère requête.
    try:
        load_model()
    except Exception as e:
        # Log l'erreur mais laisser l'app démarrer pour pouvoir investiguer
        logging.error(f"Échec du pré-chargement du modèle au démarrage: {e}", exc_info=True)
        # On pourrait choisir de stopper ici si le modèle est absolument critique:
        # raise RuntimeError(f"Échec chargement modèle au démarrage: {e}")

@app.get("/")
def root():
    # Vérifier si le modèle est chargé pour l'état
    model_status = "Chargé" if model is not None and infer is not None else "Non chargé/Échec"
    return {"message": f"API Segmentation Sémantique - Statut modèle: {model_status}"}

@app.post("/predict")
async def predict(image_file: UploadFile = File(...)):
    # Vérification du type MIME
    if image_file.content_type not in ["image/png", "image/jpeg"]:
        logging.warning(f"Type de contenu non supporté reçu: {image_file.content_type}")
        raise HTTPException(status_code=415, detail="Format d'image non supporté. Utilisez PNG ou JPEG.")

    try:
        contents = await image_file.read()
        # Vérifier si le fichier est vide
        if not contents:
            raise HTTPException(status_code=400, detail="Fichier image vide reçu.")

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        logging.info(f"Image chargée : taille={img.size}, mode={img.mode}")
    except HTTPException as he:
        raise he # Propager les erreurs HTTP spécifiques
    except Exception as e:
        logging.error(f"Erreur lecture/ouverture image : {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Impossible de lire ou ouvrir le fichier image: {e}")

    try:
        # Prétraitement
        img_tensor = preprocess_image(img)
        logging.info(f"Image prétraitée, tenseur shape: {img_tensor.shape}") # Devrait être (1, 256, 512, 3)

        # Prédiction
        pred_mask = predict_mask(img_tensor) # -> np.array (256, 512)
        logging.info(f"Masque d'indices prédit, shape: {pred_mask.shape}, min/max: {np.min(pred_mask)}/{np.max(pred_mask)}")

        # Colorisation avec la palette CORRIGÉE
        colored_mask = colorize_mask_8(pred_mask, CITYSCAPES_8_PALETTE)
        logging.info("Masque coloré généré.")

        # Encodage pour la réponse
        buffer = io.BytesIO()
        colored_mask.save(buffer, format="PNG")
        buffer.seek(0)

        logging.info("Réponse PNG prête à être envoyée.")
        return StreamingResponse(buffer, media_type="image/png")

    except HTTPException as he:
         raise he # Propager les erreurs HTTP spécifiques (ex: modèle non chargé)
    except Exception as e:
        logging.error(f"Erreur serveur inattendue pendant le traitement : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur serveur interne inattendue : {e}")