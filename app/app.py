import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ---------------------------------------------------
# 1) Définition du mapping Cityscapes → 8 classes
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

NUM_CLASSES = 8  # On segmente en 8 classes

# ---------------------------------------------------
# 2) Chargement du modèle d'inférence (SavedModel)
# ---------------------------------------------------
@st.cache_resource
def load_inference_model(model_path):
    """
    Charge le modèle TensorFlow (SavedModel) et renvoie la fonction d'inférence.
    """
    st.info("Chargement du modèle...")
    loaded_model = tf.saved_model.load(model_path)
    infer_func = loaded_model.signatures['serving_default']
    st.success("Modèle chargé avec succès.")
    return infer_func

# Chemin local vers ton modèle SavedModel
MODEL_SAVE_PATH = r"C:\Users\julien\vision_segmentation\model\saved_model_vgg16_unet"
infer = load_inference_model(MODEL_SAVE_PATH)

# ---------------------------------------------------
# 3) Fonctions utilitaires
# ---------------------------------------------------
def predict_image(infer_func, img_np):
    """
    Exécute l'inférence sur une image np.array (RGB).
    Retourne :
      - img_resized : l'image redimensionnée (256x512) pour affichage
      - pred_mask   : le masque prédit (valeurs 0..7)
    """
    # Conversion en tenseur float32 et normalisation [0..1]
    img_tensor = tf.convert_to_tensor(img_np, dtype=tf.float32) / 255.0
    # Redimension en (256,512) => (hauteur, largeur)
    img_tensor = tf.image.resize(img_tensor, (256, 512), method='bilinear')
    # Batch dimension
    img_tensor = tf.expand_dims(img_tensor, axis=0)

    # Inférence
    pred = infer_func(img_tensor)
    pred_key = list(pred.keys())[0]  # ex: 'output_1'
    pred_mask = tf.argmax(pred[pred_key], axis=-1)[0].numpy()

    # On renvoie aussi l'image redimensionnée pour affichage
    return img_tensor[0].numpy(), pred_mask

def preprocess_mask_file_from_path(mask_path):
    """
    Lit un fichier de masque Cityscapes (labelIds) depuis un chemin,
    applique le mapping, redimensionne en (256,512) et renvoie un np.array (uint8).
    """
    with open(mask_path, "rb") as f:
        msk_bytes = f.read()
    msk_tensor = tf.io.decode_png(msk_bytes, channels=1)
    # Convertir en int32 pour tf.gather
    msk_tensor = tf.cast(msk_tensor, tf.int32)
    # Application du mapping (Cityscapes → 8 classes)
    msk_tensor = tf.gather(MAPPING_TABLE, msk_tensor)
    # Redimension en (256,512)
    msk_tensor = tf.image.resize(msk_tensor, (256, 512), method='nearest')
    msk_tensor = tf.squeeze(msk_tensor, axis=-1)
    msk_tensor = tf.cast(msk_tensor, tf.uint8)
    return msk_tensor.numpy()

def calculate_iou(y_true, y_pred):
    """
    Calcule l'IoU entre deux masques booléens (y_true, y_pred).
    """
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return 0 if union == 0 else intersection / union

# ---------------------------------------------------
# 4) Chemins vers les dossiers de validation
# ---------------------------------------------------
# On suppose que tu as :
#   validation\leftImg8bit\val\<ville>\*.png
#   validation\gtFine\val\<ville>\*.png
left_val_dir = r"C:\Users\julien\vision_segmentation\validation\leftImg8bit\val"
gt_val_dir   = r"C:\Users\julien\vision_segmentation\validation\gtFine\val"

# ---------------------------------------------------
# 5) Interface Streamlit : sélection d'image par ID
# ---------------------------------------------------
st.title("Segmentation Sémantique - Inférence Cityscapes")

st.write("**But :** Sélectionner une ville, puis une image, pour lancer la prédiction du masque.")

# Lister les sous-dossiers (villes) dans left_val_dir
cities = [
    d for d in os.listdir(left_val_dir)
    if os.path.isdir(os.path.join(left_val_dir, d))
]
if not cities:
    st.error("Aucune ville trouvée dans leftImg8bit/val. Vérifie tes chemins.")
    st.stop()

# Sélection de la ville
selected_city = st.selectbox("Sélectionne la ville :", cities)

# Construire le chemin complet de la ville pour leftImg8bit
city_left_dir = os.path.join(left_val_dir, selected_city)

# Lister les fichiers terminant par "_leftImg8bit.png" dans la ville
image_files = [
    f for f in os.listdir(city_left_dir)
    if f.endswith("_leftImg8bit.png")
]
if not image_files:
    st.error(f"Aucune image '_leftImg8bit.png' trouvée dans {city_left_dir}")
    st.stop()

# Sélection du fichier image
selected_image_file = st.selectbox("Sélectionne l'image :", image_files)

# Déduire le masque correspondant en remplaçant le suffixe
mask_filename = selected_image_file.replace("_leftImg8bit.png", "_gtFine_labelIds.png")

# Construire les chemins complets
image_path = os.path.join(city_left_dir, selected_image_file)
city_mask_dir = os.path.join(gt_val_dir, selected_city)
mask_path = os.path.join(city_mask_dir, mask_filename)

st.write("**Image sélectionnée :**", selected_image_file)
st.write("**Masque correspondant :**", mask_filename)

if st.button("Lancer la prédiction"):
    # Charger l'image réelle (RGB)
    image = Image.open(image_path).convert("RGB")
    # Redimension (PIL attend (largeur, hauteur) => 512x256)
    image = image.resize((512, 256))
    img_np = np.array(image)

    # Charger et prétraiter le masque réel
    gt_mask = preprocess_mask_file_from_path(mask_path)

    # Lancer l'inférence
    img_resized, pred_mask = predict_image(infer, img_np)

    # Affichage des résultats
    st.subheader("Résultats de l'Inference")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1) Image réelle
    axes[0].imshow(img_resized)
    axes[0].set_title("Image réelle")
    axes[0].axis("off")

    # 2) Masque réel
    axes[1].imshow(gt_mask, cmap='jet', vmin=0, vmax=7)
    axes[1].set_title("Masque réel")
    axes[1].axis("off")

    # 3) Masque prédit
    axes[2].imshow(pred_mask, cmap='jet', vmin=0, vmax=7)
    axes[2].set_title("Masque prédit")
    axes[2].axis("off")

    st.pyplot(fig)

    # Calcul de la mIoU (moyenne des IoU) pour cette image
    class_ious = []
    for c in range(NUM_CLASSES):
        iou_c = calculate_iou(gt_mask == c, pred_mask == c)
        class_ious.append(iou_c)
    mIoU = np.mean(class_ious)

    st.write("**IoU par classe (cette image)** :", class_ious)
    st.write(f"**mIoU (cette image)** : {mIoU:.4f}")
