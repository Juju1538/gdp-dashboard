
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import cv2

def load_model(model_path):
    """Charge le modèle CNN."""
    if not os.path.exists(model_path):
        print(f"Erreur: Le fichier modèle '{model_path}' n'a pas été trouvé.")
        return None
    return tf.keras.models.load_model(model_path)

# STRATÉGIE 1: Seuil automatique d'Otsu + Redimensionnement adaptatif (la plus robuste en théorie)
def preprocess_auto_adaptive(image):
    img_inverted_np = 255 - np.array(image.convert('L'))
    _, img_binary = cv2.threshold(img_inverted_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return np.zeros((1, 28, 28, 1))
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = img_binary[y:y+h, x:x+w]
    pad_size = max(w, h)
    padded = np.zeros((pad_size, pad_size), dtype=np.uint8)
    pad_x, pad_y = (pad_size - w) // 2, (pad_size - h) // 2
    padded[pad_y:pad_y+h, pad_x:pad_x+w] = cropped
    target_size = 20
    interpolation = cv2.INTER_AREA if pad_size > target_size else cv2.INTER_CUBIC
    resized = cv2.resize(padded, (target_size, target_size), interpolation=interpolation)
    final_img = np.zeros((28, 28), dtype=np.uint8)
    final_img[4:24, 4:24] = resized
    return (final_img.astype('float32') / 255.0).reshape(1, 28, 28, 1)

# STRATÉGIE 2: Seuil Fixe (151) + Redimensionnement adaptatif
def preprocess_fixed_threshold(image):
    img_inverted_np = 255 - np.array(image.convert('L'))
    _, img_binary = cv2.threshold(img_inverted_np, 151, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return np.zeros((1, 28, 28, 1))
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = img_binary[y:y+h, x:x+w]
    pad_size = max(w, h)
    padded = np.zeros((pad_size, pad_size), dtype=np.uint8)
    pad_x, pad_y = (pad_size - w) // 2, (pad_size - h) // 2
    padded[pad_y:pad_y+h, pad_x:pad_x+w] = cropped
    target_size = 20
    interpolation = cv2.INTER_AREA if pad_size > target_size else cv2.INTER_CUBIC
    resized = cv2.resize(padded, (target_size, target_size), interpolation=interpolation)
    final_img = np.zeros((28, 28), dtype=np.uint8)
    final_img[4:24, 4:24] = resized
    return (final_img.astype('float32') / 255.0).reshape(1, 28, 28, 1)

# STRATÉGIE 3: Recadrage simple + Redimensionnement direct (la version "pire")
def preprocess_direct_resize(image):
    img_inverted = ImageOps.invert(image.convert('L'))
    bbox = img_inverted.getbbox()
    if not bbox: return np.zeros((1, 28, 28, 1))
    img_cropped = img_inverted.crop(bbox)
    img_final = img_cropped.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img_final)
    return (img_array.astype('float32') / 255.0).reshape(1, 28, 28, 1)

def main():
    """Script de test pour trouver la meilleure stratégie de prétraitement."""
    model_path = 'mnist_cnn_model.h5'
    model = load_model(model_path)
    if model is None:
        return

    filenames = [
        'image_1.jpeg', 'image_2.jpeg', 'image_3.jpeg',
        'image_4.jpeg', 'image_5.jpeg', 'image_6.jpeg',
        'image_7.jpeg', 'image_8.jpeg', 'image_9.jpeg'
    ]

    strategies = {
        "Seuil Auto (Otsu)": preprocess_auto_adaptive,
        "Seuil Fixe (151)": preprocess_fixed_threshold,
        "Redimensionnement Direct": preprocess_direct_resize
    }

    print("Lancement du test de prédiction sur vos images...")
    
    for filename in filenames:
        if not os.path.exists(filename):
            print(f"\n--- Fichier '{filename}' non trouvé. ---")
            continue
            
        print(f"\n--- Test de l'image : {filename} ---")
        image = Image.open(filename)
        
        for name, func in strategies.items():
            try:
                processed_image = func(image)
                if np.all(processed_image == 0):
                    prediction = "Erreur (Image vide)"
                else:
                    pred_probs = model.predict(processed_image, verbose=0)
                    prediction = np.argmax(pred_probs)
                print(f"    - Stratégie '{name}': Prédiction -> {prediction}")
            except Exception as e:
                print(f"    - Stratégie '{name}': ERREUR lors du traitement -> {e}")

if __name__ == "__main__":
    main()
