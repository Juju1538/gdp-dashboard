
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import cv2

@st.cache_resource
def load_model():
    """Charge le modèle CNN une seule fois."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'mnist_cnn_model.h5')
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_final_version(image):
    """
    VERSION FINALE : Basée sur les tests, cette version utilise la stratégie
    'Seuil Fixe (151)' qui a donné les meilleurs résultats globaux sur le jeu de test.
    """
    # 1. Convertir en niveaux de gris et inverser les couleurs
    img_inverted_np = 255 - np.array(image.convert('L'))

    # 2. Binarisation avec le seuil fixe de 151
    threshold = 151 # Seuil optimal trouvé
    _, img_binary = cv2.threshold(img_inverted_np, threshold, 255, cv2.THRESH_BINARY)
    
    # 3. Trouver les contours pour isoler le chiffre
    contours, _ = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    # 4. Extraire la boîte englobante du plus grand contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 5. Recadrer, mettre au format carré, et redimensionner à 20x20
    cropped_digit_np = img_binary[y:y+h, x:x+w]
    pad_size = max(w, h)
    padded_digit = np.zeros((pad_size, pad_size), dtype=np.uint8)
    pad_x = (pad_size - w) // 2
    pad_y = (pad_size - h) // 2
    padded_digit[pad_y:pad_y+h, pad_x:pad_x+w] = cropped_digit_np
    
    target_size = 20
    if padded_digit.shape[0] > target_size:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC
    resized_digit = cv2.resize(padded_digit, (target_size, target_size), interpolation=interpolation)
    
    # 6. Centrer sur une image finale de 28x28
    final_img = np.zeros((28, 28), dtype=np.uint8)
    final_img[4:24, 4:24] = resized_digit

    # 7. Normaliser et formater pour le modèle
    final_array = final_img.astype('float32') / 255.0
    return final_array.reshape(1, 28, 28, 1)

def main():
    st.set_page_config(page_title="Reconnaissance de Chiffres", page_icon="✍️", layout="wide")
    st.title("✍️ Reconnaissance de Chiffres Manuscrits")
    st.info(
        "**Version Finale** : Cette application utilise la méthode de prétraitement qui a fourni "
        "les meilleurs résultats sur l'ensemble de vos images de test (Seuil Fixe à 151)."
    )

    model = load_model()
    tab1, tab2 = st.tabs(["Prédiction", "Détails du Modèle"])

    with tab1:
        st.header("Testez le modèle")
        col1, col2 = st.columns([2, 3])

        with col1:
            uploaded_file = st.file_uploader("Téléchargez une image d'un chiffre (0-9)", type=["png", "jpg", "jpeg"])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, "Votre image originale")

        with col2:
            if uploaded_file:
                st.write("")
                st.write("")
                if st.button("Lancer la Prédiction", use_container_width=True, type="primary"):
                    preprocessed_image_array = preprocess_final_version(image)
                    
                    if np.all(preprocessed_image_array == 0):
                         st.error("Impossible de détecter un chiffre. Essayez une image plus contrastée.")
                    else:
                        with st.spinner("Prédiction en cours..."):
                            prediction = model.predict(preprocessed_image_array)
                            predicted_class = np.argmax(prediction)
                            
                            st.success(f"Le chiffre prédit est : **{predicted_class}**")
                            st.bar_chart(prediction[0])
    with tab2:
        st.header("Architecture du Modèle et Entraînement")
        st.write("Le modèle est un réseau de neurones convolutif (CNN) séquentiel construit avec TensorFlow/Keras. Il est conçu pour une classification d'images efficace.")
        st.code('''
# Architecture du Modèle
model = tf.keras.models.Sequential([
    # 1. Couche de Convolution : 32 filtres 3x3, activation 'relu'
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # 2. Couche de Pooling : Réduit la taille pour extraire les caractéristiques importantes
    tf.keras.layers.MaxPooling2D((2, 2)),
    # 3. Deuxième couche de Convolution : 64 filtres pour des caractéristiques plus complexes
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # 4. Deuxième couche de Pooling
    tf.keras.layers.MaxPooling2D((2, 2)),
    # 5. Couche d'aplatissement : Prépare les données pour les couches de classification
    tf.keras.layers.Flatten(),
    # 6. Couche dense : 128 neurones pour l'apprentissage des motifs
    tf.keras.layers.Dense(128, activation='relu'),
    # 7. Couche de sortie : 10 neurones (un par chiffre) avec 'softmax' pour obtenir des probabilités
    tf.keras.layers.Dense(10, activation='softmax')
])
        ''', language='python')

if __name__ == "__main__":
    try:
        import cv2
    except ImportError:
        st.error("OpenCV n'est pas installé. Exécutez : pip install opencv-python")
        st.stop()
    main()
