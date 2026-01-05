
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from PIL import Image
import numpy as np

def load_and_preprocess_data():
    """
    Charge et prépare les données MNIST pour l'entraînement d'un modèle CNN.

    Cette fonction effectue les étapes suivantes :
    1.  Charge le jeu de données MNIST inclus dans Keras.
    2.  Redimensionne les images pour qu'elles aient une dimension de canal (pour le CNN).
    3.  Normalise les valeurs des pixels pour qu'elles soient comprises entre 0 et 1.
    4.  Convertit les étiquettes de classification en format one-hot encoding.

    Retours :
        tuple: Un tuple contenant les données d'entraînement et de test préparées :
               (x_train, y_train), (x_test, y_test)
    """
    # Charger le jeu de données MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Redimensionner les images pour inclure la dimension du canal
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    # Normaliser les valeurs des pixels entre 0 et 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convertir les étiquettes en format one-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)

def preprocess_image_for_prediction(image_path):
    """
    Charge une image, la convertit en niveaux de gris, la redimensionne et la normalise
    pour la prédiction par le modèle CNN.

    Args:
        image_path (str): Le chemin vers le fichier image.

    Retours:
        numpy.ndarray: L'image prétraitée, prête pour la prédiction.
    """
    # Charger l'image
    img = Image.open(image_path)
    # Convertir en niveaux de gris
    img = img.convert('L')
    # Redimensionner à 28x28 pixels
    img = img.resize((28, 28))
    # Convertir en tableau NumPy
    img_array = np.array(img)
    # Normaliser les valeurs des pixels entre 0 et 1
    img_array = img_array.astype('float32') / 255.0
    # Reshaper pour correspondre à la forme d'entrée du modèle (batch, hauteur, largeur, canaux)
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

if __name__ == '__main__':
    # Charger les données
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # Afficher les dimensions des données pour vérification
    print("Dimensions des données d'entraînement (images) :", x_train.shape)
    print("Dimensions des données d'entraînement (étiquettes) :", y_train.shape)
    print("Dimensions des données de test (images) :", x_test.shape)
    print("Dimensions des données de test (étiquettes) :", y_test.shape)

    # Construire le modèle CNN
    model = build_cnn_model(input_shape=x_train.shape[1:], num_classes=y_train.shape[1])
    model.summary()

    # Compiler le modèle
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Entraîner le modèle
    print("\nEntraînement du modèle...")
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.1)
    print("Entraînement terminé.")

    # Évaluer le modèle
    print("\nÉvaluation du modèle sur les données de test...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Précision sur les données de test : {accuracy:.4f}")
    print(f"Perte sur les données de test : {loss:.4f}")

    # Sauvegarder le modèle entraîné
    model_save_path = "mnist_cnn_model.h5"
    model.save(model_save_path)
    print(f"\nModèle sauvegardé sous : {model_save_path}")

    # Charger le modèle entraîné pour la prédiction
    print("\nChargement du modèle entraîné pour la prédiction...")
    loaded_model = tf.keras.models.load_model(model_save_path)
    print("Modèle chargé.")

    # Exemple de prédiction sur une image réelle
    # REMPLACEZ 'path/to/your/image.png' par le chemin de votre image de chiffre manuscrit
    real_image_path = "path/to/your/image.png" 
    
    # Check if the placeholder path is still present
    if real_image_path == "path/to/your/image.png":
        print("\nATTENTION: Veuillez remplacer 'path/to/your/image.png' par le chemin réel de votre image pour la prédiction.")
    else:
        print(f"\nPrédiction sur l'image : {real_image_path}")
        preprocessed_image = preprocess_image_for_prediction(real_image_path)
        predictions = loaded_model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions)
        print(f"Le chiffre prédit est : {predicted_class}")
