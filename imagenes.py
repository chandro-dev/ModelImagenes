import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import cv2  # Asegúrate de tener instalado OpenCV
from PIL import Image
import numpy as np
import tensorflow as     tf
import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2
import os

new_size = 256
model_path = "model.keras"

if not os.path.exists(model_path):
    # 1. Cargar dataset MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # 2. Redimensionar imágenes a 128x128
    X_train_resized = np.array([cv2.resize(img, (new_size, new_size)) for img in X_train])
    X_test_resized = np.array([cv2.resize(img, (new_size, new_size)) for img in X_test])

    # 3. Normalizar
    X_train_resized = X_train_resized.reshape(-1, new_size * new_size) / 255.0
    X_test_resized = X_test_resized.reshape(-1, new_size * new_size) / 255.0

    # 4. One-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # 5. Crear modelo simple
    model = Sequential([
        Dense(128, activation='relu', input_shape=(new_size * new_size,)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 6. Entrenar
    model.fit(X_train_resized, y_train, epochs=5, validation_data=(X_test_resized, y_test))

    # 7. Guardar modelo
    model.save(model_path)
model = keras.models.load_model('model.keras')
model.summary()
model.save('model.keras')

carpeta='./imagenes'
salida = "./imagenes_redimensionadas/"

# Listar todas las imágenes (puedes ajustar si tienes otros formatos)
imagenes = [f for f in os.listdir(carpeta) if f.endswith(('.png', '.jpg', '.jpeg'))]

for nombre in imagenes:
    ruta = os.path.join(carpeta, nombre)

    # 1. Cargar imagen en escala de grises
    imagen = Image.open(ruta).convert("L")

    # 2. Redimensionar a 28x28 si es necesario
    imagen = imagen.resize((new_size, new_size))

    ruta_guardado = os.path.join(salida, nombre)
    imagen.save(ruta_guardado)
    # 3. Convertir a array NumPy y normalizar
    img_array = np.array(imagen)
    img_array = 255 - img_array  # invertir blanco/negro si el fondo es blanco
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, new_size * new_size)

    # 4. Hacer predicción
    prediccion = model.predict(img_array)
    etiqueta = np.argmax(prediccion)

    print(f"{nombre} → Predicción: {etiqueta}")
