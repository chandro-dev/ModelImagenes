import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class DigitClassifier:
    def __init__(self, img_size=78):
        """Inicializa el clasificador con configuración"""
        self.IMG_SIZE = img_size
        self.model = None
        self.model_path = "digit_classifier.keras"
        
    def create_model(self):
        """Crea una nueva arquitectura de modelo"""
        model = Sequential([
            keras.layers.Input(shape=(self.IMG_SIZE * self.IMG_SIZE,)),
            Dense(512, activation='relu'),
            Dropout(0.2),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Nuevo modelo creado con éxito")
        return model
    
    def load_dataset(self, custom_data_path=None):
        """Carga y prepara el dataset MNIST"""
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        # Redimensionar imágenes
        X_train = np.array([tf.image.resize(img[..., np.newaxis], [self.IMG_SIZE, self.IMG_SIZE]).numpy()[:, :, 0] 
                          for img in X_train])
        X_test = np.array([tf.image.resize(img[..., np.newaxis], [self.IMG_SIZE, self.IMG_SIZE]).numpy()[:, :, 0] 
                         for img in X_test])
        
        # Normalizar y aplanar
        X_train = X_train.reshape(-1, self.IMG_SIZE * self.IMG_SIZE) / 255.0
        X_test = X_test.reshape(-1, self.IMG_SIZE * self.IMG_SIZE) / 255.0
        
        # One-hot encoding
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        
        return (X_train, y_train), (X_test, y_test)
    
    def train(self, epochs=50, batch_size=128):
        """Entrena el modelo"""
        if self.model is None:
            self.create_model()
            
        (X_train, y_train), (X_test, y_test) = self.load_dataset()
        
        print("\nEntrenando modelo...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluar el modelo
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nPrecisión en test: {test_acc:.4f}")
        
        # Guardar modelo
        self.save_model()
        return history
    
    def save_model(self):
        """Guarda el modelo entrenado"""
        self.model.save(self.model_path)
        print(f"\nModelo guardado en {self.model_path}")
    
    def load_model(self):
        """Carga un modelo pre-entrenado"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Archivo de modelo no encontrado: {self.model_path}")
            
        self.model = keras.models.load_model(self.model_path)
        print(f"\nModelo cargado de {self.model_path}")
        return self.model
    
    def predict(self, image_path):
        """Predice y guarda solo la imagen final procesada"""
        try:
            output_dir = "imagenes_redimensionadas"
            os.makedirs(output_dir, exist_ok=True)
            
            img = Image.open(image_path).convert('L')
            original_name = os.path.basename(image_path)
            
            # Procesamiento completo
            img_resized = img.resize((self.IMG_SIZE, self.IMG_SIZE))
            img_array = np.array(img_resized, dtype=np.float32)
            
            if np.mean(img_array) > 127:
                img_array = 255 - img_array
                
            img_normalized = img_array / 255.0
            
            # Guardar imagen procesada final
            processed_img = Image.fromarray((img_normalized * 255).astype(np.uint8))
            processed_path = os.path.join(output_dir, f"processed_{original_name}")
            processed_img.save(processed_path)
            
            # Predecir
            img_flat = img_normalized.reshape(1, self.IMG_SIZE * self.IMG_SIZE)
            prediction = self.model.predict(img_flat, verbose=0)
            
            return np.argmax(prediction), np.max(prediction)
            
        except Exception as e:
            raise RuntimeError(f"Error en predicción: {str(e)}")

def main():
    try:
        # Configuración
        classifier = DigitClassifier(img_size=78)
        
        # Opciones:
        # 1. Crear y entrenar nuevo modelo
        if not os.path.exists(classifier.model_path):
            classifier.create_model()
            classifier.train(epochs=78)
        else:
            classifier.load_model()
        # 2. O cargar modelo existente
        # classifier.load_model()
        
        # Probar con una imagen
               # Verificar si la carpeta existe
        folder_path='./imagenes'
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Carpeta no encontrada: {folder_path}")
        
        # Obtener lista de imágenes (png, jpg, jpeg)
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"\nNo se encontraron imágenes en: {folder_path}")
            return
        
        print(f"\nProcesando {len(image_files)} imágenes de {folder_path}:")
        print("-" * 50)
        
        # Procesar cada imagen
        results = []
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            try:
                digit, confidence = classifier.predict(img_path)
                results.append((img_file, digit, confidence))
                print(f"{img_file}: Dígito {digit} | Confianza: {confidence*100:.2f}%")
            except Exception as e:
                print(f"\nError procesando {img_file}: {str(e)}")
                continue
        
        # Mostrar resumen
        print("\n" + "=" * 50)
        print("Resumen de predicciones:")
        for img_file, digit, confidence in sorted(results, key=lambda x: x[0]):
            print(f"{img_file.ljust(20)}: {digit} ({confidence*100:.1f}%)")
        
        return results
            
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()