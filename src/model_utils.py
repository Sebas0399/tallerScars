"""
Utilidades del modelo para clasificación de cicatrices.

Este módulo contiene funciones para cargar modelos entrenados,
hacer predicciones y evaluar el rendimiento del modelo.
"""

import tensorflow as tf
import cv2
import numpy as np
import os
from typing import Tuple, Optional, Dict, Any


class ScarClassifier:
    """Clasificador de cicatrices usando deep learning."""
    
    def __init__(self, model_path: str = 'modeloValido.h5'):
        """
        Inicializa el clasificador de cicatrices.
        
        Args:
            model_path: Ruta al archivo del modelo entrenado
        """
        self.model_path = model_path
        self.model = None
        self.class_names = ['Hypertrophic Scar', 'Keloid Scar']
        self.input_size = (100, 100)
        
    def load_model(self) -> bool:
        """
        Carga el modelo entrenado.
        
        Returns:
            True si el modelo se cargó correctamente, False en caso contrario
        """
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Modelo cargado correctamente desde: {self.model_path}")
                return True
            else:
                print(f"Error: No se encontró el archivo del modelo en {self.model_path}")
                return False
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            return False
    
    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Preprocesa una imagen para predicción.
        
        Args:
            image_path: Ruta a la imagen a procesar
            
        Returns:
            Array numpy con la imagen preprocesada o None si hay error
        """
        try:
            # Leer imagen en escala de grises
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Error: No se pudo cargar la imagen {image_path}")
                return None
            
            # Redimensionar
            resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_AREA)
            
            # Suavizado
            smoothed = cv2.medianBlur(resized, 5)
            
            # Detección de bordes con Sobel
            sobelx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=5)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Normalizar
            normalized = gradient_magnitude / 255.0
            
            # Reshape para el modelo
            processed = normalized.reshape(1, *self.input_size, 1)
            
            return processed
            
        except Exception as e:
            print(f"Error al procesar la imagen: {str(e)}")
            return None
    
    def predict_single_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Realiza predicción en una sola imagen.
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Diccionario con resultados de la predicción o None si hay error
        """
        if self.model is None:
            if not self.load_model():
                return None
        
        # Preprocesar imagen
        processed_image = self.preprocess_image(image_path)
        if processed_image is None:
            return None
        
        try:
            # Realizar predicción
            prediction = self.model.predict(processed_image, verbose=0)
            
            # Obtener clase predicha y confianza
            predicted_class_idx = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class_idx]
            predicted_class = self.class_names[predicted_class_idx]
            
            # Crear resultado
            result = {
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'probabilities': {
                    self.class_names[0]: float(prediction[0][0]),
                    self.class_names[1]: float(prediction[0][1])
                },
                'image_path': image_path
            }
            
            return result
            
        except Exception as e:
            print(f"Error durante la predicción: {str(e)}")
            return None
    
    def predict_batch(self, image_paths: list) -> list:
        """
        Realiza predicciones en lote para múltiples imágenes.
        
        Args:
            image_paths: Lista de rutas a las imágenes
            
        Returns:
            Lista de resultados de predicción
        """
        results = []
        for image_path in image_paths:
            result = self.predict_single_image(image_path)
            if result:
                results.append(result)
        return results


def predict_scar_type(image_path: str, model_path: str = 'modeloValido.h5') -> Optional[str]:
    """
    Función conveniente para predicción rápida del tipo de cicatriz.
    
    Args:
        image_path: Ruta a la imagen
        model_path: Ruta al modelo (opcional)
        
    Returns:
        Tipo de cicatriz predicho o None si hay error
    """
    classifier = ScarClassifier(model_path)
    result = classifier.predict_single_image(image_path)
    
    if result:
        return result['predicted_class']
    return None


def get_model_summary(model_path: str = 'modeloValido.h5') -> None:
    """
    Muestra un resumen del modelo entrenado.
    
    Args:
        model_path: Ruta al archivo del modelo
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"\n=== Resumen del Modelo: {model_path} ===")
        model.summary()
        print(f"\nNúmero total de parámetros: {model.count_params():,}")
        
    except Exception as e:
        print(f"Error al cargar el modelo para mostrar resumen: {str(e)}")


if __name__ == "__main__":
    # Ejemplo de uso
    classifier = ScarClassifier()
    
    # Probar con imagen de ejemplo
    test_images = ['a.jpg', 'c.jpg', 'kel.jpg', 'v.jpg']
    
    for img in test_images:
        if os.path.exists(img):
            result = classifier.predict_single_image(img)
            if result:
                print(f"\nImagen: {img}")
                print(f"Predicción: {result['predicted_class']}")
                print(f"Confianza: {result['confidence']:.2%}")
                print(f"Probabilidades: {result['probabilities']}")