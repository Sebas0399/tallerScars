"""
Funciones de preprocesamiento de datos para el clasificador de cicatrices.

Este módulo contiene funciones para cargar, procesar y preparar
los datos de entrenamiento y validación.
"""

import cv2
import numpy as np
import os
from typing import Tuple, List, Optional
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_image_dataset(data_dir: str, img_size: Tuple[int, int] = (100, 100)) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Carga un dataset de imágenes desde directorios organizados por clase.
    
    Args:
        data_dir: Directorio raíz que contiene subdirectorios por clase
        img_size: Tamaño objetivo para las imágenes (ancho, alto)
        
    Returns:
        Tupla con (imágenes, etiquetas, nombres_clases)
    """
    images = []
    labels = []
    class_names = []
    
    # Obtener nombres de clases de los subdirectorios
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            class_names.append(class_name)
    
    print(f"Clases encontradas: {class_names}")
    
    # Cargar imágenes para cada clase
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        print(f"Cargando imágenes de {class_name}...")
        
        image_count = 0
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, filename)
                
                # Cargar y procesar imagen
                processed_img = preprocess_single_image(img_path, img_size)
                if processed_img is not None:
                    images.append(processed_img)
                    labels.append(class_idx)
                    image_count += 1
        
        print(f"  - Cargadas {image_count} imágenes de {class_name}")
    
    return np.array(images), np.array(labels), class_names


def preprocess_single_image(image_path: str, img_size: Tuple[int, int] = (100, 100)) -> Optional[np.ndarray]:
    """
    Preprocesa una sola imagen aplicando los mismos pasos que en el entrenamiento.
    
    Args:
        image_path: Ruta a la imagen
        img_size: Tamaño objetivo (ancho, alto)
        
    Returns:
        Imagen preprocesada o None si hay error
    """
    try:
        # Leer imagen en escala de grises
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: No se pudo cargar {image_path}")
            return None
        
        # Redimensionar
        resized = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)
        
        # Suavizado con filtro mediano
        smoothed = cv2.medianBlur(resized, 5)
        
        # Detección de bordes con operador Sobel
        sobelx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalizar a rango [0, 1]
        normalized = gradient_magnitude / 255.0
        
        return normalized
        
    except Exception as e:
        print(f"Error procesando {image_path}: {str(e)}")
        return None


def prepare_data_for_training(data_dir: str, test_size: float = 0.2, val_size: float = 0.2, 
                            img_size: Tuple[int, int] = (100, 100)) -> Tuple[np.ndarray, ...]:
    """
    Prepara los datos para entrenamiento, validación y prueba.
    
    Args:
        data_dir: Directorio con los datos organizados por clase
        test_size: Proporción de datos para prueba
        val_size: Proporción de datos de entrenamiento para validación
        img_size: Tamaño de las imágenes
        
    Returns:
        Tupla con (X_train, X_val, X_test, y_train, y_val, y_test, class_names)
    """
    # Cargar dataset completo
    X, y, class_names = load_image_dataset(data_dir, img_size)
    
    # Dividir en entrenamiento+validación y prueba
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Dividir entrenamiento+validación en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    
    # Añadir dimensión de canal para CNN
    X_train = X_train.reshape(X_train.shape[0], img_size[1], img_size[0], 1)
    X_val = X_val.reshape(X_val.shape[0], img_size[1], img_size[0], 1)
    X_test = X_test.reshape(X_test.shape[0], img_size[1], img_size[0], 1)
    
    # Convertir etiquetas a formato categórico
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(class_names))
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=len(class_names))
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=len(class_names))
    
    print(f"\nDatos preparados:")
    print(f"Entrenamiento: {X_train.shape[0]} imágenes")
    print(f"Validación: {X_val.shape[0]} imágenes")
    print(f"Prueba: {X_test.shape[0]} imágenes")
    print(f"Forma de imagen: {X_train.shape[1:]}")
    print(f"Número de clases: {len(class_names)}")
    
    return X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat, class_names


def create_data_generators(train_dir: str, val_dir: str, test_dir: str = None, 
                          img_size: Tuple[int, int] = (100, 100), 
                          batch_size: int = 32, augment: bool = True) -> Tuple:
    """
    Crea generadores de datos para entrenamiento con data augmentation opcional.
    
    Args:
        train_dir: Directorio de entrenamiento
        val_dir: Directorio de validación
        test_dir: Directorio de prueba (opcional)
        img_size: Tamaño de las imágenes
        batch_size: Tamaño del lote
        augment: Si aplicar data augmentation
        
    Returns:
        Tupla con generadores (train_gen, val_gen, test_gen)
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Generador para entrenamiento con augmentation
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    # Generador para validación (sin augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Crear generadores
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale'
    )
    
    test_generator = None
    if test_dir and os.path.exists(test_dir):
        test_generator = val_datagen.flow_from_directory(
            test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='grayscale'
        )
    
    return train_generator, val_generator, test_generator


def validate_dataset_structure(data_dir: str) -> bool:
    """
    Valida que la estructura del dataset sea correcta.
    
    Args:
        data_dir: Directorio raíz del dataset
        
    Returns:
        True si la estructura es válida, False en caso contrario
    """
    if not os.path.exists(data_dir):
        print(f"Error: El directorio {data_dir} no existe")
        return False
    
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if len(subdirs) < 2:
        print(f"Error: Se encontraron menos de 2 clases en {data_dir}")
        print(f"Subdirectorios encontrados: {subdirs}")
        return False
    
    total_images = 0
    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        image_files = [f for f in os.listdir(subdir_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Clase '{subdir}': {len(image_files)} imágenes")
        total_images += len(image_files)
        
        if len(image_files) == 0:
            print(f"Advertencia: No se encontraron imágenes en {subdir}")
    
    print(f"Total de imágenes en dataset: {total_images}")
    return total_images > 0


if __name__ == "__main__":
    # Validar estructura del dataset
    print("=== Validación del Dataset ===")
    
    data_paths = ['archive/train', 'archive/val', 'archive/test']
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"\nValidando {path}:")
            validate_dataset_structure(path)
        else:
            print(f"\nAdvertencia: {path} no existe")