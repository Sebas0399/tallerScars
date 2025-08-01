"""
Funciones de visualización y evaluación para el clasificador de cicatrices.

Este módulo contiene funciones para generar gráficas, matrices de confusión
y otros elementos visuales para evaluar el modelo.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from typing import List, Dict, Any, Optional
import cv2
import os


def plot_training_history(history, save_path: Optional[str] = None) -> None:
    """
    Grafica las curvas de entrenamiento y validación.
    
    Args:
        history: Objeto History de Keras con el historial de entrenamiento
        save_path: Ruta para guardar la gráfica (opcional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfica de pérdida
    ax1.plot(history.history['loss'], label='Entrenamiento', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validación', linewidth=2)
    ax1.set_title('Función de Pérdida')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica de precisión
    metric_key = 'categorical_accuracy' if 'categorical_accuracy' in history.history else 'accuracy'
    val_metric_key = f'val_{metric_key}'
    
    ax2.plot(history.history[metric_key], label='Entrenamiento', linewidth=2)
    ax2.plot(history.history[val_metric_key], label='Validación', linewidth=2)
    ax2.set_title('Precisión del Modelo')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Precisión')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfica guardada en: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str], save_path: Optional[str] = None) -> None:
    """
    Genera y muestra una matriz de confusión.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        class_names: Nombres de las clases
        save_path: Ruta para guardar la gráfica (opcional)
    """
    # Convertir de categórico a índices si es necesario
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Crear gráfica
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matriz de confusión guardada en: {save_path}")
    
    plt.show()


def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, 
                              class_names: List[str]) -> Dict[str, float]:
    """
    Evalúa el rendimiento del modelo con múltiples métricas.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        class_names: Nombres de las clases
        
    Returns:
        Diccionario con las métricas calculadas
    """
    # Convertir de categórico a índices si es necesario
    if len(y_true.shape) > 1:
        y_true_idx = np.argmax(y_true, axis=1)
    else:
        y_true_idx = y_true
        
    if len(y_pred.shape) > 1:
        y_pred_idx = np.argmax(y_pred, axis=1)
    else:
        y_pred_idx = y_pred
    
    # Calcular métricas
    accuracy = accuracy_score(y_true_idx, y_pred_idx)
    precision_macro = precision_score(y_true_idx, y_pred_idx, average='macro')
    precision_micro = precision_score(y_true_idx, y_pred_idx, average='micro')
    recall_macro = recall_score(y_true_idx, y_pred_idx, average='macro')
    recall_micro = recall_score(y_true_idx, y_pred_idx, average='micro')
    f1_macro = f1_score(y_true_idx, y_pred_idx, average='macro')
    f1_micro = f1_score(y_true_idx, y_pred_idx, average='micro')
    
    # Métricas por clase
    precision_per_class = precision_score(y_true_idx, y_pred_idx, average=None)
    recall_per_class = recall_score(y_true_idx, y_pred_idx, average=None)
    f1_per_class = f1_score(y_true_idx, y_pred_idx, average=None)
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_micro': precision_micro,
        'recall_macro': recall_macro,
        'recall_micro': recall_micro,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro
    }
    
    # Agregar métricas por clase
    for i, class_name in enumerate(class_names):
        metrics[f'precision_{class_name}'] = precision_per_class[i]
        metrics[f'recall_{class_name}'] = recall_per_class[i]
        metrics[f'f1_{class_name}'] = f1_per_class[i]
    
    return metrics


def print_model_evaluation(y_true: np.ndarray, y_pred: np.ndarray, 
                          class_names: List[str], verbose: bool = True) -> Dict[str, float]:
    """
    Imprime un reporte completo de evaluación del modelo.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        class_names: Nombres de las clases
        verbose: Si imprimir el reporte detallado
        
    Returns:
        Diccionario con las métricas
    """
    metrics = evaluate_model_performance(y_true, y_pred, class_names)
    
    if verbose:
        print("=" * 60)
        print("EVALUACIÓN DEL MODELO")
        print("=" * 60)
        print(f"Exactitud (Accuracy): {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
        print(f"Precisión Macro: {metrics['precision_macro']:.4f}")
        print(f"Recall Macro: {metrics['recall_macro']:.4f}")
        print(f"F1-Score Macro: {metrics['f1_macro']:.4f}")
        print()
        
        print("MÉTRICAS POR CLASE:")
        print("-" * 40)
        for class_name in class_names:
            precision = metrics[f'precision_{class_name}']
            recall = metrics[f'recall_{class_name}']
            f1 = metrics[f'f1_{class_name}']
            
            print(f"{class_name}:")
            print(f"  Precisión: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print()
        
        # Convertir para classification_report
        if len(y_true.shape) > 1:
            y_true_idx = np.argmax(y_true, axis=1)
        else:
            y_true_idx = y_true
            
        if len(y_pred.shape) > 1:
            y_pred_idx = np.argmax(y_pred, axis=1)
        else:
            y_pred_idx = y_pred
        
        print("REPORTE DETALLADO:")
        print("-" * 40)
        print(classification_report(y_true_idx, y_pred_idx, 
                                  target_names=class_names, digits=4))
    
    return metrics


def visualize_sample_predictions(images: np.ndarray, y_true: np.ndarray, 
                               y_pred: np.ndarray, class_names: List[str], 
                               num_samples: int = 8, save_path: Optional[str] = None) -> None:
    """
    Visualiza muestras de predicciones del modelo.
    
    Args:
        images: Array de imágenes
        y_true: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        class_names: Nombres de las clases
        num_samples: Número de muestras a mostrar
        save_path: Ruta para guardar la gráfica (opcional)
    """
    # Convertir predicciones a índices de clase
    if len(y_pred.shape) > 1:
        pred_classes = np.argmax(y_pred, axis=1)
        pred_probs = np.max(y_pred, axis=1)
    else:
        pred_classes = y_pred
        pred_probs = np.ones_like(y_pred)
    
    if len(y_true.shape) > 1:
        true_classes = np.argmax(y_true, axis=1)
    else:
        true_classes = y_true
    
    # Seleccionar muestras aleatorias
    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    
    # Configurar gráfica
    rows = 2
    cols = num_samples // 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
            
        # Preparar imagen para visualización
        img = images[idx]
        if len(img.shape) == 3 and img.shape[-1] == 1:
            img = img.squeeze()
        
        # Mostrar imagen
        axes[i].imshow(img, cmap='gray')
        
        # Obtener información de predicción
        true_class = class_names[true_classes[idx]]
        pred_class = class_names[pred_classes[idx]]
        confidence = pred_probs[idx]
        
        # Color del título (verde si correcto, rojo si incorrecto)
        color = 'green' if true_classes[idx] == pred_classes[idx] else 'red'
        
        # Título con información
        title = f'Real: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}'
        axes[i].set_title(title, color=color, fontsize=10)
        axes[i].axis('off')
    
    # Ocultar axes no utilizados
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Muestras de Predicciones del Modelo', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualización guardada en: {save_path}")
    
    plt.show()


def plot_class_distribution(data_dir: str, save_path: Optional[str] = None) -> None:
    """
    Grafica la distribución de clases en el dataset.
    
    Args:
        data_dir: Directorio con los datos organizados por clase
        save_path: Ruta para guardar la gráfica (opcional)
    """
    class_counts = {}
    
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            class_counts[class_name] = len(image_files)
    
    # Crear gráfica de barras
    plt.figure(figsize=(10, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars = plt.bar(classes, counts, color=['skyblue', 'lightcoral'])
    plt.title('Distribución de Clases en el Dataset')
    plt.xlabel('Clase')
    plt.ylabel('Número de Imágenes')
    
    # Agregar valores en las barras
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribución de clases guardada en: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Módulo de visualización cargado correctamente.")
    print("Funciones disponibles:")
    print("- plot_training_history()")
    print("- plot_confusion_matrix()")
    print("- evaluate_model_performance()")
    print("- print_model_evaluation()")
    print("- visualize_sample_predictions()")
    print("- plot_class_distribution()")