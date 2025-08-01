#!/usr/bin/env python3
"""
Script para generar un reporte completo del proyecto de clasificación de cicatrices.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model_utils import ScarClassifier, get_model_summary
from src.data_preprocessing import validate_dataset_structure
from src.visualization import plot_class_distribution
import matplotlib.pyplot as plt


def generate_project_report():
    """Genera un reporte completo del estado del proyecto."""
    
    print("=" * 70)
    print("REPORTE DEL PROYECTO DE CLASIFICACIÓN DE CICATRICES")
    print("=" * 70)
    
    # 1. Verificar estructura del proyecto
    print("\n1. ESTRUCTURA DEL PROYECTO")
    print("-" * 40)
    
    required_files = [
        'modeloValido.h5',
        'mymodel.h5', 
        'Proyecto_Grupal.ipynb',
        'probarModelo.ipynb',
        'requirements.txt',
        'README.md'
    ]
    
    for file in required_files:
        status = "✓" if os.path.exists(file) else "✗"
        print(f"{status} {file}")
    
    # 2. Validar datasets
    print("\n2. VALIDACIÓN DE DATASETS")
    print("-" * 40)
    
    dataset_dirs = ['archive/train', 'archive/val', 'archive/test']
    for dataset_dir in dataset_dirs:
        print(f"\n{dataset_dir.upper()}:")
        if os.path.exists(dataset_dir):
            validate_dataset_structure(dataset_dir)
        else:
            print(f"  ✗ Directorio no encontrado")
    
    # 3. Información de modelos
    print("\n3. MODELOS ENTRENADOS")
    print("-" * 40)
    
    for model_file in ['modeloValido.h5', 'mymodel.h5']:
        if os.path.exists(model_file):
            print(f"\n{model_file}:")
            try:
                file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
                print(f"  Tamaño: {file_size:.2f} MB")
                # get_model_summary(model_file)
            except Exception as e:
                print(f"  Error al cargar: {str(e)}")
        else:
            print(f"\n{model_file}: ✗ No encontrado")
    
    # 4. Probar predicciones en imágenes de ejemplo
    print("\n4. PRUEBAS DE PREDICCIÓN")
    print("-" * 40)
    
    test_images = []
    if os.path.exists('test_images'):
        test_images = [f for f in os.listdir('test_images') 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if test_images and os.path.exists('modeloValido.h5'):
        classifier = ScarClassifier('modeloValido.h5')
        
        for img_file in test_images[:4]:  # Probar máximo 4 imágenes
            img_path = os.path.join('test_images', img_file)
            result = classifier.predict_single_image(img_path)
            
            if result:
                print(f"\n{img_file}:")
                print(f"  Predicción: {result['predicted_class']}")
                print(f"  Confianza: {result['confidence']:.2%}")
            else:
                print(f"\n{img_file}: Error en predicción")
    else:
        print("No se encontraron imágenes de prueba o modelo válido")
    
    # 5. Visualizar distribución de clases
    print("\n5. DISTRIBUCIÓN DE CLASES")
    print("-" * 40)
    
    if os.path.exists('archive/train'):
        try:
            plot_class_distribution('archive/train', 'results/class_distribution.png')
        except Exception as e:
            print(f"Error generando visualización: {str(e)}")
    
    # 6. Recomendaciones
    print("\n6. RECOMENDACIONES PARA MEJORAS")
    print("-" * 40)
    
    recommendations = [
        "✓ Documentación completa agregada (README.md)",
        "✓ Gestión de dependencias implementada (requirements.txt)",
        "✓ Código modularizado en el directorio src/",
        "✓ Scripts de utilidad creados (predict_scar.py)",
        "Considerar implementar data augmentation para mejorar robustez",
        "Agregar validación cruzada para evaluación más robusta",
        "Implementar transfer learning con modelos pre-entrenados",
        "Crear interfaz web para uso más accesible",
        "Agregar logging detallado para debugging",
        "Implementar pruebas unitarias para las funciones principales"
    ]
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print("\n" + "=" * 70)
    print("REPORTE COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    # Crear directorio de resultados si no existe
    os.makedirs('results', exist_ok=True)
    
    generate_project_report()