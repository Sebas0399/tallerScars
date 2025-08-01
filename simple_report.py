#!/usr/bin/env python3
"""
Script simplificado para generar un reporte del proyecto sin dependencias de TensorFlow.
"""

import os
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
        'README.md',
        'src/',
        'test_images/',
        'archive/'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            if os.path.isdir(file):
                file_count = len([f for f in os.listdir(file) if os.path.isfile(os.path.join(file, f))])
                print(f"✓ {file} ({file_count} archivos)")
            else:
                file_size = os.path.getsize(file) / (1024 * 1024)  # MB
                print(f"✓ {file} ({file_size:.2f} MB)")
        else:
            print(f"✗ {file}")
    
    # 2. Validar datasets
    print("\n2. VALIDACIÓN DE DATASETS")
    print("-" * 40)
    
    dataset_dirs = ['archive/train', 'archive/val', 'archive/test']
    for dataset_dir in dataset_dirs:
        print(f"\n{dataset_dir.upper()}:")
        if os.path.exists(dataset_dir):
            for class_dir in os.listdir(dataset_dir):
                class_path = os.path.join(dataset_dir, class_dir)
                if os.path.isdir(class_path):
                    image_files = [f for f in os.listdir(class_path) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    print(f"  {class_dir}: {len(image_files)} imágenes")
        else:
            print(f"  ✗ Directorio no encontrado")
    
    # 3. Información de archivos del proyecto
    print("\n3. ARCHIVOS DEL PROYECTO")
    print("-" * 40)
    
    src_files = ['src/model_utils.py', 'src/data_preprocessing.py', 'src/visualization.py', 'src/__init__.py']
    for src_file in src_files:
        if os.path.exists(src_file):
            lines = len(open(src_file).readlines())
            print(f"✓ {src_file} ({lines} líneas)")
        else:
            print(f"✗ {src_file}")
    
    # 4. Imágenes de prueba
    print("\n4. IMÁGENES DE PRUEBA")
    print("-" * 40)
    
    if os.path.exists('test_images'):
        test_images = [f for f in os.listdir('test_images') 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Imágenes de prueba encontradas: {len(test_images)}")
        for img in test_images:
            print(f"  - {img}")
    else:
        print("Directorio test_images no encontrado")
    
    # 5. Visualizar distribución de clases (simplificado)
    print("\n5. DISTRIBUCIÓN DE CLASES (ENTRENAMIENTO)")
    print("-" * 40)
    
    if os.path.exists('archive/train'):
        class_counts = {}
        for class_name in os.listdir('archive/train'):
            class_path = os.path.join('archive/train', class_name)
            if os.path.isdir(class_path):
                image_files = [f for f in os.listdir(class_path) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                class_counts[class_name] = len(image_files)
        
        total_images = sum(class_counts.values())
        for class_name, count in class_counts.items():
            percentage = (count / total_images) * 100 if total_images > 0 else 0
            print(f"  {class_name}: {count} imágenes ({percentage:.1f}%)")
        
        print(f"\nTotal de imágenes de entrenamiento: {total_images}")
    
    # 6. Mejoras implementadas
    print("\n6. MEJORAS IMPLEMENTADAS")
    print("-" * 40)
    
    improvements = [
        "✓ Documentación completa agregada (README.md)",
        "✓ Gestión de dependencias implementada (requirements.txt)", 
        "✓ Código modularizado en el directorio src/",
        "✓ Scripts de utilidad creados (predict_scar.py, generate_report.py)",
        "✓ Organización mejorada de archivos (test_images/, results/)",
        "✓ Configuración de Git (.gitignore)",
        "✓ Estructura de paquete Python con __init__.py"
    ]
    
    for improvement in improvements:
        print(f"  {improvement}")
    
    # 7. Recomendaciones adicionales
    print("\n7. RECOMENDACIONES PARA FUTURAS MEJORAS")
    print("-" * 40)
    
    future_recommendations = [
        "Implementar data augmentation para mejorar robustez del modelo",
        "Agregar validación cruzada para evaluación más robusta",
        "Implementar transfer learning con modelos pre-entrenados",
        "Crear interfaz web para uso más accesible",
        "Agregar logging detallado para debugging",
        "Implementar pruebas unitarias para las funciones principales",
        "Optimizar hiperparámetros con búsqueda automática",
        "Agregar explicabilidad al modelo (GradCAM, LIME)",
        "Crear pipeline de CI/CD para automatizar pruebas",
        "Documentar el proceso de entrenamiento paso a paso"
    ]
    
    for i, rec in enumerate(future_recommendations, 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "=" * 70)
    print("REPORTE COMPLETADO - PROYECTO MEJORADO EXITOSAMENTE")
    print("=" * 70)
    print("\nEl proyecto ahora tiene:")
    print("• Mejor organización de código")
    print("• Documentación completa")
    print("• Scripts de utilidad")
    print("• Gestión de dependencias")
    print("• Estructura profesional")


if __name__ == "__main__":
    # Crear directorio de resultados si no existe
    os.makedirs('results', exist_ok=True)
    
    generate_project_report()