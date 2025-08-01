#!/usr/bin/env python3
"""
Script de predicción simple para el clasificador de cicatrices.

Uso: python predict_scar.py <ruta_imagen>
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model_utils import ScarClassifier
import argparse


def main():
    parser = argparse.ArgumentParser(description='Clasificar tipo de cicatriz en una imagen')
    parser.add_argument('image_path', help='Ruta a la imagen a clasificar')
    parser.add_argument('--model', default='modeloValido.h5', 
                       help='Ruta al modelo entrenado (default: modeloValido.h5)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Mostrar información detallada')
    
    args = parser.parse_args()
    
    # Verificar que la imagen existe
    if not os.path.exists(args.image_path):
        print(f"Error: No se encontró la imagen en {args.image_path}")
        return 1
    
    # Verificar que el modelo existe
    if not os.path.exists(args.model):
        print(f"Error: No se encontró el modelo en {args.model}")
        return 1
    
    # Crear clasificador y realizar predicción
    print(f"Cargando modelo desde: {args.model}")
    classifier = ScarClassifier(args.model)
    
    print(f"Analizando imagen: {args.image_path}")
    result = classifier.predict_single_image(args.image_path)
    
    if result is None:
        print("Error: No se pudo procesar la imagen")
        return 1
    
    # Mostrar resultados
    print("\n" + "="*50)
    print("RESULTADO DE LA CLASIFICACIÓN")
    print("="*50)
    print(f"Imagen: {result['image_path']}")
    print(f"Tipo de cicatriz: {result['predicted_class']}")
    print(f"Confianza: {result['confidence']:.2%}")
    
    if args.verbose:
        print("\nProbabilidades por clase:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.2%}")
    
    return 0


if __name__ == "__main__":
    exit(main())