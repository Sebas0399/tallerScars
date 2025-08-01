"""
Paquete de utilidades para el clasificador de cicatrices.

Este paquete contiene módulos para:
- model_utils: Funciones para cargar y usar modelos entrenados
- data_preprocessing: Funciones para procesar y preparar datos
- visualization: Funciones para generar gráficas y visualizaciones
"""

__version__ = "1.0.0"
__author__ = "Sebas0399"

from .model_utils import ScarClassifier, predict_scar_type, get_model_summary
from .data_preprocessing import (
    load_image_dataset, 
    preprocess_single_image, 
    prepare_data_for_training,
    validate_dataset_structure
)
from .visualization import (
    plot_training_history,
    plot_confusion_matrix, 
    evaluate_model_performance,
    print_model_evaluation
)

__all__ = [
    'ScarClassifier',
    'predict_scar_type', 
    'get_model_summary',
    'load_image_dataset',
    'preprocess_single_image',
    'prepare_data_for_training', 
    'validate_dataset_structure',
    'plot_training_history',
    'plot_confusion_matrix',
    'evaluate_model_performance',
    'print_model_evaluation'
]