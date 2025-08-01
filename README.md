# Clasificador de Cicatrices usando Deep Learning

## Descripción del Proyecto

Este proyecto implementa un sistema de clasificación automática de cicatrices usando técnicas de deep learning. El modelo es capaz de distinguir entre dos tipos principales de cicatrices:

- **Cicatrices queloides (Keloid scars)**: Cicatrices que crecen más allá de los límites de la herida original
- **Cicatrices hipertróficas (Hypertrophic scars)**: Cicatrices elevadas que permanecen dentro de los límites de la herida original

## Estructura del Proyecto

```
tallerScars/
├── README.md                    # Este archivo
├── requirements.txt             # Dependencias del proyecto
├── Proyecto_Grupal.ipynb       # Notebook principal con entrenamiento del modelo
├── probarModelo.ipynb          # Notebook para pruebas del modelo
├── modeloValido.h5             # Modelo entrenado principal
├── mymodel.h5                  # Modelo alternativo
├── src/                        # Código fuente modularizado
│   ├── model_utils.py          # Utilidades del modelo
│   ├── data_preprocessing.py   # Preprocesamiento de datos
│   └── visualization.py       # Funciones de visualización
├── archive/                    # Dataset principal
│   ├── train/                  # Datos de entrenamiento
│   ├── val/                    # Datos de validación
│   └── test/                   # Datos de prueba
├── test_images/                # Imágenes de prueba individuales
└── results/                    # Resultados y visualizaciones
```

## Tecnologías Utilizadas

- **Python 3.11+**
- **TensorFlow 2.15**: Framework de deep learning
- **OpenCV**: Procesamiento de imágenes
- **Scikit-learn**: Métricas de evaluación
- **Matplotlib**: Visualización de resultados
- **NumPy**: Operaciones numéricas

## Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone https://github.com/Sebas0399/tallerScars.git
cd tallerScars
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Verificar la estructura de datos
Asegúrate de que el directorio `archive/` contenga los datos organizados en las siguientes carpetas:
- `train/keloid_scars/`
- `train/hypertrophic_scar/`
- `val/keloid_scars/`
- `val/hypertrophic_scar/`
- `test/keloid_scars/`
- `test/hypertrophic_scar/`

## Uso del Proyecto

### Entrenamiento del Modelo
Para entrenar un nuevo modelo, ejecuta el notebook `Proyecto_Grupal.ipynb` paso a paso.

### Predicción en Nuevas Imágenes
Para probar el modelo con imágenes nuevas:

1. Usar el notebook `probarModelo.ipynb`
2. O usar el script de predicción:
```python
from src.model_utils import predict_scar_type
resultado = predict_scar_type('ruta/a/imagen.jpg')
print(f"Tipo de cicatriz: {resultado}")
```

### Evaluación del Modelo
El modelo incluye las siguientes métricas de evaluación:
- Exactitud (Accuracy)
- Precisión (Precision)
- Matriz de confusión
- Curvas de pérdida y precisión durante el entrenamiento

## Características del Modelo

### Preprocesamiento de Imágenes
- Redimensionamiento a 100x100 píxeles
- Conversión a escala de grises
- Suavizado con filtro mediano
- Detección de bordes con operador Sobel
- Normalización de valores de píxeles

### Arquitectura
- Red neuronal convolucional (CNN)
- Capas de convolución para extracción de características
- Capas densas para clasificación
- Función de activación softmax para clasificación binaria

### Métricas de Rendimiento
- **Exactitud**: >80% en conjunto de prueba
- **Precisión macro**: Balanceada entre ambas clases
- **Recall**: Buena detección de ambos tipos de cicatriz

## Archivos de Ejemplo

El proyecto incluye imágenes de ejemplo para pruebas:
- `a.jpg`, `c.jpg`, `kel.jpg`, `v.jpg`: Imágenes de prueba
- `4-Figure5-1.png`: Diagrama de arquitectura o resultados

## Contribuciones

Para contribuir al proyecto:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -am 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crea un Pull Request

## Mejoras Futuras

- [ ] Implementar data augmentation para mejorar la robustez
- [ ] Agregar más tipos de cicatrices al modelo
- [ ] Crear una interfaz web para uso fácil
- [ ] Implementar transfer learning con modelos pre-entrenados
- [ ] Optimizar hiperparámetros con búsqueda automática
- [ ] Agregar explicabilidad al modelo (GradCAM, LIME)

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Autores

- **Sebas0399** - Desarrollo principal

## Agradecimientos

- Dataset de cicatrices utilizado para entrenamiento
- Comunidad de TensorFlow y OpenCV por las herramientas