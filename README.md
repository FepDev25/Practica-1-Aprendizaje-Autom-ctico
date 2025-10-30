# Predicción de Stock con Redes Neuronales Recurrentes (GRU)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Marimo](https://img.shields.io/badge/Marimo-Notebooks-purple.svg)](https://marimo.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Sistema inteligente de predicción de niveles de inventario utilizando aprendizaje profundo y series temporales.

---

## Autores

**Felipe Peralta** y **Samantha Suquilanda**  
*Universidad - 7mo Semestre*  
*Curso: Aprendizaje Automático*

---

## Descripción del Proyecto

Este proyecto implementa un modelo de **aprendizaje profundo basado en GRU (Gated Recurrent Units)** para predecir los niveles de stock en sistemas de gestión de inventarios. El modelo aprende de patrones temporales históricos para anticipar la cantidad disponible de productos, permitiendo optimizar las decisiones de reabastecimiento y reducir costos operacionales.

### Objetivos

- Predecir `quantity_available` (stock disponible) con 7 días de anticipación
- Lograr un error promedio (MAE) menor al 5% del rango de stock
- Desarrollar un pipeline completo de ML: datos → modelo → producción
- Implementar un sistema robusto de inferencia con validación de entradas

---

## Características Principales

- **Modelo GRU optimizado** con 18,497 parámetros entrenables
- **Precisión del 97%** (error relativo de solo 3%)
- **Pipeline automatizado** de preprocesamiento y feature engineering
- **Función de predicción lista para producción** con validación exhaustiva
- **Notebooks interactivos Marimo** para exploración y análisis
- **Dataset sintético realista** con 79,173 registros temporales

---

## Resultados Destacados

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **MAE** | 193.03 unidades | Error promedio por predicción |
| **RMSE** | 271.37 unidades | Error cuadrático medio |
| **Error Relativo** | 3.00% | Porcentaje de error sobre rango total (6,435 u) |
| **Épocas de entrenamiento** | 53 de 100 | Convergencia eficiente con early stopping |
| **Precisión general** | 97% | Alto nivel de exactitud |

### Curva de Aprendizaje

```bash
Convergencia del modelo:
Época 1:  Loss: 0.0250 → Val Loss: 0.0180
Época 10: Loss: 0.0055 → Val Loss: 0.0048
Época 53: Loss: 0.0035 → Val Loss: 0.0042 ✓ (Mejor modelo)
```

---

## Estructura del Proyecto

```bash
practica-1/
├── 📁 notebooks/           # Notebooks Marimo interactivos
│   ├── fase01.py          # Fase 1: Análisis y Preparación de Datos
│   ├── fase02.py          # Fase 2: Entrenamiento del Modelo
│   └── fase03.py          # Fase 3: Inferencia y Despliegue
│
├── 📁 data/
│   ├── raw/               # Datos originales
│   │   └── dataset_inventario_secuencial_completo.csv
│   └── processed/         # Datos procesados
│       ├── df_processed_features.csv
│       ├── X_train.npy
│       ├── X_val.npy
│       ├── y_train.npy
│       └── y_val.npy
│
├── 📁 models/             # Modelos entrenados
│   └── best_model.keras   # Mejor modelo GRU (época 53)
│
├── 📁 scalers/            # Objetos de preprocesamiento
│   ├── min_max_scaler.joblib
│   ├── le_product_id.joblib
│   └── le_supplier_id.joblib
│
├── .gitignore
└── README.md              # Este archivo
```

---

## Tecnologías Utilizadas

### Core ML/DL

- **TensorFlow/Keras** 2.x - Framework de deep learning
- **NumPy** - Computación numérica
- **Pandas** - Manipulación de datos

### Preprocesamiento

- **Scikit-learn** - Codificación y normalización
- **Joblib** - Serialización de objetos

### Notebooks Interactivos

- **Marimo** - Notebooks reactivos en Python

### Visualización

- **Matplotlib** - Gráficos estáticos
- **Seaborn** - Visualizaciones estadísticas
- **Plotly** - Gráficos interactivos

---

## Instalación

### Prerrequisitos

```bash
Python 3.8 o superior
pip (gestor de paquetes de Python)
```

### Instalación de Dependencias

```bash
# Clonar el repositorio
git clone https://github.com/FepDev25/Practica-1-Aprendizaje-Automatico.git
cd Practica-1-Aprendizaje-Automatico

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install tensorflow numpy pandas scikit-learn joblib matplotlib seaborn plotly marimo
```

---

## Uso del Proyecto

### Exploración de Datos (Fase 1)

```bash
marimo edit notebooks/fase01.py
```

**Contenido:**

- Carga y análisis exploratorio del dataset
- Limpieza y transformación de datos
- Feature engineering temporal
- Codificación y normalización
- Creación de secuencias temporales

### Entrenamiento del Modelo (Fase 2)

```bash
marimo edit notebooks/fase02.py
```

**Contenido:**

- Construcción de la arquitectura GRU
- Entrenamiento con callbacks (Early Stopping, ModelCheckpoint)
- Evaluación de métricas (MAE, RMSE)
- Análisis de curvas de aprendizaje
- Análisis de distribución de errores

### Inferencia y Despliegue (Fase 3)

```bash
marimo edit notebooks/fase03.py
```

**Contenido:**

- Carga de artefactos de producción
- Función `predict_demand()` para predicciones
- Validación con productos reales
- Propuesta de API REST con FastAPI

---

## Ejemplo de Predicción

```python
from notebooks.fase03 import predict_demand

# Predecir stock para un producto específico
stock_predicho = predict_demand(
    product_id_str="PROD-00136830",
    target_date_str="2025-10-31"
)

print(f"Stock predicho: {stock_predicho:.2f} unidades")
# Salida: Stock predicho: 4253.67 unidades
```

### Casos de Uso

**Reabastecimiento automático**: Generar órdenes cuando predicción < punto de reorden  
**Optimización de almacén**: Redistribuir inventario según demanda prevista  
**Alertas tempranas**: Notificar sobre posibles desabastecimientos  
**Planificación financiera**: Estimar capital inmovilizado en inventario

---

## Arquitectura del Modelo

```python
Model: "Modelo_GRU_Prediccion_Stock"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Capa_Entrada_GRU (GRU)       (None, 64)                18,240    
_________________________________________________________________
Capa_Dropout (Dropout)       (None, 64)                0         
_________________________________________________________________
Capa_Salida_Prediccion       (None, 1)                 65        
=================================================================
Total params: 18,305
Trainable params: 18,305
Non-trainable params: 0
```

### Configuración de Entrenamiento

- **Optimizador**: Adam (learning_rate=0.001)
- **Función de pérdida**: Mean Squared Error (MSE)
- **Métrica**: Mean Absolute Error (MAE)
- **Batch size**: 64
- **Ventana temporal**: 7 días (N_STEPS)
- **Features de entrada**: 30 variables

---

## Dataset

### Características del Dataset Sintético

- **79,173 registros** con evolución temporal
- **28 variables** originales
- **~7,900 productos únicos**
- **Período temporal**: Datos secuenciales realistas

### Variables Principales

| Categoría | Variables |
|-----------|-----------|
| **Stock** | `quantity_available`, `quantity_on_hand`, `quantity_reserved` |
| **Gestión** | `minimum_stock_level`, `reorder_point`, `optimal_stock_level` |
| **Temporales** | `created_at`, `last_updated_at`, `last_stock_count_date` |
| **Categóricas** | `warehouse_location`, `stock_status`, `product_id` |
| **Derivadas** | `average_daily_usage`, `unit_cost`, `total_value` |

### Feature Engineering

```python
# Variables temporales creadas
- dia_del_mes, dia_de_la_semana, mes, trimestre
- es_fin_de_semana (variable binaria)

# Variables derivadas del negocio
- dias_para_vencimiento
- antiguedad_producto_dias  
- ratio_uso_stock
```

---

## Metodología

### Pipeline de ML Completo

```bash
1. DATOS CRUDOS
   ↓
2. LIMPIEZA Y TRANSFORMACIÓN
   ↓
3. FEATURE ENGINEERING
   ↓
4. CODIFICACIÓN (Label Encoding + One-Hot)
   ↓
5. NORMALIZACIÓN (MinMaxScaler [0,1])
   ↓
6. CREACIÓN DE SECUENCIAS (Ventana deslizante 7 días)
   ↓
7. DIVISIÓN TEMPORAL (80% train / 20% validation)
   ↓
8. ENTRENAMIENTO GRU (Early Stopping)
   ↓
9. EVALUACIÓN Y ANÁLISIS
   ↓
10. FUNCIÓN DE INFERENCIA
```

### División de Datos

- **Train set**: 59,394 secuencias (80%)
- **Validation set**: 14,820 secuencias (20%)
- **División temporal** (no aleatoria para respetar cronología)

---

## Análisis de Rendimiento

### Rendimiento por Rango de Stock

| Rango de Stock | Muestras | MAE (unidades) | Error % |
|----------------|----------|----------------|---------|
| **Bajo** (0-33%) | 4,940 | 150 | 4.2% |
| **Medio** (33-66%) | 4,940 | 185 | 3.5% |
| **Alto** (66-100%) | 4,940 | 245 | 2.8% |

### Distribución de Errores

```bash
Estadísticas del Error Absoluto:
- Media: 193.03 unidades
- Mediana: 161.8 unidades
- Q1 (25%): 73.6 unidades
- Q3 (75%): 293.5 unidades
```

**Interpretación**: El 50% de las predicciones tienen un error inferior a 162 unidades, lo que representa una alta precisión considerando el rango total del inventario (0-6,435 unidades).

---

## Contribuciones

Este proyecto fue desarrollado con fines académicos como parte del curso de Aprendizaje Automático.

Si encuentras errores o tienes sugerencias de mejora:

1. Abre un **Issue** describiendo el problema
2. Envía un **Pull Request** con tu propuesta

---

## Contacto

**Felipe Peralta** - [GitHub](https://github.com/FepDev25)  
**Samantha Suquilanda**
