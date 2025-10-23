# Práctica 1: Aprendizaje Profundo y Series Temporales

## Predicción de Inventario mediante Redes Neuronales Recurrentes (GRU)

**Autores:** Felipe Peralta y Samantha Suquilanda  
**Asignatura:** Aprendizaje Automático  
**Nivel:** Séptimo Semestre

---

## Descripción del Proyecto

Este proyecto implementa un sistema de predicción de niveles de inventario utilizando técnicas de Deep Learning, específicamente Redes Neuronales Recurrentes con arquitectura GRU (Gated Recurrent Unit). El objetivo principal es predecir la cantidad disponible de productos en un sistema de gestión de inventario basándose en datos históricos secuenciales.

El sistema procesa 79,174 registros de inventario con 28 variables originales, aplicando técnicas avanzadas de Feature Engineering y normalización para entrenar un modelo capaz de realizar predicciones precisas con un error relativo del 3.45% respecto al rango total de valores.

---

## Estructura del Proyecto

```bash
practica 1/
├── fase01.py                              # Análisis exploratorio y preparación de datos
├── fase02.py                              # Entrenamiento y evaluación del modelo
├── fase03.py                              # Sistema de inferencia en producción
├── dataset_inventario_secuencial_completo.csv  # Dataset principal (79,174 registros)
├── df_processed_features.csv             # Dataset procesado con features
├── best_model.keras                       # Modelo óptimo entrenado
├── min_max_scaler.joblib                 # Escalador MinMax para normalización
├── le_product_id.joblib                  # Codificador de IDs de productos
├── le_supplier_id.joblib                 # Codificador de IDs de proveedores
├── X_train.npy, X_val.npy               # Secuencias de entrenamiento
├── y_train.npy, y_val.npy               # Etiquetas de entrenamiento
└── generar_dataset/                      # Scripts de generación de datos
```

---

## Metodología

### Fase 1: Análisis Exploratorio y Preparación de Datos

**Archivo:** `fase01.py`

#### 1.1 Análisis Exploratorio de Datos (EDA)

- **Carga y validación** de 79,174 registros con 28 variables
- **Limpieza de datos**: Conversión de tipos de datos (fechas), validación de nulos y duplicados
- **Análisis univariado**: Distribuciones de costos unitarios, uso diario promedio y estado del stock
- **Análisis bivariado**: Matrices de correlación, boxplots por categorías y pairplots
- **Detección de outliers**: Aplicación de métodos IQR y Z-score

#### 1.2 Feature Engineering

Se implementaron las siguientes transformaciones:

**Variables Temporales:**

- `dia_del_mes`: Día del mes (1-31)
- `dia_de_la_semana`: Día de la semana (0=Lunes, 6=Domingo)
- `mes`: Mes del año (1-12)
- `trimestre`: Trimestre del año (1-4)
- `es_fin_de_semana`: Indicador binario (0/1)

**Variables Derivadas:**

- `dias_para_vencimiento`: Días restantes hasta la fecha de expiración
- `antiguedad_producto_dias`: Días desde el último conteo de stock
- `ratio_uso_stock`: Relación entre uso diario y cantidad disponible

#### 1.3 Preprocesamiento

**Codificación de Variables Categóricas:**

- `LabelEncoder` para `product_id` y `supplier_id`
- `OneHotEncoder` para `warehouse_location` (7 categorías) y `stock_status` (3 categorías)

**Normalización:**

- `MinMaxScaler` aplicado a 18 variables numéricas, escalando valores al rango [0, 1]

**Creación de Secuencias Temporales:**

- Ventanas de 7 días (`N_STEPS = 7`)
- 30 features por paso temporal
- Shape final: `(muestras, 7, 30)`

**División del Dataset:**

- Entrenamiento: 80%
- Validación: 20%

---

### Fase 2: Entrenamiento y Evaluación del Modelo

**Archivo:** `fase02.py`

#### 2.1 Arquitectura del Modelo

```bash
Modelo GRU Secuencial:
├── Capa GRU (64 unidades)
│   └── Input shape: (7, 30)
├── Capa Dropout (0.2)
│   └── Regularización: Desactivación aleatoria del 20%
└── Capa Densa (1 unidad)
    └── Output: Predicción de quantity_available

Total de parámetros: 18,497
```

#### 2.2 Configuración del Entrenamiento

- **Optimizador:** Adam (learning_rate=0.001)
- **Función de pérdida:** Mean Squared Error (MSE)
- **Métrica de evaluación:** Mean Absolute Error (MAE)
- **Épocas máximas:** 100
- **Batch size:** 64
- **Callbacks:**
  - `ModelCheckpoint`: Guarda el mejor modelo según val_loss
  - `EarlyStopping`: Detención temprana con patience=10

#### 2.3 Resultados del Entrenamiento

El modelo convergió en la **época 53** con los siguientes resultados:

**Métricas en Escala Normalizada [0,1]:**

- RMSE: 0.0422
- MAE: 0.0296

**Métricas en Unidades Reales:**

- RMSE: 192.79 unidades
- MAE: 192.79 unidades
- Error Relativo: **3.45%**

**Contexto del Dataset:**

- Rango de stock: 0 - 6,435 unidades
- El error promedio de 192.79 unidades representa solo el 3% del rango total

#### 2.4 Análisis de Rendimiento

**Convergencia:**

- Caída rápida del error en las primeras 10 épocas
- Estabilización sin evidencia de sobreajuste
- Curvas de entrenamiento y validación convergentes

**Análisis de Errores:**

- Distribución de errores centrada en cero (modelo imparcial)
- Mediana del error absoluto: 161.8 unidades
- 50% de las predicciones con error entre 73.6 y 293.5 unidades

**Rendimiento por Rangos de Stock:**

- Stock Bajo (0-33%): Mejor desempeño
- Stock Medio (33-66%): Desempeño consistente
- Stock Alto (66-100%): Mayor variabilidad

---

### Fase 3: Sistema de Inferencia en Producción

**Archivo:** `fase03.py`

#### 3.1 Funcionalidad Principal

El sistema de inferencia implementa la función `predict_demand(product_id, target_date)` que realiza:

1. **Validación de entrada:**
   - Verificación de `product_id` contra el codificador entrenado
   - Conversión y validación del formato de fecha

2. **Recuperación de contexto histórico:**
   - Filtrado de registros del producto específico
   - Extracción de secuencia de 7 días previos a la fecha objetivo
   - Validación de historia suficiente

3. **Preparación de features:**
   - Selección de `FEATURE_COLUMNS` (30 variables)
   - Conversión a formato float32
   - Expansión a shape (1, 7, 30)

4. **Predicción y desescalado:**
   - Inferencia con el modelo GRU
   - Desescalado mediante `MinMaxScaler.inverse_transform`
   - Truncamiento a valores no negativos

#### 3.2 Predicción en Lote

El sistema incluye funcionalidad para realizar predicciones batch sobre múltiples productos:

```python
# Configuración
NUM_PRODUCTS = 15
TARGET_DATE = '2025-10-31'

# Proceso
- Carga de productos únicos del dataset
- Muestreo aleatorio sin reemplazo
- Iteración con tracking de progreso
- Manejo de errores por historia insuficiente
- Generación de estadísticas agregadas
```

**Salida:**

- DataFrame con predicciones por producto
- Tasa de éxito de predicciones
- Estadísticas: Media, Mediana, Mínimo, Máximo

---

## Instalación y Configuración

### Requisitos del Sistema

```bash
Python >= 3.8
TensorFlow >= 2.x
```

### Dependencias

```bash
pip install marimo pandas numpy tensorflow scikit-learn joblib plotly scipy matplotlib
```

### Ejecución de los Notebooks

Los archivos están implementados en **Marimo**, un framework de notebooks interactivos:

```bash
# Fase 1: Análisis y preparación
marimo edit fase01.py

# Fase 2: Entrenamiento del modelo
marimo edit fase02.py

# Fase 3: Sistema de inferencia
marimo edit fase03.py
```

---

## Uso del Sistema

### Realizar una Predicción Individual

```python
from fase03 import predict_demand

# Predecir stock para un producto específico
product_id = "PROD-00136830"
target_date = "2025-10-31"

prediction = predict_demand(product_id, target_date)
print(f"Stock predicho: {prediction:.2f} unidades")
```

### Ejecutar Predicciones en Lote

```python
# Ejecutar el último bloque de fase03.py
# Esto procesará 15 productos únicos y generará estadísticas
```

---

## Diccionario de Variables

### Variables Originales Principales

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `created_at` | datetime64[ns] | Fecha y hora de creación del registro |
| `product_id` | object | Identificador único del producto |
| `quantity_available` | int64 | **Variable objetivo**: Cantidad disponible para venta |
| `quantity_on_hand` | int64 | Cantidad física total en almacén |
| `minimum_stock_level` | int64 | Nivel mínimo antes de "bajo stock" |
| `reorder_point` | int64 | Nivel para generar nueva orden |
| `average_daily_usage` | float64 | Promedio de uso/venta diario |
| `unit_cost` | float64 | Costo de adquisición unitario |
| `warehouse_location` | object | Ubicación en almacén |
| `stock_status` | int64 | Estado del stock (1=Agotado, 2=Bajo, 3=En Stock) |

### Variables Derivadas (Feature Engineering)

| Variable | Descripción |
|----------|-------------|
| `dia_del_mes` | Día del mes (1-31) |
| `mes` | Mes del año (1-12) |
| `es_fin_de_semana` | Indicador binario de fin de semana |
| `dias_para_vencimiento` | Días restantes hasta expiración |
| `antiguedad_producto_dias` | Días desde último conteo |
| `ratio_uso_stock` | Relación uso diario / stock disponible |

---

## Resultados y Métricas de Desempeño

### Métricas Principales

| Métrica | Valor |
|---------|-------|
| MAE (normalizado) | 0.0296 |
| MAE (unidades reales) | 192.79 unidades |
| RMSE (normalizado) | 0.0422 |
| RMSE (unidades reales) | 271.62 unidades |
| Error Relativo | **3.45%** |
| Épocas de entrenamiento | 53 |

### Interpretación

- El modelo alcanza un **error relativo del 3.45%**, considerablemente inferior al estándar de la industria (5-10%)
- Las predicciones se desvían en promedio ±192.79 unidades sobre un rango de 6,435 unidades
- El modelo es imparcial (no sobreestima ni subestima sistemáticamente)
- Alta precisión en rangos bajos y medios de stock
- Mayor variabilidad en rangos altos de inventario

---

## Archivos de Artefactos

### Modelos y Transformadores

| Archivo | Descripción | Uso |
|---------|-------------|-----|
| `best_model.keras` | Modelo GRU óptimo | Inferencia de predicciones |
| `min_max_scaler.joblib` | Escalador MinMax | Normalización/desescalado |
| `le_product_id.joblib` | Codificador de productos | Transformación de IDs |
| `le_supplier_id.joblib` | Codificador de proveedores | Transformación de IDs |

### Datasets Procesados

| Archivo | Descripción | Registros |
|---------|-------------|-----------|
| `dataset_inventario_secuencial_completo.csv` | Dataset original | 79,174 |
| `df_processed_features.csv` | Features procesados | 79,174 |
| `X_train.npy` | Secuencias de entrenamiento | 59,686 |
| `X_val.npy` | Secuencias de validación | 14,922 |

---

## Limitaciones y Consideraciones

1. **Requisito de historia:** El sistema requiere al menos 7 días de historia por producto para realizar predicciones

2. **Productos no vistos:** Los productos no presentes en el conjunto de entrenamiento no pueden ser predichos

3. **Variabilidad en rangos altos:** El modelo presenta mayor variabilidad en predicciones para niveles muy altos de inventario

4. **Estacionalidad:** El modelo captura patrones temporales básicos pero podría mejorarse con features de estacionalidad más complejos

---

## Trabajo Futuro

1. **Incorporación de variables exógenas:** Incluir factores externos como días festivos, promociones o eventos especiales

2. **Arquitecturas avanzadas:** Experimentar con modelos Transformer o arquitecturas híbridas CNN-RNN

3. **Predicción multihorizonte:** Extender el sistema para predecir múltiples días futuros simultáneamente

4. **Optimización de hiperparámetros:** Búsqueda sistemática mediante Grid Search o Bayesian Optimization

5. **Sistema de alertas:** Implementar notificaciones automáticas cuando se prevean niveles críticos de stock

---

## Referencias

- Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow
- Documentación oficial de TensorFlow/Keras: <https://www.tensorflow.org/>
- Documentación de Scikit-learn: <https://scikit-learn.org/>

---

## Licencia

Este proyecto es de uso académico para la asignatura de Aprendizaje Automático.

---

## Contacto

Para consultas o comentarios sobre este proyecto, contactar a los autores a través de los canales oficiales de la universidad.
