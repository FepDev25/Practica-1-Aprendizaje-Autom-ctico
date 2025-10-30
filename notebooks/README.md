# Notebooks Directory

## Notebooks Marimo - Fases del Proyecto

Este directorio contiene los notebooks interactivos desarrollados con **Marimo** que documentan las tres fases del proyecto de predicción de stock.

### Archivos

#### `fase01.py` - Análisis y Preparación de Datos

**Contenido:**

- Carga y exploración del dataset de inventario
- Análisis exploratorio de datos (EDA)
  - Análisis univariado
  - Análisis bivariado y multivariado
  - Detección de outliers
- Limpieza y transformación de datos
- Feature Engineering
  - Variables temporales (día, mes, trimestre, etc.)
  - Variables derivadas del negocio
- Codificación de variables categóricas
- Normalización con MinMaxScaler
- Creación de secuencias temporales (ventana de 7 días)
- División temporal Train/Validation (80/20)

**Ejecutar:**

```bash
marimo edit notebooks/fase01.py
```

#### `fase02.py` - Entrenamiento del Modelo

**Contenido:**

- Carga de datos procesados (X_train, y_train, X_val, y_val)
- Construcción de arquitectura GRU
  - Capa GRU (64 unidades)
  - Capa Dropout (0.2)
  - Capa Dense de salida
- Configuración de callbacks
  - Early Stopping (patience=10)
  - ModelCheckpoint (guardar mejor modelo)
- Entrenamiento del modelo
- Visualización de curvas de aprendizaje
- Evaluación de métricas (RMSE, MAE)
- Análisis de rendimiento
  - Predicciones vs valores reales
  - Distribución de errores
  - Rendimiento por rango de stock

**Ejecutar:**

```bash
marimo edit notebooks/fase02.py
```

#### `fase03.py` - Inferencia y Despliegue

**Contenido:**

- Carga de artefactos de producción
  - Modelo entrenado (best_model.keras)
  - Escaladores (MinMaxScaler)
  - Codificadores (LabelEncoder)
- Implementación de función `predict_demand()`
  - Validación de product_id
  - Validación de fecha
  - Construcción de secuencia histórica
  - Predicción y desescalado
- Validación con productos reales
- Ejemplos de uso
- Propuesta de API REST con FastAPI

**Ejecutar:**

```bash
marimo edit notebooks/fase03.py
```

## Sobre Marimo

**Marimo** es un framework de notebooks reactivos para Python que ofrece:

- **Reactividad automática**: Las celdas se actualizan automáticamente
- **Notebooks como scripts**: Archivos .py estándar (fácil versionamiento)
- **Interactividad**: Widgets y visualizaciones interactivas
- **Reproducibilidad**: Orden de ejecución determinístico

### Instalación de Marimo

```bash
pip install marimo
```

### Comandos Útiles

```bash
# Ejecutar notebook en modo edición
marimo edit notebooks/fase01.py

# Ejecutar notebook en modo app (solo lectura)
marimo run notebooks/fase01.py

# Exportar a HTML
marimo export html notebooks/fase01.py > fase01.html
```

## Flujo de Ejecución Recomendado

1. **Fase 1**: Ejecutar primero para generar datos procesados
2. **Fase 2**: Ejecutar para entrenar y evaluar el modelo
3. **Fase 3**: Ejecutar para probar el sistema de inferencia
