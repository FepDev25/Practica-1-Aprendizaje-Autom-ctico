# 🔬 Guía de Evaluación Comparativa Multi-Dataset

## Descripción

Este sistema permite evaluar tu modelo de predicción de stock con **múltiples datasets** de forma automatizada. Ejecuta todo el pipeline completo (preprocesamiento → entrenamiento → evaluación) para cada dataset y genera comparaciones visuales.

## 🎯 ¿Qué hace este sistema?

Para cada dataset disponible:
1. **Carga** los datos
2. **Preprocesa** (feature engineering, codificación, normalización)
3. **Crea secuencias temporales** (con ventana de 7 días)
4. **Entrena** un modelo GRU
5. **Evalúa** el modelo y calcula métricas
6. **Genera visualizaciones** completas
7. **Compara** resultados entre datasets

## 📋 Estructura del Notebook

### Sección 1: Clase Principal de Evaluación
- `ModeloStockEvaluador`: Encapsula todo el pipeline para un dataset

### Sección 2: Sistema Multi-Dataset
- `EvaluadorMultiDataset`: Ejecuta evaluación en múltiples datasets

### Sección 3: Ejecutar Evaluación Completa
- Ejecuta el pipeline para **TODOS** los datasets disponibles

### Sección 4: Generar Reporte Completo
- Genera visualizaciones comparativas
- Crea tablas resumen
- Identifica el mejor dataset

### Sección 5: Análisis Individual
- Análisis detallado de un dataset específico

### Sección 6: Exportar Resultados
- Guarda predicciones, métricas, modelos y historia

### Sección 7: Prueba Rápida
- Evalúa solo un dataset para pruebas

## 🚀 Uso Rápido

### Opción 1: Evaluar TODOS los datasets (Recomendado)

```python
# En la Sección 3, ejecutar las celdas en orden:
# 1. Configuración
# 2. Iniciar evaluación
# 3. Generar reporte completo
```

**Tiempo estimado:** ~2-5 minutos por dataset

### Opción 2: Prueba Rápida con UN dataset

```python
# En la Sección 7, configurar:
DATASET_PRUEBA = './data/subset_10_productos.csv'
# Y ejecutar
```

## 📊 Visualizaciones Generadas

### Por cada dataset:
1. **Curvas de entrenamiento** (Loss y MAE)
2. **Scatter plot** (Predicciones vs Reales)
3. **Histograma de errores**
4. **Boxplot de errores**
5. **Panel de métricas** detallado

### Comparativas entre datasets:
1. **MAE vs Tamaño** del dataset
2. **RMSE vs Tamaño**
3. **Error relativo vs Tamaño**
4. **Tiempo de entrenamiento vs Tamaño**
5. **Épocas entrenadas** por dataset
6. **Tabla comparativa** completa

## 📈 Métricas Calculadas

Para cada dataset se calculan:

### Métricas Normalizadas (0-1):
- **RMSE** (escalado)
- **MAE** (escalado)

### Métricas Reales (unidades de stock):
- **RMSE** (unidades)
- **MAE** (unidades)
- **Error relativo** (%)

### Información del Dataset:
- Total de productos únicos
- Total de filas
- Secuencias de train/validación
- Rango de stock (min/max)

### Tiempos:
- Tiempo de carga
- Tiempo de preprocesamiento
- Tiempo de creación de secuencias
- Tiempo de entrenamiento
- Tiempo de evaluación
- **Tiempo total**

## 🎓 Interpretación de Resultados

### MAE (Mean Absolute Error)
- **Bajo es mejor**
- Representa el error promedio en unidades de stock
- Ejemplo: MAE = 25.5 significa que las predicciones se desvían ±25.5 unidades en promedio

### RMSE (Root Mean Squared Error)
- **Bajo es mejor**
- Penaliza más los errores grandes
- Siempre mayor o igual que MAE

### Error Relativo (%)
- **Bajo es mejor**
- Error como porcentaje del rango total de stock
- Facilita comparación entre datasets con diferentes escalas
- Ejemplo: 5% es excelente, 15% es aceptable, >25% es mejorable

### Épocas Entrenadas
- Si es cercano al máximo (100), el modelo quizás necesite más tiempo
- Si es bajo (<20), el modelo converge rápido (puede ser overfitting o underfitting)
- Ideal: 30-70 épocas con early stopping

## 💡 Casos de Uso

### 1. Comparar Tamaños de Dataset
**Pregunta:** ¿Cuántos datos necesito para un buen modelo?

```python
# Ejecutar Sección 3 y 4
# Observar gráficos "MAE vs Tamaño del Dataset"
# Identificar punto de rendimientos decrecientes
```

### 2. Optimizar Tiempo de Entrenamiento
**Pregunta:** ¿Cuál es el mejor balance entre precisión y tiempo?

```python
# Revisar gráfico "Tiempo de Entrenamiento vs Tamaño"
# Comparar con MAE para encontrar sweet spot
```

### 3. Validar Modelo en Producción
**Pregunta:** ¿Qué dataset usar para deployment?

```python
# Usar evaluador_multi.mejor_dataset(metrica='mae_real')
# Considerar también tiempo de inferencia
```

### 4. Debugging de Modelo
**Pregunta:** ¿Por qué el modelo no mejora?

```python
# Sección 5: Análisis individual
# Revisar histograma de errores
# Ver si hay patrones en scatter plot
```

## 📁 Archivos Generados

Después de ejecutar el sistema, se generarán:

```
resultados_comparacion.csv          # Tabla comparativa completa
resultados_evaluacion/
    ├── predicciones_subset_XXX.csv  # Predicciones de cada dataset
    ├── metricas_subset_XXX.json     # Métricas detalladas
    ├── modelo_subset_XXX.keras      # Modelo entrenado
    └── historia_subset_XXX.csv      # Historia de entrenamiento
```

## ⚙️ Configuración Avanzada

### Ajustar Parámetros de Entrenamiento

En la **Sección 3**:

```python
EPOCHS = 100          # Más épocas = más tiempo, posiblemente mejor modelo
BATCH_SIZE = 64       # Mayor = más rápido pero más memoria
VERBOSE = 0           # 1 para ver progreso
```

### Cambiar Arquitectura del Modelo

Modificar en `ModeloStockEvaluador.entrenar_modelo()`:

```python
# Cambiar unidades GRU
self.model.add(GRU(units=128, ...))  # Era 64

# Añadir más capas
self.model.add(GRU(units=64, return_sequences=True, ...))
self.model.add(GRU(units=32, ...))

# Ajustar Dropout
self.model.add(Dropout(0.3, ...))  # Era 0.2
```

### Cambiar División Train/Val

```python
evaluador = ModeloStockEvaluador(
    dataset_path='...',
    nombre_experimento='...',
    split_percentage=0.85  # Era 0.8 (80/20)
)
```

## 🐛 Solución de Problemas

### "No se encontraron datasets"
- Verifica que los archivos estén en `./data/`
- Verifica el patrón de búsqueda: `subset_*.csv`

### "Out of memory"
- Reduce `BATCH_SIZE`
- Procesa menos datasets a la vez
- Usa datasets más pequeños primero

### "El entrenamiento es muy lento"
- Reduce `EPOCHS`
- Aumenta `BATCH_SIZE` (si hay memoria)
- Usa GPU si está disponible

### "Las métricas no mejoran"
- Revisa la distribución de errores (histograma)
- Verifica que hay suficientes datos de entrenamiento
- Considera ajustar la arquitectura del modelo
- Revisa el preprocesamiento de datos

## 📞 Flujo de Trabajo Recomendado

1. **Prueba rápida** (Sección 7)
   - Ejecuta con el dataset más pequeño (10 productos)
   - Verifica que todo funciona
   - Tiempo: ~30 segundos

2. **Evaluación parcial**
   - Selecciona 2-3 datasets de diferentes tamaños
   - Ejecuta Sección 3 con solo esos datasets
   - Tiempo: ~5-10 minutos

3. **Evaluación completa**
   - Ejecuta con todos los datasets
   - Genera reporte completo
   - Tiempo: ~15-30 minutos

4. **Análisis profundo**
   - Revisa el mejor dataset (Sección 5)
   - Exporta resultados (Sección 6)
   - Documenta hallazgos

## 🎯 Ejemplo Completo

```python
# 1. Crear evaluador multi-dataset
evaluador_multi = EvaluadorMultiDataset(datasets_dir='./data')

# 2. Detectar datasets
datasets = evaluador_multi.detectar_datasets(patron='subset_*.csv')

# 3. Evaluar todos
evaluador_multi.evaluar_todos(
    datasets=datasets,
    epochs=100,
    batch_size=64,
    verbose=0
)

# 4. Generar reporte
df_resultados = evaluador_multi.generar_reporte_completo(guardar=True)

# 5. Encontrar mejor
mejor = evaluador_multi.mejor_dataset(metrica='mae_real')

# 6. Analizar mejor dataset
eval_mejor = evaluador_multi.evaluadores[mejor['nombre']]
eval_mejor.generar_visualizaciones()
```

## 📚 Recursos Adicionales

- **Pipeline original**: Ver celdas iniciales del notebook para entender cada paso
- **Generador de datasets**: Ver `generador_subsets.py` para crear nuevos subsets
- **Documentación datasets**: Ver `README_SUBSETS.md`

---

**¡Feliz análisis de modelos! 🚀📊**
