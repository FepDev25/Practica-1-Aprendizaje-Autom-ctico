# Models Directory

## Modelos Entrenados

### `best_model.keras`

Modelo óptimo de red neuronal GRU para predicción de stock.

#### Especificaciones Técnicas

**Arquitectura:**

```bash
Model: "Modelo_GRU_Prediccion_Stock"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Capa_Entrada_GRU (GRU)       (None, 64)                18,240    
Capa_Dropout (Dropout)       (None, 64)                0         
Capa_Salida_Prediccion       (None, 1)                 65        
=================================================================
Total params: 18,305
Trainable params: 18,305
Non-trainable params: 0
```

**Configuración de Entrenamiento:**

- Optimizador: Adam (learning_rate=0.001)
- Función de pérdida: Mean Squared Error (MSE)
- Métrica: Mean Absolute Error (MAE)
- Batch size: 64
- Épocas totales: 53 (detenido por Early Stopping)

**Rendimiento:**

- MAE (normalizado): 0.0296
- RMSE (normalizado): 0.0422
- MAE (unidades reales): 193.03 unidades
- RMSE (unidades reales): 271.37 unidades
- Error relativo: 3.00%

**Input Shape:**

- (None, 7, 30)
  - 7 pasos temporales (días)
  - 30 features por paso

**Output:**

- (None, 1) - Predicción de quantity_available normalizada [0, 1]

#### Uso

```python
from tensorflow.keras.models import load_model

# Cargar modelo
model = load_model('models/best_model.keras')

# Predicción
# input_sequence shape: (1, 7, 30)
prediction = model.predict(input_sequence)
```

#### Guardado

El modelo fue guardado usando el callback `ModelCheckpoint` que monitoreaba `val_loss` y guardaba solo el mejor modelo durante el entrenamiento.

```python
ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)
```

#### Notas

- El modelo está en formato Keras nativo (.keras)
- Requiere TensorFlow 2.x para carga
- Tamaño aproximado: ~248 KB
- Época del mejor modelo: 53

---

**Proyecto:** Predicción de Stock con GRU  
**Autores:** Felipe Peralta y Samantha Suquilanda
