# Data Directory

## Estructura

```bash
data/
├── raw/                    # Datos originales sin procesar
│   └── dataset_inventario_secuencial_completo.csv
└── processed/              # Datos procesados y listos para entrenamiento
    ├── df_processed_features.csv
    ├── X_train.npy
    ├── X_val.npy
    ├── y_train.npy
    └── y_val.npy
```

## Descripción de Archivos

### Raw Data

**`dataset_inventario_secuencial_completo.csv`**

- Dataset sintético de inventario con evolución temporal
- **79,173 registros** × 28 variables
- ~7,900 productos únicos con múltiples registros temporales
- Periodo: Datos secuenciales realistas
- Formato: CSV con encoding UTF-8

### Processed Data

**`df_processed_features.csv`**

- Dataset con feature engineering aplicado
- Incluye variables temporales creadas (día, mes, trimestre, etc.)
- Variables codificadas y normalizadas
- Listo para creación de secuencias

**`X_train.npy`**

- Secuencias de entrenamiento (features)
- Shape: (59,394, 7, 30)
  - 59,394 secuencias
  - 7 pasos temporales (días)
  - 30 características por paso

**`y_train.npy`**

- Targets de entrenamiento (quantity_available)
- Shape: (59,394,)
- Valores normalizados [0, 1]

**`X_val.npy`**

- Secuencias de validación (features)
- Shape: (14,820, 7, 30)

**`y_val.npy`**

- Targets de validación
- Shape: (14,820,)

**Generado para:** Práctica 1 - Aprendizaje Automático  
**Autores:** Felipe Peralta y Samantha Suquilanda
