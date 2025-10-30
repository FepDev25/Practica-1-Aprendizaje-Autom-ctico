# Scalers Directory

## Objetos de Preprocesamiento

Este directorio contiene los objetos serializados necesarios para el preprocesamiento y postprocesamiento de datos en el sistema de predicción.

### Archivos

#### `min_max_scaler.joblib`

**Tipo:** MinMaxScaler de scikit-learn  
**Propósito:** Normalización de variables numéricas al rango [0, 1]

**Variables escaladas (18 variables numéricas):**

1. quantity_on_hand
2. quantity_reserved
3. quantity_available (TARGET)
4. minimum_stock_level
5. reorder_point
6. optimal_stock_level
7. reorder_quantity
8. average_daily_usage
9. unit_cost
10. total_value
11. is_active
12. dia_del_mes
13. dia_de_la_semana
14. mes
15. trimestre
16. es_fin_de_semana
17. dias_para_vencimiento
18. antiguedad_producto_dias

**Rango de quantity_available (variable objetivo):**

- Mínimo: 0 unidades
- Máximo: 6,435 unidades
- Índice en el scaler: 2

**Uso:**

```python
import joblib

# Cargar scaler
scaler = joblib.load('scalers/min_max_scaler.joblib')

# Transformar datos nuevos
data_scaled = scaler.transform(data)

# Inversa (desescalar predicciones)
data_original = scaler.inverse_transform(data_scaled)
```

**Nota importante:** Para desescalar solo la variable objetivo (quantity_available), se debe crear un array "dummy" con 18 columnas donde solo la posición 2 contiene el valor:

```python
import numpy as np

# Predicción normalizada
pred_scaled = 0.466

# Crear dummy array
dummy = np.zeros((1, 18))
dummy[:, 2] = pred_scaled

# Desescalar
pred_real = scaler.inverse_transform(dummy)[0, 2]
# Output: 2999 unidades aproximadamente
```

---

#### `le_product_id.joblib`

**Tipo:** LabelEncoder de scikit-learn  
**Propósito:** Codificación de IDs de productos a valores numéricos

**Características:**

- ~7,900 productos únicos codificados
- Rango de salida: [0, 7899]
- Usado para agrupar series temporales por producto

**Uso:**

```python
import joblib

# Cargar encoder
le_product = joblib.load('scalers/le_product_id.joblib')

# Codificar
product_encoded = le_product.transform(['PROD-00136830'])
# Output: [1234]

# Decodificar
product_id = le_product.inverse_transform([1234])
# Output: ['PROD-00136830']

# Validar si un producto fue visto en entrenamiento
try:
    encoded = le_product.transform(['PROD-NUEVO'])
except ValueError:
    print("Producto no visto en entrenamiento")
```

---

#### `le_supplier_id.joblib`

**Tipo:** LabelEncoder de scikit-learn  
**Propósito:** Codificación de IDs de proveedores a valores numéricos

**Características:**

- Proveedores únicos codificados
- Usado para análisis de patrones por proveedor

**Uso:**

```python
import joblib

# Cargar encoder
le_supplier = joblib.load('scalers/le_supplier_id.joblib')

# Codificar
supplier_encoded = le_supplier.transform(['SUP-12345'])
```

---

## Flujo de Preprocesamiento

### Entrenamiento (Fase 1)

```bash
1. Cargar datos raw
2. Feature engineering
3. Crear y ajustar (fit) escaladores
   - LabelEncoder para product_id
   - LabelEncoder para supplier_id
   - MinMaxScaler para variables numéricas
4. Transformar datos
5. Guardar escaladores (.joblib)
6. Crear secuencias temporales
```

### Inferencia (Fase 3)

```bash
1. Cargar escaladores guardados
2. Validar product_id con LabelEncoder
3. Buscar histórico del producto
4. Aplicar mismo preprocesamiento (transform)
5. Hacer predicción
6. Desescalar resultado con inverse_transform
```

## Notas Técnicas

**Importante:** Los escaladores deben aplicarse en el mismo orden y con las mismas columnas que durante el entrenamiento. Cualquier desviación causará errores en las predicciones.

**Consistencia:** Estos objetos garantizan que el preprocesamiento en producción sea idéntico al del entrenamiento.

**Serialización:** Se usa `joblib` en lugar de `pickle` por ser más eficiente con arrays NumPy grandes.

---

**Proyecto:** Predicción de Stock con GRU  
**Autores:** Felipe Peralta y Samantha Suquilanda
