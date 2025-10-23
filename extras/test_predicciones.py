"""
Script de prueba para verificar que las predicciones ahora están en valores reales
"""

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

print("="*60)
print("PRUEBA DE PREDICCIONES CON SCALER CORREGIDO")
print("="*60)

N_STEPS = 7
TARGET_COLUMN_INDEX = 2
NUM_NUMERIC_FEATURES = 18

# Cargar artefactos
print("\nCargando modelo y artefactos...")
model = load_model('best_model.keras', compile=False)
scaler = joblib.load('min_max_scaler.joblib')
le_product_id = joblib.load('le_product_id.joblib')
df_features = pd.read_csv('df_processed_features.csv')
df_features['created_at'] = pd.to_datetime(df_features['created_at'])

print("✓ Todos los artefactos cargados correctamente")

# Verificar el scaler
print(f"\nVerificación del scaler:")
print(f"  Range de quantity_available: [{scaler.data_min_[2]:.2f}, {scaler.data_max_[2]:.2f}]")

FEATURE_COLUMNS = [
    'quantity_on_hand', 'quantity_reserved', 'quantity_available',
    'minimum_stock_level', 'reorder_point', 'optimal_stock_level',
    'reorder_quantity', 'average_daily_usage', 'unit_cost', 'total_value',
    'is_active', 'dia_del_mes', 'dia_de_la_semana', 'mes', 'trimestre',
    'es_fin_de_semana', 'dias_para_vencimiento', 'antiguedad_producto_dias',
    'ratio_uso_stock', 'product_id_encoded', 'supplier_id_encoded',
    'warehouse_location_Almacén Este', 'warehouse_location_Almacén Norte',
    'warehouse_location_Almacén Oeste', 'warehouse_location_Almacén Sur',
    'warehouse_location_Centro Distribución 1',
    'warehouse_location_Centro Distribución 2', 'stock_status_1',
    'stock_status_2', 'stock_status_3'
]

def predict_demand(product_id_str, target_date_str):
    try:
        product_id_encoded = le_product_id.transform([product_id_str])[0]
    except ValueError:
        return f"Error: El ID de producto '{product_id_str}' no fue visto durante el entrenamiento."
    
    try:
        target_date = pd.to_datetime(target_date_str)
    except ValueError:
        return f"Error: Formato de fecha incorrecto '{target_date_str}'."
    
    product_data = df_features[
        (df_features['product_id_encoded'] == product_id_encoded)
    ].sort_values(by='created_at')
    
    historical_data = product_data[product_data['created_at'] < target_date]
    
    if len(historical_data) < N_STEPS:
        return (f"Error: No hay suficiente historia ({len(historical_data)} días) "
                f"para predecir. Se necesitan {N_STEPS} días.")
    
    sequence_df = historical_data.tail(N_STEPS)
    input_features_df = sequence_df[FEATURE_COLUMNS]
    
    input_features_scaled = input_features_df.astype(np.float32).values
    input_sequence = np.expand_dims(input_features_scaled, axis=0)
    
    # Predicción
    pred_scaled = model.predict(input_sequence, verbose=0)[0][0]
    
    # Desescalar
    dummy_pred = np.zeros((1, NUM_NUMERIC_FEATURES))
    dummy_pred[:, TARGET_COLUMN_INDEX] = pred_scaled
    
    pred_real = scaler.inverse_transform(dummy_pred)[0][TARGET_COLUMN_INDEX]
    
    return max(0, pred_real)

# Realizar pruebas
print("\n" + "="*60)
print("PRUEBAS DE PREDICCIÓN")
print("="*60)

# Prueba 1
print("\n[PRUEBA 1]")
TEST_ID = 'PROD-00136830'
TEST_DATE = '2025-10-31'
print(f"Producto: {TEST_ID}")
print(f"Fecha objetivo: {TEST_DATE}")

# Mostrar valores históricos reales
df_original = pd.read_csv('dataset_inventario_secuencial_completo.csv')
df_original['created_at'] = pd.to_datetime(df_original['created_at'])
hist = df_original[df_original['product_id'] == TEST_ID].sort_values('created_at').tail(10)
print(f"\nÚltimos 10 valores reales de quantity_available:")
print(hist[['created_at', 'quantity_available']].to_string(index=False))

prediccion = predict_demand(TEST_ID, TEST_DATE)
print(f"\n>>> PREDICCIÓN: ", end="")
if isinstance(prediccion, (int, float)):
    print(f"{prediccion:.2f} unidades")
    print(f"    (Rango esperado: ~0 a ~6435 unidades)")
else:
    print(prediccion)

# Prueba 2
print("\n" + "-"*60)
print("\n[PRUEBA 2]")
TEST_ID = 'PROD-023D0E26'
TEST_DATE = '2025-10-31'
print(f"Producto: {TEST_ID}")
print(f"Fecha objetivo: {TEST_DATE}")

hist = df_original[df_original['product_id'] == TEST_ID].sort_values('created_at').tail(10)
print(f"\nÚltimos 10 valores reales de quantity_available:")
print(hist[['created_at', 'quantity_available']].to_string(index=False))

prediccion = predict_demand(TEST_ID, TEST_DATE)
print(f"\n>>> PREDICCIÓN: ", end="")
if isinstance(prediccion, (int, float)):
    print(f"{prediccion:.2f} unidades")
    print(f"    (Rango esperado: ~0 a ~6435 unidades)")
else:
    print(prediccion)

print("\n" + "="*60)
print("✓ PRUEBAS COMPLETADAS")
print("="*60)
print("\nCONCLUSIÓN:")
print("Si las predicciones ahora muestran valores en el rango de cientos o miles")
print("de unidades (en lugar de 0.XX unidades), el problema está RESUELTO.")
print("="*60)
