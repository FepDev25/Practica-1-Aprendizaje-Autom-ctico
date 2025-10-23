"""
Script para regenerar el scaler correcto con los rangos originales
Este script debe ejecutarse para corregir el problema de escalado
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

print("=== REGENERANDO SCALER CORRECTO ===\n")

# 1. Cargar el dataset procesado con features (ya escalado)
df_features = pd.read_csv('df_processed_features.csv')
df_features['created_at'] = pd.to_datetime(df_features['created_at'])

print(f"Dataset cargado: {len(df_features)} registros")

# 2. Cargar el dataset ORIGINAL (sin escalar)
df_original = pd.read_csv('dataset_inventario_secuencial_completo.csv')
df_original['created_at'] = pd.to_datetime(df_original['created_at'])

print(f"Dataset original cargado: {len(df_original)} registros")

# 3. Las columnas que fueron escaladas
columnas_numericas = [
    'quantity_on_hand', 'quantity_reserved', 'quantity_available',
    'minimum_stock_level', 'reorder_point', 'optimal_stock_level',
    'reorder_quantity', 'average_daily_usage', 'unit_cost', 'total_value',
    'dia_del_mes', 'dia_de_la_semana', 'mes', 'trimestre', 'es_fin_de_semana',
    'dias_para_vencimiento', 'antiguedad_producto_dias', 'ratio_uso_stock'
]

# 4. Necesitamos reconstruir el DataFrame con las features originales
# Las primeras 10 columnas vienen del dataset original
columnas_originales_base = [
    'quantity_on_hand', 'quantity_reserved', 'quantity_available',
    'minimum_stock_level', 'reorder_point', 'optimal_stock_level',
    'reorder_quantity', 'average_daily_usage', 'unit_cost', 'total_value'
]

# Las features temporales necesitan ser recalculadas desde created_at
df_original_feat = df_original.copy()
base_date = df_original_feat['created_at']

df_original_feat['dia_del_mes'] = base_date.dt.day
df_original_feat['dia_de_la_semana'] = base_date.dt.dayofweek
df_original_feat['mes'] = base_date.dt.month
df_original_feat['trimestre'] = base_date.dt.quarter
df_original_feat['es_fin_de_semana'] = df_original_feat['dia_de_la_semana'].isin([5, 6]).astype(int)

# Convertir fechas
df_original_feat['expiration_date'] = pd.to_datetime(df_original_feat['expiration_date'])
df_original_feat['last_stock_count_date'] = pd.to_datetime(df_original_feat['last_stock_count_date'])

# Calcular features derivadas
df_original_feat['dias_para_vencimiento'] = (df_original_feat['expiration_date'] - base_date).dt.days
df_original_feat['dias_para_vencimiento'] = df_original_feat['dias_para_vencimiento'].fillna(0)
df_original_feat['dias_para_vencimiento'] = df_original_feat['dias_para_vencimiento'].apply(lambda x: max(0, x))

df_original_feat['antiguedad_producto_dias'] = (base_date - df_original_feat['last_stock_count_date']).dt.days
df_original_feat['antiguedad_producto_dias'] = df_original_feat['antiguedad_producto_dias'].fillna(0)
df_original_feat['antiguedad_producto_dias'] = df_original_feat['antiguedad_producto_dias'].apply(lambda x: max(0, x))

df_original_feat['ratio_uso_stock'] = df_original_feat['average_daily_usage'] / (df_original_feat['quantity_available'] + 1)

print("\nFeatures originales calculadas")

# 5. Extraer las columnas numéricas en el orden correcto
datos_originales = df_original_feat[columnas_numericas].values

print(f"\nEstadísticas de las features ORIGINALES (sin escalar):")
print(f"quantity_available: min={datos_originales[:, 2].min():.2f}, max={datos_originales[:, 2].max():.2f}")
print(f"unit_cost: min={datos_originales[:, 8].min():.2f}, max={datos_originales[:, 8].max():.2f}")

# 6. Crear y entrenar el scaler con los datos ORIGINALES
scaler_correcto = MinMaxScaler()
scaler_correcto.fit(datos_originales)

print(f"\n✓ Scaler entrenado correctamente")
print(f"  Número de features: {scaler_correcto.n_features_in_}")
print(f"  Range de quantity_available (índice 2): [{scaler_correcto.data_min_[2]:.2f}, {scaler_correcto.data_max_[2]:.2f}]")

# 7. Guardar el scaler correcto
joblib.dump(scaler_correcto, 'min_max_scaler_CORRECTO.joblib')
print(f"\n✓ Scaler guardado como 'min_max_scaler_CORRECTO.joblib'")

# 8. Prueba de desescalado
print("\n=== PRUEBA DE DESESCALADO ===")
valor_escalado = 0.45
dummy = np.zeros((1, 18))
dummy[:, 2] = valor_escalado
valor_real = scaler_correcto.inverse_transform(dummy)[0][2]

print(f"Valor escalado: {valor_escalado}")
print(f"Valor desescalado: {valor_real:.2f} unidades")
print(f"\n✓ ¡Ahora el desescalado funciona correctamente!")

print("\n" + "="*50)
print("PRÓXIMOS PASOS:")
print("1. Renombra 'min_max_scaler.joblib' a 'min_max_scaler_ANTIGUO.joblib'")
print("2. Renombra 'min_max_scaler_CORRECTO.joblib' a 'min_max_scaler.joblib'")
print("3. Ejecuta nuevamente fase03.py para obtener predicciones en valores reales")
print("="*50)
