"""
Script mejorado para generar dataset secuencial SIN valores NaN
Rellena automáticamente los valores iniciales con estrategias apropiadas
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

print("="*80)
print("GENERADOR DE DATASET SECUENCIAL - VERSIÓN MEJORADA (SIN NaN)")
print("="*80)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

CONFIG = {
    'fecha_inicio': '2025-10-01',
    'fecha_fin': '2025-10-31',
    'consumo_base_min': 5,
    'consumo_base_max': 30,
    'variacion_diaria': 0.3,
    'prob_pico_consumo': 0.10,
    'factor_pico': 2.5,
    'stock_inicial_min': 100,
    'stock_inicial_max': 5000,
    'ventana_promedio': 7,
    'ventana_tendencia': 3,
}

# ============================================================================
# FUNCIONES
# ============================================================================

def generar_consumo_diario(consumo_base, config):
    """Genera consumo diario con variación"""
    variacion = np.random.uniform(
        1 - config['variacion_diaria'],
        1 + config['variacion_diaria']
    )
    consumo = consumo_base * variacion
    
    if np.random.random() < config['prob_pico_consumo']:
        consumo *= config['factor_pico']
    
    return int(consumo)

def calcular_stock_status(stock, min_stock, reorder_point):
    """Determina estado del stock"""
    if stock == 0:
        return 0
    elif stock < min_stock:
        return 1
    elif stock < reorder_point:
        return 2
    else:
        return 3

# ============================================================================
# GENERACIÓN DEL DATASET
# ============================================================================

print("\n[1/7] Cargando dataset original...")
df_original = pd.read_csv('dataset_inventario.csv')
print(f"    ✓ {len(df_original)} productos cargados")

print("\n[2/7] Configurando fechas...")
fecha_inicio = pd.to_datetime(CONFIG['fecha_inicio'])
fecha_fin = pd.to_datetime(CONFIG['fecha_fin'])
fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
print(f"    ✓ Período: {fecha_inicio.date()} a {fecha_fin.date()} ({len(fechas)} días)")

print("\n[3/7] Generando series temporales...")
datos = []
total_productos = len(df_original)

for idx, row in df_original.iterrows():
    product_id = row['product_id']
    
    # Parámetros
    stock_inicial = np.random.randint(
        CONFIG['stock_inicial_min'],
        CONFIG['stock_inicial_max']
    )
    consumo_base = np.random.uniform(
        CONFIG['consumo_base_min'],
        CONFIG['consumo_base_max']
    )
    min_stock = row.get('minimum_stock_level', 50)
    reorder_point = row.get('reorder_point', 150)
    
    # Generar serie temporal
    stock_actual = stock_inicial
    
    for fecha in fechas:
        consumo_hoy = generar_consumo_diario(consumo_base, CONFIG)
        stock_actual = max(stock_actual - consumo_hoy, 0)
        status = calcular_stock_status(stock_actual, min_stock, reorder_point)
        
        registro = {
            'product_id': product_id,
            'product_name': row['product_name'],
            'timestamp': fecha,
            'quantity_available': stock_actual,
            'stock_status': status,
            'supplier_id': row['supplier_id'],
            'warehouse_location': row['warehouse_location'],
            'unit_cost': row['unit_cost'],
            'minimum_stock_level': min_stock,
            'reorder_point': reorder_point,
            'dia_semana': fecha.dayofweek,
            'dia_mes': fecha.day,
            'mes': fecha.month,
            'es_fin_semana': 1 if fecha.dayofweek >= 5 else 0,
            'semana_del_año': fecha.isocalendar()[1],
        }
        datos.append(registro)
    
    if (idx + 1) % 250 == 0 or idx == total_productos - 1:
        progreso = ((idx + 1) / total_productos) * 100
        print(f"    Progreso: {idx + 1}/{total_productos} ({progreso:.1f}%)")

print("\n[4/7] Creando DataFrame...")
df = pd.DataFrame(datos)
df = df.sort_values(['product_id', 'timestamp']).reset_index(drop=True)
print(f"    ✓ {len(df):,} registros generados")

print("\n[5/7] Calculando features de ventana móvil...")

# Lags
for i in range(1, 4):
    col_name = f'cantidad_t_minus_{i}'
    df[col_name] = df.groupby('product_id')['quantity_available'].shift(i)

# Consumo diario
df['consumo_diario'] = df.groupby('product_id')['quantity_available'].diff() * -1

# Promedio móvil
ventana = CONFIG['ventana_promedio']
df['promedio_movil'] = df.groupby('product_id')['quantity_available'] \
    .rolling(window=ventana, min_periods=1).mean() \
    .reset_index(0, drop=True)

# Desviación estándar móvil
df['std_movil'] = df.groupby('product_id')['quantity_available'] \
    .rolling(window=ventana, min_periods=1).std() \
    .reset_index(0, drop=True)

# Tendencia
ventana_tend = CONFIG['ventana_tendencia']
df['tendencia'] = df.groupby('product_id')['quantity_available'].diff(ventana_tend)

# Variación porcentual
df['var_porcentual'] = df.groupby('product_id')['quantity_available'].pct_change() * 100

print(f"    ✓ Features calculadas")

print("\n[6/7] Rellenando valores NaN...")

# Estrategia de relleno para cada columna
for producto in df['product_id'].unique():
    mask = df['product_id'] == producto
    df_producto = df[mask].copy()
    
    # Para los lags: rellenar con el valor actual
    for i in range(1, 4):
        col_name = f'cantidad_t_minus_{i}'
        primer_valor = df_producto['quantity_available'].iloc[0]
        df.loc[mask, col_name] = df.loc[mask, col_name].fillna(primer_valor)
    
    # Para consumo_diario: rellenar con 0 el primer día
    df.loc[mask, 'consumo_diario'] = df.loc[mask, 'consumo_diario'].fillna(0)
    
    # Para std_movil: rellenar con 0 los primeros días
    df.loc[mask, 'std_movil'] = df.loc[mask, 'std_movil'].fillna(0)
    
    # Para tendencia: rellenar con 0
    df.loc[mask, 'tendencia'] = df.loc[mask, 'tendencia'].fillna(0)
    
    # Para var_porcentual: rellenar con 0
    df.loc[mask, 'var_porcentual'] = df.loc[mask, 'var_porcentual'].fillna(0)

# Verificar que no quedan NaN
nan_count = df.isna().sum().sum()
if nan_count > 0:
    print(f"    ⚠️  Aún quedan {nan_count} valores NaN")
    print(f"    Columnas con NaN:")
    for col in df.columns:
        nan_col = df[col].isna().sum()
        if nan_col > 0:
            print(f"      - {col}: {nan_col} valores")
else:
    print(f"    ✓ No hay valores NaN en el dataset")

print("\n[7/7] Guardando dataset...")
archivo_salida = 'dataset_inventario_secuencial_sin_nan.csv'
df.to_csv(archivo_salida, index=False, encoding='utf-8')
print(f"    ✓ Guardado en: {archivo_salida}")

# ============================================================================
# RESUMEN
# ============================================================================

print("\n" + "="*80)
print("RESUMEN DEL DATASET GENERADO")
print("="*80)

print(f"\nDimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"Productos únicos: {df['product_id'].nunique():,}")
print(f"Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
print(f"Días por producto: {len(df) // df['product_id'].nunique()}")

print("\n✅ VERIFICACIÓN DE CALIDAD:")
print(f"   Valores NaN: {df.isna().sum().sum()}")
print(f"   Valores duplicados: {df.duplicated().sum()}")
print(f"   Registros completos: {len(df)}")

print("\nColumnas generadas:")
for i, col in enumerate(df.columns, 1):
    tipo = df[col].dtype
    nan = df[col].isna().sum()
    print(f"  {i:2d}. {col:25s} - {str(tipo):10s} - NaN: {nan}")

print("\nEjemplo de primeras filas de un producto:")
producto_ej = df['product_id'].iloc[0]
print(df[df['product_id'] == producto_ej][[
    'timestamp', 'quantity_available', 'stock_status',
    'cantidad_t_minus_1', 'consumo_diario', 'promedio_movil'
]].head(10).to_string(index=False))

print("\n" + "="*80)
print("✓ DATASET GENERADO EXITOSAMENTE SIN VALORES NaN")
print("="*80)
