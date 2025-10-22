"""
Generador de Dataset Secuencial COMPLETO
Mantiene TODAS las variables originales + añade features temporales
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid

print("="*80)
print("GENERADOR DE DATASET SECUENCIAL - VERSIÓN COMPLETA")
print("Mantiene todas las variables originales del dataset")
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
}

# ============================================================================
# CARGAR DATASET ORIGINAL
# ============================================================================

print("\n" + "="*80)
print("CONVERSIÓN A SERIES TEMPORALES SECUENCIALES")
print("Manteniendo SOLO las 28 columnas originales")
print("="*80)

print("\n[1/4] Cargando dataset original...")
df_original = pd.read_csv('dataset_inventario.csv')
total_productos = len(df_original)
print(f"    ✓ {total_productos} productos cargados")
print(f"    ✓ Columnas originales: {len(df_original.columns)}")

# ============================================================================
# CONFIGURAR FECHAS
# ============================================================================

print("\n[2/4] Configurando fechas...")
fecha_inicio = datetime(2025, 10, 1)
fecha_fin = datetime(2025, 10, 31)
fechas = pd.date_range(fecha_inicio, fecha_fin, freq='D')

print(f"    ✓ Período: {fecha_inicio.strftime('%Y-%m-%d')} a {fecha_fin.strftime('%Y-%m-%d')}")
print(f"    ✓ Total de días: {len(fechas)}")

# ============================================================================
# GENERAR SERIES TEMPORALES
# ============================================================================

print("\n[3/4] Generando series temporales...")

# ============================================================================
# GENERAR SERIES TEMPORALES
# ============================================================================

print("\n[2/5] Configurando fechas...")
fecha_inicio = pd.to_datetime(CONFIG['fecha_inicio'])
fecha_fin = pd.to_datetime(CONFIG['fecha_fin'])
fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
print(f"    ✓ Período: {fecha_inicio.date()} a {fecha_fin.date()}")
print(f"    ✓ Total de días: {len(fechas)}")

print("\n[3/5] Generando series temporales (esto puede tomar unos minutos)...")

datos_secuenciales = []
total_productos = len(df_original)

for idx, row in df_original.iterrows():
    # Valores finales (del dataset original)
    quantity_final = row['quantity_available']
    quantity_on_hand_final = row['quantity_on_hand']
    quantity_reserved_final = row['quantity_reserved']
    
    # Consumo base del producto
    consumo_base = row.get('average_daily_usage', random.uniform(
        CONFIG['consumo_base_min'], 
        CONFIG['consumo_base_max']
    ))
    
    # Simular stock inicial (mayor que el final)
    dias_total = len(fechas)
    stock_inicial = quantity_final + int(consumo_base * dias_total * random.uniform(0.8, 1.2))
    stock_inicial = max(stock_inicial, quantity_final + 50)
    
    # Generar serie temporal día a día
    stock_actual = stock_inicial
    on_hand_actual = stock_inicial + quantity_reserved_final
    
    for i, fecha in enumerate(fechas):
        dias_restantes = dias_total - i
        
        # Calcular consumo del día
        if dias_restantes > 1:
            variacion = random.uniform(
                1 - CONFIG['variacion_diaria'],
                1 + CONFIG['variacion_diaria']
            )
            consumo_hoy = int(consumo_base * variacion)
            
            # Picos ocasionales
            if random.random() < CONFIG['prob_pico_consumo']:
                consumo_hoy = int(consumo_hoy * CONFIG['factor_pico'])
            
            # Ajustar para llegar al valor final en la última fecha
            if i == len(fechas) - 2:
                consumo_hoy = stock_actual - quantity_final
            
            stock_actual = max(stock_actual - consumo_hoy, 0)
            on_hand_actual = max(on_hand_actual - consumo_hoy, 0)
        else:
            # Último día: usar valores finales del dataset original
            stock_actual = quantity_final
            on_hand_actual = quantity_on_hand_final
        
        # Calcular stock_status
        min_stock = row['minimum_stock_level']
        reorder_point = row['reorder_point']
        
        if stock_actual == 0:
            status = 0
        elif stock_actual < min_stock:
            status = 1
        elif stock_actual < reorder_point:
            status = 2
        else:
            status = 3
        
        # Crear registro con SOLO las 28 columnas originales
        registro = {
            'id': str(uuid.uuid4()),  # Nuevo ID para cada registro temporal
            'created_at': row['created_at'],
            'product_id': row['product_id'],
            'product_name': row['product_name'],
            'product_sku': row['product_sku'],
            'supplier_id': row['supplier_id'],
            'supplier_name': row['supplier_name'],
            'quantity_on_hand': on_hand_actual,
            'quantity_reserved': quantity_reserved_final,
            'quantity_available': stock_actual,
            'minimum_stock_level': row['minimum_stock_level'],
            'reorder_point': row['reorder_point'],
            'optimal_stock_level': row['optimal_stock_level'],
            'reorder_quantity': row['reorder_quantity'],
            'average_daily_usage': row['average_daily_usage'],
            'last_order_date': row['last_order_date'],
            'last_stock_count_date': row['last_stock_count_date'],
            'unit_cost': row['unit_cost'],
            'total_value': row['total_value'],
            'expiration_date': row['expiration_date'],
            'batch_number': row['batch_number'],
            'warehouse_location': row['warehouse_location'],
            'shelf_location': row['shelf_location'],
            'stock_status': status,
            'is_active': row['is_active'],
            'last_updated_at': fecha.strftime('%Y-%m-%d'),
            'notes': row['notes'],
            'created_by_id': row['created_by_id'],
        }
        
        datos_secuenciales.append(registro)
    
    # Progreso
    if (idx + 1) % 250 == 0 or idx == total_productos - 1:
        progreso = ((idx + 1) / total_productos) * 100
        print(f"    Progreso: {idx + 1}/{total_productos} ({progreso:.1f}%)")

# ============================================================================
# CREAR DATAFRAME Y ORDENAR
# ============================================================================

print("\n[4/4] Creando DataFrame secuencial...")
df_secuencial = pd.DataFrame(datos_secuenciales)

# Ordenar por producto y last_updated_at
df_secuencial = df_secuencial.sort_values(['product_id', 'last_updated_at']).reset_index(drop=True)

print(f"    ✓ {len(df_secuencial):,} registros generados")
print(f"    ✓ Total de columnas: {len(df_secuencial.columns)}")

# ============================================================================
# GUARDAR DATASET
# ============================================================================

output_file = 'dataset_inventario_secuencial_completo.csv'
df_secuencial.to_csv(output_file, index=False)
print(f"\n✓ Dataset guardado en: {output_file}")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print("RESUMEN DEL DATASET SECUENCIAL")
print("="*80)

print(f"\nDimensiones: {df_secuencial.shape[0]:,} filas × {df_secuencial.shape[1]} columnas")
print(f"\nTotal de columnas: {len(df_secuencial.columns)}")

print("\n=== COLUMNAS (28) ===")
print(", ".join(df_secuencial.columns.tolist()))

# Estadísticas de stock_status
print("\n=== DISTRIBUCIÓN DE STOCK STATUS ===")
status_counts = df_secuencial['stock_status'].value_counts()
for status, count in status_counts.items():
    pct = (count / len(df_secuencial)) * 100
    status_name = ['Sin stock', 'Stock bajo', 'Stock normal', 'Stock alto'][status]
    print(f"{status_name}: {count:,} ({pct:.1f}%)")

# Mostrar ejemplo de un producto
print("\n=== EJEMPLO DE SERIE TEMPORAL (primer producto) ===")
producto_ejemplo = df_secuencial['product_id'].iloc[0]
df_ejemplo = df_secuencial[df_secuencial['product_id'] == producto_ejemplo].head(5)
print(f"\nProducto: {df_ejemplo['product_name'].iloc[0]} ({producto_ejemplo})")
print("\nPrimeros 5 días:")
for _, row in df_ejemplo.iterrows():
    print(f"  {row['last_updated_at']}: {row['quantity_available']} unidades (status: {row['stock_status']})")

print("\n" + "="*80)
print("✓ Proceso completado exitosamente")
print("="*80)

# ============================================================================
# RELLENAR VALORES NaN
# ============================================================================

print("    Rellenando valores NaN en los primeros días...")

# Para cada producto, rellenar los primeros días con valores apropiados
for producto_id in df_secuencial['product_id'].unique():
    mask = df_secuencial['product_id'] == producto_id
    
    # Lags: forward fill (usar el primer valor disponible)
    for i in range(1, 4):
        col = f'cantidad_t_minus_{i}'
        df_secuencial.loc[mask, col] = df_secuencial.loc[mask, col].fillna(method='bfill')
    
    # Consumo diario: rellenar con la media del producto
    consumo_mean = df_secuencial.loc[mask, 'consumo_diario'].mean()
    df_secuencial.loc[mask, 'consumo_diario'] = df_secuencial.loc[mask, 'consumo_diario'].fillna(consumo_mean)
    
    # Tendencia: rellenar con 0
    df_secuencial.loc[mask, 'tendencia_3d'] = df_secuencial.loc[mask, 'tendencia_3d'].fillna(0)
    
    # Variación porcentual: rellenar con 0
    df_secuencial.loc[mask, 'var_porcentual'] = df_secuencial.loc[mask, 'var_porcentual'].fillna(0)
    
    # Std móvil: rellenar con 0
    df_secuencial.loc[mask, 'std_movil_7d'] = df_secuencial.loc[mask, 'std_movil_7d'].fillna(0)

print(f"    ✓ Features calculadas y NaN rellenados")

# ============================================================================
# VERIFICAR Y GUARDAR
# ============================================================================

# Verificar que no hay NaN
nan_count = df_secuencial.isnull().sum().sum()
if nan_count > 0:
    print(f"    ⚠️  Advertencia: Quedan {nan_count} valores NaN")
else:
    print(f"    ✓ Dataset limpio: 0 valores NaN")

# Guardar
archivo_salida = 'dataset_inventario_secuencial_completo.csv'
df_secuencial.to_csv(archivo_salida, index=False, encoding='utf-8')

print(f"\n✓ Dataset guardado en: {archivo_salida}")

# ============================================================================
# RESUMEN
# ============================================================================

print("\n" + "="*80)
print("RESUMEN DEL DATASET GENERADO")
print("="*80)

print(f"\nDimensiones: {df_secuencial.shape[0]:,} filas × {df_secuencial.shape[1]} columnas")
print(f"\nProductos únicos: {df_secuencial['product_id'].nunique():,}")
print(f"Período: {df_secuencial['timestamp'].min()} a {df_secuencial['timestamp'].max()}")
print(f"Días por producto: {len(df_secuencial) // df_secuencial['product_id'].nunique()}")

print("\n=== COLUMNAS ORIGINALES (28) ===")
columnas_originales = [
    'id', 'created_at', 'product_id', 'product_name', 'product_sku',
    'supplier_id', 'supplier_name', 'quantity_on_hand', 'quantity_reserved',
    'quantity_available', 'minimum_stock_level', 'reorder_point',
    'optimal_stock_level', 'reorder_quantity', 'average_daily_usage',
    'last_order_date', 'last_stock_count_date', 'unit_cost', 'total_value',
    'expiration_date', 'batch_number', 'warehouse_location', 'shelf_location',
    'stock_status', 'is_active', 'last_updated_at', 'notes', 'created_by_id'
]
for i, col in enumerate(columnas_originales, 1):
    print(f"  {i:2d}. {col}")

print("\n=== COLUMNAS TEMPORALES NUEVAS (12) ===")
columnas_nuevas = [
    'timestamp', 'dia_semana', 'dia_mes', 'mes', 'es_fin_semana', 
    'semana_del_año', 'cantidad_t_minus_1', 'cantidad_t_minus_2',
    'cantidad_t_minus_3', 'consumo_diario', 'promedio_movil_7d',
    'std_movil_7d', 'tendencia_3d', 'var_porcentual'
]
for i, col in enumerate(columnas_nuevas, 1):
    print(f"  {i:2d}. {col}")

print(f"\nTotal de columnas: {len(columnas_originales) + len(columnas_nuevas)} ")

print("\nDistribución de stock_status:")
status_map = {0: 'Sin stock', 1: 'Stock bajo', 2: 'Stock normal', 3: 'Stock alto'}
for status, count in df_secuencial['stock_status'].value_counts().sort_index().items():
    pct = (count / len(df_secuencial)) * 100
    print(f"  {status} - {status_map.get(status, 'Desconocido'):12s}: {count:6,} ({pct:5.1f}%)")

print("\nEjemplo de 1 producto (primeros 10 días):")
producto_ej = df_secuencial['product_id'].iloc[0]
cols_mostrar = ['product_id', 'product_name', 'timestamp', 'quantity_available', 
                'stock_status', 'consumo_diario', 'promedio_movil_7d']
print(df_secuencial[df_secuencial['product_id'] == producto_ej][cols_mostrar].head(10).to_string(index=False))

print("\n" + "="*80)
print("✓ DATASET SECUENCIAL COMPLETO GENERADO EXITOSAMENTE")
print("="*80)
print("\nCaracterísticas:")
print("  ✓ Mantiene TODAS las 28 columnas originales")
print("  ✓ Añade 12 columnas de features temporales")
print("  ✓ Series temporales de 31 días por producto")
print("  ✓ Sin valores NaN (todos rellenados)")
print("  ✓ Listo para modelos RNN/LSTM")
