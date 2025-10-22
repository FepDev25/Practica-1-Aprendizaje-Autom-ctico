"""
Script para generar un dataset de series temporales de inventario
completamente personalizable.

Características:
- Series temporales por producto
- Rango de fechas configurable
- Simulación realista de consumo diario
- Features temporales automáticas
- Compatible con modelos RNN/LSTM
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ============================================================================
# CONFIGURACIÓN PERSONALIZABLE
# ============================================================================

CONFIG = {
    # Fechas del período
    'fecha_inicio': '2025-10-01',  # Formato: 'YYYY-MM-DD'
    'fecha_fin': '2025-10-31',     # Generar 1 mes de datos
    
    # Parámetros de consumo
    'consumo_base_min': 5,         # Consumo mínimo diario
    'consumo_base_max': 30,        # Consumo máximo diario
    'variacion_diaria': 0.3,       # Variación aleatoria (30%)
    'prob_pico_consumo': 0.10,     # 10% de probabilidad de pico
    'factor_pico': 2.5,            # Multiplicador en días de pico
    
    # Stock inicial
    'stock_inicial_min': 100,      # Stock inicial mínimo
    'stock_inicial_max': 5000,     # Stock inicial máximo
    
    # Ventanas para features
    'ventana_promedio': 7,         # Promedio móvil de 7 días
    'ventana_tendencia': 3,        # Tendencia de 3 días
}

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def generar_consumo_diario(consumo_base, config):
    """Genera un consumo diario con variación realista"""
    # Variación aleatoria
    variacion = np.random.uniform(
        1 - config['variacion_diaria'],
        1 + config['variacion_diaria']
    )
    consumo = consumo_base * variacion
    
    # Pico de consumo ocasional
    if np.random.random() < config['prob_pico_consumo']:
        consumo *= config['factor_pico']
    
    return int(consumo)

def calcular_stock_status(stock, min_stock, reorder_point):
    """Determina el estado del stock"""
    if stock == 0:
        return 0  # Sin stock
    elif stock < min_stock:
        return 1  # Stock bajo
    elif stock < reorder_point:
        return 2  # Stock normal
    else:
        return 3  # Stock alto

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def generar_dataset_secuencial(archivo_entrada, config):
    """
    Genera dataset secuencial a partir del dataset estático
    
    Args:
        archivo_entrada: Ruta al CSV del dataset original
        config: Diccionario con configuración
    
    Returns:
        DataFrame con series temporales
    """
    print("="*80)
    print("GENERADOR DE DATASET SECUENCIAL PARA PREDICCIÓN DE INVENTARIO")
    print("="*80)
    
    # --- 1. CARGAR DATOS ---
    print("\n[1/6] Cargando dataset original...")
    df_original = pd.read_csv(archivo_entrada)
    print(f"    ✓ {len(df_original)} productos cargados")
    
    # --- 2. CONFIGURAR FECHAS ---
    print("\n[2/6] Configurando rango de fechas...")
    fecha_inicio = pd.to_datetime(config['fecha_inicio'])
    fecha_fin = pd.to_datetime(config['fecha_fin'])
    fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
    print(f"    ✓ Período: {fecha_inicio.date()} a {fecha_fin.date()}")
    print(f"    ✓ Total de días: {len(fechas)}")
    
    # --- 3. GENERAR SERIES TEMPORALES ---
    print("\n[3/6] Generando series temporales...")
    datos = []
    total_productos = len(df_original)
    
    for idx, row in df_original.iterrows():
        # Parámetros del producto
        product_id = row['product_id']
        
        # Stock inicial aleatorio
        stock_inicial = np.random.randint(
            config['stock_inicial_min'],
            config['stock_inicial_max']
        )
        
        # Consumo base del producto
        consumo_base = np.random.uniform(
            config['consumo_base_min'],
            config['consumo_base_max']
        )
        
        # Niveles de referencia
        min_stock = row.get('minimum_stock_level', 50)
        reorder_point = row.get('reorder_point', 150)
        
        # Generar serie temporal día a día
        stock_actual = stock_inicial
        
        for fecha in fechas:
            # Calcular consumo del día
            consumo_hoy = generar_consumo_diario(consumo_base, config)
            
            # Actualizar stock (no puede ser negativo)
            stock_actual = max(stock_actual - consumo_hoy, 0)
            
            # Calcular stock_status
            status = calcular_stock_status(stock_actual, min_stock, reorder_point)
            
            # Crear registro
            registro = {
                # Identificación
                'product_id': product_id,
                'product_name': row['product_name'],
                'timestamp': fecha,
                
                # Variables objetivo
                'quantity_available': stock_actual,
                'stock_status': status,
                
                # Features del producto
                'supplier_id': row['supplier_id'],
                'warehouse_location': row['warehouse_location'],
                'unit_cost': row['unit_cost'],
                'minimum_stock_level': min_stock,
                'reorder_point': reorder_point,
                
                # Features temporales
                'dia_semana': fecha.dayofweek,      # 0=Lunes, 6=Domingo
                'dia_mes': fecha.day,
                'mes': fecha.month,
                'es_fin_semana': 1 if fecha.dayofweek >= 5 else 0,
                'semana_del_año': fecha.isocalendar()[1],
            }
            
            datos.append(registro)
        
        # Mostrar progreso
        if (idx + 1) % 250 == 0 or idx == total_productos - 1:
            progreso = ((idx + 1) / total_productos) * 100
            print(f"    Progreso: {idx + 1}/{total_productos} productos ({progreso:.1f}%)")
    
    # --- 4. CREAR DATAFRAME ---
    print("\n[4/6] Creando DataFrame...")
    df = pd.DataFrame(datos)
    df = df.sort_values(['product_id', 'timestamp']).reset_index(drop=True)
    print(f"    ✓ {len(df):,} registros generados")
    
    # --- 5. FEATURES DE VENTANA MÓVIL ---
    print("\n[5/6] Calculando features de ventana móvil...")
    
    # Lags (valores anteriores)
    for i in range(1, 4):  # 3 días anteriores
        df[f'cantidad_t_minus_{i}'] = df.groupby('product_id')['quantity_available'].shift(i)
    
    # Consumo diario
    df['consumo_diario'] = df.groupby('product_id')['quantity_available'].diff() * -1
    
    # Promedio móvil
    ventana = config['ventana_promedio']
    df['promedio_movil'] = df.groupby('product_id')['quantity_available'] \
        .rolling(window=ventana, min_periods=1).mean() \
        .reset_index(0, drop=True)
    
    # Desviación estándar móvil
    df['std_movil'] = df.groupby('product_id')['quantity_available'] \
        .rolling(window=ventana, min_periods=1).std() \
        .reset_index(0, drop=True)
    
    # Tendencia
    ventana_tend = config['ventana_tendencia']
    df['tendencia'] = df.groupby('product_id')['quantity_available'].diff(ventana_tend)
    
    # Variación porcentual
    df['var_porcentual'] = df.groupby('product_id')['quantity_available'].pct_change() * 100
    
    print(f"    ✓ Features calculadas")
    
    # --- 6. GUARDAR ---
    print("\n[6/6] Guardando dataset...")
    archivo_salida = 'dataset_inventario_secuencial_completo.csv'
    df.to_csv(archivo_salida, index=False, encoding='utf-8')
    print(f"    ✓ Guardado en: {archivo_salida}")
    
    return df

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    # Generar dataset
    df = generar_dataset_secuencial('dataset_inventario.csv', CONFIG)
    
    # Mostrar resumen
    print("\n" + "="*80)
    print("RESUMEN DEL DATASET GENERADO")
    print("="*80)
    
    print(f"\nDimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"\nProductos únicos: {df['product_id'].nunique():,}")
    print(f"Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
    print(f"Días por producto: {len(df) // df['product_id'].nunique()}")
    
    print("\nColumnas generadas:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print("\nDistribución de stock_status:")
    status_map = {0: 'Sin stock', 1: 'Stock bajo', 2: 'Stock normal', 3: 'Stock alto'}
    for status, count in df['stock_status'].value_counts().sort_index().items():
        pct = (count / len(df)) * 100
        print(f"  {status} - {status_map[status]:12s}: {count:6,} ({pct:5.1f}%)")
    
    print("\nEjemplo de un producto:")
    producto_ej = df['product_id'].iloc[0]
    print(df[df['product_id'] == producto_ej][[
        'timestamp', 'quantity_available', 'stock_status', 
        'consumo_diario', 'promedio_movil'
    ]].head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("✓ DATASET GENERADO EXITOSAMENTE")
    print("="*80)
