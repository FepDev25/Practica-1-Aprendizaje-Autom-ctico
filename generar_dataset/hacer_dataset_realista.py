#!/usr/bin/env python3
"""
Script para convertir el dataset secuencial en uno REALISTA con todas las variables dinámicas.
- Simula consumo diario realista
- Actualiza stock coherentemente (on_hand, reserved, available)
- Actualiza valores calculados (total_value, average_daily_usage)
- Actualiza fechas (last_order_date, last_stock_count_date, last_updated_at)
- Cambia stock_status según niveles de inventario
- Simula reabastecimientos cuando stock baja del reorder_point
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid

print("\n" + "="*70)
print("GENERANDO DATASET REALISTA CON TODAS LAS VARIABLES DINÁMICAS")
print("="*70)

# ============================================================================
# [1] CARGAR DATASET
# ============================================================================
print("\n[1/6] Cargando dataset...")
df = pd.read_csv('dataset_inventario_secuencial_completo.csv')
print(f"    ✓ {len(df)} registros cargados")
print(f"    ✓ {df['product_id'].nunique()} productos únicos")

# Ordenar por producto y fecha
df['last_updated_at'] = pd.to_datetime(df['last_updated_at'])
df = df.sort_values(['product_id', 'last_updated_at']).reset_index(drop=True)

# ============================================================================
# [2] FUNCIÓN PARA SIMULAR CONSUMO REALISTA
# ============================================================================
def calcular_consumo_realista(dia_semana, stock_actual, consumo_base):
    """
    Calcula consumo diario considerando:
    - Patrón semanal (más consumo lunes-viernes)
    - Variabilidad aleatoria (±30%)
    - Límite por stock disponible
    """
    # Factor día de semana (0=lunes, 6=domingo)
    if dia_semana in [5, 6]:  # Fin de semana
        factor_dia = random.uniform(0.4, 0.7)
    else:  # Entre semana
        factor_dia = random.uniform(0.9, 1.3)
    
    # Variabilidad aleatoria
    variacion = random.uniform(0.7, 1.3)
    
    # Calcular consumo
    consumo = int(consumo_base * factor_dia * variacion)
    
    # No consumir más de lo disponible
    consumo = min(consumo, stock_actual)
    consumo = max(0, consumo)  # No negativo
    
    return consumo

# ============================================================================
# [3] FUNCIÓN PARA SIMULAR REABASTECIMIENTO
# ============================================================================
def simular_reabastecimiento(stock_actual, reorder_point, reorder_quantity):
    """
    Simula reabastecimiento cuando stock baja del punto de reorden
    Retorna: (nuevo_stock, hubo_reabastecimiento)
    """
    if stock_actual <= reorder_point:
        # Reabastecer con variabilidad (±20% de reorder_quantity)
        cantidad_reabasto = int(reorder_quantity * random.uniform(0.8, 1.2))
        return stock_actual + cantidad_reabasto, True
    return stock_actual, False

# ============================================================================
# [4] FUNCIÓN PARA CALCULAR STOCK_STATUS
# ============================================================================
def calcular_stock_status(available, minimum, optimal):
    """
    Calcula el estado del stock basado en niveles
    """
    if available == 0:
        return 0  # sin stock
    elif available <= minimum:
        return 1  # stock bajo
    elif available <= optimal * 0.7:
        return 2  # stock normal
    else:
        return 3  # stock alto

# ============================================================================
# [5] PROCESAR CADA PRODUCTO DÍA POR DÍA
# ============================================================================
print("\n[2/6] Procesando productos día por día...")

productos_unicos = df['product_id'].unique()
registros_nuevos = []

for idx, product_id in enumerate(productos_unicos, 1):
    if idx % 5 == 0 or idx == len(productos_unicos):
        print(f"    Progreso: {idx}/{len(productos_unicos)} ({idx/len(productos_unicos)*100:.1f}%)", end='\r')
    
    # Obtener registros del producto
    mask = df['product_id'] == product_id
    producto_df = df[mask].copy()
    
    # Valores iniciales del primer día
    primer_registro = producto_df.iloc[0].to_dict()
    
    # Variables que se mantienen constantes
    product_name = primer_registro['product_name']
    product_sku = primer_registro['product_sku']
    supplier_id = primer_registro['supplier_id']
    supplier_name = primer_registro['supplier_name']
    minimum_stock = primer_registro['minimum_stock_level']
    reorder_point = primer_registro['reorder_point']
    optimal_stock = primer_registro['optimal_stock_level']
    reorder_qty = primer_registro['reorder_quantity']
    unit_cost = primer_registro['unit_cost']
    expiration_date = primer_registro['expiration_date']
    batch_number = primer_registro['batch_number']
    warehouse = primer_registro['warehouse_location']
    shelf = primer_registro['shelf_location']
    is_active = primer_registro['is_active']
    notes = primer_registro['notes']
    created_by_id = primer_registro['created_by_id']
    created_at = primer_registro['created_at']
    
    # Variables dinámicas iniciales
    consumo_base = primer_registro['average_daily_usage']
    quantity_on_hand = primer_registro['quantity_on_hand']
    quantity_reserved = primer_registro['quantity_reserved']
    quantity_available = quantity_on_hand - quantity_reserved
    
    # Historial de consumos para calcular promedio móvil
    consumos_historicos = [consumo_base] * 7
    
    # Fechas dinámicas
    last_order_date = pd.to_datetime(primer_registro['last_order_date'])
    last_stock_count_date = pd.to_datetime(primer_registro['last_stock_count_date'])
    
    # Procesar cada día
    for i in range(len(producto_df)):
        fecha_actual = producto_df.iloc[i]['last_updated_at']
        dia_semana = fecha_actual.dayofweek
        
        # ====================================
        # CONSUMO DIARIO
        # ====================================
        consumo_dia = calcular_consumo_realista(dia_semana, quantity_available, consumo_base)
        
        # Actualizar stock
        quantity_available -= consumo_dia
        quantity_on_hand = quantity_available + quantity_reserved
        
        # ====================================
        # REABASTECIMIENTO SI ES NECESARIO
        # ====================================
        quantity_on_hand, hubo_reabasto = simular_reabastecimiento(
            quantity_on_hand, reorder_point, reorder_qty
        )
        quantity_available = quantity_on_hand - quantity_reserved
        
        # Si hubo reabastecimiento, actualizar fecha de orden
        if hubo_reabasto:
            last_order_date = fecha_actual
        
        # ====================================
        # ACTUALIZAR PROMEDIO DE CONSUMO
        # ====================================
        consumos_historicos.append(consumo_dia)
        if len(consumos_historicos) > 30:  # Mantener últimos 30 días
            consumos_historicos.pop(0)
        average_daily_usage = round(np.mean(consumos_historicos), 2)
        
        # ====================================
        # ACTUALIZAR VALORES CALCULADOS
        # ====================================
        total_value = round(quantity_on_hand * unit_cost, 2)
        
        # ====================================
        # ACTUALIZAR STOCK STATUS
        # ====================================
        stock_status = calcular_stock_status(quantity_available, minimum_stock, optimal_stock)
        
        # ====================================
        # ACTUALIZAR CANTIDAD RESERVADA (varía ligeramente)
        # ====================================
        if random.random() < 0.2:  # 20% de probabilidad de cambio
            cambio = random.randint(-5, 10)
            quantity_reserved = max(0, quantity_reserved + cambio)
            quantity_reserved = min(quantity_reserved, quantity_on_hand)  # No puede ser mayor al stock
            quantity_available = quantity_on_hand - quantity_reserved
        
        # ====================================
        # ACTUALIZAR FECHA DE CONTEO (cada 7-10 días)
        # ====================================
        dias_desde_conteo = (fecha_actual - last_stock_count_date).days
        if dias_desde_conteo >= random.randint(7, 10):
            last_stock_count_date = fecha_actual
        
        # ====================================
        # CREAR REGISTRO DEL DÍA
        # ====================================
        registro_dia = {
            'id': str(uuid.uuid4()),
            'created_at': created_at,
            'product_id': product_id,
            'product_name': product_name,
            'product_sku': product_sku,
            'supplier_id': supplier_id,
            'supplier_name': supplier_name,
            'quantity_on_hand': int(quantity_on_hand),
            'quantity_reserved': int(quantity_reserved),
            'quantity_available': int(quantity_available),
            'minimum_stock_level': int(minimum_stock),
            'reorder_point': int(reorder_point),
            'optimal_stock_level': int(optimal_stock),
            'reorder_quantity': int(reorder_qty),
            'average_daily_usage': average_daily_usage,
            'last_order_date': last_order_date.strftime('%Y-%m-%d'),
            'last_stock_count_date': last_stock_count_date.strftime('%Y-%m-%d'),
            'unit_cost': unit_cost,
            'total_value': total_value,
            'expiration_date': expiration_date,
            'batch_number': batch_number,
            'warehouse_location': warehouse,
            'shelf_location': shelf,
            'stock_status': int(stock_status),
            'is_active': is_active,
            'last_updated_at': fecha_actual.strftime('%Y-%m-%d'),
            'notes': notes,
            'created_by_id': created_by_id
        }
        
        registros_nuevos.append(registro_dia)

print(f"\n    ✓ {len(registros_nuevos)} registros procesados")

# ============================================================================
# [6] CREAR DATAFRAME FINAL
# ============================================================================
print("\n[3/6] Creando DataFrame final...")
df_realista = pd.DataFrame(registros_nuevos)
print(f"    ✓ DataFrame creado: {len(df_realista)} filas × {len(df_realista.columns)} columnas")

# ============================================================================
# [7] VERIFICAR CALIDAD DE DATOS
# ============================================================================
print("\n[4/6] Verificando calidad de datos...")
print(f"    • Valores NaN totales: {df_realista.isnull().sum().sum()}")
print(f"    • Productos únicos: {df_realista['product_id'].nunique()}")
print(f"    • Rango de fechas: {df_realista['last_updated_at'].min()} a {df_realista['last_updated_at'].max()}")
print(f"    • Distribución stock_status:")
for status, nombre in [(0, 'Sin stock'), (1, 'Bajo'), (2, 'Normal'), (3, 'Alto')]:
    count = (df_realista['stock_status'] == status).sum()
    pct = count / len(df_realista) * 100
    print(f"      - {nombre}: {count} ({pct:.1f}%)")

# ============================================================================
# [8] GUARDAR DATASET
# ============================================================================
print("\n[5/6] Guardando dataset realista...")
output_file = 'dataset_inventario_secuencial_completo-reducido.csv'
df_realista.to_csv(output_file, index=False)
print(f"    ✓ Guardado en: {output_file}")

# ============================================================================
# [9] MOSTRAR EJEMPLO
# ============================================================================
print("\n[6/6] Ejemplo de datos realistas generados:")
print("="*70)
producto_ejemplo = df_realista['product_id'].iloc[0]
ejemplo = df_realista[df_realista['product_id'] == producto_ejemplo].head(10)
print(f"\nProducto: {ejemplo.iloc[0]['product_name']} ({producto_ejemplo})")
print("\nEvolución temporal:")
print(ejemplo[['last_updated_at', 'quantity_available', 'quantity_reserved', 'quantity_on_hand', 
               'average_daily_usage', 'total_value', 'stock_status']].to_string(index=False))

print("\n" + "="*70)
print("✓ DATASET REALISTA GENERADO EXITOSAMENTE")
print("="*70)
print(f"\n📊 Archivo: {output_file}")
print(f"📈 {len(df_realista)} registros × {len(df_realista.columns)} columnas")
print(f"🎯 Listo para entrenar RNN/LSTM")
print("\n")
