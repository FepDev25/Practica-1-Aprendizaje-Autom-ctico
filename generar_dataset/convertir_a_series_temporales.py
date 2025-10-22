import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

print("="*70)
print("CONVERSIÓN DE DATASET ESTÁTICO A SERIES TEMPORALES")
print("="*70)

# --- 1. CARGAR EL DATASET ACTUAL ---
print("\n[1] Cargando dataset actual...")
df_actual = pd.read_csv('dataset_inventario.csv')
print(f"✓ Dataset cargado: {len(df_actual)} productos")
print(f"  Columnas: {list(df_actual.columns)[:5]}...")

# --- 2. CONFIGURACIÓN DE PARÁMETROS ---
print("\n[2] Configurando parámetros de la serie temporal...")

# Definir el rango de fechas
fecha_inicio = datetime(2025, 10, 1)  # Fecha inicial
fecha_fin = datetime(2025, 10, 11)    # Fecha final (ajustar según last_updated_at)

# Generar todas las fechas del período
fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
print(f"  Período: {fecha_inicio.date()} a {fecha_fin.date()}")
print(f"  Total de días: {len(fechas)}")

# --- 3. GENERAR SERIES TEMPORALES PARA CADA PRODUCTO ---
print("\n[3] Generando series temporales...")

datos_secuenciales = []

for idx, row in df_actual.iterrows():
    product_id = row['product_id']
    product_name = row['product_name']
    
    # Valores finales (en la última fecha)
    quantity_final = row['quantity_available']
    stock_status_final = row['stock_status']
    
    # Generar histórico: simular consumo diario hacia atrás
    # El stock fue disminuyendo día a día hasta llegar al valor actual
    
    # Calcular consumo diario promedio
    dias_total = len(fechas)
    consumo_diario_base = row.get('average_daily_usage', random.uniform(5, 20))
    
    # Simular stock inicial (mayor que el final)
    stock_inicial = quantity_final + int(consumo_diario_base * dias_total * random.uniform(0.8, 1.2))
    stock_inicial = max(stock_inicial, quantity_final + 50)  # Asegurar que sea mayor
    
    stock_actual = stock_inicial
    
    for i, fecha in enumerate(fechas):
        # Calcular días restantes hasta el final
        dias_restantes = dias_total - i
        
        if dias_restantes > 1:
            # Consumo diario con variación aleatoria
            variacion = random.uniform(0.7, 1.3)
            consumo_hoy = int(consumo_diario_base * variacion)
            
            # Añadir picos de consumo aleatorios (10% de probabilidad)
            if random.random() < 0.1:
                consumo_hoy = int(consumo_hoy * random.uniform(2, 3))
            
            # Calcular nuevo stock (no puede ser negativo)
            stock_siguiente = stock_actual - consumo_hoy
            
            # Asegurar que llegamos al valor final en la última fecha
            if i == len(fechas) - 2:  # Penúltimo día
                stock_siguiente = quantity_final + int(consumo_diario_base)
            
            stock_actual = max(stock_siguiente, 0)
        else:
            # Último día: usar el valor final real
            stock_actual = quantity_final
        
        # Determinar stock_status basado en quantity_available
        minimum_stock = row.get('minimum_stock_level', 50)
        reorder_point = row.get('reorder_point', 100)
        
        if stock_actual == 0:
            status = 0  # Sin stock
        elif stock_actual < minimum_stock:
            status = 1  # Stock bajo
        elif stock_actual < reorder_point:
            status = 2  # Stock normal
        else:
            status = 3  # Stock alto
        
        # Crear registro para este día
        registro = {
            'product_id': product_id,
            'timestamp': fecha,
            'quantity_available': stock_actual,
            'stock_status': status,
            
            # Otras features relevantes para predicción
            'product_name': product_name,
            'supplier_id': row['supplier_id'],
            'supplier_name': row['supplier_name'],
            'warehouse_location': row['warehouse_location'],
            'unit_cost': row['unit_cost'],
            'minimum_stock_level': row['minimum_stock_level'],
            'reorder_point': row['reorder_point'],
            'average_daily_usage': consumo_diario_base,
            
            # Features derivadas temporales
            'dia_semana': fecha.dayofweek,  # 0=Lunes, 6=Domingo
            'dia_mes': fecha.day,
            'mes': fecha.month,
            'es_fin_semana': 1 if fecha.dayofweek >= 5 else 0,
            
            # Indicadores de tendencia (calculados sobre la ventana)
            'dias_hasta_fin': dias_restantes,
        }
        
        datos_secuenciales.append(registro)
    
    # Mostrar progreso cada 100 productos
    if (idx + 1) % 100 == 0:
        print(f"  Procesados {idx + 1}/{len(df_actual)} productos...")

# --- 4. CREAR DATAFRAME SECUENCIAL ---
print("\n[4] Creando DataFrame secuencial...")
df_secuencial = pd.DataFrame(datos_secuenciales)

# Ordenar por producto y fecha
df_secuencial = df_secuencial.sort_values(['product_id', 'timestamp']).reset_index(drop=True)

print(f"✓ DataFrame creado con {len(df_secuencial)} registros")
print(f"  Productos únicos: {df_secuencial['product_id'].nunique()}")
print(f"  Registros por producto: ~{len(df_secuencial) // df_secuencial['product_id'].nunique()}")

# --- 5. AÑADIR FEATURES DE VENTANA MÓVIL ---
print("\n[5] Calculando features de ventana móvil...")

# Agrupar por producto y calcular features
df_secuencial['cantidad_ayer'] = df_secuencial.groupby('product_id')['quantity_available'].shift(1)
df_secuencial['cantidad_hace_2_dias'] = df_secuencial.groupby('product_id')['quantity_available'].shift(2)
df_secuencial['cantidad_hace_3_dias'] = df_secuencial.groupby('product_id')['quantity_available'].shift(3)

# Consumo diario (diferencia con el día anterior)
df_secuencial['consumo_diario'] = df_secuencial.groupby('product_id')['quantity_available'].diff() * -1

# Media móvil de 3 días
df_secuencial['promedio_3_dias'] = df_secuencial.groupby('product_id')['quantity_available'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)

# Tendencia (pendiente de los últimos 3 días)
df_secuencial['tendencia_3_dias'] = df_secuencial.groupby('product_id')['quantity_available'].diff(3)

print("✓ Features de ventana móvil calculadas")

# --- 6. GUARDAR DATASET SECUENCIAL ---
print("\n[6] Guardando dataset secuencial...")

nombre_archivo = 'dataset_inventario_secuencial.csv'
df_secuencial.to_csv(nombre_archivo, index=False, encoding='utf-8')

print(f"✓ Dataset guardado en: {nombre_archivo}")

# --- 7. MOSTRAR RESUMEN ---
print("\n" + "="*70)
print("RESUMEN DEL DATASET SECUENCIAL")
print("="*70)

print(f"\nDimensiones: {df_secuencial.shape[0]} filas × {df_secuencial.shape[1]} columnas")

print(f"\nColumnas principales:")
for col in df_secuencial.columns:
    print(f"  - {col}")

print(f"\nPrimeras filas de un producto (ejemplo):")
producto_ejemplo = df_secuencial['product_id'].iloc[0]
print(df_secuencial[df_secuencial['product_id'] == producto_ejemplo][
    ['product_id', 'timestamp', 'quantity_available', 'stock_status', 'consumo_diario', 'promedio_3_dias']
].to_string(index=False))

print(f"\nEstadísticas de quantity_available:")
print(df_secuencial['quantity_available'].describe())

print(f"\nDistribución de stock_status:")
print(df_secuencial['stock_status'].value_counts().sort_index())

print("\n" + "="*70)
print("✓ CONVERSIÓN COMPLETADA EXITOSAMENTE")
print("="*70)
