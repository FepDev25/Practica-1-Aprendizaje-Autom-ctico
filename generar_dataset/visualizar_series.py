import pandas as pd

# Cargar el dataset secuencial
df = pd.read_csv('dataset_inventario_secuencial.csv')

print("\n" + "="*80)
print("VISUALIZACIÓN DEL DATASET SECUENCIAL - SERIES TEMPORALES")
print("="*80)

# Obtener 3 productos de ejemplo
productos = df['product_id'].unique()[:3]

for producto_id in productos:
    # Filtrar datos de este producto
    df_producto = df[df['product_id'] == producto_id].copy()
    
    # Información del producto
    nombre_producto = df_producto['product_name'].iloc[0]
    
    print(f"\n{'='*80}")
    print(f"PRODUCTO: {producto_id}")
    print(f"Nombre: {nombre_producto}")
    print(f"{'='*80}")
    
    # Mostrar la serie temporal completa
    print(f"\nSerie temporal completa ({len(df_producto)} días):")
    print(df_producto[[
        'timestamp', 
        'quantity_available', 
        'stock_status', 
        'consumo_diario', 
        'promedio_3_dias'
    ]].to_string(index=False))
    
    print(f"\nEstadísticas del producto:")
    print(f"  Stock inicial: {df_producto['quantity_available'].iloc[0]}")
    print(f"  Stock final: {df_producto['quantity_available'].iloc[-1]}")
    print(f"  Consumo total: {df_producto['quantity_available'].iloc[0] - df_producto['quantity_available'].iloc[-1]}")
    print(f"  Consumo promedio diario: {df_producto['consumo_diario'].mean():.2f}")
    print(f"  Consumo máximo en un día: {df_producto['consumo_diario'].max():.2f}")

print("\n" + "="*80)
print("RESUMEN GENERAL DEL DATASET")
print("="*80)
print(f"\nTotal de registros: {len(df):,}")
print(f"Total de productos: {df['product_id'].nunique():,}")
print(f"Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
print(f"Registros por producto: {len(df) // df['product_id'].nunique()}")

print("\nDistribución de stock_status:")
status_map = {0: 'Sin stock', 1: 'Stock bajo', 2: 'Stock normal', 3: 'Stock alto'}
for status, count in df['stock_status'].value_counts().sort_index().items():
    porcentaje = (count / len(df)) * 100
    print(f"  {status} ({status_map[status]}): {count:,} registros ({porcentaje:.1f}%)")

print("\n" + "="*80)
