"""
Ejemplo Rápido: Generar Subset del Dataset

Este script muestra cómo usar el generador de subsets de forma simple.
"""

from generador_subsets import DatasetSliceGenerator

# Ruta al dataset completo
DATASET_PATH = 'data/dataset_inventario_secuencial_completo.csv'

# Número de productos que quieres en tu subset
NUM_PRODUCTOS = 1000

# Método de selección: 'aleatorio', 'primeros', 'mas_registros', 'menos_registros'
METODO = 'aleatorio'

# Archivo de salida
OUTPUT_FILE = f'data/subset_{NUM_PRODUCTOS}_productos.csv'

# ============================================================================
# EJECUTAR
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("GENERADOR DE SUBSET - EJEMPLO RÁPIDO")
    print("=" * 70)
    
    # Crear generador
    print("\n1. Cargando dataset...")
    generator = DatasetSliceGenerator(DATASET_PATH)
    
    # Mostrar estadísticas
    print("\n2. Estadísticas del dataset completo:")
    stats = generator.get_stats()
    for key, value in stats.items():
        formatted_value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
        print(f"   - {key}: {formatted_value}")
    
    # Generar subset
    print(f"\n3. Generando subset con {NUM_PRODUCTOS} productos...")
    df_subset = generator.generate_subset(
        num_productos=NUM_PRODUCTOS,
        metodo=METODO,
        random_state=42
    )
    
    # Guardar subset
    print("\n4. Guardando subset...")
    generator.save_subset(df_subset, OUTPUT_FILE)
    
    print("\n" + "=" * 70)
    print("✓ PROCESO COMPLETADO")
    print("=" * 70)
    print(f"\nTu subset está listo en: {OUTPUT_FILE}")
    print(f"Puedes usar este archivo para entrenar tu modelo.\n")
