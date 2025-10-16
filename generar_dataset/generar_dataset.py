import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import timedelta, datetime
import uuid

# Inicializamos Faker para generar datos de prueba en español
fake = Faker('es_ES')

# --- 1. Cargar productos desde productos.txt ---
print("Cargando productos desde productos.txt...")
ruta_productos = '../productos.txt'
with open(ruta_productos, 'r', encoding='utf-8') as f:
    productos_lista = [linea.strip() for linea in f if linea.strip()]

num_productos_unicos = len(productos_lista)
numero_de_registros = num_productos_unicos  # Uno por cada producto

print(f"Se cargaron {num_productos_unicos} productos únicos")
print(f"Generando {numero_de_registros} registros (uno por producto)...")

# --- 2. Generación de Datos para cada Columna según variables.txt ---

# id: uuid
ids = [str(uuid.uuid4()) for _ in range(numero_de_registros)]

# created_at: timestamp
created_at = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=numero_de_registros, freq='h'))

# product_id: varchar(255)
product_ids = [f"PROD-{str(uuid.uuid4())[:8].upper()}" for _ in range(numero_de_registros)]

# product_name: varchar(500) - desde productos.txt
product_names = productos_lista.copy()

# product_sku: varchar(255)
product_skus = [f"SKU-{random.randint(100000, 999999)}" for _ in range(numero_de_registros)]

# supplier_id: varchar(255)
num_proveedores = 50
proveedores_ids = [f"SUP-{str(uuid.uuid4())[:8].upper()}" for _ in range(num_proveedores)]
supplier_ids = [random.choice(proveedores_ids) for _ in range(numero_de_registros)]

# supplier_name: varchar(500)
proveedores_nombres = [fake.company() for _ in range(num_proveedores)]
supplier_names = [random.choice(proveedores_nombres) for _ in range(numero_de_registros)]

# quantity_on_hand: integer (stock en bodega)
quantity_on_hand = np.random.randint(0, 5000, size=numero_de_registros)

# quantity_reserved: integer (cantidad reservada, menor que on_hand)
quantity_reserved = [random.randint(0, min(qoh, 500)) for qoh in quantity_on_hand]

# quantity_available: integer (disponible = on_hand - reserved)
quantity_available = [qoh - qr for qoh, qr in zip(quantity_on_hand, quantity_reserved)]

# minimum_stock_level: integer
minimum_stock_level = np.random.randint(10, 200, size=numero_de_registros)

# reorder_point: integer (punto de reorden, mayor que el mínimo)
reorder_point = [msl + random.randint(10, 100) for msl in minimum_stock_level]

# optimal_stock_level: integer (nivel óptimo, mayor que reorder_point)
optimal_stock_level = [rp + random.randint(100, 500) for rp in reorder_point]

# reorder_quantity: integer (cantidad a reordenar)
reorder_quantity = np.random.randint(100, 1000, size=numero_de_registros)

# average_daily_usage: numeric(10,2)
average_daily_usage = np.round(np.random.uniform(1.0, 50.0, size=numero_de_registros), 2)

# last_order_date: date (últimos 6 meses)
last_order_date = [fake.date_between(start_date='-180d', end_date='today') for _ in range(numero_de_registros)]

# last_stock_count_date: date (últimos 30 días)
last_stock_count_date = [fake.date_between(start_date='-30d', end_date='today') for _ in range(numero_de_registros)]

# unit_cost: numeric(12,2) (costo unitario en dólares)
unit_cost = np.round(np.random.uniform(0.50, 500.0, size=numero_de_registros), 2)

# total_value: numeric(15,2) (valor total = quantity_on_hand * unit_cost)
total_value = np.round([qoh * uc for qoh, uc in zip(quantity_on_hand, unit_cost)], 2)

# expiration_date: date (fecha de vencimiento, entre 30 días y 2 años en el futuro)
expiration_date = [fake.date_between(start_date='+30d', end_date='+730d') for _ in range(numero_de_registros)]

# batch_number: varchar(255)
batch_numbers = [f"BATCH-{random.randint(1000, 9999)}-{random.randint(10, 99)}" for _ in range(numero_de_registros)]

# warehouse_location: varchar(255)
almacenes = ['Almacén Central', 'Almacén Norte', 'Almacén Sur', 'Almacén Este', 'Almacén Oeste', 'Centro Distribución 1', 'Centro Distribución 2']
warehouse_locations = [random.choice(almacenes) for _ in range(numero_de_registros)]

# shelf_location: varchar(255)
shelf_locations = [f"Pasillo-{random.randint(1, 20)}-Estante-{random.randint(1, 10)}-Nivel-{random.randint(1, 5)}" for _ in range(numero_de_registros)]

# stock_status: smallint (0=Sin stock, 1=Stock bajo, 2=Stock normal, 3=Stock alto)
stock_status = []
for qa, msl, rp in zip(quantity_available, minimum_stock_level, reorder_point):
    if qa == 0:
        status = 0  # Sin stock
    elif qa < msl:
        status = 1  # Stock bajo
    elif qa < rp:
        status = 2  # Stock normal
    else:
        status = 3  # Stock alto
    stock_status.append(status)

# is_active: boolean
is_active = [random.choice([True, False]) for _ in range(numero_de_registros)]

# last_updated_at: timestamp (actualización reciente)
last_updated_at = [fake.date_time_between(start_date='-7d', end_date='now') for _ in range(numero_de_registros)]

# notes: text
notas_posibles = [
    "Producto en buen estado",
    "Requiere revisión de inventario",
    "Próximo a vencimiento",
    "Stock bajo, programar reorden",
    "Producto de alta rotación",
    "Verificar con proveedor",
    "Almacenamiento refrigerado",
    "",  # algunos sin notas
    "",
    "Control de calidad aprobado"
]
notes = [random.choice(notas_posibles) for _ in range(numero_de_registros)]

# created_by_id: uuid
usuarios = [str(uuid.uuid4()) for _ in range(10)]  # 10 usuarios diferentes
created_by_ids = [random.choice(usuarios) for _ in range(numero_de_registros)]


# --- 3. Creación del DataFrame ---

data = {
    'id': ids,
    'created_at': created_at,
    'product_id': product_ids,
    'product_name': product_names,
    'product_sku': product_skus,
    'supplier_id': supplier_ids,
    'supplier_name': supplier_names,
    'quantity_on_hand': quantity_on_hand,
    'quantity_reserved': quantity_reserved,
    'quantity_available': quantity_available,
    'minimum_stock_level': minimum_stock_level,
    'reorder_point': reorder_point,
    'optimal_stock_level': optimal_stock_level,
    'reorder_quantity': reorder_quantity,
    'average_daily_usage': average_daily_usage,
    'last_order_date': last_order_date,
    'last_stock_count_date': last_stock_count_date,
    'unit_cost': unit_cost,
    'total_value': total_value,
    'expiration_date': expiration_date,
    'batch_number': batch_numbers,
    'warehouse_location': warehouse_locations,
    'shelf_location': shelf_locations,
    'stock_status': stock_status,
    'is_active': is_active,
    'last_updated_at': last_updated_at,
    'notes': notes,
    'created_by_id': created_by_ids
}

df = pd.DataFrame(data)

# --- 4. Guardado del Dataset ---

nombre_archivo = 'dataset_inventario.csv'
df.to_csv(nombre_archivo, index=False, encoding='utf-8')

print(f"\n¡Dataset generado y guardado exitosamente en '{nombre_archivo}'!")
print(f"\nTotal de registros: {len(df)}")
print(f"Total de columnas: {len(df.columns)}")
print("\nPrimeras 5 filas del dataset:")
print(df.head())
print("\nInformación del dataset:")
print(df.info())
print("\nEstadísticas descriptivas:")
print(df.describe())
