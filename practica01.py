import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    return mo, pd


@app.cell
def _(mo):
    mo.md(r"""## 1. Carga de datos""")
    return


@app.cell
def _(pd):
    df = pd.read_csv('dataset_inventario.csv')
    df.head(10)
    return (df,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Diccionario de Datos (Descripción de Variables)

    Basado en la estructura del dataset, este diccionario de datos describe cada columna:

    | Variable | Tipo de Dato | Descripción |
    | :--- | :--- | :--- |
    | `id` | object | Identificador único para cada registro o movimiento de inventario. |
    | `created_at` | datetime64[ns] | Fecha y hora en que se creó el registro en el sistema. |
    | `product_id` | object | Identificador único para el producto. |
    | `product_name` | object | Nombre descriptivo del producto. |
    | `product_sku` | object | (Stock Keeping Unit) Código único interno del producto. |
    | `supplier_id` | object | Identificador único del proveedor del producto. |
    | `supplier_name` | object | Nombre del proveedor. |
    | `quantity_on_hand` | int64 | Cantidad física total del producto actualmente en el almacén. |
    | `quantity_reserved` | int64 | Cantidad del producto que está apartada para pedidos pendientes. |
    | `quantity_available` | int64 | Cantidad real disponible para la venta (`on_hand` - `reserved`). |
    | `minimum_stock_level` | int64 | Nivel mínimo de stock antes de que se considere "bajo stock". |
    | `reorder_point` | int64 | Nivel de stock en el cual se debe generar una nueva orden de compra. |
    | `optimal_stock_level` | int64 | La cantidad ideal de stock que se desea mantener. |
    | `reorder_quantity` | int64 | Cantidad estándar que se pide en una nueva orden de compra. |
    | `average_daily_usage` | float64 | Promedio de unidades de este producto usadas o vendidas por día. |
    | `last_order_date` | datetime64[ns] | Fecha en que se realizó la última orden de compra de este producto. |
    | `last_stock_count_date` | datetime64[ns] | Fecha del último conteo físico de este producto en el almacén. |
    | `unit_cost` | float64 | El costo de adquirir una sola unidad del producto. |
    | `total_value` | float64 | Valor total del stock a mano (`quantity_on_hand` * `unit_cost`). |
    | `expiration_date` | datetime64[ns] | Fecha de caducidad del lote del producto (si aplica). |
    | `batch_number` | object | Número de lote para trazabilidad. |
    | `warehouse_location` | object | Ubicación general dentro del almacén (ej. "Bodega A", "Zona Fría"). |
    | `shelf_location` | object | Ubicación específica en la estantería (ej. "Pasillo 3, Rack B"). |
    | `stock_status` | int64 | Código numérico que representa el estado del stock (ej. 1=En Stock, 2=Bajo, 0=Agotado). |
    | `is_active` | bool | Indica si el producto es un ítem activo (`True`) o descontinuado (`False`). |
    | `last_updated_at` | datetime64[ns] | Fecha y hora de la última modificación de este registro. |
    | `notes` | object | Notas adicionales o comentarios sobre el producto o lote. |
    | `created_by_id` | object | Identificador del usuario o sistema que creó el registro. |
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 2. Análisis descriptivo inicial""")
    return


@app.cell
def _(df):
    print(f"Dimensiones: {df.shape}")
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(mo):
    mo.md(r"""## 3. Limpieza de datos""")
    return


@app.cell
def _(df):
    print("Valores nulos: ")
    df.isnull().sum()
    return


@app.cell
def _(df):
    print(f"Duplicados: {df.duplicated().sum()}")
    return


@app.cell
def _(df, pd):
    print("Corrección de tipos de datos a datos correctos: ")

    # Marcar productos como activos
    df["is_active"] = True

    # Transformar datos a tipo fecha
    df.created_at = pd.to_datetime(df.created_at)
    df.last_order_date = pd.to_datetime(df.last_order_date)
    df.last_updated_at = pd.to_datetime(df.last_updated_at)
    df.last_stock_count_date = pd.to_datetime(df.last_stock_count_date)
    df.expiration_date = pd.to_datetime(df.expiration_date)

    df.info()

    return


if __name__ == "__main__":
    app.run()
