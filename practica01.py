import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    import pandas as pd
    import numpy as np
    from scipy.stats import zscore
    return mo, np, pd, px, zscore


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


@app.cell
def _(mo):
    mo.md(r"""## 4. Análisis univriado""")
    return


@app.cell
def _(df, px):
    # Histograma interactivo de la cantidad disponible
    fig_cantidad = px.histogram(
        df,
        x="quantity_available",
        title="Distribución de la Cantidad Disponible",
        marginal="box",
        hover_data=df.columns,
    )

    # AÑADE UN BORDE a las barras
    fig_cantidad.update_traces(
        marker_line_color='rgb(8,48,107)', # Un color de borde oscuro
        marker_line_width=1
    )

    # Muestra el gráfico
    fig_cantidad
    return


@app.cell
def _(df, px):
    # Histograma interactivo del costo unitario
    fig_costo = px.histogram(
        df,
        x="unit_cost",
        title="Distribución del Costo Unitario",
        marginal="box"
    )

    fig_costo.update_traces(
        marker_line_color='rgb(8,48,107)', # Un color de borde oscuro
        marker_line_width=1
    )

    # Muestra el gráfico
    fig_costo
    return


@app.cell
def _(df, px):
    # Histograma interactivo del uso diario
    fig_uso = px.histogram(
        df,
        x="average_daily_usage",
        title="Distribución del Uso Diario Promedio",
        marginal="box"
    )

    fig_uso.update_traces(
        marker_line_color='rgb(8,48,107)', # Un color de borde oscuro
        marker_line_width=1
    )

    # Muestra el gráfico
    fig_uso
    return


@app.cell
def _(df):
    # Conteo de la columna 'stock_status'
    df['stock_status'].value_counts()
    return


@app.cell
def _(df, px):
    # Gráfico de barras interactivo para 'stock_status'
    # Usamos el DataFrame de value_counts() para un mejor control
    status_counts = df['stock_status'].value_counts().reset_index()
    status_counts.columns = ['Stock Status', 'Count']

    fig_status = px.bar(
        status_counts,
        x='Stock Status',
        y='Count',
        title='Frecuencia del Estado del Stock',
        color='Stock Status',
        template='plotly_white',
    )

    # Muestra el gráfico
    fig_status
    return


@app.cell
def _(df, px):
    # 1. Calcular el Top 10 de proveedores
    top_10_suppliers = df['supplier_name'].value_counts().nlargest(10).reset_index()
    top_10_suppliers.columns = ['Proveedor', 'Registros']

    # 2. Mostrar la tabla (Marimo la renderiza bonito)
    print(top_10_suppliers)

    # 3. Graficar el Top 10
    fig_suppliers = px.bar(
        top_10_suppliers,
        x='Proveedor',
        y='Registros',
        title='Top 10 Proveedores por Nro. de Registros de Producto',
        template='plotly_white'
    )

    # Borde a las barras
    fig_suppliers.update_traces(
        marker_line_color='rgb(0,0,0)', # Color del borde negro
        marker_line_width=1.5           # Ancho del borde
    )

    # Muestra el gráfico
    fig_suppliers
    return


@app.cell
def _(mo):
    mo.md(r"""## 5. Análisis Bivariado y Multivariado""")
    return


@app.cell
def _(df, px):
    # Scatter plot: Costo Unitario vs. Cantidad Disponible
    fig_scatter = px.scatter(
        df,
        x="unit_cost",
        y="quantity_available",
        title="Relación entre Costo Unitario y Cantidad Disponible",
        template="plotly_white",
        hover_data=["product_name"]  # Muestra el nombre del producto al pasar el mouse
    )

    fig_scatter.update_traces(
        marker=dict(line=dict(width=1, color='DarkSlateGrey'))
    )

    fig_scatter
    return


@app.cell
def _(df, px):
    # 1. Seleccionar solo las columnas numéricas
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])

    # 2. Calcular la matriz de correlación
    corr_matrix = numeric_cols.corr()

    # 3. Visualizar el heatmap
    # 'color_continuous_scale' nos da un mejor rango de color (Rojo-Azul)
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,  # Muestra los valores numéricos en las celdas
        aspect="auto",
        title="Mapa de Calor de Correlación Numérica",
        color_continuous_scale='RdBu_r', # 'r' invierte la escala (Rojo=1, Azul=-1)
        zmin=-1, # Fija los límites del color
        zmax=1
    )

    fig_heatmap
    return (numeric_cols,)


@app.cell
def _(df, px):
    # Boxplot: Cantidad Disponible vs. Estado del Stock
    fig_box = px.box(
        df,
        x="stock_status",
        y="quantity_available",
        color="stock_status",  # Da un color a cada categoría
        title="Distribución de Cantidad Disponible por Estado del Stock",
        template="plotly_white"
    )

    fig_box
    return


@app.cell
def _(df, pd, px):
    # 1. Crear la tabla de contingencia (crosstab)
    cross_tab = pd.crosstab(df['warehouse_location'], df['stock_status'])

    # 2. "Derretir" (melt) la tabla para que Plotly pueda usarla
    cross_tab_tidy = cross_tab.reset_index().melt(id_vars='warehouse_location')

    # 3. Mostrar la tabla "tidy" (opcional, para que veas cómo queda)
    cross_tab_tidy.head()

    # 4. Crear el gráfico de barras agrupado
    fig_grouped_bar = px.bar(
        cross_tab_tidy,
        x="warehouse_location",
        y="value",
        color="stock_status",
        title="Estado del Stock por Ubicación del Almacén",
        template="plotly_white",
        barmode="group"  # <-- Esto crea las barras agrupadas
    )

    fig_grouped_bar
    return


@app.cell
def _(df, px):
    # Lista de columnas clave para el pairplot
    pairplot_cols = [
        'quantity_available',
        'unit_cost',
        'average_daily_usage',
        'reorder_point',
        'total_value',
        'stock_status' # La usaremos para el color
    ]

    # Creamos el 'scatter_matrix' (equivalente a pairplot)
    fig_pairplot = px.scatter_matrix(
        df[pairplot_cols],
        dimensions=['quantity_available', 'unit_cost', 'average_daily_usage', 'reorder_point'],
        color="stock_status",  # Colorea los puntos por estado de stock
        title="Pairplot de Variables Clave (Coloreado por Estado de Stock)"
    )

    # Hacemos los puntos más pequeños y con borde para que se vea mejor
    fig_pairplot.update_traces(
        marker=dict(size=3, line=dict(width=0.5, color='DarkSlateGrey'))
    )

    fig_pairplot
    return


@app.cell
def _(mo):
    mo.md(r"""## 6. Detección de outliers""")
    return


@app.cell
def _(df, numeric_cols):
    # Iterar sobre cada columna para calcular outliers
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
    
        # Definir los límites
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    
        # Identificar los outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
        if not outliers.empty:
            print(f"\nColumna: '{col}'")
            print(f"Límites (IQR): ({lower_bound:.2f}, {upper_bound:.2f})")
            print(f"Total de outliers detectados: {len(outliers)}")
            # print(outliers[[col, 'product_name']].sort_values(by=col, ascending=False).head())
        else:
            print(f"\nColumna: '{col}' -> Sin outliers (según IQR).")
    return (outliers,)


@app.cell
def _(df, np, numeric_cols, outliers, zscore):
    threshold = 3 # Umbral estándar

    for col_2 in numeric_cols:
        z_scores = np.abs(zscore(df[col_2]))
    
        outliers_2 = df[z_scores > threshold]
    
        if not outliers.empty:
            print(f"\nColumna: '{col_2}'")
            print(f"Umbral (Z-score): {threshold}")
            print(f"Total de outliers detectados: {len(outliers_2)}")
            # print(outliers[[col_2, 'product_name']].sort_values(by=col_2, ascending=False).head())
        else:
            print(f"\nColumna: '{col_2}' -> Sin outliers (Z-score < {threshold}).")
    return


if __name__ == "__main__":
    app.run()
