import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Practica 1: APRENDIZAJE PROFUNDO Y SERIES TEMPORALES
    ##Fase 1: Análisis y Preparación del Dataset
    **Nombres:** Felipe Peralta y Samantha Suquilanda
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    import pandas as pd
    import numpy as np
    from scipy.stats import zscore

    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    import joblib
    return LabelEncoder, MinMaxScaler, joblib, mo, np, pd, px, zscore


@app.cell
def _(mo):
    mo.md(r"""## 1. Data Engineer: Análisis exploratorio de datos""")
    return


@app.cell
def _(mo):
    mo.md(r"""### 1. Carga de datos""")
    return


@app.cell
def _(pd):
    df = pd.read_csv('dataset_inventario_secuencial_completo.csv')
    df.tail(10)
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
    mo.md(r"""### 2. Análisis descriptivo inicial""")
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
    mo.md(
        r"""
    **Análisis de los Resultados:**
    - Hay 79173 entradas (filas) y 28 columnas (features).

    - Hallazgo Crítico: existen 16 columnas de tipo object. Python usa object para texto (strings). Damos énfasis a las columnas created_at, last_order_date, last_stock_count_date y expiration_date, que están listados como "object".

    - Se puede observar un problema de Tipos de Datos (Dates), dado que vamos a trabajar con series temporales, no se puede trabajar con fechas si están en formato de texto (object).
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### 3. Limpieza de datos""")
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
    mo.md(r"""**Análisis:** El primer paso para cualquier análisis de series temporales es garantizar que el índice principal del conjunto de datos sea el tiempo. Se cargó el conjunto de datos original y se determinó que la columna created_at es nuestra variable temporal. Utilizando pd.to_datetime, se llevó a cabo una transformación explícita al formato de fecha y hora.""")
    return


@app.cell
def _(mo):
    mo.md(r"""### 4. Análisis univariado""")
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
        marker_line_color='rgb(8,48,107)',
        marker_line_width=1
    )

    fig_costo
    return


@app.cell
def _(mo):
    mo.md(r"""**Análisis:** La distribución de los valores del coste unitario se presenta en el histograma. Se nota una amplia dispersión, lo cual revela que los costos oscilan de manera significativa entre periodos o productos. El boxplot superior verifica que hay valores extremos (outliers), si bien la mayor parte de los datos se agrupan en el rango medio. Esto indica una considerable variabilidad que podría tener un impacto en la predicción de costos a través del tiempo.""")
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
        marker_line_color='rgb(8,48,107)',
        marker_line_width=1
    )

    fig_uso
    return


@app.cell
def _(mo):
    mo.md(r"""**Análisis:** La variable average_daily_usage muestra una distribución más o menos uniforme, lo que indica que los niveles de uso diario no se agrupan en una tendencia, sino que se distribuyen de manera equitativa. El boxplot muestra una amplitud significativa sin valores atípicos extremos, lo que sugiere que el consumo o uso diario de los productos es estable en términos generales.""")
    return


@app.cell
def _(df):
    df['stock_status'].value_counts()
    return


@app.cell
def _(df, px):
    # Gráfico de barras interactivo para 'stock_status'
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

    fig_status
    return


@app.cell
def _(mo):
    mo.md(r"""**Análisis:** La variable stock_status muestra una fuerte desproporción: la categoría “3” domina con más de 77 000 registros, mientras que los otros estados son minoritarios""")
    return


@app.cell
def _(df, px):
    top_10_suppliers = df['supplier_name'].value_counts().nlargest(10).reset_index()
    top_10_suppliers.columns = ['Proveedor', 'Registros']

    print(top_10_suppliers)

    fig_suppliers = px.bar(
        top_10_suppliers,
        x='Proveedor',
        y='Registros',
        title='Top 10 Proveedores por Nro. de Registros de Producto',
        template='plotly_white'
    )

    fig_suppliers.update_traces(
        marker_line_color='rgb(0,0,0)', 
        marker_line_width=1.5           
    )

    fig_suppliers
    return


@app.cell
def _(mo):
    mo.md(r"""**Análisis:** El gráfico muestra los diez proveedores con mayor número de registros en el dataset. Se observa una distribución bastante homogénea, destacando Banca Privada JWW S.L. y Hnos Raya S.L. como los principales. Esto sugiere una concentración moderada en pocos proveedores, lo cual puede influir en la disponibilidad y frecuencia de productos dentro de las series temporales.""")
    return


@app.cell
def _(mo):
    mo.md(r"""### 5. Análisis Bivariado y Multivariado""")
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
        hover_data=["product_name"]
    )

    fig_scatter.update_traces(
        marker=dict(line=dict(width=1, color='DarkSlateGrey'))
    )

    fig_scatter
    return


@app.cell
def _(mo):
    mo.md(r"""**Análisis:** La relación entre el costo unitario y la cantidad disponible no es lineal, como lo demuestra el diagrama de dispersión. El precio no determina directamente el nivel del stock, dado que los puntos se distribuyen al azar. Esto subraya la importancia de incorporar variables temporales o categóricas adicionales para representar la demanda de manera precisa.""")
    return


@app.cell
def _(df, px):
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])

    corr_matrix = numeric_cols.corr()

    # Visualizar el heatmap
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Mapa de Calor de Correlación Numérica",
        color_continuous_scale='RdBu_r', 
        zmin=-1,
        zmax=1
    )

    fig_heatmap
    return (numeric_cols,)


@app.cell
def _(mo):
    mo.md(r"""**Análisis:** El mapa de correlación muestra relaciones fuertes entre ciertas variables, como quantity_on_hand y quantity_available (r ≈ 0.99), o entre reorder_point y minimum_stock_level (r ≈ 0.90). Estas correlaciones indican redundancia informativa entre variables relacionadas con el inventario. En contraste, unit_cost y average_daily_usage presentan baja correlación con las demás, sugiriendo independencia y potencial valor predictivo.""")
    return


@app.cell
def _(df, px):
    # Boxplot: Cantidad Disponible vs. Estado del Stock
    fig_box = px.box(
        df,
        x="stock_status",
        y="quantity_available",
        color="stock_status", 
        title="Distribución de Cantidad Disponible por Estado del Stock",
        template="plotly_white"
    )

    fig_box
    return


@app.cell
def _(mo):
    mo.md(r"""**Análisis:** El boxplot muestra que el estado del stock 3 concentra la mayor cantidad de productos disponibles, con un rango amplio de valores. Los demás estados presentan cantidades muy reducidas. Esto refuerza la desproporción en la variable stock_status detectada previamente y puede requerir normalización o reagrupación antes del modelado.""")
    return


@app.cell
def _(df, pd, px):
    cross_tab = pd.crosstab(df['warehouse_location'], df['stock_status'])
    cross_tab_tidy = cross_tab.reset_index().melt(id_vars='warehouse_location')
    cross_tab_tidy.head()

    # Crear el gráfico de barras agrupado
    fig_grouped_bar = px.bar(
        cross_tab_tidy,
        x="warehouse_location",
        y="value",
        color="stock_status",
        title="Estado del Stock por Ubicación del Almacén",
        template="plotly_white",
        barmode="group"
    )

    fig_grouped_bar
    return


@app.cell
def _(mo):
    mo.md(r"""**Análisis:** La distribución del stock por ubicación evidencia una homogeneidad entre los almacenes, con el estado "3" predominando en todos los casos. Esto indica que la administración de inventarios es parecida entre los centros y no hay diferencias operacionales importantes en función de la ubicación, a pesar de que el desbalance general en el estado del stock persiste.""")
    return


@app.cell
def _(df, px):
    pairplot_cols = [
        'quantity_available',
        'unit_cost',
        'average_daily_usage',
        'reorder_point',
        'total_value',
        'stock_status'
    ]

    fig_pairplot = px.scatter_matrix(
        df[pairplot_cols],
        dimensions=['quantity_available', 'unit_cost', 'average_daily_usage', 'reorder_point'],
        color="stock_status",  # Colorea los puntos por estado de stock
        title="Pairplot de Variables Clave (Coloreado por Estado de Stock)"
    )

    fig_pairplot.update_traces(
        marker=dict(size=3, line=dict(width=0.5, color='DarkSlateGrey'))
    )

    fig_pairplot
    return


@app.cell
def _(mo):
    mo.md(r"""**Análisis:** El pairplot permite observar relaciones bivariadas entre variables numéricas clave. Se confirma una fuerte relación entre quantity_on_hand y quantity_available, mientras que unit_cost y average_daily_usage no muestran patrones evidentes. Los colores del estado de stock revelan que la mayoría de registros pertenecen a la clase 3, indicando un sesgo que deberá controlarse en las fases de modelado.""")
    return


@app.cell
def _(mo):
    mo.md(r"""### 6. Detección de outliers""")
    return


@app.cell
def _(df, numeric_cols):
    # Iterar sobre cada columna para calcular outliers
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

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
def _(mo):
    mo.md(r"""**Análisis:** El análisis mediante el rango intercuartílico (IQR) muestra ausencia de outliers en la mayoría de variables numéricas, excepto en total_value (899 valores extremos) y stock_status (2058 registros fuera del rango esperado). Esto indica que, aunque el dataset es consistente, existen valores anómalos en variables relacionadas con el valor total y estado del stock, que podrían afectar la estabilidad del modelo si no se tratan adecuadamente.""")
    return


@app.cell
def _(df, np, numeric_cols, outliers, zscore):
    threshold = 3 

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


@app.cell
def _(mo):
    mo.md(r"""**Análisis:** El método del Z-score confirmó la ausencia de valores atípicos en la mayoría de las variables numéricas. Solo se detectaron outliers en total_value (341 casos) y en stock_status (2058 registros), coincidiendo con los resultados obtenidos por el método IQR. Esto refuerza la consistencia general del dataset, pero también evidencia la necesidad de revisar estas dos variables para evitar sesgos en el entrenamiento del modelo.""")
    return


@app.cell
def _(mo):
    mo.md(r"""## 2. Feature Engineering""")
    return


@app.cell
def _(df):
    print("Iniciando Feature Engineering...")

    df_feat = df.copy()

    # Usando 'created_at' como la fecha principal del registro
    base_date = df_feat['created_at']

    df_feat['dia_del_mes'] = base_date.dt.day
    df_feat['dia_de_la_semana'] = base_date.dt.dayofweek # Lunes=0, Domingo=6
    df_feat['mes'] = base_date.dt.month
    df_feat['trimestre'] = base_date.dt.quarter
    df_feat['es_fin_de_semana'] = df_feat['dia_de_la_semana'].isin([5, 6]).astype(int)

    print("Variables temporales creadas.")

    # 1. Días restantes hasta vencimiento
    df_feat['dias_para_vencimiento'] = (df_feat['expiration_date'] - base_date).dt.days
    # Manejar valores negativos (si 'created_at' es posterior a 'expiration_date')
    df_feat['dias_para_vencimiento'] = df_feat['dias_para_vencimiento'].fillna(0)
    df_feat['dias_para_vencimiento'] = df_feat['dias_para_vencimiento'].apply(lambda x: max(0, x))

    # 2. Antigüedad del producto (Sugerido en la guía)
    df_feat['antiguedad_producto_dias'] = (base_date - df_feat['last_stock_count_date']).dt.days
    df_feat['antiguedad_producto_dias'] = df_feat['antiguedad_producto_dias'].fillna(0)
    df_feat['antiguedad_producto_dias'] = df_feat['antiguedad_producto_dias'].apply(lambda x: max(0, x))


    # 3. Ratio de uso sobre stock (Sugerido en la guía)
    df_feat['ratio_uso_stock'] = df_feat['average_daily_usage'] / (df_feat['quantity_available'] + 1)

    # Mostramos las columnas clave y las nuevas que creamos
    columnas_a_mostrar = [
        'created_at', 
        'product_id', 
        'quantity_available', 
        'average_daily_usage',
        'expiration_date',
        # --- Nuevas ---
        'dia_de_la_semana', 
        'mes', 
        'es_fin_de_semana',
        'dias_para_vencimiento',
        'antiguedad_producto_dias',
        'ratio_uso_stock'
    ]

    print(df_feat[columnas_a_mostrar].head())
    print(df_feat[columnas_a_mostrar].info())
    return (df_feat,)


@app.cell
def _(mo):
    mo.md(
        r"""
    **Análisis:**
    - En esta etapa se generaron variables temporales clave como "dia_de_la_semana", "mes", "es_fin_de_semana", "dias_para_vencimiento" y "antiguedad_producto_dias", junto con el indicador ratio_uso_stock.
    Estas nuevas columnas permitirán que el modelo capture patrones estacionales y cíclicos de demanda, fundamentales para las series temporales. Además, todas las variables fueron correctamente tipadas y sin valores nulos, garantizando calidad en los datos de entrada.
    - El nuevo DataFrame tiene 79.174 entradas y 11 columnas, sin valores ausentes. Las variables poseen tipos apropiados (int32, int64, float64, datetime64), lo que indica una conversión adecuada de valores numéricos y fechas.
    Este control garantiza que los datos estén preparados para ser codificados, escalados y utilizados en la creación de secuencias temporales.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 3. Codificación y Escalado""")
    return


@app.cell
def _():
    return


@app.cell
def _(LabelEncoder, df_feat, joblib, pd):
    df_proc = df_feat.copy()

    # Codificador para product_id
    le_product_id = LabelEncoder()
    df_proc['product_id_encoded'] = le_product_id.fit_transform(df_proc['product_id'])
    joblib.dump(le_product_id, 'le_product_id.joblib') # Guardar

    # Codificador para supplier_id
    le_supplier_id = LabelEncoder()
    df_proc['supplier_id_encoded'] = le_supplier_id.fit_transform(df_proc['supplier_id'])
    joblib.dump(le_supplier_id, 'le_supplier_id.joblib') # Guardar

    categorias_onehot = ['warehouse_location', 'stock_status']
    df_proc = pd.get_dummies(df_proc, columns=categorias_onehot, drop_first=True)

    print("\nColumnas después de One-Hot Encoding:")
    print([col for col in df_proc.columns if 'warehouse_location_' in col or 'stock_status_' in col])
    return (df_proc,)


@app.cell
def _(mo):
    mo.md(
        r"""
    **Análisis:** Se aplicó Label Encoding a los identificadores (product_id, supplier_id) para convertirlos en valores numéricos únicos, y One-Hot Encoding para las variables categóricas (warehouse_location, stock_status).
    Esta combinación preserva la información nominal sin introducir jerarquías falsas, lo que resulta esencial para modelos neuronales que procesan datos categóricos junto con variables continuas.
    """
    )
    return


@app.cell
def _(MinMaxScaler, df_proc, joblib):
    columnas_numericas = [
        'quantity_on_hand', 'quantity_reserved', 'quantity_available',
        'minimum_stock_level', 'reorder_point', 'optimal_stock_level',
        'reorder_quantity', 'average_daily_usage', 'unit_cost', 'total_value',
        'dia_del_mes', 'dia_de_la_semana', 'mes', 'trimestre', 'es_fin_de_semana',
        'dias_para_vencimiento', 'antiguedad_producto_dias', 'ratio_uso_stock'
    ]

    scaler = MinMaxScaler()
    df_proc[columnas_numericas] = scaler.fit_transform(df_proc[columnas_numericas])
    joblib.dump(scaler, 'min_max_scaler.joblib')

    print("\nEscalar Variables Numéricas")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **Análisis:**
    - Mediante el MinMaxScaler, todas las variables numéricas fueron normalizadas al rango [0,1]. Esto facilita la convergencia del modelo y evita que las variables con magnitudes grandes dominen el aprendizaje.
    - El escalador se guardó con joblib, asegurando su reutilización durante la fase de predicción para mantener la coherencia entre entrenamiento y despliegue.
    """
    )
    return


@app.cell
def _(df_proc):
    print("\nDataFrame Procesado")
    print(df_proc.head())

    columnas_modelo = df_proc.select_dtypes(exclude=['object', 'datetime64[ns]']).columns
    print(df_proc[columnas_modelo].info())
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **Análisis:**
    - El DataFrame final integra las variables originales y las transformadas, alcanzando 45 columnas. Se observan variables numéricas escaladas, categóricas codificadas y booleanas derivadas del One-Hot Encoding.

    - El resultado es un dataset completamente limpio, estructurado y listo para modelado secuencial, cumpliendo con los criterios de la Fase 1 de la guía de práctica
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### 4. Preparación de Secuencias""")
    return


@app.cell
def _(df_feat, df_proc):
    df_proc['created_at'] = df_feat['created_at']
    print("DataFrame 'df_proc' listo para la creación de secuencias.")
    return


@app.cell
def _(mo):
    mo.md(r"""**Análisis:** El DataFrame "df_proc" está totalmente listo para generar secuencias temporales. Se asegura el orden cronológico apropiado para crear ventanas de tiempo que nutran el modelo RNN si se conserva la columna created_at en formato de fecha.""")
    return


@app.cell
def _(df_proc):
    # 7 días.
    N_STEPS = 7 

    # Columna objetivo
    TARGET_COLUMN = 'quantity_available'

    FEATURE_COLUMNS = [
        'quantity_on_hand', 'quantity_reserved', 'quantity_available',
        'minimum_stock_level', 'reorder_point', 'optimal_stock_level',
        'reorder_quantity', 'average_daily_usage', 'unit_cost', 'total_value',
        'is_active', 'dia_del_mes', 'dia_de_la_semana', 'mes', 'trimestre',
        'es_fin_de_semana', 'dias_para_vencimiento', 'antiguedad_producto_dias',
        'ratio_uso_stock', 'product_id_encoded', 'supplier_id_encoded',
        'warehouse_location_Almacén Este', 'warehouse_location_Almacén Norte',
        'warehouse_location_Almacén Oeste', 'warehouse_location_Almacén Sur',
        'warehouse_location_Centro Distribución 1',
        'warehouse_location_Centro Distribución 2', 'stock_status_1',
        'stock_status_2', 'stock_status_3'
    ]

    missing_cols = [col for col in FEATURE_COLUMNS if col not in df_proc.columns]
    if missing_cols:
        print(f"Faltan las columnas: {missing_cols}")
    else:
        print(f"Todas las {len(FEATURE_COLUMNS)} features están presentes.")

    if TARGET_COLUMN not in FEATURE_COLUMNS:
        print(f"Target '{TARGET_COLUMN}' no en features.")
    return FEATURE_COLUMNS, N_STEPS, TARGET_COLUMN


@app.cell
def _(mo):
    mo.md(
        r"""
    **Análisis:**
    - La variable objetivo quantity_available se define y se fijan siete pasos temporales (N_STEPS = 7), que corresponden a una semana de observaciones.
    - Las 30 características elegidas contienen variables numéricas, categóricas codificadas y temporales derivadas, lo cual posibilita la identificación de patrones conductuales en el inventario a través del tiempo.
    """
    )
    return


@app.cell
def _(df_proc):
    print("\nDividiendo en Train y Validation")

    df_sorted = df_proc.sort_values(by='created_at')

    split_percentage = 0.8
    split_index = int(len(df_sorted) * split_percentage)

    train_df = df_sorted.iloc[:split_index]
    val_df = df_sorted.iloc[split_index:]

    print(f"Total de registros: {len(df_sorted)}")
    print(f"Set de Entrenamiento (Train): {len(train_df)} registros")
    print(f"Set de Validación (Val): {len(val_df)} registros")
    print(f"Corte temporal en: {val_df['created_at'].min()}")
    return train_df, val_df


@app.cell
def _(mo):
    mo.md(
        r"""
    **Análisis:**
    - El dataset se ordena cronológicamente por created_at y se divide en 80 % para entrenamiento y 20 % para validación.
    - El corte temporal garantiza que los datos futuros no influyan en el entrenamiento, manteniendo la coherencia temporal fundamental en series temporales.
    - El resultado muestra 79 174 registros totales: 63 339 para Train y 15 835 para Validation.
    """
    )
    return


@app.cell
def _(np):
    def create_sequences(data_df, product_group, n_steps, feature_cols, target_col):
        product_data = data_df[data_df['product_id_encoded'] == product_group].copy()

        product_data = product_data.sort_values(by='created_at')

        features = product_data[feature_cols].values
        target = product_data[target_col].values

        X, y = [], []

        for i in range(n_steps, len(product_data)):
            X.append(features[i-n_steps:i])

            y.append(target[i])

        if len(X) > 0:
            return np.array(X), np.array(y)
        else:
            return None, None
    return (create_sequences,)


@app.cell
def _(mo):
    mo.md(
        r"""
    **Análisis:**
    - Esta función genera secuencias deslizantes de N_STEPS observaciones previas para cada producto.
    - Cada ventana temporal contiene los valores de las features definidas y un valor objetivo asociado, lo que transforma los datos tabulares en estructuras tridimensionales
    - Este formato es indispensable para alimentar modelos RNN, LSTM o GRU en Keras/TensorFlow.
    """
    )
    return


@app.cell
def _(
    FEATURE_COLUMNS,
    N_STEPS,
    TARGET_COLUMN,
    create_sequences,
    np,
    train_df,
    val_df,
):
    print("\nProcesando secuencias para Train y Validation")

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    print("procesar set entrenamiento")
    unique_products_train = train_df['product_id_encoded'].unique()
    for product_id in unique_products_train:
        X_prod, y_prod = create_sequences(train_df, product_id, N_STEPS, FEATURE_COLUMNS, TARGET_COLUMN)

        if X_prod is not None:
            X_train_list.append(X_prod)
            y_train_list.append(y_prod)

    print("procesar set validacion...")
    unique_products_val = val_df['product_id_encoded'].unique()
    for product_id in unique_products_val:
        X_prod, y_prod = create_sequences(val_df, product_id, N_STEPS, FEATURE_COLUMNS, TARGET_COLUMN)

        if X_prod is not None:
            X_val_list.append(X_prod)
            y_val_list.append(y_prod)

    if len(X_train_list) > 0:
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)

        print(f"Forma de X_train (Muestras, Pasos, Features): {X_train.shape}")
        print(f"Forma de y_train (Muestras,): {y_train.shape}")
        print(f"Forma de X_val (Muestras, Pasos, Features): {X_val.shape}")
        print(f"Forma de y_val (Muestras,): {y_val.shape}")

        np.save('X_train.npy', X_train)
        np.save('y_train.npy', y_train)
        np.save('X_val.npy', X_val)
        np.save('y_val.npy', y_val)

    else:
        print("\nNo hay secuencias.")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **Análisis:**
    - Se utilizó la función create_sequences() para cada producto individualmente, lo que produjo las matrices X_train, y_train, X_val y y_val.
    - La conclusión revela que los conjuntos poseen la estructura (muestras, 7, 30), lo que confirma 30 variables por observación y 7 pasos temporales.
    """
    )
    return


@app.cell
def _(df_proc, pd):
    df_proc_path = 'df_processed_features.csv'
    df_proc['created_at'] = pd.to_datetime(df_proc['created_at'])
    df_proc.to_csv(df_proc_path, index=False)
    print(f"DataFrame procesado guardado en '{df_proc_path}'")
    return


if __name__ == "__main__":
    app.run()
