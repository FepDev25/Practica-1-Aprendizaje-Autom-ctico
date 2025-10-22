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

    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    import joblib # Para guardar los modelos de preprocesamiento
    return LabelEncoder, MinMaxScaler, joblib, mo, np, pd, px, zscore


@app.cell
def _(mo):
    mo.md(r"""## 1. Análisis exploratorio de datos""")
    return


@app.cell
def _(mo):
    mo.md(r"""### 1. Carga de datos""")
    return


@app.cell
def _(pd):
    df = pd.read_csv('dataset_inventario_secuencial_completo.csv')
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
    mo.md(r"""### 4. Análisis univriado""")
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
    mo.md(r"""### 6. Detección de outliers""")
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


@app.cell
def _(mo):
    mo.md(r"""## 2. Feature Engineering""")
    return


@app.cell
def _(df):
    print("Iniciando Feature Engineering...")

    df_feat = df.copy()

    # Usaremos 'created_at' como la fecha principal del registro
    base_date = df_feat['created_at']

    df_feat['dia_del_mes'] = base_date.dt.day
    df_feat['dia_de_la_semana'] = base_date.dt.dayofweek # Lunes=0, Domingo=6
    df_feat['mes'] = base_date.dt.month
    df_feat['trimestre'] = base_date.dt.quarter
    df_feat['es_fin_de_semana'] = df_feat['dia_de_la_semana'].isin([5, 6]).astype(int)

    print("Variables temporales creadas.")

    # --- B. Ingeniería de Variables Basadas en Dominio ---

    # 1. Días restantes hasta vencimiento
    #    (Calcula la diferencia y la extrae en días)
    df_feat['dias_para_vencimiento'] = (df_feat['expiration_date'] - base_date).dt.days

    # Manejar valores negativos (si 'created_at' es posterior a 'expiration_date')
    # o nulos si los hubiera 
    df_feat['dias_para_vencimiento'] = df_feat['dias_para_vencimiento'].fillna(0)
    df_feat['dias_para_vencimiento'] = df_feat['dias_para_vencimiento'].apply(lambda x: max(0, x))

    # [cite_start]2. Antigüedad del producto (Sugerido en la guía [cite: 1])
    #    (Usando 'last_stock_count_date' como proxy de "elaboración" o "ingreso")
    df_feat['antiguedad_producto_dias'] = (base_date - df_feat['last_stock_count_date']).dt.days
    df_feat['antiguedad_producto_dias'] = df_feat['antiguedad_producto_dias'].fillna(0)
    df_feat['antiguedad_producto_dias'] = df_feat['antiguedad_producto_dias'].apply(lambda x: max(0, x))


    # [cite_start]3. Ratio de uso sobre stock (Sugerido en la guía [cite: 1])
    #    Usamos 'average_daily_usage' / 'quantity_available'
    #    Sumamos 1 a 'quantity_available' para evitar división por cero.
    df_feat['ratio_uso_stock'] = df_feat['average_daily_usage'] / (df_feat['quantity_available'] + 1)

    print("Variables de dominio creadas.")

    # --- Revisión de las nuevas variables ---
    print("\n--- Vista previa del DataFrame con nuevas features ---")

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

    print("\n--- Tipos de datos de las nuevas features ---")
    print(df_feat[columnas_a_mostrar].info())
    return (df_feat,)


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

    print("--- 2. Codificando Variables Categóricas ---")

    # --- A. Label Encoding (para IDs de alta cardinalidad) ---
    # Guardaremos los encoders para poder "traducir" nuevos IDs en el futuro

    # Codificador para product_id
    le_product_id = LabelEncoder()
    df_proc['product_id_encoded'] = le_product_id.fit_transform(df_proc['product_id'])
    joblib.dump(le_product_id, 'le_product_id.joblib') # Guardar

    # Codificador para supplier_id
    le_supplier_id = LabelEncoder()
    df_proc['supplier_id_encoded'] = le_supplier_id.fit_transform(df_proc['supplier_id'])
    joblib.dump(le_supplier_id, 'le_supplier_id.joblib') # Guardar

    print("LabelEncoding aplicado a 'product_id' y 'supplier_id'.")


    # --- B. One-Hot Encoding (para categóricas de baja cardinalidad) ---
    # pd.get_dummies es la forma más fácil de hacerlo
    categorias_onehot = ['warehouse_location', 'stock_status']
    df_proc = pd.get_dummies(df_proc, columns=categorias_onehot, drop_first=True)

    print("One-Hot Encoding aplicado a 'warehouse_location' y 'stock_status'.")

    print("\nColumnas después de One-Hot Encoding:")
    print([col for col in df_proc.columns if 'warehouse_location_' in col or 'stock_status_' in col])
    return (df_proc,)


@app.cell
def _(MinMaxScaler, df_proc, joblib):
    print("\n--- 3. Escalando Variables Numéricas ---")

    # 1. Identificar TODAS las columnas numéricas para escalar
    #    (Incluyendo nuestro futuro 'target', quantity_available)
    columnas_numericas = [
        'quantity_on_hand', 'quantity_reserved', 'quantity_available',
        'minimum_stock_level', 'reorder_point', 'optimal_stock_level',
        'reorder_quantity', 'average_daily_usage', 'unit_cost', 'total_value',
        # Features que creamos en el paso anterior
        'dia_del_mes', 'dia_de_la_semana', 'mes', 'trimestre', 'es_fin_de_semana',
        'dias_para_vencimiento', 'antiguedad_producto_dias', 'ratio_uso_stock'
    ]

    # 2. Inicializar y "entrenar" el escalador
    scaler = MinMaxScaler()

    # 3. Aplicar el escalador y reemplazar las columnas
    #    Usamos fit_transform para "aprender" la escala y transformarla
    df_proc[columnas_numericas] = scaler.fit_transform(df_proc[columnas_numericas])

    # 4. Guardar el escalador para la Fase 3 (¡Muy importante!)
    #    Lo necesitaremos para revertir las predicciones
    joblib.dump(scaler, 'min_max_scaler.joblib')

    print("Todas las columnas numéricas han sido escaladas a [0, 1].")
    print("Escalador guardado como 'min_max_scaler.joblib'.")
    return


@app.cell
def _(df_proc):
    # --- 4. Revisión Final ---

    print("\n--- DataFrame Procesado (Vista Previa) ---")
    print(df_proc.head())

    print("\n--- Tipos de Datos Finales ---")
    # Filtramos solo las columnas que no son 'object' o 'datetime'
    columnas_modelo = df_proc.select_dtypes(exclude=['object', 'datetime64[ns]']).columns
    print(f"Total de features listas para el modelo: {len(columnas_modelo)}")
    print(df_proc[columnas_modelo].info())
    return


@app.cell
def _(mo):
    mo.md(r"""### 4. Preparación de Secuencias""")
    return


@app.cell
def _(df_feat, df_proc):
    # Cargamos las columnas originales necesarias que se perdieron en df_proc.select_dtypes
    df_proc['created_at'] = df_feat['created_at']
    # df_proc['product_id_encoded'] ya debería estar, pero por si acaso:
    # df_proc['product_id_encoded'] = df_feat['product_id_encoded']

    print("DataFrame 'df_proc' listo para la creación de secuencias.")
    return


@app.cell
def _(df_proc):
    # --- Parámetros de Secuencia ---

    # n_steps: ¿Cuántos días de historia usaremos para predecir el siguiente?
    # 7 días (una semana) es un buen punto de partida.
    N_STEPS = 7 

    # Columna objetivo que queremos predecir
    TARGET_COLUMN = 'quantity_available'

    # --- Listas de Columnas ---

    # Lista de TODAS las features (pistas) que usará el modelo
    # (Tu 'df_proc.select_dtypes' tenía 30 columnas. Las listamos aquí)
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

    # Verificamos que todas las columnas existan
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df_proc.columns]
    if missing_cols:
        print(f"¡ADVERTENCIA! Faltan columnas: {missing_cols}")
    else:
        print(f"Todas las {len(FEATURE_COLUMNS)} features están presentes.")

    # Asegurarnos de que el target está en las features (es común usarlo)
    if TARGET_COLUMN not in FEATURE_COLUMNS:
        print(f"Advertencia: El target '{TARGET_COLUMN}' no está en las features.")
    return FEATURE_COLUMNS, N_STEPS, TARGET_COLUMN


@app.cell
def _(df_proc):
    print("\n--- 3. Dividiendo en Train y Validation (Split Temporal) ---")

    # 1. Asegurar el orden cronológico
    df_sorted = df_proc.sort_values(by='created_at')

    # 2. Definir el punto de corte (ej. 80% para train, 20% para val)
    split_percentage = 0.8
    split_index = int(len(df_sorted) * split_percentage)

    # 3. Dividir el DataFrame
    train_df = df_sorted.iloc[:split_index]
    val_df = df_sorted.iloc[split_index:]

    print(f"Total de registros: {len(df_sorted)}")
    print(f"Set de Entrenamiento (Train): {len(train_df)} registros")
    print(f"Set de Validación (Val): {len(val_df)} registros")
    print(f"Corte temporal en: {val_df['created_at'].min()}")
    return train_df, val_df


@app.cell
def _(np):
    def create_sequences(data_df, product_group, n_steps, feature_cols, target_col):
        """
        Crea secuencias (ventanas deslizantes) para un grupo de productos.
        """
        # 1. Obtener los datos solo para este producto
        product_data = data_df[data_df['product_id_encoded'] == product_group].copy()

        # 2. Ordenar por si acaso (aunque ya debería estarlo)
        product_data = product_data.sort_values(by='created_at')

        # 3. Extraer los arrays de features y target
        features = product_data[feature_cols].values
        target = product_data[target_col].values

        X, y = [], []

        # 4. Iterar para crear las ventanas
        # Empezamos desde n_steps porque necesitamos 'n_steps' días de historia
        for i in range(n_steps, len(product_data)):
            # La ventana de features (X) es [i-n_steps] hasta [i-1]
            X.append(features[i-n_steps:i])

            # El objetivo (y) es el valor en el instante [i]
            y.append(target[i])

        # Si no hay suficientes datos para este producto (menos de n_steps),
        # las listas estarán vacías, lo cual está bien.
        if len(X) > 0:
            return np.array(X), np.array(y)
        else:
            return None, None
    return (create_sequences,)


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
    print("\n--- 5. Procesando secuencias para Train y Validation ---")

    # Listas para guardar los "mini-batches" de cada producto
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    # --- Procesar Set de Entrenamiento (Train) ---
    print("Procesando set de Entrenamiento...")
    unique_products_train = train_df['product_id_encoded'].unique()
    for product_id in unique_products_train:
        X_prod, y_prod = create_sequences(train_df, product_id, N_STEPS, FEATURE_COLUMNS, TARGET_COLUMN)

        if X_prod is not None:
            X_train_list.append(X_prod)
            y_train_list.append(y_prod)

    # --- Procesar Set de Validación (Val) ---
    print("Procesando set de Validación...")
    unique_products_val = val_df['product_id_encoded'].unique()
    for product_id in unique_products_val:
        X_prod, y_prod = create_sequences(val_df, product_id, N_STEPS, FEATURE_COLUMNS, TARGET_COLUMN)

        if X_prod is not None:
            X_val_list.append(X_prod)
            y_val_list.append(y_prod)

    # --- Combinar todo en grandes arrays de NumPy ---
    if len(X_train_list) > 0:
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)

        print("\n--- ¡Procesamiento Completado! ---")
        print(f"Forma de X_train (Muestras, Pasos, Features): {X_train.shape}")
        print(f"Forma de y_train (Muestras,): {y_train.shape}")
        print(f"Forma de X_val (Muestras, Pasos, Features): {X_val.shape}")
        print(f"Forma de y_val (Muestras,): {y_val.shape}")

        # --- Guardar los archivos para la Fase 2 ---
        np.save('X_train.npy', X_train)
        np.save('y_train.npy', y_train)
        np.save('X_val.npy', X_val)
        np.save('y_val.npy', y_val)

        print("\nArchivos .npy guardados exitosamente.")
        print("¡Fase 1 Completada!")

    else:
        print("\n¡ERROR! No se generaron secuencias. Revisa N_STEPS o los datos.")
    return


if __name__ == "__main__":
    app.run()
