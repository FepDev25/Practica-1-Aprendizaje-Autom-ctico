import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell
def _():
    import plotly.express as px
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import zscore
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    import joblib
    import os
    import math
    import marimo as mo

    # ***** 

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from tensorflow.keras.models import load_model
    return (
        Adam,
        Dense,
        Dropout,
        EarlyStopping,
        GRU,
        LabelEncoder,
        MinMaxScaler,
        ModelCheckpoint,
        Sequential,
        joblib,
        load_model,
        math,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pd,
        plt,
        tf,
        zscore,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Fase 01""")
    return


@app.cell
def _(pd):
    df = pd.read_csv('./data/dataset_inventario_secuencial_completo.csv')
    return (df,)


@app.cell
def _(df, pd):
    df["is_active"] = True

    # Transformar datos a tipo fecha
    df.created_at = pd.to_datetime(df.created_at)
    df.last_order_date = pd.to_datetime(df.last_order_date)
    df.last_updated_at = pd.to_datetime(df.last_updated_at)
    df.last_stock_count_date = pd.to_datetime(df.last_stock_count_date)
    df.expiration_date = pd.to_datetime(df.expiration_date)
    return


@app.cell
def _(df, np, zscore):
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])

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
        else:
            print(f"\nColumna: '{col}' -> Sin outliers (según IQR).")


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
def _(df_proc):
    print("\nDataFrame Procesado")
    print(df_proc.head())

    columnas_modelo = df_proc.select_dtypes(exclude=['object', 'datetime64[ns]']).columns
    print(df_proc[columnas_modelo].info())
    return


@app.cell
def _(df_feat, df_proc):
    df_proc['created_at'] = df_feat['created_at']
    print("DataFrame 'df_proc' listo para la creación de secuencias.")
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
def _(
    FEATURE_COLUMNS,
    N_STEPS,
    TARGET_COLUMN,
    create_sequences,
    np,
    train_df,
    val_df,
):
    print('\nProcesando secuencias para Train y Validation')
    X_train_list, y_train_list = ([], [])
    X_val_list, y_val_list = ([], [])
    print('procesar set entrenamiento')
    unique_products_train = train_df['product_id_encoded'].unique()
    for _product_id in unique_products_train:
        X_prod, y_prod = create_sequences(train_df, _product_id, N_STEPS, FEATURE_COLUMNS, TARGET_COLUMN)
        if X_prod is not None:
            X_train_list.append(X_prod)
            y_train_list.append(y_prod)
    print('procesar set validacion...')
    unique_products_val = val_df['product_id_encoded'].unique()
    for _product_id in unique_products_val:
        X_prod, y_prod = create_sequences(val_df, _product_id, N_STEPS, FEATURE_COLUMNS, TARGET_COLUMN)
        if X_prod is not None:
            X_val_list.append(X_prod)
            y_val_list.append(y_prod)
    if len(X_train_list) > 0:
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)
        print(f'Forma de X_train (Muestras, Pasos, Features): {X_train.shape}')
        print(f'Forma de y_train (Muestras,): {y_train.shape}')
        print(f'Forma de X_val (Muestras, Pasos, Features): {X_val.shape}')
        print(f'Forma de y_val (Muestras,): {y_val.shape}')
        np.save('X_train.npy', X_train)
        np.save('y_train.npy', y_train)
        np.save('X_val.npy', X_val)
        np.save('y_val.npy', y_val)
    else:
        print('\nNo hay secuencias.')
    return


@app.cell
def _(df_proc, pd):
    df_proc_path = 'df_processed_features.csv'
    df_proc['created_at'] = pd.to_datetime(df_proc['created_at'])
    df_proc.to_csv(df_proc_path, index=False)
    print(f"DataFrame procesado guardado en '{df_proc_path}'")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Fase 02""")
    return


@app.cell
def _(np, tf):
    np.random.seed(42)
    tf.random.set_seed(42)
    PATH_X_TRAIN = 'X_train.npy'
    PATH_Y_TRAIN = 'y_train.npy'
    PATH_X_VAL = 'X_val.npy'
    PATH_Y_VAL = 'y_val.npy'
    X_train_1 = np.load(PATH_X_TRAIN, allow_pickle=True)
    y_train_1 = np.load(PATH_Y_TRAIN, allow_pickle=True)
    X_val_1 = np.load(PATH_X_VAL, allow_pickle=True)
    y_val_1 = np.load(PATH_Y_VAL, allow_pickle=True)
    print("\nConvirtiendo arrays a dtype 'float32'...")
    X_train_1 = X_train_1.astype('float32')
    y_train_1 = y_train_1.astype('float32')
    # Convertir arrays de 'object' a 'float32' para TensorFlow
    X_val_1 = X_val_1.astype('float32')
    y_val_1 = y_val_1.astype('float32')
    print('\n--- 2. Verificación de Formas (Shapes) ---')
    print(f'Forma de X_train (Muestras, Pasos, Features): {X_train_1.shape}')
    print(f'Forma de y_train (Muestras,): {y_train_1.shape}')
    print(f'Forma de X_val (Muestras, Pasos, Features): {X_val_1.shape}')
    print(f'Forma de y_val (Muestras,): {y_val_1.shape}')
    INPUT_SHAPE = (X_train_1.shape[1], X_train_1.shape[2])
    print('Datos cargados')
    return INPUT_SHAPE, X_train_1, X_val_1, y_train_1, y_val_1


@app.cell
def _(Dense, Dropout, GRU, INPUT_SHAPE, Sequential):
    # Arquitectura del Modelo (GRU)

    model_gru = Sequential(name="Modelo_GRU_Prediccion_Stock")

    # Capa 1: Capa GRU
    # units=64: El número de "neuronas" en la capa.
    # input_shape: (7, 30) -> (N_STEPS, N_FEATURES)
    model_gru.add(GRU(units=64, input_shape=INPUT_SHAPE, name="Capa_Entrada_GRU"))

    # Capa 2: Dropout (Regularización)
    # Apagamos el 20% de las neuronas aleatoriamente en cada época
    # para evitar que el modelo "memorice" los datos de entrenamiento.
    model_gru.add(Dropout(0.2, name="Capa_Dropout"))

    # Capa 3: Capa de Salida
    # units=1: predicción de 'quantity_available'
    model_gru.add(Dense(units=1, name="Capa_Salida_Prediccion"))

    model_gru.summary()
    model = model_gru
    return (model,)


@app.cell
def _(Adam, EarlyStopping, ModelCheckpoint, model):
    # --- Compilar el modelo 

    # Evaluar con RMSE o MAE como pide la guía.

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error', 
        metrics=['mean_absolute_error']
    )

    print("Compilado")


    # Callbacks

    # Esto es para guardar solo el modelo que tenga el val_loss más bajo.
    checkpoint_path = 'best_model.keras'
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss', 
        save_best_only=True,
        mode='min',
        verbose=1 
    )

    # Detener el entrenamiento si no hay mejora
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=1,
    )

    print("Callbacks")
    return early_stopping, model_checkpoint


@app.cell
def _(
    X_train_1,
    X_val_1,
    early_stopping,
    model,
    model_checkpoint,
    y_train_1,
    y_val_1,
):
    # Entrenar el Modelo
    EPOCHS = 100
    BATCH_SIZE = 64
    history = model.fit(X_train_1, y_train_1, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val_1, y_val_1), callbacks=[model_checkpoint, early_stopping], verbose=1)
    print('Entrenado')  # ¡Clave para monitorear!  # Muestra el progreso en cada época
    return (history,)


@app.cell
def _(history, plt):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mean_absolute_error']
    val_mae = history.history['val_mean_absolute_error']

    epochs_range = range(len(loss)) # El número de épocas que realmente corrió

    plt.figure(figsize=(14, 6))

    # Gráfico de Pérdida (Loss - MSE)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, label='Pérdida de Entrenamiento (MSE)')
    plt.plot(epochs_range, val_loss, label='Pérdida de Validación (MSE)')
    plt.legend(loc='upper right')
    plt.title('Pérdida (Loss) de Entrenamiento y Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (MSE)')

    # Gráfico de Métrica (MAE)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, mae, label='Error Absoluto Medio (MAE) de Entrenamiento')
    plt.plot(epochs_range, val_mae, label='Error Absoluto Medio (MAE) de Validación')
    plt.legend(loc='upper right')
    plt.title('Métrica (MAE) de Entrenamiento y Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Error (MAE)')

    plt.show()
    return


@app.cell
def _(
    X_val_1,
    load_model,
    math,
    mean_absolute_error,
    mean_squared_error,
    y_val_1,
):
    best_model = load_model('best_model.keras')
    y_pred_scaled = best_model.predict(X_val_1)
    # Predicciones
    rmse_scaled = math.sqrt(mean_squared_error(y_val_1, y_pred_scaled))
    mae_scaled = mean_absolute_error(y_val_1, y_pred_scaled)
    # Métricas
    print(f'Métricas del Modelo (en datos escalados [0, 1]):')
    print(f'RMSE: {rmse_scaled:.4f}')
    print(f'MAE:  {mae_scaled:.4f}')
    return mae_scaled, rmse_scaled, y_pred_scaled


@app.cell
def _(
    joblib,
    mae_scaled,
    math,
    mean_absolute_error,
    mean_squared_error,
    np,
    rmse_scaled,
    y_pred_scaled,
    y_val_1,
):
    # Des escalar
    scaler_1 = joblib.load('min_max_scaler.joblib')
    TARGET_COLUMN_INDEX = 2
    num_numeric_features = 18
    dummy_y_val = np.zeros((len(y_val_1), num_numeric_features))
    dummy_y_val[:, TARGET_COLUMN_INDEX] = y_val_1.ravel()
    y_val_real = scaler_1.inverse_transform(dummy_y_val)[:, TARGET_COLUMN_INDEX]
    dummy_y_pred = np.zeros((len(y_pred_scaled), num_numeric_features))
    dummy_y_pred[:, TARGET_COLUMN_INDEX] = y_pred_scaled.ravel()
    y_pred_real = scaler_1.inverse_transform(dummy_y_pred)[:, TARGET_COLUMN_INDEX]  # .ravel() lo aplana
    rmse_real = math.sqrt(mean_squared_error(y_val_real, y_pred_real))
    mae_real = mean_absolute_error(y_val_real, y_pred_real)
    min_stock = scaler_1.data_min_[TARGET_COLUMN_INDEX]
    max_stock = scaler_1.data_max_[TARGET_COLUMN_INDEX]
    rango_stock = max_stock - min_stock
    error_relativo = mae_real / rango_stock * 100
    # Calcular métricas finales en unidades reales de stock
    print('MÉTRICAS FINALES DEL MODELO')
    print(f'\nContexto del Dataset:')
    print(f'   • Rango de stock: {min_stock:.0f} - {max_stock:.0f} unidades')
    # Obtener el rango de valores reales para contexto
    print(f'   • Rango total: {rango_stock:.0f} unidades')
    print(f'\nMétricas en Escala Normalizada [0,1]:')
    print(f'   • RMSE: {rmse_scaled:.4f}')
    print(f'   • MAE:  {mae_scaled:.4f}')
    # Error relativo porcentual
    print(f'\nMétricas en Unidades Reales:')
    print(f'   • RMSE: {rmse_real:.2f} unidades')
    print(f'   • MAE:  {mae_real:.2f} unidades')
    print(f'   • Error Relativo: {error_relativo:.2f}%')
    print(f'\nInterpretación:')
    print(f'   En promedio, las predicciones se desvían ±{mae_real:.2f} unidades,')
    print(f'   lo que representa un error del {error_relativo:.2f}% respecto al rango total.')
    print(f'   Esto es equivalente a un MAE normalizado de {mae_scaled:.4f}.')
    return error_relativo, mae_real, rmse_real, y_pred_real, y_val_real


@app.cell
def _(plt, y_pred_real, y_val_real):
    plt.figure(figsize=(14, 6))

    # Scatter plot de predicciones vs reales
    plt.subplot(1, 2, 1)
    plt.scatter(y_val_real, y_pred_real, alpha=0.3, s=10, edgecolors='none', color='steelblue')

    # Línea diagonal perfecta
    min_val = min(y_val_real.min(), y_pred_real.min())
    max_val = max(y_val_real.max(), y_pred_real.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')

    plt.xlabel('Valor Real (unidades)', fontsize=11)
    plt.ylabel('Valor Predicho (unidades)', fontsize=11)
    plt.title('Predicciones vs Valores Reales', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Histograma comparativo
    plt.subplot(1, 2, 2)
    plt.hist(y_val_real, bins=50, alpha=0.5, label='Valores Reales', color='blue', edgecolor='black')
    plt.hist(y_pred_real, bins=50, alpha=0.5, label='Predicciones', color='orange', edgecolor='black')
    plt.xlabel('Stock (unidades)', fontsize=11)
    plt.ylabel('Frecuencia', fontsize=11)
    plt.title('Distribución: Real vs Predicho', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(np, plt, y_pred_real, y_val_real):
    errors = y_pred_real - y_val_real
    abs_errors = np.abs(errors)
    percent_errors = (abs_errors / (y_val_real + 1)) * 100

    plt.figure(figsize=(15, 5))

    # Subplot 1: Histograma de errores (con signo)
    plt.subplot(1, 3, 1)
    plt.hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error = 0')
    plt.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, 
                label=f'Media = {np.mean(errors):.2f}')
    plt.xlabel('Error (Predicción - Real)', fontsize=11)
    plt.ylabel('Frecuencia', fontsize=11)
    plt.title('Distribución de Errores', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # Subplot 2: Histograma de errores absolutos
    plt.subplot(1, 3, 2)
    plt.hist(abs_errors, bins=50, color='coral', edgecolor='black', alpha=0.7)
    plt.axvline(x=np.mean(abs_errors), color='darkred', linestyle='--', linewidth=2, 
                label=f'MAE = {np.mean(abs_errors):.2f}')
    plt.xlabel('Error Absoluto (unidades)', fontsize=11)
    plt.ylabel('Frecuencia', fontsize=11)
    plt.title('Distribución de Errores Absolutos', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # Subplot 3: Boxplot de errores absolutos
    plt.subplot(1, 3, 3)
    box = plt.boxplot(abs_errors, vert=True, patch_artist=True, 
                      boxprops=dict(facecolor='lightgreen', alpha=0.7))
    plt.ylabel('Error Absoluto (unidades)', fontsize=11)
    plt.title('Boxplot de Errores Absolutos', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # Añadir estadísticas
    q1, median, q3 = np.percentile(abs_errors, [25, 50, 75])
    plt.text(1.15, median, f'Mediana: {median:.1f}', fontsize=9, va='center')
    plt.text(1.15, q1, f'Q1: {q1:.1f}', fontsize=9, va='center')
    plt.text(1.15, q3, f'Q3: {q3:.1f}', fontsize=9, va='center')

    plt.tight_layout()
    plt.show()
    return abs_errors, percent_errors


@app.cell
def _(abs_errors, np, plt, y_val_real):
    percentiles = [0, 33, 66, 100]
    bins = np.percentile(y_val_real, percentiles)

    stock_ranges = ['Bajo (0-33%)', 'Medio (33-66%)', 'Alto (66-100%)']
    range_indices = [
        (y_val_real >= bins[0]) & (y_val_real < bins[1]),
        (y_val_real >= bins[1]) & (y_val_real < bins[2]),
        (y_val_real >= bins[2])
    ]

    # Calcular métricas por rango
    print("ANÁLISIS DE RENDIMIENTO POR RANGO DE STOCK")

    range_stats = []
    for i, (range_name, indices) in enumerate(zip(stock_ranges, range_indices)):
        range_errors = abs_errors[indices]
        range_vals = y_val_real[indices]

        mae_range = np.mean(range_errors)
        count = indices.sum()
        pct = (count / len(y_val_real)) * 100

        range_stats.append({
            'name': range_name,
            'count': count,
            'percentage': pct,
            'mae': mae_range,
            'median_error': np.median(range_errors),
            'std_error': range_errors.std(),
            'min_stock': range_vals.min(),
            'max_stock': range_vals.max()
        })

        print(f"\n{range_name}:")
        print(f"   • Rango: [{range_vals.min():.0f} - {range_vals.max():.0f}] unidades")
        print(f"   • Cantidad de muestras: {count:,} ({pct:.1f}%)")
        print(f"   • MAE: {mae_range:.2f} unidades")
        print(f"   • Error mediano: {np.median(range_errors):.2f} unidades")
        print(f"   • Desviación estándar: {range_errors.std():.2f} unidades")


    # Visualización por rangos
    plt.figure(figsize=(15, 5))

    # Gráfico 1: MAE por rango
    plt.subplot(1, 3, 1)
    maes = [stat['mae'] for stat in range_stats]
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
    bars = plt.bar(stock_ranges, maes, color=colors, edgecolor='black', alpha=0.7)
    plt.ylabel('MAE (unidades)', fontsize=11)
    plt.title('Error Absoluto Medio por Rango', fontsize=12, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    for bar, mae_iter in zip(bars, maes):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae_iter:.1f}', ha='center', va='bottom', fontweight='bold')

    # Gráfico 2: Distribución de muestras
    plt.subplot(1, 3, 2)
    counts = [stat['count'] for stat in range_stats]
    plt.pie(counts, labels=stock_ranges, autopct='%1.1f%%', colors=colors, startangle=90,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    plt.title('Distribución de Muestras por Rango', fontsize=12, fontweight='bold')

    # Gráfico 3: Boxplot comparativo
    plt.subplot(1, 3, 3)
    error_data = [abs_errors[indices] for indices in range_indices]
    box_i = plt.boxplot(error_data, labels=['Bajo', 'Medio', 'Alto'], 
                      patch_artist=True, notch=True)

    for patch, color in zip(box_i['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.ylabel('Error Absoluto (unidades)', fontsize=11)
    plt.xlabel('Rango de Stock', fontsize=11)
    plt.title('Distribución de Errores por Rango', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()
    return (range_stats,)


@app.cell
def _(error_relativo, mae_real, percent_errors, range_stats, rmse_real):
    print("ANÁLISIS DE RENDIMIENTO DEL MODELO")

    print("\nMÉTRICAS GLOBALES:")
    print(f"   • MAE: {mae_real:.2f} unidades ({error_relativo:.2f}% del rango)")
    print(f"   • RMSE: {rmse_real:.2f} unidades")
    print(f"   • Ratio RMSE/MAE: {rmse_real/mae_real:.2f}")

    print("\nDISTRIBUCIÓN DE CALIDAD:")
    excellent = (percent_errors < 5).sum()
    good = ((percent_errors >= 5) & (percent_errors < 10)).sum()
    fair = ((percent_errors >= 10) & (percent_errors < 20)).sum()
    poor = (percent_errors >= 20).sum()
    total = len(percent_errors)

    print(f"   • Excelente (<5% error):  {excellent:,} predicciones ({excellent/total*100:.1f}%)")
    print(f"   • Bueno (5-10% error):    {good:,} predicciones ({good/total*100:.1f}%)")
    print(f"   • Aceptable (10-20%):     {fair:,} predicciones ({fair/total*100:.1f}%)")
    print(f"   • Necesita mejora (>20%): {poor:,} predicciones ({poor/total*100:.1f}%)")

    print("\nRENDIMIENTO POR RANGO DE STOCK:")
    for stat in range_stats:
        print(f"   • {stat['name']:15} → MAE: {stat['mae']:6.2f} unidades ({stat['percentage']:5.1f}% de datos)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Fase 03""")
    return


@app.cell
def _(joblib, load_model, pd):
    N_STEPS_1 = 7
    TARGET_COLUMN_INDEX_1 = 2
    NUM_NUMERIC_FEATURES = 18
    try:
    # modelo
        model_1 = load_model('best_model.keras')
        print("Modelo 'best_model.keras' cahrgado.")
    except Exception as e:
        print(f"Error al cargar 'best_model.keras': {e}")
    try:
        scaler_2 = joblib.load('min_max_scaler.joblib')
    # escalador
        print("Escalador 'min_max_scaler.joblib' cargado.")
    except Exception as e:
        print(f"Error al cargar 'min_max_scaler.joblib': {e}")
    try:
        le_product_id_1 = joblib.load('le_product_id.joblib')
        print("Codificador 'le_product_id.joblib' cargado.")
    # codificador de productos
    except Exception as e:
        print(f"Error al cargar 'le_product_id.joblib': {e}")
    try:
        df_features = pd.read_csv('df_processed_features.csv')
        df_features['created_at'] = pd.to_datetime(df_features['created_at'])
        print(f'Base de datos de features cargada ({len(df_features)} registros).')
    # features
    except Exception as e:
        print(f"Error al cargar 'df_processed_features.csv': {e}")
    # Lista de columnas
    FEATURE_COLUMNS_1 = ['quantity_on_hand', 'quantity_reserved', 'quantity_available', 'minimum_stock_level', 'reorder_point', 'optimal_stock_level', 'reorder_quantity', 'average_daily_usage', 'unit_cost', 'total_value', 'is_active', 'dia_del_mes', 'dia_de_la_semana', 'mes', 'trimestre', 'es_fin_de_semana', 'dias_para_vencimiento', 'antiguedad_producto_dias', 'ratio_uso_stock', 'product_id_encoded', 'supplier_id_encoded', 'warehouse_location_Almacén Este', 'warehouse_location_Almacén Norte', 'warehouse_location_Almacén Oeste', 'warehouse_location_Almacén Sur', 'warehouse_location_Centro Distribución 1', 'warehouse_location_Centro Distribución 2', 'stock_status_1', 'stock_status_2', 'stock_status_3']
    return (
        FEATURE_COLUMNS_1,
        NUM_NUMERIC_FEATURES,
        N_STEPS_1,
        TARGET_COLUMN_INDEX_1,
        df_features,
        le_product_id_1,
        model_1,
        scaler_2,
    )


@app.cell
def _(
    FEATURE_COLUMNS_1,
    NUM_NUMERIC_FEATURES,
    N_STEPS_1,
    TARGET_COLUMN_INDEX_1,
    df_features,
    le_product_id_1,
    model_1,
    np,
    pd,
    scaler_2,
):
    def predict_demand(product_id_str, target_date_str):
        try:
            product_id_encoded = le_product_id_1.transform([product_id_str])[0]  # Validar y Codificar ID de Producto ---
        except ValueError:
            return f"Error: El ID de producto '{product_id_str}' no fue visto durante el entrenamiento."
        try:
            target_date = pd.to_datetime(target_date_str)
        except ValueError:
            return f"Error: Formato de fecha incorrecto '{target_date_str}'."  # validar fecha
        product_data = df_features[df_features['product_id_encoded'] == product_id_encoded].sort_values(by='created_at')
        historical_data = product_data[product_data['created_at'] < target_date]
        if len(historical_data) < N_STEPS_1:
            return f'Error: No hay suficiente historia ({len(historical_data)} días) para predecir. Se necesitan {N_STEPS_1} días.'
        sequence_df = historical_data.tail(N_STEPS_1)
        input_features_df = sequence_df[FEATURE_COLUMNS_1]  # secuencia histórica
        input_features_scaled = input_features_df.astype(np.float32).values
        input_sequence = np.expand_dims(input_features_scaled, axis=0)
        pred_scaled = model_1.predict(input_sequence)[0][0]
        dummy_pred = np.zeros((1, NUM_NUMERIC_FEATURES))
        dummy_pred[:, TARGET_COLUMN_INDEX_1] = pred_scaled
        pred_real = scaler_2.inverse_transform(dummy_pred)[0][TARGET_COLUMN_INDEX_1]
        print('predicción realizada')  # validar historia
        return max(0, pred_real)  # prediccion  # des-escalar
    return (predict_demand,)


@app.cell
def _(np, pd, predict_demand):
    df_original = pd.read_csv('./data/dataset_inventario_secuencial_completo.csv')
    unique_products = df_original['product_id'].unique()
    NUM_PRODUCTS = 20
    TARGET_DATE = '2025-10-31'
    np.random.seed(42)
    sample_products = np.random.choice(unique_products, size=min(NUM_PRODUCTS, len(unique_products)), replace=False)
    print(f'PREDICCIONES PARA {len(sample_products)} PRODUCTOS ÚNICOS')
    print(f'Fecha objetivo: {TARGET_DATE}')
    results = []
    success_count = 0
    for idx, _product_id in enumerate(sample_products, 1):
        print(f'\n[{idx}/{len(sample_products)}] {_product_id}')
        prediction = predict_demand(_product_id, TARGET_DATE)
        if isinstance(prediction, (int, float)):
            success_count = success_count + 1
            print(f'Stock predicho: {prediction:.2f} unidades')
            results.append({'product_id': _product_id, 'prediction': prediction, 'status': 'success'})
        else:
            print(f'{prediction}')
            results.append({'product_id': _product_id, 'prediction': None, 'status': 'failed'})
    results_df = pd.DataFrame(results)
    if success_count > 0:
        successful = results_df[results_df['status'] == 'success']['prediction']
        print(f'\nEstadísticas:')
        print(f'   Media: {successful.mean():.2f} unidades')
        print(f'   Mediana: {successful.median():.2f} unidades')
        print(f'   Mínimo: {successful.min():.2f} unidades')
        print(f'   Máximo: {successful.max():.2f} unidades')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Sistema de Evaluación Comparativa con Múltiples Datasets

    Este sistema permite probar el modelo completo con diferentes datasets de forma automatizada, ejecutando todo el pipeline desde la preparación de datos hasta las predicciones y visualizaciones.
    """
    )
    return


@app.cell
def _(np, tf):
    import warnings
    warnings.filterwarnings('ignore')
    from pathlib import Path
    import time
    from datetime import datetime
    import json

    # Configurar para reproducibilidad
    np.random.seed(42)
    tf.random.set_seed(42)

    print("✓ Librerías importadas para evaluación comparativa")
    print(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return Path, datetime, json, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. Clase Principal de Evaluación

    Esta clase encapsula todo el pipeline de procesamiento, entrenamiento y evaluación.
    """
    )
    return


@app.cell
def _(
    Adam,
    Dense,
    Dropout,
    EarlyStopping,
    GRU,
    LabelEncoder,
    MinMaxScaler,
    ModelCheckpoint,
    Path,
    Sequential,
    load_model,
    math,
    mean_absolute_error,
    mean_squared_error,
    np,
    pd,
    plt,
    time,
):
    class ModeloStockEvaluador:
        """
        Clase para evaluar el modelo de predicción de stock con diferentes datasets.
        Ejecuta todo el pipeline: preprocesamiento, entrenamiento y evaluación.
        """

        def __init__(self, dataset_path, nombre_experimento, n_steps=7, split_percentage=0.8):
            """
            Inicializa el evaluador.
        
            Args:
                dataset_path: Ruta al archivo CSV del dataset
                nombre_experimento: Nombre descriptivorformance: subset_500_productos
      • MAE: 192.17 unidades
      • RMSE: 262.17 unidades
      • Error relativo: 2.99 del experimento
                n_steps: Número de pasos temporales (default: 7)
                split_percentage: Porcentaje para división train/val (default: 0.8)
            """
            self.dataset_path = dataset_path
            self.nombre_experimento = nombre_experimento
            self.n_steps = n_steps
            self.split_percentage = split_percentage
            self.TARGET_COLUMN = 'quantity_available'
            self.TARGET_COLUMN_INDEX = 2  # Constantes
            self.resultados = {}
            self.tiempos = {}
            self.df_original = None
            self.df_proc = None  # Resultados
            self.model = None
            self.scaler = None
            print(f"\n{'=' * 70}")
            print(f'EXPERIMENTO: {self.nombre_experimento}')
            print(f'Dataset: {Path(dataset_path).name}')
            print(f"{'=' * 70}\n")

        def cargar_datos(self):
            """Paso 1: Cargar dataset"""
            inicio = time.time()
            print('Paso 1: Cargando datos...')
            self.df_original = pd.read_csv(self.dataset_path)
            self.resultados['total_filas'] = len(self.df_original)
            self.resultados['productos_unicos'] = self.df_original['product_id'].nunique()
            print(f"   ✓ Filas cargadas: {self.resultados['total_filas']:,}")
            print(f"   ✓ Productos únicos: {self.resultados['productos_unicos']:,}")
            self.tiempos['carga'] = time.time() - inicio

        def preprocesar_datos(self):
            """Paso 2: Preprocesamiento completo"""
            inicio = time.time()
            print('\nPaso 2: Preprocesamiento de datos...')
            df = self.df_original.copy()
            df['is_active'] = True
            df.created_at = pd.to_datetime(df.created_at)
            df.last_order_date = pd.to_datetime(df.last_order_date)
            df.last_updated_at = pd.to_datetime(df.last_updated_at)
            df.last_stock_count_date = pd.to_datetime(df.last_stock_count_date)
            df.expiration_date = pd.to_datetime(df.expiration_date)
            base_date = df['created_at']
            df['dia_del_mes'] = base_date.dt.day
            df['dia_de_la_semana'] = base_date.dt.dayofweek
            df['mes'] = base_date.dt.month
            df['trimestre'] = base_date.dt.quarter
            df['es_fin_de_semana'] = df['dia_de_la_semana'].isin([5, 6]).astype(int)
            df['dias_para_vencimiento'] = (df['expiration_date'] - base_date).dt.days  # Convertir a datetime
            df['dias_para_vencimiento'] = df['dias_para_vencimiento'].fillna(0).apply(lambda x: max(0, x))
            df['antiguedad_producto_dias'] = (base_date - df['last_stock_count_date']).dt.days
            df['antiguedad_producto_dias'] = df['antiguedad_producto_dias'].fillna(0).apply(lambda x: max(0, x))
            df['ratio_uso_stock'] = df['average_daily_usage'] / (df['quantity_available'] + 1)
            le_product_id = LabelEncoder()
            df['product_id_encoded'] = le_product_id.fit_transform(df['product_id'])
            le_supplier_id = LabelEncoder()  # Feature Engineering
            df['supplier_id_encoded'] = le_supplier_id.fit_transform(df['supplier_id'])
            categorias_onehot = ['warehouse_location', 'stock_status']
            df = pd.get_dummies(df, columns=categorias_onehot, drop_first=True)
            columnas_numericas = ['quantity_on_hand', 'quantity_reserved', 'quantity_available', 'minimum_stock_level', 'reorder_point', 'optimal_stock_level', 'reorder_quantity', 'average_daily_usage', 'unit_cost', 'total_value', 'dia_del_mes', 'dia_de_la_semana', 'mes', 'trimestre', 'es_fin_de_semana', 'dias_para_vencimiento', 'antiguedad_producto_dias', 'ratio_uso_stock']
            self.scaler = MinMaxScaler()
            df[columnas_numericas] = self.scaler.fit_transform(df[columnas_numericas])
            df['created_at'] = self.df_original['created_at']
            self.df_proc = df
            print(f'   ✓ Features creadas')
            print(f'   ✓ Variables codificadas')
            print(f'   ✓ Normalización aplicada')
            self.tiempos['preprocesamiento'] = time.time() - inicio

        def crear_secuencias(self):
            """Paso 3: Crear secuencias temporales"""
            inicio = time.time()  # Codificación
            print('\nPaso 3: Creando secuencias temporales...')
            FEATURE_COLUMNS = ['quantity_on_hand', 'quantity_reserved', 'quantity_available', 'minimum_stock_level', 'reorder_point', 'optimal_stock_level', 'reorder_quantity', 'average_daily_usage', 'unit_cost', 'total_value', 'is_active', 'dia_del_mes', 'dia_de_la_semana', 'mes', 'trimestre', 'es_fin_de_semana', 'dias_para_vencimiento', 'antiguedad_producto_dias', 'ratio_uso_stock', 'product_id_encoded', 'supplier_id_encoded', 'warehouse_location_Almacén Este', 'warehouse_location_Almacén Norte', 'warehouse_location_Almacén Oeste', 'warehouse_location_Almacén Sur', 'warehouse_location_Centro Distribución 1', 'warehouse_location_Centro Distribución 2', 'stock_status_1', 'stock_status_2', 'stock_status_3']
            df_sorted = self.df_proc.sort_values(by='created_at')
            split_index = int(len(df_sorted) * self.split_percentage)
            train_df = df_sorted.iloc[:split_index]
            val_df = df_sorted.iloc[split_index:]
      # One-Hot Encoding
            def create_sequences(data_df, product_group, n_steps, feature_cols, target_col):
                product_data = data_df[data_df['product_id_encoded'] == product_group].copy()
                product_data = product_data.sort_values(by='created_at')
                features = product_data[feature_cols].values  # Normalización
                target = product_data[target_col].values
                X, y = ([], [])
                for i in range(n_steps, len(product_data)):
                    X.append(features[i - n_steps:i])
                    y.append(target[i])
                if len(X) > 0:
                    return (np.array(X), np.array(y))
                else:
                    return (None, None)
            X_train_list, y_train_list = ([], [])
            unique_products_train = train_df['product_id_encoded'].unique()
            for _product_id in unique_products_train:  # Guardar fecha original
                X_prod, y_prod = create_sequences(train_df, _product_id, self.n_steps, FEATURE_COLUMNS, self.TARGET_COLUMN)
                if X_prod is not None:
                    X_train_list.append(X_prod)
                    y_train_list.append(y_prod)
            X_val_list, y_val_list = ([], [])
            unique_products_val = val_df['product_id_encoded'].unique()
            for _product_id in unique_products_val:
                X_prod, y_prod = create_sequences(val_df, _product_id, self.n_steps, FEATURE_COLUMNS, self.TARGET_COLUMN)
                if X_prod is not None:
                    X_val_list.append(X_prod)
                    y_val_list.append(y_prod)
            self.X_train = np.concatenate(X_train_list, axis=0).astype('float32')
            self.y_train = np.concatenate(y_train_list, axis=0).astype('float32')
            self.X_val = np.concatenate(X_val_list, axis=0).astype('float32')
            self.y_val = np.concatenate(y_val_list, axis=0).astype('float32')
            print(f'   ✓ Train: {self.X_train.shape[0]:,} secuencias')
            print(f'   ✓ Validation: {self.X_val.shape[0]:,} secuencias')
            print(f'   ✓ Shape: ({self.n_steps} pasos, {self.X_train.shape[2]} features)')
            self.resultados['secuencias_train'] = self.X_train.shape[0]
            self.resultados['secuencias_val'] = self.X_val.shape[0]
            self.tiempos['secuencias'] = time.time() - inicio

        def entrenar_modelo(self, epochs=100, batch_size=64, verbose=0):
            """Paso 4: Entrenar modelo GRU"""
            inicio = time.time()
            print('\nPaso 4: Entrenando modelo...')
            INPUT_SHAPE = (self.X_train.shape[1], self.X_train.shape[2])
            self.model = Sequential(name=f'GRU_{self.nombre_experimento}')
            self.model.add(GRU(units=64, input_shape=INPUT_SHAPE, name='Capa_GRU'))
            self.model.add(Dropout(0.2, name='Dropout'))  # División temporal
            self.model.add(Dense(units=1, name='Salida'))
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error'])
            checkpoint_path = f'temp_model_{self.nombre_experimento}.keras'
            model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=0)
            history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_val, self.y_val), callbacks=[model_checkpoint, early_stopping], verbose=verbose)
            self.model = load_model(checkpoint_path)  # Función para crear secuencias
            self.history = history
            self.resultados['epocas_entrenadas'] = len(history.history['loss'])
            print(f'   ✓ Entrenamiento completado')
            print(f"   ✓ Épocas ejecutadas: {self.resultados['epocas_entrenadas']}")
            self.tiempos['entrenamiento'] = time.time() - inicio
            if Path(checkpoint_path).exists():
                Path(checkpoint_path).unlink()

        def evaluar_modelo(self):
            """Paso 5: Evaluar modelo y calcular métricas"""
            inicio = time.time()
            print('\nPaso 5: Evaluando modelo...')
            y_pred_scaled = self.model.predict(self.X_val, verbose=0)
            rmse_scaled = math.sqrt(mean_squared_error(self.y_val, y_pred_scaled))
            mae_scaled = mean_absolute_error(self.y_val, y_pred_scaled)
            num_numeric_features = 18
            dummy_y_val = np.zeros((len(self.y_val), num_numeric_features))
            dummy_y_val[:, self.TARGET_COLUMN_INDEX] = self.y_val.ravel()  # Procesar train
            y_val_real = self.scaler.inverse_transform(dummy_y_val)[:, self.TARGET_COLUMN_INDEX]
            dummy_y_pred = np.zeros((len(y_pred_scaled), num_numeric_features))
            dummy_y_pred[:, self.TARGET_COLUMN_INDEX] = y_pred_scaled.ravel()
            y_pred_real = self.scaler.inverse_transform(dummy_y_pred)[:, self.TARGET_COLUMN_INDEX]
            rmse_real = math.sqrt(mean_squared_error(y_val_real, y_pred_real))
            mae_real = mean_absolute_error(y_val_real, y_pred_real)
            min_stock = self.scaler.data_min_[self.TARGET_COLUMN_INDEX]
            max_stock = self.scaler.data_max_[self.TARGET_COLUMN_INDEX]
            rango_stock = max_stock - min_stock
            error_relativo = mae_real / rango_stock * 100
            self.resultados.update({'rmse_scaled': rmse_scaled, 'mae_scaled': mae_scaled, 'rmse_real': rmse_real, 'mae_real': mae_real, 'min_stock': min_stock, 'max_stock': max_stock, 'rango_stock': rango_stock, 'error_relativo_pct': error_relativo})  # Procesar validation
            self.y_val_real = y_val_real
            self.y_pred_real = y_pred_real
            print(f'   ✓ RMSE: {rmse_real:.2f} unidades')
            print(f'   ✓ MAE: {mae_real:.2f} unidades')
            print(f'   ✓ Error relativo: {error_relativo:.2f}%')
            self.tiempos['evaluacion'] = time.time() - inicio

        def ejecutar_pipeline_completo(self, epochs=100, batch_size=64, verbose=0):
            """Ejecuta todo el pipeline de forma secuencial"""
            tiempo_total_inicio = time.time()
            try:  # Concatenar
                self.cargar_datos()
                self.preprocesar_datos()
                self.crear_secuencias()
                self.entrenar_modelo(epochs=epochs, batch_size=batch_size, verbose=verbose)
                self.evaluar_modelo()
                self.tiempos['total'] = time.time() - tiempo_total_inicio
                print(f"\nPipeline completado en {self.tiempos['total']:.2f}s")
                return True
            except Exception as e:
                print(f'\nError en el pipeline: {str(e)}')
                import traceback
                traceback.print_exc()
                return False

        def generar_visualizaciones(self, mostrar=True):
            """Genera todas las visualizaciones del modelo"""
            fig = plt.figure(figsize=(18, 12))
            ax1 = plt.subplot(2, 3, 1)
            epochs_range = range(len(self.history.history['loss']))
            ax1.plot(epochs_range, self.history.history['loss'], label='Train Loss', linewidth=2)
            ax1.plot(epochs_range, self.history.history['val_loss'], label='Val Loss', linewidth=2)
            ax1.set_xlabel('Épocas')  # Arquitectura GRU
            ax1.set_ylabel('Loss (MSE)')
            ax1.set_title('Curvas de Pérdida')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax2 = plt.subplot(2, 3, 2)
            ax2.plot(epochs_range, self.history.history['mean_absolute_error'], label='Train MAE', linewidth=2)  # Compilar
            ax2.plot(epochs_range, self.history.history['val_mean_absolute_error'], label='Val MAE', linewidth=2)
            ax2.set_xlabel('Épocas')
            ax2.set_ylabel('MAE')
            ax2.set_title('Curvas de Error Absoluto Medio')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax3 = plt.subplot(2, 3, 3)  # Callbacks
            ax3.scatter(self.y_val_real, self.y_pred_real, alpha=0.3, s=10, edgecolors='none', color='steelblue')
            min_val = min(self.y_val_real.min(), self.y_pred_real.min())
            max_val = max(self.y_val_real.max(), self.y_pred_real.max())
            ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
            ax3.set_xlabel('Valores Reales')
            ax3.set_ylabel('Predicciones')
            ax3.set_title('Predicciones vs Valores Reales')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax4 = plt.subplot(2, 3, 4)
            errores = self.y_pred_real - self.y_val_real
            ax4.hist(errores, bins=50, edgecolor='black', alpha=0.7, color='coral')
            ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
            ax4.set_xlabel('Error (Predicción - Real)')
            ax4.set_ylabel('Frecuencia')
            ax4.set_title('Distribución de Errores')
            ax4.legend()  # Entrenar
            ax4.grid(True, alpha=0.3)
            ax5 = plt.subplot(2, 3, 5)
            ax5.boxplot(errores, vert=True)
            ax5.set_ylabel('Error (unidades)')
            ax5.set_title('Boxplot de Errores')
            ax5.grid(True, alpha=0.3)
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('off')
            metricas_texto = f"\n        MÉTRICAS DEL MODELO\n        {'=' * 35}\n        \n        Dataset: {Path(self.dataset_path).name}\n        Experimento: {self.nombre_experimento}\n        \n        Datos:\n        • Productos: {self.resultados['productos_unicos']:,}\n        • Filas totales: {self.resultados['total_filas']:,}\n        • Secuencias train: {self.resultados['secuencias_train']:,}\n        • Secuencias val: {self.resultados['secuencias_val']:,}\n        \n        Entrenamiento:\n        • Épocas: {self.resultados['epocas_entrenadas']}\n        • Tiempo: {self.tiempos['entrenamiento']:.1f}s\n        \n        Métricas (Escala Normalizada):\n        • RMSE: {self.resultados['rmse_scaled']:.4f}\n        • MAE: {self.resultados['mae_scaled']:.4f}\n        \n        Métricas (Unidades Reales):\n        • RMSE: {self.resultados['rmse_real']:.2f} unidades\n        • MAE: {self.resultados['mae_real']:.2f} unidades\n        • Error relativo: {self.resultados['error_relativo_pct']:.2f}%\n        \n        Rango de Stock:\n        • Mínimo: {self.resultados['min_stock']:.0f}\n        • Máximo: {self.resultados['max_stock']:.0f}\n        • Rango: {self.resultados['rango_stock']:.0f}\n        "
            ax6.text(0.1, 0.95, metricas_texto, transform=ax6.transAxes, fontsize=10, verticalalignment='top', family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            plt.suptitle(f'Análisis Completo: {self.nombre_experimento}', fontsize=16, fontweight='bold', y=0.995)  # Cargar mejor modelo
            plt.tight_layout()
            if mostrar:
                plt.show()  # Guardar historia
            return fig

        def resumen(self):
            """Imprime un resumen ejecutivo del experimento"""
            print(f"\n{'=' * 70}")
            print(f'RESUMEN EJECUTIVO: {self.nombre_experimento}')
            print(f"{'=' * 70}")
            print(f'\nDATOS:')
            print(f'   • Dataset: {Path(self.dataset_path).name}')  # Limpiar archivo temporal
            print(f"   • Productos únicos: {self.resultados['productos_unicos']:,}")
            print(f"   • Total de filas: {self.resultados['total_filas']:,}")
            print(f"   • Secuencias train: {self.resultados['secuencias_train']:,}")
            print(f"   • Secuencias validación: {self.resultados['secuencias_val']:,}")
            print(f'\nENTRENAMIENTO:')
            print(f"   • Épocas ejecutadas: {self.resultados['epocas_entrenadas']}")
            print(f"   • Tiempo de entrenamiento: {self.tiempos['entrenamiento']:.1f}s")
            print(f'\nMÉTRICAS:')
            print(f"   • RMSE (real): {self.resultados['rmse_real']:.2f} unidades")  # Predicciones
            print(f"   • MAE (real): {self.resultados['mae_real']:.2f} unidades")
            print(f"   • Error relativo: {self.resultados['error_relativo_pct']:.2f}%")
            print(f'\n⏱TIEMPOS:')  # Métricas escaladas
            for proceso, tiempo in self.tiempos.items():
                print(f'   • {proceso.capitalize()}: {tiempo:.2f}s')
            print(f"\n{'=' * 70}\n")
    print('Clase ModeloStockEvaluador creada')  # Des-escalar  # Métricas reales  # Contexto  # Guardar resultados  # Guardar predicciones  # 1. Curvas de entrenamiento  # 2. Curvas de MAE  # 3. Scatter: Predicción vs Real  # 4. Histograma de errores  # 5. Boxplot de errores  # 6. Métricas resumen
    return (ModeloStockEvaluador,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Sistema de Evaluación Multi-Dataset

    Este sistema ejecuta el pipeline completo para todos los datasets disponibles y compara resultados.
    """
    )
    return


@app.cell
def _(ModeloStockEvaluador, Path, pd, plt, tf):
    class EvaluadorMultiDataset:
        """
        Sistema para evaluar el modelo con múltiples datasets y comparar resultados.
        """
    
        def __init__(self, datasets_dir='./data'):
            """
            Inicializa el evaluador multi-dataset.
        
            Args:
                datasets_dir: Directorio donde están los datasets
            """
            self.datasets_dir = Path(datasets_dir)
            self.evaluadores = {}
            self.resultados_comparativos = []
    
        def detectar_datasets(self, patron='subset_*.csv'):
            """
            Detecta todos los datasets en el directorio.
        
            Args:
                patron: Patrón de búsqueda de archivos (default: 'subset_*.csv')
        
            Returns:
                Lista de rutas de datasets encontrados
            """
            datasets = sorted(self.datasets_dir.glob(patron))
        
            if not datasets:
                print(f"No se encontraron datasets con el patrón '{patron}' en {self.datasets_dir}")
                return []
        
            print(f"\nDatasets encontrados: {len(datasets)}")
            for i, ds in enumerate(datasets, 1):
                tamanio_mb = ds.stat().st_size / (1024**2)
                print(f"   {i}. {ds.name} ({tamanio_mb:.2f} MB)")
        
            return datasets
    
        def evaluar_todos(self, datasets=None, epochs=100, batch_size=64, verbose=0):
            """
            Evalúa el modelo con todos los datasets.
        
            Args:
                datasets: Lista de rutas de datasets. Si es None, detecta automáticamente
                epochs: Número de épocas para entrenamiento
                batch_size: Tamaño del batch
                verbose: Nivel de verbosidad del entrenamiento
            """
            if datasets is None:
                datasets = self.detectar_datasets()
        
            if not datasets:
                return
        
            print(f"\n{'='*70}")
            print(f"INICIANDO EVALUACIÓN DE {len(datasets)} DATASETS")
            print(f"{'='*70}\n")
        
            for idx, dataset_path in enumerate(datasets, 1):
                nombre_dataset = dataset_path.stem  # nombre sin extensión
            
                print(f"\n{'#'*70}")
                print(f"# DATASET {idx}/{len(datasets)}: {nombre_dataset}")
                print(f"{'#'*70}")
            
                # Crear evaluador
                evaluador = ModeloStockEvaluador(
                    dataset_path=str(dataset_path),
                    nombre_experimento=nombre_dataset
                )
            
                # Ejecutar pipeline
                exito = evaluador.ejecutar_pipeline_completo(
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=verbose
                )
            
                if exito:
                    # Guardar evaluador
                    self.evaluadores[nombre_dataset] = evaluador
                
                    # Guardar resultados para comparación
                    resultado = {
                        'nombre': nombre_dataset,
                        'dataset_path': str(dataset_path),
                        **evaluador.resultados,
                        **{f'tiempo_{k}': v for k, v in evaluador.tiempos.items()}
                    }
                    self.resultados_comparativos.append(resultado)
                
                    # Mostrar resumen
                    evaluador.resumen()
                else:
                    print(f"\nFalló la evaluación de {nombre_dataset}\n")
            
                # Liberar memoria
                tf.keras.backend.clear_session()
        
            print(f"\n{'='*70}")
            print(f"EVALUACIÓN COMPLETADA: {len(self.evaluadores)}/{len(datasets)} exitosos")
            print(f"{'='*70}\n")
    
        def generar_comparacion(self):
            """Genera visualizaciones comparativas entre todos los datasets"""
            if not self.resultados_comparativos:
                print("No hay resultados para comparar")
                return
        
            df_comp = pd.DataFrame(self.resultados_comparativos)
        
            # Ordenar por número de productos
            df_comp = df_comp.sort_values('productos_unicos')
        
            fig = plt.figure(figsize=(20, 12))
        
            # 1. MAE vs Tamaño del dataset
            ax1 = plt.subplot(2, 3, 1)
            ax1.plot(df_comp['productos_unicos'], df_comp['mae_real'], 
                    marker='o', linewidth=2, markersize=8, color='steelblue')
            ax1.set_xlabel('Número de Productos')
            ax1.set_ylabel('MAE (unidades)')
            ax1.set_title('MAE vs Tamaño del Dataset')
            ax1.grid(True, alpha=0.3)
        
            # 2. RMSE vs Tamaño del dataset
            ax2 = plt.subplot(2, 3, 2)
            ax2.plot(df_comp['productos_unicos'], df_comp['rmse_real'], 
                    marker='s', linewidth=2, markersize=8, color='coral')
            ax2.set_xlabel('Número de Productos')
            ax2.set_ylabel('RMSE (unidades)')
            ax2.set_title('RMSE vs Tamaño del Dataset')
            ax2.grid(True, alpha=0.3)
        
            # 3. Error relativo vs Tamaño
            ax3 = plt.subplot(2, 3, 3)
            ax3.plot(df_comp['productos_unicos'], df_comp['error_relativo_pct'], 
                    marker='D', linewidth=2, markersize=8, color='mediumseagreen')
            ax3.set_xlabel('Número de Productos')
            ax3.set_ylabel('Error Relativo (%)')
            ax3.set_title('Error Relativo vs Tamaño del Dataset')
            ax3.grid(True, alpha=0.3)
        
            # 4. Tiempo de entrenamiento vs Tamaño
            ax4 = plt.subplot(2, 3, 4)
            ax4.plot(df_comp['productos_unicos'], df_comp['tiempo_entrenamiento'], 
                    marker='^', linewidth=2, markersize=8, color='purple')
            ax4.set_xlabel('Número de Productos')
            ax4.set_ylabel('Tiempo (segundos)')
            ax4.set_title('Tiempo de Entrenamiento vs Tamaño')
            ax4.grid(True, alpha=0.3)
        
            # 5. Épocas vs Tamaño
            ax5 = plt.subplot(2, 3, 5)
            ax5.bar(range(len(df_comp)), df_comp['epocas_entrenadas'], 
                   color='orange', alpha=0.7, edgecolor='black')
            ax5.set_xlabel('Dataset')
            ax5.set_ylabel('Épocas')
            ax5.set_title('Épocas Entrenadas por Dataset')
            ax5.set_xticks(range(len(df_comp)))
            ax5.set_xticklabels([f"{p}p" for p in df_comp['productos_unicos']], rotation=45)
            ax5.grid(True, alpha=0.3, axis='y')
        
            # 6. Tabla comparativa
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('tight')
            ax6.axis('off')
        
            # Crear tabla
            tabla_data = []
            for _, row in df_comp.iterrows():
                tabla_data.append([
                    f"{row['productos_unicos']}",
                    f"{row['total_filas']:,}",
                    f"{row['mae_real']:.2f}",
                    f"{row['error_relativo_pct']:.1f}%",
                    f"{row['tiempo_total']:.0f}s"
                ])
        
            tabla = ax6.table(
                cellText=tabla_data,
                colLabels=['Productos', 'Filas', 'MAE', 'Error %', 'Tiempo'],
                cellLoc='center',
                loc='center',
                colWidths=[0.15, 0.15, 0.15, 0.15, 0.15]
            )
            tabla.auto_set_font_size(False)
            tabla.set_fontsize(9)
            tabla.scale(1, 2)
        
            # Estilo de la tabla
            for i in range(len(tabla_data) + 1):
                for j in range(5):
                    cell = tabla[(i, j)]
                    if i == 0:
                        cell.set_facecolor('#4CAF50')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
            plt.suptitle('Comparación de Resultados entre Datasets', 
                        fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            plt.show()
        
            return df_comp
    
        def generar_reporte_completo(self, guardar=True):
            """Genera un reporte completo con todas las visualizaciones"""
            if not self.evaluadores:
                print("No hay evaluadores para generar reporte")
                return
        
            print(f"\n{'='*70}")
            print("GENERANDO REPORTE COMPLETO")
            print(f"{'='*70}\n")
        
            # 1. Comparación general
            print("Generando gráficos comparativos...")
            df_comp = self.generar_comparacion()
        
            # 2. Visualizaciones individuales
            print(f"\nGenerando visualizaciones individuales ({len(self.evaluadores)} datasets)...")
            for nombre, evaluador in self.evaluadores.items():
                print(f"   • {nombre}")
                evaluador.generar_visualizaciones(mostrar=True)
        
            # 3. Tabla resumen
            print("\nTABLA RESUMEN:")
            print("="*100)
            print(df_comp[['nombre', 'productos_unicos', 'total_filas', 'secuencias_train', 
                          'mae_real', 'rmse_real', 'error_relativo_pct', 
                          'tiempo_total']].to_string(index=False))
            print("="*100)
        
            if guardar:
                # Guardar resultados en CSV
                output_path = Path('resultados_comparacion.csv')
                df_comp.to_csv(output_path, index=False)
                print(f"\n✓ Resultados guardados en: {output_path}")
        
            return df_comp
    
        def mejor_dataset(self, metrica='mae_real'):
            """
            Encuentra el mejor dataset según una métrica.
        
            Args:
                metrica: Métrica a optimizar (default: 'mae_real')
        
            Returns:
                Diccionario con información del mejor dataset
            """
            if not self.resultados_comparativos:
                return None
        
            df_comp = pd.DataFrame(self.resultados_comparativos)
            mejor_idx = df_comp[metrica].idxmin()
            mejor = df_comp.iloc[mejor_idx]
        
            print(f"\nMEJOR DATASET según {metrica}:")
            print(f"   • Nombre: {mejor['nombre']}")
            print(f"   • Productos: {mejor['productos_unicos']}")
            print(f"   • {metrica}: {mejor[metrica]:.4f}")
        
            return mejor.to_dict()

    print("Clase EvaluadorMultiDataset creada")
    return (EvaluadorMultiDataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Ejecutar Evaluación Completa

    - Cada dataset pasará por: preprocesamiento → entrenamiento → evaluación
    - Se generarán gráficos y métricas para cada uno
    """
    )
    return


@app.cell
def _(EvaluadorMultiDataset):
    # ============================================================================
    # CONFIGURACIÓN DE LA EVALUACIÓN
    EPOCHS_1 = 100
    BATCH_SIZE_1 = 64
    # Parámetros de entrenamiento
    VERBOSE = 0  # Número máximo de épocas
    DATASETS_DIR = './data'  # Tamaño del batch
    evaluador_multi = EvaluadorMultiDataset(datasets_dir=DATASETS_DIR)  # 0=silencioso, 1=barra de progreso, 2=una línea por época
    datasets_disponibles = evaluador_multi.detectar_datasets(patron='subset_*.csv')
    # Directorio de datasets
    # EJECUTAR EVALUACIÓN
    # Crear evaluador multi-dataset
    # Detectar datasets disponibles
    print(f'\nSe evaluarán {len(datasets_disponibles)} datasets')
    return (
        BATCH_SIZE_1,
        EPOCHS_1,
        VERBOSE,
        datasets_disponibles,
        evaluador_multi,
    )


@app.cell
def _(
    BATCH_SIZE_1,
    EPOCHS_1,
    VERBOSE,
    datasets_disponibles,
    evaluador_multi,
    time,
):
    # ============================================================================
    # INICIAR EVALUACIÓN DE TODOS LOS DATASETS
    print('INICIANDO EVALUACIÓN MULTI-DATASET')
    print('=' * 70)
    tiempo_inicio_global = time.time()
    evaluador_multi.evaluar_todos(datasets=datasets_disponibles, epochs=EPOCHS_1, batch_size=BATCH_SIZE_1, verbose=VERBOSE)
    tiempo_total = time.time() - tiempo_inicio_global
    print(f'\nTiempo total de evaluación: {tiempo_total / 60:.2f} minutos')
    # Ejecutar evaluación completa
    print(f'Evaluación completada exitosamente')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. Generar Reporte Completo

    Genera todas las visualizaciones comparativas y reportes detallados de cada dataset.
    """
    )
    return


@app.cell
def _(evaluador_multi):
    # Generar reporte completo con todas las visualizaciones
    df_resultados = evaluador_multi.generar_reporte_completo(guardar=True)

    # Mostrar el mejor dataset
    mejor = evaluador_multi.mejor_dataset(metrica='mae_real')

    print("\n" + "="*70)
    print("CONCLUSIONES")
    print("="*70)
    print(f"\nMejor performance: {mejor['nombre']}")
    print(f"  • MAE: {mejor['mae_real']:.2f} unidades")
    print(f"  • RMSE: {mejor['rmse_real']:.2f} unidades")
    print(f"  • Error relativo: {mejor['error_relativo_pct']:.2f}%")
    print(f"  • Productos: {mejor['productos_unicos']}")
    print(f"  • Tiempo de entrenamiento: {mejor['tiempo_entrenamiento']:.1f}s")
    print("\n" + "="*70)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Análisis Individual de un Dataset Específico

    Si quieres analizar en detalle un dataset específico, usa esta sección.
    """
    )
    return


@app.cell
def _(evaluador_multi, np):
    # Seleccionar un dataset específico para análisis detallado
    DATASET_ESPECIFICO = 'subset_1000_productos'  # Cambia esto al dataset que quieras analizar

    if DATASET_ESPECIFICO in evaluador_multi.evaluadores:
        eval_especifico = evaluador_multi.evaluadores[DATASET_ESPECIFICO]
    
        print(f"\n{'='*70}")
        print(f"ANÁLISIS DETALLADO: {DATASET_ESPECIFICO}")
        print(f"{'='*70}\n")
    
        # Resumen
        eval_especifico.resumen()
    
        # Visualizaciones
        eval_especifico.generar_visualizaciones(mostrar=True)
    
        # Análisis de predicciones
        print("\nANÁLISIS DE PREDICCIONES:")
        print(f"   • Predicción mínima: {eval_especifico.y_pred_real.min():.2f}")
        print(f"   • Predicción máxima: {eval_especifico.y_pred_real.max():.2f}")
        print(f"   • Predicción promedio: {eval_especifico.y_pred_real.mean():.2f}")
        print(f"\n   • Real mínimo: {eval_especifico.y_val_real.min():.2f}")
        print(f"   • Real máximo: {eval_especifico.y_val_real.max():.2f}")
        print(f"   • Real promedio: {eval_especifico.y_val_real.mean():.2f}")
    
        # Estadísticas de errores
        errores = eval_especifico.y_pred_real - eval_especifico.y_val_real
        print(f"\nESTADÍSTICAS DE ERRORES:")
        print(f"   • Error medio: {errores.mean():.2f}")
        print(f"   • Desviación estándar: {errores.std():.2f}")
        print(f"   • Error mínimo: {errores.min():.2f}")
        print(f"   • Error máximo: {errores.max():.2f}")
        print(f"   • Percentil 25: {np.percentile(errores, 25):.2f}")
        print(f"   • Percentil 75: {np.percentile(errores, 75):.2f}")
    
    else:
        print(f"Dataset '{DATASET_ESPECIFICO}' no encontrado")
        print(f"Datasets disponibles: {list(evaluador_multi.evaluadores.keys())}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 6. Exportar Resultados y Predicciones

    Guarda las predicciones y métricas de un dataset específico para análisis posterior.
    """
    )
    return


@app.cell
def _(Path, datetime, evaluador_multi, json, np, pd):
    # Seleccionar dataset para exportar
    DATASET_EXPORTAR = 'subset_500_productos'  # Cambia esto

    if DATASET_EXPORTAR in evaluador_multi.evaluadores:
        eval_exp = evaluador_multi.evaluadores[DATASET_EXPORTAR]
    
        # Crear directorio de resultados
        resultados_dir = Path('resultados_evaluacion')
        resultados_dir.mkdir(exist_ok=True)
    
        # 1. Exportar predicciones
        df_predicciones = pd.DataFrame({
            'valor_real': eval_exp.y_val_real,
            'prediccion': eval_exp.y_pred_real,
            'error': eval_exp.y_pred_real - eval_exp.y_val_real,
            'error_abs': np.abs(eval_exp.y_pred_real - eval_exp.y_val_real)
        })
    
        pred_path = resultados_dir / f'predicciones_{DATASET_EXPORTAR}.csv'
        df_predicciones.to_csv(pred_path, index=False)
        print(f"Predicciones exportadas a: {pred_path}")
    
        # 2. Exportar métricas
        metricas_dict = {
            'experimento': DATASET_EXPORTAR,
            'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **eval_exp.resultados,
            **{f'tiempo_{k}': v for k, v in eval_exp.tiempos.items()}
        }
    
        metricas_path = resultados_dir / f'metricas_{DATASET_EXPORTAR}.json'
        with open(metricas_path, 'w') as f:
            json.dump(metricas_dict, f, indent=2, default=str)
        print(f"Métricas exportadas a: {metricas_path}")
    
        # 3. Guardar modelo
        modelo_path = resultados_dir / f'modelo_{DATASET_EXPORTAR}.keras'
        eval_exp.model.save(modelo_path)
        print(f"Modelo guardado en: {modelo_path}")
    
        # 4. Guardar historia de entrenamiento
        historia_path = resultados_dir / f'historia_{DATASET_EXPORTAR}.csv'
        df_historia = pd.DataFrame(eval_exp.history.history)
        df_historia.to_csv(historia_path, index=False)
        print(f"Historia de entrenamiento guardada en: {historia_path}")
    
        print(f"\nTodos los resultados exportados a: {resultados_dir}/")
    
        # Mostrar muestra de predicciones
        print(f"\nMuestra de predicciones:")
        print(df_predicciones.head(10).to_string(index=False))
        print(f"\n   Total de predicciones: {len(df_predicciones):,}")
    
    else:
        print(f"Dataset '{DATASET_EXPORTAR}' no encontrado")
    return


if __name__ == "__main__":
    app.run()

