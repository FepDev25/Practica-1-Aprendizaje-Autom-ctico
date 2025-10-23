import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import joblib
    from tensorflow.keras.models import load_model
    import math

    print("--- Fase 3: Carga de Artefactos de Producción ---")

    N_STEPS = 7
    TARGET_COLUMN_INDEX = 2
    NUM_NUMERIC_FEATURES = 18

    # modelo
    try:
        model = load_model('best_model.keras')
        print("Modelo 'best_model.keras' cahrgado.")
    except Exception as e:
        print(f"Error al cargar 'best_model.keras': {e}")

    # escalador
    try:
        scaler = joblib.load('min_max_scaler.joblib')
        print("Escalador 'min_max_scaler.joblib' cargado.")
    except Exception as e:
        print(f"Error al cargar 'min_max_scaler.joblib': {e}")

    # codificador de productos
    try:
        le_product_id = joblib.load('le_product_id.joblib')
        print("Codificador 'le_product_id.joblib' cargado.")
    except Exception as e:
        print(f"Error al cargar 'le_product_id.joblib': {e}")

    # features
    try:
        df_features = pd.read_csv('df_processed_features.csv')
        df_features['created_at'] = pd.to_datetime(df_features['created_at'])
        print(f"Base de datos de features cargada ({len(df_features)} registros).")
    except Exception as e:
        print(f"Error al cargar 'df_processed_features.csv': {e}")

    # Lista de columnas
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
    return (
        FEATURE_COLUMNS,
        NUM_NUMERIC_FEATURES,
        N_STEPS,
        TARGET_COLUMN_INDEX,
        df_features,
        le_product_id,
        model,
        np,
        pd,
        scaler,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    Se cargan los artefactos de producción: 
    - Modelo óptimo `best_model.keras` 
    - Escalador `min_max_scaler.joblib` para desescalar predicciones.
    - Codificador `le_product_id.joblib` para validar/convertir `product_id`.

    Además se lee `df_processed_features.csv` (79 174 filas) y se fija la lista de `FEATURE_COLUMNS`. 
    Estos objetos garantizan que la inferencia use **exactamente** las mismas transformaciones que en el entrenamiento (consistencia entre train y producción).

    """
    )
    return


@app.cell
def _(
    FEATURE_COLUMNS,
    NUM_NUMERIC_FEATURES,
    N_STEPS,
    TARGET_COLUMN_INDEX,
    df_features,
    le_product_id,
    model,
    np,
    pd,
    scaler,
):
    def predict_demand(product_id_str, target_date_str):

        # Validar y Codificar ID de Producto ---
        try:
            product_id_encoded = le_product_id.transform([product_id_str])[0]
        except ValueError:
            return f"Error: El ID de producto '{product_id_str}' no fue visto durante el entrenamiento."

        # validar fecha
        try:
            target_date = pd.to_datetime(target_date_str)
        except ValueError:
            return f"Error: Formato de fecha incorrecto '{target_date_str}'."

        # secuencia histórica
        product_data = df_features[
            (df_features['product_id_encoded'] == product_id_encoded)
        ].sort_values(by='created_at')

        historical_data = product_data[product_data['created_at'] < target_date]

        # validar historia
        if len(historical_data) < N_STEPS:
            return (f"Error: No hay suficiente historia ({len(historical_data)} días) "
                    f"para predecir. Se necesitan {N_STEPS} días.")

        sequence_df = historical_data.tail(N_STEPS)
        input_features_df = sequence_df[FEATURE_COLUMNS]

        input_features_scaled = input_features_df.astype(np.float32).values
        input_sequence = np.expand_dims(input_features_scaled, axis=0)

        # prediccion
        pred_scaled = model.predict(input_sequence)[0][0]

        # des-escalar
        dummy_pred = np.zeros((1, NUM_NUMERIC_FEATURES))
        dummy_pred[:, TARGET_COLUMN_INDEX] = pred_scaled

        pred_real = scaler.inverse_transform(dummy_pred)[0][TARGET_COLUMN_INDEX]

        print("predicción realizada")
        return max(0, pred_real)
    return (predict_demand,)


@app.cell
def _(mo):
    mo.md(
        r"""
    Flujo de inferencia:

    1) **Validación**: 
       - `product_id` se transforma con `LabelEncoder`; si no fue visto en train, se devuelve un mensaje de error.
       - `target_date` se convierte a `datetime`; si el formato es inválido, se informa al usuario.

    2) **Construcción del contexto temporal**:
       - Se filtra el historial del producto **previo** a `target_date` y se ordena por `created_at`.
       - Se requieren **N_STEPS (=7)** observaciones; si no hay suficientes días, se detiene con mensaje.

    3) **Preparación del input**:
       - Se toman las últimas 7 filas (`tail(N_STEPS)`), se seleccionan `FEATURE_COLUMNS`, se castea a `float32` y se expande a forma `(1, 7, 30)`.

    4) **Predicción y desescalado**:
       - El modelo devuelve `pred_scaled` en [0,1]. 
       - Se embebe en un vector “dummy” con `NUM_NUMERIC_FEATURES` para aplicar `inverse_transform` y recuperar unidades reales de `quantity_available` (índice `TARGET_COLUMN_INDEX`).
       - Se trunca a `>= 0` y se retorna el valor en **unidades de stock**.
    Con esto, la función es **determinista y segura** ante entradas no vistas o con historial insuficiente

    """
    )
    return


@app.cell
def _(predict_demand):
    print("Prueba 1:")
    TEST_ID = 'PROD-00136830'
    TEST_DATE = '2025-10-31'

    prediccion_1 = predict_demand(TEST_ID, TEST_DATE)
    print(f"\nPredicción para {TEST_ID} el {TEST_DATE}:")

    if isinstance(prediccion_1, (int, float)):
        print(f"Stock predicho: {prediccion_1:.2f} unidades")
    else:
        print(f"Resultado: {prediccion_1}")

    print()

    print("Prueba 2:")
    TEST_ID = 'PROD-023D0E26'
    TEST_DATE = '2025-10-31'

    prediccion_1 = predict_demand(TEST_ID, TEST_DATE)
    print(f"\nPredicción para {TEST_ID} el {TEST_DATE}:")

    if isinstance(prediccion_1, (int, float)):
        print(f"Stock predicho: {prediccion_1:.2f} unidades")
    else:
        print(f"Resultado: {prediccion_1}")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
