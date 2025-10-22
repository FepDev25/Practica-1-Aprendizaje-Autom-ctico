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

    # --- Parámetros Constantes (deben coincidir con el entrenamiento) ---
    N_STEPS = 7
    TARGET_COLUMN_INDEX = 2 # El índice de 'quantity_available' en el scaler
    NUM_NUMERIC_FEATURES = 18 # Nro de columnas que escalamos con MinMaxScaler

    # --- 1. Cargar el Modelo Entrenado ---
    try:
        model = load_model('best_model.keras')
        print("Modelo 'best_model.keras' cargado.")
    except Exception as e:
        print(f"Error al cargar 'best_model.keras': {e}")

    # --- 2. Cargar el Escalador ---
    try:
        scaler = joblib.load('min_max_scaler.joblib')
        print("Escalador 'min_max_scaler.joblib' cargado.")
    except Exception as e:
        print(f"Error al cargar 'min_max_scaler.joblib': {e}")

    # --- 3. Cargar el Codificador de Productos ---
    try:
        le_product_id = joblib.load('le_product_id.joblib')
        print("Codificador 'le_product_id.joblib' cargado.")
    except Exception as e:
        print(f"Error al cargar 'le_product_id.joblib': {e}")

    # --- 4. Cargar la "Base de Datos" de Features ---
    try:
        df_features = pd.read_csv('df_processed_features.csv')
        df_features['created_at'] = pd.to_datetime(df_features['created_at'])
        print(f"Base de datos de features cargada ({len(df_features)} registros).")
    except Exception as e:
        print(f"Error al cargar 'df_processed_features.csv': {e}")
    
    # --- 5. Definir la lista de columnas (debe ser idéntica a la Fase 1) ---
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

    print("\n--- ¡Entorno de predicción listo! ---")
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
        """
        Predice la 'quantity_available' de un producto para una fecha específica.
        """
        print(f"--- Iniciando predicción para '{product_id_str}' en '{target_date_str}' ---")
    
        # --- 1. Validar y Codificar ID de Producto ---
        try:
            product_id_encoded = le_product_id.transform([product_id_str])[0]
        except ValueError:
            return f"Error: El ID de producto '{product_id_str}' no fue visto durante el entrenamiento."
    
        # --- 2. Validar Fecha ---
        try:
            target_date = pd.to_datetime(target_date_str)
        except ValueError:
            return f"Error: Formato de fecha incorrecto '{target_date_str}'."
    
        # --- 3. Obtener Secuencia Histórica ---
        product_data = df_features[
            (df_features['product_id_encoded'] == product_id_encoded)
        ].sort_values(by='created_at')
    
        historical_data = product_data[product_data['created_at'] < target_date]
    
        # --- 4. Validar si hay suficiente historia ---
        if len(historical_data) < N_STEPS:
            return (f"Error: No hay suficiente historia ({len(historical_data)} días) "
                    f"para predecir. Se necesitan {N_STEPS} días.")
    
        # --- 5. Preparar la Secuencia de Entrada (Input) ---
        sequence_df = historical_data.tail(N_STEPS)
    
        # Extraemos solo las columnas de features, en el orden correcto
        input_features_df = sequence_df[FEATURE_COLUMNS]
    
        # --- !!! CORRECCIÓN CRÍTICA AQUÍ !!! ---
        # Forzamos que todos los datos sean float32. 
        # El error 'dtype: object' ocurre si pd.read_csv cargó
        # los booleanos (True/False) como strings u objetos.
        input_features_scaled = input_features_df.astype(np.float32).values
        # --- Fin de la corrección ---
    
        # Añadimos la dimensión de "batch" (1, N_STEPS, N_FEATURES)
        input_sequence = np.expand_dims(input_features_scaled, axis=0)
    
        # --- 6. Hacer la Predicción (Escalada) ---
        pred_scaled = model.predict(input_sequence)[0][0] # Extraemos el valor
    
        # --- 7. Des-escalar la Predicción ---
        dummy_pred = np.zeros((1, NUM_NUMERIC_FEATURES))
        dummy_pred[:, TARGET_COLUMN_INDEX] = pred_scaled
    
        pred_real = scaler.inverse_transform(dummy_pred)[0][TARGET_COLUMN_INDEX]
    
        print("--- Predicción completada ---")
        return max(0, pred_real)
    return (predict_demand,)


@app.cell
def _(predict_demand):
    # --- Pruebas de la Función 'predict_demand' ---

    print("--- Prueba 1: Predicción Válida (Día Siguiente) ---")
    TEST_ID = 'PROD-00136830' # El ID de tu "Té oolong"
    TEST_DATE = '2025-10-31'  # El día DESPUÉS de tu último dato

    prediccion_1 = predict_demand(TEST_ID, TEST_DATE)
    print(f"\nPredicción para {TEST_ID} el {TEST_DATE}:")

    # Verificamos si la salida es un número (int o float)
    if isinstance(prediccion_1, (int, float)):
        print(f"Stock predicho: {prediccion_1:.2f} unidades")
    else:
        # Si es un string, es un mensaje de error, solo lo imprimimos
        print(f"Resultado: {prediccion_1}")


    print("--- Prueba 1: Predicción Válida (Día Siguiente) ---")
    TEST_ID = 'PROD-023D0E26' # El ID de tu "Té oolong"
    TEST_DATE = '2025-10-31'  # El día DESPUÉS de tu último dato

    prediccion_1 = predict_demand(TEST_ID, TEST_DATE)
    print(f"\nPredicción para {TEST_ID} el {TEST_DATE}:")

    # Verificamos si la salida es un número (int o float)
    if isinstance(prediccion_1, (int, float)):
        print(f"Stock predicho: {prediccion_1:.2f} unidades")
    else:
        # Si es un string, es un mensaje de error, solo lo imprimimos
        print(f"Resultado: {prediccion_1}")



    return


if __name__ == "__main__":
    app.run()
