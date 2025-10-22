import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from tensorflow.keras.optimizers import Adam
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import os

    from tensorflow.keras.models import load_model
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import math
    import joblib
    return (
        Adam,
        Dense,
        Dropout,
        EarlyStopping,
        GRU,
        ModelCheckpoint,
        Sequential,
        joblib,
        load_model,
        math,
        mean_absolute_error,
        mean_squared_error,
        np,
        plt,
        tf,
    )


@app.cell
def _(np, tf):

    # Opcional: para asegurar reproducibilidad
    np.random.seed(42)
    tf.random.set_seed(42)

    # Opcional: Asegurarse de que TensorFlow use la GPU si está disponible
    # (La guía recomienda usar Colab con GPU)
    print("Versión de TensorFlow:", tf.__version__)
    print("GPUs Disponibles: ", tf.config.list_physical_devices('GPU'))

    # --- 1. Carga de Datos ---
    print("\n--- 1. Cargando datos preprocesados ---")

    # Definir la ruta (asumiendo que están en la misma carpeta)
    PATH_X_TRAIN = 'X_train.npy'
    PATH_Y_TRAIN = 'y_train.npy'
    PATH_X_VAL = 'X_val.npy'
    PATH_Y_VAL = 'y_val.npy'

    # Cargar los arrays (permitiendo 'pickle' ya que confiamos en los archivos)
    X_train = np.load(PATH_X_TRAIN, allow_pickle=True)
    y_train = np.load(PATH_Y_TRAIN, allow_pickle=True)
    X_val = np.load(PATH_X_VAL, allow_pickle=True)
    y_val = np.load(PATH_Y_VAL, allow_pickle=True)

    # --- !! AQUÍ ESTÁ LA SOLUCIÓN !! ---
    # Convertir arrays de 'object' a 'float32' para TensorFlow
    print("\nConvirtiendo arrays a dtype 'float32'...")
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_val = X_val.astype('float32')
    y_val = y_val.astype('float32')
    print("¡Conversión completada!")

    print("¡Datos cargados exitosamente!")

    # --- 2. Verificación de Formas y Definición de Parámetros ---
    print("\n--- 2. Verificación de Formas (Shapes) ---")
    print(f"Forma de X_train (Muestras, Pasos, Features): {X_train.shape}")
    print(f"Forma de y_train (Muestras,): {y_train.shape}")
    print(f"Forma de X_val (Muestras, Pasos, Features): {X_val.shape}")
    print(f"Forma de y_val (Muestras,): {y_val.shape}")

    # Extraer las dimensiones para la entrada del modelo
    # X_train.shape[1] = n_pasos (7)
    # X_train.shape[2] = n_features (30)
    INPUT_SHAPE = (X_train.shape[1], X_train.shape[2])

    print(f"\nParámetros de entrada para el modelo (Input Shape): {INPUT_SHAPE}")
    return INPUT_SHAPE, X_train, X_val, y_train, y_val


@app.cell
def _(Dense, Dropout, GRU, INPUT_SHAPE, Sequential):
    # --- 3. Definir la Arquitectura del Modelo (Opción GRU) ---

    print("\n--- 3. Construyendo el modelo con GRU ---")

    model_gru = Sequential(name="Modelo_GRU_Prediccion_Stock")

    # Capa 1: Capa GRU
    # units=64: El número de "neuronas" en la capa. 64 es un buen balance.
    # input_shape: (7, 30) -> (N_STEPS, N_FEATURES)
    model_gru.add(GRU(units=64, input_shape=INPUT_SHAPE, name="Capa_Entrada_GRU"))

    # Capa 2: Dropout (Regularización)
    # Apagamos el 20% de las neuronas aleatoriamente en cada época
    # para evitar que el modelo "memorice" los datos de entrenamiento.
    model_gru.add(Dropout(0.2, name="Capa_Dropout"))

    # Capa 3: Capa de Salida
    # units=1: Solo queremos 1 predicción (la 'quantity_available')
    # No usamos 'activation' (es lineal), lo cual es correcto para regresión.
    model_gru.add(Dense(units=1, name="Capa_Salida_Prediccion"))

    # --- Mostrar un resumen del modelo ---
    print("¡Modelo construido! Mostrando resumen...")
    model_gru.summary()

    # Asignamos este modelo como el 'model' principal
    model = model_gru
    return (model,)


@app.cell
def _(Adam, EarlyStopping, ModelCheckpoint, model):
    # --- 4. Compilar el Modelo y Definir Callbacks ---

    print("\n--- 4. Compilando el modelo ---")

    # La guía pide evaluar con RMSE o MAE.
    # - 'mean_squared_error' (MSE) como 'loss' optimiza indirectamente el RMSE.
    # - 'mean_absolute_error' (MAE) lo añadimos como métrica explícita.
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error', 
        metrics=['mean_absolute_error']
    )

    print("¡Modelo compilado exitosamente!")
    print("Optimizador: Adam")
    print("Función de Pérdida (Loss): MSE (para optimizar RMSE)")
    print("Métricas: MAE")


    # --- Definir Callbacks ---

    # 1. Guardar el mejor modelo
    #    Guarda solo el modelo que tenga el 'val_loss' (error en validación) más bajo.
    checkpoint_path = 'best_model.keras'
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss', # Monitorea el error en los datos de validación
        save_best_only=True,
        mode='min', # Queremos minimizar el error
        verbose=1 # Muestra un mensaje cuando guarda
    )

    # 2. Detener el entrenamiento si no hay mejora
    #    'patience=10' significa: espera 10 épocas. Si el 'val_loss' no mejora
    #    en esas 10 épocas, detén el entrenamiento.
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=1,
        restore_best_weights=True # Restaura los pesos del mejor modelo guardado
    )

    print("\nCallbacks 'ModelCheckpoint' y 'EarlyStopping' definidos.")
    return checkpoint_path, early_stopping, model_checkpoint


@app.cell
def _(
    X_train,
    X_val,
    checkpoint_path,
    early_stopping,
    model,
    model_checkpoint,
    y_train,
    y_val,
):
    # --- 5. Entrenar el Modelo ---

    print("\n--- 5. ¡Iniciando el entrenamiento del modelo! ---")
    print("Esto puede tardar varios minutos...")

    # Definimos los hiperparámetros de entrenamiento
    EPOCHS = 100         # Un número máximo, EarlyStopping decidirá
    BATCH_SIZE = 64    # Cuántas muestras procesar a la vez

    # Llamamos a model.fit()
    # Guardamos el historial (history) para graficar las curvas de aprendizaje
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val), # ¡Clave para monitorear!
        callbacks=[model_checkpoint, early_stopping],
        verbose=1 # Muestra el progreso en cada época
    )

    print("\n--- ¡Entrenamiento completado! ---")
    print(f"El mejor modelo ha sido guardado en '{checkpoint_path}'")
    return (history,)


@app.cell
def _(
    X_val,
    history,
    joblib,
    load_model,
    math,
    mean_absolute_error,
    mean_squared_error,
    np,
    plt,
    y_val,
):
    # --- 6.1. Visualizar el Historial de Entrenamiento ---
    print("\n--- 6.1. Graficando historial de entrenamiento ---")

    # 'history' contiene los datos del 'model.fit()'
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

    # --- 6.2. Cargar el mejor modelo y Evaluar ---
    print("\n--- 6.2. Evaluando el mejor modelo guardado ---")

    # Cargar el modelo que guardó ModelCheckpoint
    best_model = load_model('best_model.keras')

    # 1. Hacer predicciones en el set de validación
    y_pred_scaled = best_model.predict(X_val)

    # 2. Calcular métricas (en datos escalados)
    #    Estas métricas son las que vio el modelo (rango 0-1)
    rmse_scaled = math.sqrt(mean_squared_error(y_val, y_pred_scaled))
    mae_scaled = mean_absolute_error(y_val, y_pred_scaled)

    print(f"Métricas del Modelo (en datos escalados [0, 1]):")
    print(f"  - RMSE (escalado): {rmse_scaled:.4f}")
    print(f"  - MAE (escalado):  {mae_scaled:.4f}")
    print("Nota: 'val_loss' fue MSE, por lo que RMSE debe ser la raíz cuadrada de val_loss.")


    # --- 6.3. Des-escalar Predicciones (¡Paso clave!) ---
    #    Para entender el error en unidades reales de stock

    print("\n--- 6.3. Des-escalando predicciones a unidades reales ---")

    # 1. Cargar el escalador que guardamos en la Fase 1
    scaler = joblib.load('min_max_scaler.joblib')

    # 2. Revertir la escala de y_val (valores reales)
    #    Necesitamos "engañar" al escalador para que revierta solo nuestra columna 'quantity_available'
    #    Creamos un array de "dummys" con la misma forma que escalamos (18 features)
    #    y ponemos nuestros datos de 'y_val' en la columna correcta.

    # Debemos saber en qué índice estaba 'quantity_available'
    # Basado en tu código de Fase 1, era la columna 2 (índice 2)
    # 'quantity_on_hand' (0), 'quantity_reserved' (1), 'quantity_available' (2)
    TARGET_COLUMN_INDEX = 2 

    # Crear arrays dummy para 'y_val' y 'y_pred'
    # Usamos el número de features numéricas originales (18)
    num_numeric_features = 18 # Revisa este número si es diferente

    dummy_y_val = np.zeros((len(y_val), num_numeric_features))
    dummy_y_val[:, TARGET_COLUMN_INDEX] = y_val.ravel() # .ravel() lo aplana
    y_val_real = scaler.inverse_transform(dummy_y_val)[:, TARGET_COLUMN_INDEX]

    dummy_y_pred = np.zeros((len(y_pred_scaled), num_numeric_features))
    dummy_y_pred[:, TARGET_COLUMN_INDEX] = y_pred_scaled.ravel()
    y_pred_real = scaler.inverse_transform(dummy_y_pred)[:, TARGET_COLUMN_INDEX]

    # 3. Calcular métricas finales en unidades reales de stock
    rmse_real = math.sqrt(mean_squared_error(y_val_real, y_pred_real))
    mae_real = mean_absolute_error(y_val_real, y_pred_real)

    print("\n--- ¡Métricas Finales (en Unidades de Stock Reales)! ---")
    print(f"  - RMSE (unidades): {rmse_real:.2f} unidades")
    print(f"  - MAE (unidades):  {mae_real:.2f} unidades")
    print("\nInterpretación del MAE:")
    print(f"En promedio, las predicciones del modelo se equivocan por +/- {mae_real:.2f} unidades de stock.")
    return


if __name__ == "__main__":
    app.run()
