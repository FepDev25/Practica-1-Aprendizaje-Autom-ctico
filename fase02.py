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
    import marimo as mo
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
        mo,
        np,
        plt,
        tf,
    )


@app.cell
def _(np, tf):
    np.random.seed(42)
    tf.random.set_seed(42)

    PATH_X_TRAIN = 'X_train.npy'
    PATH_Y_TRAIN = 'y_train.npy'
    PATH_X_VAL = 'X_val.npy'
    PATH_Y_VAL = 'y_val.npy'

    X_train = np.load(PATH_X_TRAIN, allow_pickle=True)
    y_train = np.load(PATH_Y_TRAIN, allow_pickle=True)
    X_val = np.load(PATH_X_VAL, allow_pickle=True)
    y_val = np.load(PATH_Y_VAL, allow_pickle=True)

    # Convertir arrays de 'object' a 'float32' para TensorFlow
    print("\nConvirtiendo arrays a dtype 'float32'...")
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_val = X_val.astype('float32')
    y_val = y_val.astype('float32')

    print("\n--- 2. Verificación de Formas (Shapes) ---")
    print(f"Forma de X_train (Muestras, Pasos, Features): {X_train.shape}")
    print(f"Forma de y_train (Muestras,): {y_train.shape}")
    print(f"Forma de X_val (Muestras, Pasos, Features): {X_val.shape}")
    print(f"Forma de y_val (Muestras,): {y_val.shape}")

    INPUT_SHAPE = (X_train.shape[1], X_train.shape[2])

    print("Datos cargados")
    return INPUT_SHAPE, X_train, X_val, y_train, y_val


@app.cell
def _(mo):
    mo.md(
        r"""
    **Resultado:**
        - Se cargan los archivos X_train.npy, y_train.npy, X_val.npy y y_val.npy, previamente generados en la Fase 1.
        - Todos los arreglos se convierten al tipo float32, formato compatible con TensorFlow y más eficiente en memoria.
        - La verificación de formas confirma que las matrices mantienen la estructura esperada (muestras, pasos, características) para X y (muestras,) para y.
        - El modelo va a trabajar con un INPUT_SHAPE de (7, 30), equivalente a una ventana de 7 días con 30 variables predictoras.
    """
    )
    return


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
def _(mo):
    mo.md(
        r"""
    **Resultado:** Nuestro modelo esta compuesto por tres capas:

      - Capa GRU (64 unidades): captura dependencias temporales a lo largo de los 7 pasos temporales.

      - Capa Dropout (0.2): regulariza el modelo apagando aleatoriamente el 20 % de las neuronas para evitar sobreajuste.

      - Capa Densa (1 unidad): produce la predicción continua del nivel de stock (quantity_available).

      El resumen del modelo muestra un total de 18 497 parámetros entrenables, confirmando una arquitectura ligera y eficiente para series de inventario multivariadas.
    """
    )
    return


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
def _(mo):
    mo.md(
        r"""
    **Análisis:**
        - El modelo se compila con el optimizador Adam 
        - Se utiliza la función de pérdida Error Cuadrático Medio (MSE) y la métrica Error Absoluto Medio (MAE)
        - Estas métricas permiten medir tanto la precisión como la estabilidad del modelo frente a valores extremos

        **Parte 2:**
        - Se configuran dos callbacks esenciales:

          - ModelCheckpoint: guarda el modelo solo cuando alcanza su mejor desempeño en validación
          - EarlyStopping: detiene el entrenamiento si no hay mejora después de 10 épocas, evitando sobreentrenamiento.

            Esto garantiza una ejecución eficiente y con control automático de convergencia.
    """
    )
    return


@app.cell
def _(X_train, X_val, early_stopping, model, model_checkpoint, y_train, y_val):
    # Entrenar el Modelo

    EPOCHS = 100        
    BATCH_SIZE = 64    

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val), # ¡Clave para monitorear!
        callbacks=[model_checkpoint, early_stopping],
        verbose=1 # Muestra el progreso en cada época
    )

    print("Entrenado")
    return (history,)


@app.cell
def _(mo):
    mo.md(
        r"""
    **Análisis del Resultado:**
    Se entrenó el modelo con:

    - Épocas máximas: 100

    - Tamaño de batch: 64

    - Parada temprana: época 53

    Resultados finales:

    - loss: 0.0035

    - mean_absolute_error: 0.0443

    El modelo mostró una convergencia estable y rápida, estabilizando las métricas después de las primeras 10 épocas sin evidencias de sobreajuste.
    """
    )
    return


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
def _(mo):
    mo.md(
        r"""
    **Análisis de las gráficas:**
    1. Los gráficos de error (MAE) y pérdida (MSE) exhiben una caída rápida durante las primeras diez épocas, después de lo cual se estabilizan.
    2. Los gráficos de pérdida y error muestran que las curvas de entrenamiento y validación convergen hacia valores bajos y estables, lo que refleja un aprendizaje consistente y una buena capacidad de generalización.
    """
    )
    return


@app.cell
def _(X_val, load_model, math, mean_absolute_error, mean_squared_error, y_val):
    best_model = load_model('best_model.keras')

    # Predicciones
    y_pred_scaled = best_model.predict(X_val)

    # Métricas
    rmse_scaled = math.sqrt(mean_squared_error(y_val, y_pred_scaled))
    mae_scaled = mean_absolute_error(y_val, y_pred_scaled)

    print(f"Métricas del Modelo (en datos escalados [0, 1]):")
    print(f"RMSE: {rmse_scaled:.4f}")
    print(f"MAE:  {mae_scaled:.4f}")
    return (y_pred_scaled,)


@app.cell
def _(mo):
    mo.md(
        r"""
    Sobre los datos normalizados [0, 1]:

    - RMSE: 0.0422

    - MAE: 0.0296

    Estos valores confirman un error promedio muy bajo dentro del rango de los datos escalados.
    """
    )
    return


@app.cell
def _(
    joblib,
    math,
    mean_absolute_error,
    mean_squared_error,
    np,
    y_pred_scaled,
    y_val,
):
    # Des escalar

    scaler = joblib.load('min_max_scaler.joblib')

    TARGET_COLUMN_INDEX = 2 

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

    print("\nMétricas Finales")
    print(f"RMSE (unidades reales): {rmse_real:.2f} unidades")
    print(f"MAE (unidades reales):  {mae_real:.2f} unidades")
    print("\nInterpretación del MAE:")
    print(f"En promedio, las predicciones del modelo se equivocan por +/- {mae_real:.2f} unidades de stock.")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **Resultados:**
    En promedio, el modelo se equivoca en alrededor de ±0.03 unidades de stock, lo que representa un desempeño muy bueno para predicciones de demanda o niveles de inventario con alto volumen de movimiento.
    """
    )
    return


if __name__ == "__main__":
    app.run()
