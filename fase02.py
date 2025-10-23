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
def _(mo):
    mo.md(r"""## Entrenamiento del modelo""")
    return


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
    return mae_scaled, rmse_scaled, y_pred_scaled


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
    mae_scaled,
    math,
    mean_absolute_error,
    mean_squared_error,
    np,
    rmse_scaled,
    y_pred_scaled,
    y_val,
):
    # Des escalar

    scaler = joblib.load('min_max_scaler.joblib')

    TARGET_COLUMN_INDEX = 2 

    num_numeric_features = 18 

    dummy_y_val = np.zeros((len(y_val), num_numeric_features))
    dummy_y_val[:, TARGET_COLUMN_INDEX] = y_val.ravel() # .ravel() lo aplana
    y_val_real = scaler.inverse_transform(dummy_y_val)[:, TARGET_COLUMN_INDEX]

    dummy_y_pred = np.zeros((len(y_pred_scaled), num_numeric_features))
    dummy_y_pred[:, TARGET_COLUMN_INDEX] = y_pred_scaled.ravel()
    y_pred_real = scaler.inverse_transform(dummy_y_pred)[:, TARGET_COLUMN_INDEX]

    # Calcular métricas finales en unidades reales de stock
    rmse_real = math.sqrt(mean_squared_error(y_val_real, y_pred_real))
    mae_real = mean_absolute_error(y_val_real, y_pred_real)

    # Obtener el rango de valores reales para contexto
    min_stock = scaler.data_min_[TARGET_COLUMN_INDEX]
    max_stock = scaler.data_max_[TARGET_COLUMN_INDEX]
    rango_stock = max_stock - min_stock

    # Error relativo porcentual
    error_relativo = (mae_real / rango_stock) * 100

    print("MÉTRICAS FINALES DEL MODELO")
    print(f"\nContexto del Dataset:")
    print(f"   • Rango de stock: {min_stock:.0f} - {max_stock:.0f} unidades")
    print(f"   • Rango total: {rango_stock:.0f} unidades")

    print(f"\nMétricas en Escala Normalizada [0,1]:")
    print(f"   • RMSE: {rmse_scaled:.4f}")
    print(f"   • MAE:  {mae_scaled:.4f}")

    print(f"\nMétricas en Unidades Reales:")
    print(f"   • RMSE: {rmse_real:.2f} unidades")
    print(f"   • MAE:  {mae_real:.2f} unidades")
    print(f"   • Error Relativo: {error_relativo:.2f}%")

    print(f"\nInterpretación:")
    print(f"   En promedio, las predicciones se desvían ±{mae_real:.2f} unidades,")
    print(f"   lo que representa un error del {error_relativo:.2f}% respecto al rango total.")
    print(f"   Esto es equivalente a un MAE normalizado de {mae_scaled:.4f}.")
    return error_relativo, mae_real, rmse_real, y_pred_real, y_val_real


@app.cell
def _(mo):
    mo.md(
        r"""
    **Resultados:**
    El modelo tiene un rendimiento excelente. Con un error relativo del ~3%, 
    las predicciones son altamente precisas considerando el rango completo del inventario.
    El MAE de ~193 unidades sobre un rango de 6435 unidades es equivalente 
    al MAE normalizado de 0.03 en escala [0,1].
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Análisis de rendimiento""")
    return


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
def _(mo):
    mo.md(
        r"""
    - Los puntos forman una nube alargada que sigue de cerca la línea roja. Significa que el modelo ha aprendido la tendencia general de los datos.
    - Sin embargo, el modelo no refleja la distribución real. Las dos formas de distribución son bastante diferentes. Los datos reales (azules) tienen varios "picos" o modas. Las predicciones (naranja), son más "suaves" y centradas, con un gran pico alrededor de 4500 que no existe en los datos reales.
    """
    )
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
def _(mo):
    mo.md(
        r"""
    - El modelo es imparcial (unbiased). No tiene una tendencia sistemática a sobreestimar o subestimar el stock. Comete tantos errores pequeños por encima como por debajo.
    - El modelo es extremadamente preciso. La gran mayoría de las predicciones tienen un error minúsculo. El MAE de 0.03 (en una escala de 0 a 1) es un resultado sobresaliente.
    - En general (el 50% de las veces), el modelo se equivoca por un valor entre 73.6 y 293.5 unidades, con un error "promedio" (mediana) de 161.8.
    - Sin embargo, el modelo es inconsistente. Aunque normalmente funciona dentro de un rango aceptable (la caja), con frecuencia produce predicciones muy alejadas de su valor real.
    """
    )
    return


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
def _(mo):
    mo.md(
        r"""
    - El error de el modelo aumenta a medida que aumenta el valor del stock.
    - El modelo es bueno prediciendo stocks bajos, regular prediciendo stocks medios, e inconsistente prediciendo stocks altos.
    """
    )
    return


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


if __name__ == "__main__":
    app.run()
