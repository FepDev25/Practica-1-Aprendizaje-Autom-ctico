import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Practica 1 - Parte 1
    ## RNN y series temporales
    **Nombres:** Felipe Peralta y Samantha Suquilanda
    """
    )
    return


@app.cell
def _():
    #  2. Importación de módulos
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    print(" Librerías importadas correctamente")
    return (
        Dense,
        LSTM,
        LabelEncoder,
        MinMaxScaler,
        Sequential,
        mean_squared_error,
        np,
        pd,
        plt,
        sqrt,
    )


@app.cell
def _(plt):
    #  3. Carga y limpieza del dataset
    from pandas import read_csv
    from datetime import datetime

    # Carga del CSV original
    dataset = read_csv('pollution.csv', header=0, index_col=0)
    print(dataset.head())

    # Reemplaza valores nulos
    dataset['pollution'].fillna(0, inplace=True)
    print(f"Datos cargados: {dataset.shape[0]} filas y {dataset.shape[1]} columnas")

    # Gráfico rápido de la variable objetivo
    dataset['pollution'].plot(figsize=(10,4), title='Contaminación (pollution)')
    plt.show()
    return (dataset,)


@app.cell
def _(dataset, plt):
    _values = dataset.values
    groups = [0, 1, 2, 3, 5, 6, 7]
    i = 1
    plt.figure(figsize=(10, 10))
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(_values[:, group])
        plt.title(dataset.columns[group], y=0.5, loc='right')
        i = i + 1
    plt.show()
    return


@app.cell
def _(pd):
    #  5. Transformación a formato supervisado (2 horas)
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        (cols, names) = ([], [])
        for i in range(n_in, 0, -1):  # entradas pasadas (t-n, ... t-1)
            cols.append(df.shift(i))
            names = names + [f'var{j + 1}(t-{i})' for j in range(n_vars)]
        for i in range(0, n_out):
            cols.append(df.shift(-i))  # predicción (t)
            if i == 0:
                names = names + [f'var{j + 1}(t)' for j in range(n_vars)]
            else:
                names = names + [f'var{j + 1}(t+{i})' for j in range(n_vars)]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    return (series_to_supervised,)


@app.cell
def _(LabelEncoder, MinMaxScaler, dataset, series_to_supervised):
    #  6. Codificación y normalización
    encoder = LabelEncoder()
    dataset['wnd_dir'] = encoder.fit_transform(dataset['wnd_dir'])
    _values = dataset.values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(_values)
    (n_in, n_out) = (2, 1)
    reframed = series_to_supervised(scaled, n_in, n_out)
    to_drop = [f'var{i}(t)' for i in range(2, 9)]
    reframed.drop(to_drop, axis=1, inplace=True)  # Dos horas 
    # Eliminamos salidas 
    print('Forma final del dataset supervisado:', reframed.shape)
    return encoder, n_in, reframed, scaler


@app.cell
def _(n_in, reframed):
    #  7. Preparación de datos para el modelo
    _values = reframed.values
    n_features = 8
    n_train_hours = 365 * 24
    train = _values[:n_train_hours, :]
    test = _values[n_train_hours:, :]
    (train_X, train_y) = (train[:, :-1], train[:, -1])
    (test_X, test_y) = (test[:, :-1], test[:, -1])
    train_X = train_X.reshape((train_X.shape[0], n_in, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_in, n_features))
    print('train_X:', train_X.shape, 'train_y:', train_y.shape)
    print('test_X:', test_X.shape, 'test_y:', test_y.shape)
    return n_features, test, test_X, test_y, train_X, train_y


@app.cell
def _(
    Dense,
    LSTM,
    Sequential,
    n_features,
    n_in,
    plt,
    test_X,
    test_y,
    train_X,
    train_y,
):
    #  8. Creación y entrenamiento del modelo LSTM
    model = Sequential()
    model.add(LSTM(50, input_shape=(n_in, n_features)))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    history = model.fit(
        train_X, train_y,
        epochs=50, batch_size=72,
        validation_data=(test_X, test_y),
        verbose=2, shuffle=False
    )

    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.title("Evolución del error (MAE)")
    plt.show()
    return (model,)


@app.cell
def _(
    mean_squared_error,
    model,
    n_features,
    np,
    plt,
    scaler,
    sqrt,
    test,
    test_X,
    test_y,
):
    #  9. Evaluación del modelo
    yhat = model.predict(test_X, verbose=0)
    test_X_2d = test[:, :-1]
    inv_yhat = np.concatenate((yhat, test_X_2d[:, -(n_features - 1):]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    test_y_1 = test[:, -1].reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y_1, test_X_2d[:, -(n_features - 1):]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print(f'RMSE: {rmse:.3f}')
    plt.figure(figsize=(10, 4))
    plt.plot(inv_y[:200], label='Real')
    plt.plot(inv_yhat[:200], label='Predicho')
    plt.legend()
    plt.title('Comparación Real vs Predicho (primeras 200 horas)')
    plt.show()
    return


@app.cell
def _(np):
    #  10. Función de predicción con 2 horas
    def predict_next_hour_from_two(x_t2, x_t1, encoder, scaler, model):
        cols = ['pollution','dew','temp','press','wnd_dir','wnd_spd','snow','rain']
        row_t2 = [x_t2[c] for c in cols]
        row_t1 = [x_t1[c] for c in cols]
        if isinstance(row_t2[4], str):
            row_t2[4] = encoder.transform([row_t2[4]])[0]
        if isinstance(row_t1[4], str):
            row_t1[4] = encoder.transform([row_t1[4]])[0]
        X_raw = np.array([row_t2, row_t1], dtype='float32')
        X_scaled = scaler.transform(X_raw)
        X_in = X_scaled.reshape((1, 2, 8))
        yhat = model.predict(X_in, verbose=0)
        last_step = X_scaled[-1]
        inv_base = np.concatenate([yhat.ravel(), last_step[1:]], axis=0).reshape(1, -1)
        yhat_inv = scaler.inverse_transform(inv_base)[:, 0]
        return float(yhat_inv[0])
    return (predict_next_hour_from_two,)


@app.cell
def _(encoder, model, predict_next_hour_from_two, scaler):
    #  11. Ejemplo de predicción
    x_t2 = {'pollution': 42.0, 'dew': 3.5, 'temp': 17.2, 'press': 1012.0, 'wnd_dir': 'NW', 'wnd_spd': 3.1, 'snow': 0.0, 'rain': 0.0}
    x_t1 = {'pollution': 45.0, 'dew': 3.8, 'temp': 17.6, 'press': 1011.5, 'wnd_dir': 'NW', 'wnd_spd': 3.4, 'snow': 0.0, 'rain': 0.0}

    pred = predict_next_hour_from_two(x_t2, x_t1, encoder, scaler, model)
    print("Predicción de pollution(t):", pred)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Conclusiones

    1. Configuración y arquitectura:

    - Se entrenó una red neuronal recurrente LSTM utilizando una ventana temporal de 2 horas (16 valores de entrada).

    - Esto permite capturar dependencias de corto plazo en la serie temporal.

    2. Desempeño del modelo:

    - El modelo alcanzó una pérdida MAE ≈ 0.013 y un error RMSE ≈ 26.23, equivalentes a un error relativo <9%.

    - Las curvas de entrenamiento y validación muestran convergencia estable y sin sobreajuste.

    3. Calidad de las predicciones:

    - El modelo logra predecir con alta precisión los valores futuros de contaminación, siguiendo las tendencias reales.

    - Los picos de contaminación y los descensos fueron correctamente anticipados por la red.

    - Las ligeras desviaciones se presentan en los puntos de cambio abrupto, lo cual es esperable en modelos secuenciales.

    4. Predicción puntual:

    Para un caso de prueba con pollution(t-2)=42 y pollution(t-1)=45, la red predice pollution(t)≈44.3, manteniendo una tendencia coherente con los datos anteriores.

    5. Conclusión general:

    - La red LSTM con 2 horas de historia es capaz de aprender patrones temporales complejos en series multivariadas.

    - Su rendimiento demuestra que puede utilizarse como una herramienta confiable de predicción a corto plazo, útil en contextos de monitoreo ambiental o pronóstico de variables físicas.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
