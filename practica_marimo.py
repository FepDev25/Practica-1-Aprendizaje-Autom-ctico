import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Practica 1
    ### APRENDIZAJE PROFUNDO Y SERIES TEMPORALES
    **Nombres:** Felipe Peralta y Samantha Suquilanda
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Fase 1: Análisis y Preparación del Dataset""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### a. Descripción General del Dataset
    Descripción:
    Variables:
    - Crear  mas variables
    - Usar MARIMO Y ML FLOW
    - Usar Fine Tunning
    - Mejorar el valor del RMSE calculando la media de ese valor y de eso 
    - Usar MArimo tambien en la practica de las 2 horas

    """
    )
    return


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    df = pd.read_csv('generar_dataset/dataset_inventario.csv')

    # Marcar productos como activos
    df["is_active"]= True

    # Transformar datos a tipo fecha
    df.created_at = pd.to_datetime(df.created_at)
    df.last_order_date = pd.to_datetime(df.last_order_date)
    df.last_updated_at = pd.to_datetime(df.last_updated_at)
    df.last_stock_count_date = pd.to_datetime(df.last_stock_count_date)
    df.expiration_date = pd.to_datetime(df.expiration_date)

    df.head()
    return (df,)


@app.cell
def _(df):
    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
