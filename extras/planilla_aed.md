# Pasos para un Análisis Exploratorio de Datos (EDA) Profesional

## 1. Entendimiento del Problema y Carga de Datos

El primer paso es siempre contextual. Antes de escribir una línea de código, debes entender qué problema intentas resolver y qué significan las columnas de tus datos.

* **Definir el objetivo:** ¿Qué preguntas de negocio o de investigación buscas responder?
* **Revisar el diccionario de datos:** Entender qué representa cada columna (variable), sus unidades y su tipo de dato esperado.
* **Cargar los datos:** Importar el conjunto de datos a un DataFrame (usualmente con Pandas en Python).
* **Primer vistazo:** Usar `df.head()`, `df.tail()` y `df.sample(5)` para ver una muestra de los registros.

### 2. Análisis Descriptivo Inicial (Overview)

Obtén un resumen general y de alto nivel del conjunto de datos. Esta es una revisión rápida para entender la estructura y el contenido.

* **Dimensiones:** Revisa cuántas filas y columnas tiene el dataset con `df.shape`.
* **Tipos de datos:** Comprueba los tipos de datos de cada columna con `df.info()`. Esto es crucial para detectar columnas numéricas que están como texto, fechas mal formateadas, etc.
* **Resumen estadístico:** Utiliza `df.describe(include='all')` para obtener estadísticas clave:
  * **Para variables numéricas:** conteo, media, desviación estándar, mínimo, máximo y los percentiles.
  * **Para variables categóricas:** conteo, número de valores únicos (`unique`), el valor más frecuente (`top`) y su frecuencia (`freq`).

### 3. Limpieza de Datos (Data Cleaning)

Un dataset rara vez es perfecto. Este paso es fundamental para asegurar la calidad del análisis.

* **Valores Nulos:** Identifica cuántos valores nulos hay por columna con `df.isnull().sum()`. Decide una estrategia para manejarlos: eliminarlos (`dropna`), imputarlos con la media, mediana, moda (`fillna`), o un valor constante.
* **Valores Duplicados:** Busca y elimina filas completamente duplicadas con `df.duplicated().sum()` y `df.drop_duplicates()`.
* **Corrección de Tipos de Datos:** Convierte las columnas a su tipo de dato correcto si es necesario (ej. de `object` a `datetime` o `int`) usando `.astype()`.
* **Inconsistencias:** Revisa valores categóricos en busca de errores tipográficos o inconsistencias (ej. "México" vs "mexico").

### 4. Análisis Univariado

Analiza cada variable de forma individual para entender su distribución y características.

* **Variables Numéricas:**
  * **Distribución:** Visualiza con **histogramas** (`sns.histplot`) o **gráficos de densidad (KDE)** (`sns.kdeplot`) para ver si la distribución es normal, sesgada, bimodal, etc.
  * **Dispersión y Outliers:** Usa **diagramas de caja (boxplots)** (`sns.boxplot`) para identificar la mediana, los cuartiles y los valores atípicos (outliers).
* **Variables Categóricas:**
  * **Frecuencia:** Calcula la frecuencia de cada categoría con `df['columna'].value_counts()`.
  * **Visualización:** Usa **gráficos de barras** (`sns.countplot`) para visualizar las frecuencias.

### 5. Análisis Bivariado y Multivariado

Aquí es donde comienzas a descubrir relaciones entre las variables.

* **Numérica vs. Numérica:**

  * **Relación lineal:** Usa **diagramas de dispersión (scatter plots)** (`sns.scatterplot`) para ver la relación entre dos variables.
  * **Correlación:** Calcula la matriz de correlación (`df.corr()`) y visualízala con un **mapa de calor (heatmap)** (`sns.heatmap`). Esto te muestra la fuerza y dirección de la relación lineal entre pares de variables numéricas.

* **Numérica vs. Categórica:**

  * **Comparación de distribuciones:** Compara la distribución de una variable numérica a través de diferentes categorías usando **boxplots** o **gráficos de violín** (`sns.violinplot`). Por ejemplo, comparar el salario (numérica) por departamento (categórica).

* **Categórica vs. Categórica:**

  * **Tabla de contingencia:** Usa `pd.crosstab()` para ver la frecuencia de combinación entre dos variables categóricas.
  * **Visualización:** Un gráfico de barras agrupadas o apiladas puede ser muy útil.

* **Análisis Multivariado:**

  * **Pairplot:** Utiliza `sns.pairplot(df)` para obtener una matriz de diagramas de dispersión para cada par de variables numéricas y histogramas en la diagonal. Puedes colorear los puntos por una variable categórica (`hue`) para añadir una tercera dimensión al análisis.

### 6. Detección Profunda de Outliers

Aunque ya los viste en el análisis univariado, aquí puedes aplicar técnicas más formales si es necesario.

* **Métodos estadísticos:** Como el **Rango Intercuartílico (IQR)** o el **Z-score** para identificar y decidir si eliminar, transformar o mantener estos valores atípicos.

### 7. Feature Engineering (Opcional, pero recomendado)

Basado en los hallazgos, puedes crear nuevas variables que podrían ser más útiles para un futuro modelo de machine learning.

* **Ejemplos:**
  * Extraer el mes o el día de la semana de una fecha.
  * Combinar dos variables (ej. `ancho * alto` para crear `area`).
  * Crear categorías a partir de una variable numérica (ej. agrupar edades en rangos).

### 8. Documentación y Resumen de Hallazgos

Este es el paso final y uno de los más importantes. Un buen EDA no es solo código, sino las conclusiones que extraes.

* **Resume los insights clave:** ¿Qué patrones interesantes encontraste? ¿Qué relaciones son las más fuertes?
* **Menciona las anomalías:** ¿Hubo datos faltantes o outliers significativos? ¿Cómo los trataste?
* **Genera nuevas preguntas:** ¿Qué nuevas preguntas surgieron durante el análisis?
* **Propón los siguientes pasos:** Basado en tu análisis, ¿qué recomendarías hacer a continuación? (Ej. "La variable X está fuertemente correlacionada con el objetivo, debería ser clave en el modelo predictivo").

-----

## Plantilla de Código en Python

```python
# 1. IMPORTACIÓN DE LIBRERÍAS Y CARGA DE DATOS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de visualización
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

# Carga de datos
df = pd.read_csv('tu_dataset.csv')

# --------------------------------------------------

# 2. ANÁLISIS DESCRIPTIVO INICIAL
print("--- Primeras 5 filas ---")
print(df.head())
print("\n--- Dimensiones del DataFrame (filas, columnas) ---")
print(df.shape)
print("\n--- Información General y Tipos de Datos ---")
df.info()
print("\n--- Resumen Estadístico ---")
print(df.describe(include='all'))

# --------------------------------------------------

# 3. LIMPIEZA DE DATOS
print("\n--- Conteo de Valores Nulos por Columna ---")
print(df.isnull().sum())

# Ejemplo de imputación (reemplazar nulos en una columna numérica con la media)
# df['columna_numerica'].fillna(df['columna_numerica'].mean(), inplace=True)

print("\n--- Conteo de Filas Duplicadas ---")
print(f"Hay {df.duplicated().sum()} filas duplicadas.")
# df.drop_duplicates(inplace=True)

# --------------------------------------------------

# 4. ANÁLISIS UNIVARIADO
# Variables numéricas
numerical_cols = df.select_dtypes(include=np.number).columns
for col in numerical_cols:
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Histograma de {col}')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot de {col}')
    
    plt.show()

# Variables categóricas
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(y=df[col], order=df[col].value_counts().index)
    plt.title(f'Conteo de {col}')
    plt.show()

# --------------------------------------------------

# 5. ANÁLISIS BIVARIADO Y MULTIVARIADO
# Mapa de calor de correlaciones (solo para variables numéricas)
plt.figure(figsize=(12, 8))
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Mapa de Calor de Correlaciones')
plt.show()

# Pairplot para visualizar relaciones entre todas las variables numéricas
# Se puede colorear por una variable categórica usando el parámetro 'hue'
# sns.pairplot(df, hue='una_columna_categorica')
# plt.show()

# Ejemplo: Numérica vs. Categórica
# sns.boxplot(x='col_categorica', y='col_numerica', data=df)
# plt.title('Boxplot de Columna Numérica por Categoría')
# plt.show()

# --------------------------------------------------

# 8. RESUMEN DE HALLAZGOS (en Markdown)
# - Insight 1: La variable 'precio' tiene una distribución sesgada a la derecha.
# - Insight 2: Existe una fuerte correlación positiva entre 'edad' y 'salario'.
# - Insight 3: La categoría 'A' es la más frecuente en la columna 'tipo_producto'.
# - Siguientes pasos: Investigar los outliers detectados en 'ventas_diarias'.

```
