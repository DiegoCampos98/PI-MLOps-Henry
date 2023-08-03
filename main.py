import numpy as np
import pandas as pd
import ast
from fastapi import FastAPI
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error


app = FastAPI(title = "Proyecto MLOps")

rows = []
with open("steam_games.json") as f:
    for line in f.readlines():
        rows.append(ast.literal_eval(line))
df = pd.DataFrame(rows)

df['anio'] = df['release_date'].str.extract(r'(\d{4})')
df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce')
df['anio'] = pd.to_numeric(df['anio'], errors='coerce')

df_filtrado = pd.read_csv(r'steam_games_model.csv')
  # Supongamos que tienes un DataFrame llamado df_filtrado con las columnas price, early_access, sentiment, metascore, year, month y day


@app.get('/genero/{anio}')
def genero(anio: str):
    # Filtrar el DataFrame por el año proporcionado
    df_filtered = df[df['anio'] == int(anio)]

    # Unir todas las listas de géneros en una sola lista y eliminar valores nulos
    all_genres = [genre for sublist in df_filtered['genres'].dropna() for genre in sublist]

    # Contar la cantidad de veces que aparece cada género, incluyendo los nulos
    genre_counts = pd.Series(all_genres).value_counts()

    # Obtener los 5 géneros más vendidos
    top_5_genres = genre_counts.index[:5].tolist()

    return top_5_genres

@app.get('/juegos/{año}')
def juegos(Año: str):
    # Filtrar el DataFrame por el año proporcionado
    df_filtered = df[df['anio'] == int(Año)]

    # Obtener la lista de nombres de los juegos lanzados en el año
    juegos_lanzados = df_filtered['app_name'].tolist()

    return juegos_lanzados

@app.get('/specs/{año}')
def specs(Año: str):
    # Filtrar el DataFrame por el año proporcionado
    df_filtered = df[df['anio'] == int(Año)]

    # Unir todas las listas de géneros en una sola lista y eliminar valores nulos
    all_specs = [specs for sublist in df_filtered['specs'].dropna() for specs in sublist]

    # Contar la cantidad de veces que aparece cada género, incluyendo los nulos
    specs_counts = pd.Series(all_specs).value_counts()

    # Obtener los 5 géneros más vendidos
    top_5_specs = specs_counts.index[:5].tolist()

    return top_5_specs


@app.get('/earlyacces/{año}')
def earlyacces(Año: str):
    
    # Filtrar el DataFrame por el año proporcionado y que tenga "early access"
    df_filtered = df[(df['anio'] == int(Año)) & (df['early_access'] == True)]

    # Contar la cantidad de juegos con "early access" lanzados en el año
    cantidad_early_access = len(df_filtered)

    return cantidad_early_access


@app.get('/sentiment/{año}')
def sentiment(Año: str):
    # Filtrar el DataFrame por el año proporcionado
    df_filtered = df[df['anio'] == int(Año) ]

    # Contar la cantidad de registros con cada análisis de sentimiento para el año
    sentiment_counts = df_filtered['sentiment'].value_counts().to_dict()

    return sentiment_counts

@app.get('/metascore/{año}')
def metascore(Año: str):
    # Filtrar el DataFrame por el año proporcionado y eliminar filas con valores nulos en 'metascore'
    df_filtered = df[(df['anio'] == int(Año)) & ~df['metascore'].isnull()]

    # Ordenar el DataFrame filtrado por Metascore de manera descendente
    df_sorted = df_filtered.sort_values(by='metascore', ascending=False)

    # Obtener los nombres de los 5 juegos con el mayor Metascore
    top_5_juegos = df_sorted['app_name'].head(5).tolist()

    return top_5_juegos

@app.get('/prediccion/{year}/{genero}/{metascore}')
def prediccion(year, genero, metascore):


    # Supongamos que tienes un DataFrame llamado df_filtrado con las columnas price, early_access, sentiment, metascore, year, month y day

    # Seleccionar las variables independientes (predictores) y la variable dependiente (precio)
    y = df_filtrado['price']
    X = df_filtrado.drop(columns=['price'])


    # Dividir el conjunto de datos en datos de entrenamiento y datos de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear características polinómicas de grado 2
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Crear y entrenar el modelo de regresión lineal con características polinómicas
    poly_regression_model = LinearRegression()
    poly_regression_model.fit(X_train_poly, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred_poly = poly_regression_model.predict(X_test_poly)

    mse_poly = mean_squared_error(y_test, y_pred_poly)
    rmse_poly = (mse_poly ** 0.5)


    available_genres = ['Indie', 'Early Access', 'Massively Multiplayer', 'Strategy', 'RPG', 'Action', 'Casual', 'Free to Play', 'Racing', 'Adventure', 'Simulation', 'Sports']
    assert genero in available_genres, f"El género ingresado '{genero}' no es válido. Los géneros válidos son: {', '.join(available_genres)}"

    # Resto del código para realizar la predicción...

    # Creamos el DataFrame X_new con las características para el nuevo producto
    new_data = {
        'metascore': [metascore],
        'year': [year],
        'Indie': [0],
        'Early Access': [0],
        'Massively Multiplayer': [0],
        'Strategy': [0],
        'RPG': [0],
        'Action': [0],
        'Casual': [0],
        'Free to Play': [0],
        'Racing': [0],
        'Adventure': [0],
        'Simulation': [0],
        'Sports': [0]
    }

    X_new = pd.DataFrame(new_data)

    X_new[genero] = 1

    # Verificamos si "Free to Play" está presente en la lista de géneros
    if genero == "Free to Play":
        return f"Precio: 0  RMSE del Modelo: {rmse_poly:.2f}"

    # Asegurarse de que las columnas en X_new tengan el mismo orden que en X_train
    X_new = X_new[X_train.columns]

    # Generamos las características polinómicas para X_new usando el mismo objeto 'poly'
    X_new_poly = poly.transform(X_new)

    # Realizamos la predicción de precios para X_new
    y_pred_new = poly_regression_model.predict(X_new_poly)[0]

    return f"Precio: {y_pred_new:.2f}  RMSE del Modelo: {rmse_poly:.2f}"


