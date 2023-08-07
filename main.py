# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
import ast
from fastapi import FastAPI, HTTPException
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

# Crear una instancia de la aplicación FastAPI
app = FastAPI(title="Proyecto MLOps")

# Leer los datos desde un archivo JSON y almacenarlos en un DataFrame llamado 'df'
rows = []
with open("steam_games.json") as f:
    for line in f.readlines():
        rows.append(ast.literal_eval(line))
df = pd.DataFrame(rows)

# Procesar el DataFrame para extraer información relevante y almacenarla en 'df_filtrado'
df = df[df['release_date'].str.contains(r'\d{4}-\d{2}-\d{2}', na=False)]
df['anio'] = df['release_date'].str.extract(r'(\d{4})')
df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce')
df['anio'] = pd.to_numeric(df['anio'], errors='coerce')

#Leer el csv para instanciar el modelo
df_filtrado = pd.read_csv(r'steam_games_model.csv')

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

# Calcular el Mean Squared Error (MSE) y el Root Mean Squared Error (RMSE)
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = (mse_poly ** 0.5)

# Endpoint para obtener los géneros más populares en un año específico
@app.get('/generos/{anio}')
def genero(anio: int):
    # Obtener la lista de años únicos con registros de géneros
    anio_unico_genero = sorted(df[df['genres'].notnull()]['anio'].unique())
    
    # Verificar si el año está presente en la lista de años con registros de géneros
    if anio not in anio_unico_genero:
        return {"mensaje": f"El año ingresado no tiene registros de géneros disponibles. Los años con registros de géneros son: {anio_unico_genero}"}

    # Verificar si hay registros disponibles para el año ingresado
    if df[df['anio'] == anio].empty:
        return {"mensaje": f"No hay registros disponibles para el año {anio}"}

    # Unir todas las listas de géneros en una sola lista y eliminar valores nulos
    all_genres = [genre for sublist in df[df['anio'] == anio]['genres'].dropna() for genre in sublist]

    # Contar la cantidad de veces que aparece cada género, incluyendo los nulos
    genre_counts = pd.Series(all_genres).value_counts()

    # Obtener los 5 géneros más lanzados
    top_5_genres = genre_counts.index[:5].tolist()

    return {"Top 5 géneros": top_5_genres}

# Endpoint para obtener los juegos lanzados en un año específico
@app.get('/juegos/{anio}')
def juegos(anio: int):
    # Obtener la lista de años únicos con registros de juegos
    anio_unico_juegos = sorted(df[df['app_name'].notnull()]['anio'].unique())
    if anio not in anio_unico_juegos:
        return {"mensaje": f"El año ingresado no tiene registros de juegos disponibles. Los años con registros de juegos son: {anio_unico_juegos}"}

    # Verificar si hay registros disponibles para el año ingresado
    if df[df['anio'] == anio].empty:
        return {"mensaje": f"No hay registros disponibles para el año {anio}"}

    # Obtener la lista de juegos lanzados en el año
    juegos_lanzados = df[df['anio'] == anio]['app_name'].tolist()
    return {"Juegos lanzados en el año": juegos_lanzados}

# Endpoint para obtener las especificaciones más comunes en un año específico
@app.get('/specs/{anio}')
def specs(anio: int):
    # Obtener la lista de años únicos con registros de especificaciones
    anio_unico_specs = sorted(df[df['specs'].notnull()]['anio'].unique())
    if anio not in anio_unico_specs:
        return {"mensaje": f"El año ingresado no tiene registros de especificaciones disponibles. Los años con registros de especificaciones son: {anio_unico_specs}"}

    # Verificar si hay registros disponibles para el año ingresado
    if df[df['anio'] == anio].empty:
        return {"mensaje": f"No hay registros disponibles para el año {anio}"}

    # Unir todas las listas de especificaciones en una sola lista y eliminar valores nulos
    all_specs = [specs for sublist in df[df['anio'] == anio]['specs'].dropna() for specs in sublist]
    specs_counts = pd.Series(all_specs).value_counts()
    top_5_specs = specs_counts.index[:5].tolist()

    return {"Top 5 especificaciones más comunes": top_5_specs}

# Endpoint para obtener la cantidad de juegos con 'early access' en un año específico
@app.get('/earlyacces/{anio}')
def earlyacces(anio: int):
    # Obtener la lista de años únicos con registros de 'early access'
    anio_unico_earlyacces = sorted(df[df['early_access'] == True]['anio'].unique())
    if anio not in anio_unico_earlyacces:
        return {"mensaje": f"El año ingresado no tiene registros de 'early access' disponibles. Los años con registros de 'early access' son: {anio_unico_earlyacces}"}

    # Verificar si hay registros disponibles para el año ingresado
    if df[(df['anio'] == anio) & (df['early_access'] == True)].empty:
        return {"mensaje": f"No hay registros disponibles de 'early access' para el año {anio}"}

    # Contar la cantidad de juegos con 'early access' en el año
    cantidad_early_access = len(df[(df['anio'] == anio) & (df['early_access'] == True)])
    return {"Cantidad de juegos con early access": cantidad_early_access}

# Endpoint para obtener el análisis de sentimiento en un año específico
@app.get('/sentiment/{anio}')
def sentiment(anio: int):
    # Obtener la lista de años únicos con registros de sentimiento
    anio_unico_sentiment = sorted(df[df['sentiment'].notnull()]['anio'].unique())
    if anio not in anio_unico_sentiment:
        return {"mensaje": f"El año ingresado no tiene registros de sentimiento disponibles. Los años con registros de sentimiento son: {anio_unico_sentiment}"}

    # Verificar si hay registros disponibles para el año ingresado
    if df[df['anio'] == anio].empty:
        return {"mensaje": f"No hay registros disponibles para el año {anio}"}

    # Contar los análisis de sentimiento en el año y modificar el diccionario para quitar "user reviews"
    sentiment_counts = df[df['anio'] == anio]['sentiment'].value_counts().to_dict()
    modified_sentiment_counts = {}
    for clave, valor in sentiment_counts.items():
        if 'user reviews' not in clave:
            modified_sentiment_counts[clave] = valor

    return {"Análisis de sentimiento": modified_sentiment_counts}

# Endpoint para obtener los juegos con mayor Metascore en un año específico
@app.get('/metascore/{anio}')
def metascore(anio: int):
    # Obtener la lista de años únicos con registros de Metascore
    anio_unico_metascore = sorted(df[df['metascore'].notnull()]['anio'].unique())
    if anio not in anio_unico_metascore:
        return {"mensaje": f"El año ingresado no tiene registros de Metascore disponibles. Los años con registros de Metascore son: {anio_unico_metascore}"}
    if df[(df['anio'] == anio) & ~df['metascore'].isnull()].empty:
        return {"mensaje": f"No hay registros de Metascore disponibles para el año {anio}"}

    # Filtrar el DataFrame por el año proporcionado y eliminar filas con valores nulos en 'metascore'
    df_filtered = df[(df['anio'] == anio) & ~df['metascore'].isnull()]

    # Ordenar el DataFrame por 'metascore' de manera descendente y obtener los 5 juegos con mayor Metascore
    df_sorted = df_filtered.sort_values(by='metascore', ascending=False)
    top_5_juegos = df_sorted['app_name'].head(5).tolist()

    return {"Top 5 juegos con mayor Metascore": top_5_juegos}


@app.post('/prediccion/{year}/{generos}/{metascore}')
def prediccion(year: int, generos: str, metascore: int):

    input_genres = generos.split(',')
    available_genres = ['Indie', 'Early Access', 'Massively Multiplayer', 'Strategy', 'RPG', 'Action', 'Casual', 'Free to Play', 'Racing', 'Adventure', 'Simulation', 'Sports']

    # Verificar si los géneros ingresados son válidos
    for genero in input_genres:
        if genero.strip() not in available_genres:
            raise HTTPException(status_code=400, detail=f"El género ingresado '{genero}' no es válido. Los géneros válidos son: {', '.join(available_genres)}")

    # Verificar si el Metascore está entre 0 y 100
    if not (0 <= metascore <= 100):
        raise HTTPException(status_code=400, detail="El Metascore debe estar entre 0 y 100.")

    # Crear un nuevo DataFrame 'X_new' con las características para el nuevo producto
    new_data = {
        'metascore': [metascore],
        'year': [year]
    }

    for genre in available_genres:
        new_data[genre] = [1 if genre in input_genres else 0]

    X_new = pd.DataFrame(new_data)

    # Verificar si el género es "Free to Play", en cuyo caso el precio predicho será 0
    if 'Free to Play' in input_genres:
        return {
            "Precio": 0,
            "RMSE del Modelo": round(rmse_poly, 2)
        }

    # Asegurarse de que las columnas en X_new tengan el mismo orden que en X_train
    X_new = X_new[X_train.columns]

    # Generar las características polinómicas para X_new usando el mismo objeto 'poly' que se utilizó en el entrenamiento
    X_new_poly = poly.transform(X_new)

    # Realizar la predicción de precios para X_new utilizando el modelo de regresión lineal con características polinómicas
    y_pred_new = poly_regression_model.predict(X_new_poly)[0]

    # Retornar el resultado de la predicción
    return {
        "Precio": round(y_pred_new, 2),
        "RMSE del Modelo": round(rmse_poly, 2)
    }

