import numpy as np
import pandas as pd
import ast
from fastapi import FastAPI

app = FastAPI()

rows = []
with open("steam_games.json") as f:
    for line in f.readlines():
        rows.append(ast.literal_eval(line))
df = pd.DataFrame(rows)

df['anio'] = df['release_date'].str.extract(r'(\d{4})')
df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce')
df['anio'] = pd.to_numeric(df['anio'], errors='coerce')

@app.get('/genero/{año}')
def genero(año: str):
    # Filtrar el DataFrame por el año proporcionado
    df_filtered = df[df['año'] == int(anio)]

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