import numpy as np
import pandas as pd
import ast
from fastapi import FastAPI

app = FastAPI(title = "Proyecto MLOps")

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
    df_filtered = df[df['anio'] == int(año)]

    # Unir todas las listas de géneros en una sola lista y eliminar valores nulos
    all_genres = [genre for sublist in df_filtered['genres'].dropna() for genre in sublist]

    # Contar la cantidad de veces que aparece cada género, incluyendo los nulos
    genre_counts = pd.Series(all_genres).value_counts()

    # Obtener los 5 géneros más vendidos
    top_5_genres = genre_counts.index[:5].tolist()

    return top_5_genres

