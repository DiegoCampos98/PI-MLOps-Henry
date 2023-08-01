import numpy as np
import pandas as pd
import ast
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse


rows = []
with open("steam_games.json") as f:
    for line in f.readlines():
        rows.append(ast.literal_eval(line))
df = pd.DataFrame(rows)

df['anio'] = df['release_date'].str.extract(r'(\d{4})')
df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce')
df['anio'] = pd.to_numeric(df['anio'], errors='coerce')

# Creamos la página de inicio de la api
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    output = """
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Poppins Light', sans-serif;
            }
            h1 {
                font-size: 36px;
                font-weight: bold;
            }
            p {
                font-size: 24px;
            }
            ol li {
                font-size: 22px;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <div style="display:flex; flex-direction:row;">
            <div style="display:flex; flex-direction:row; background-color:#f2f2f2;">
                <div style="display:flex; flex-direction:column; margin-left:20px;">
                    <h1>¡Te damos la bienvenida a nuestra plataforma en línea, donde podrás realizar consultas sobre películas y series de diversas plataformas, como Netflix, Amazon, Hulu y Disney!</h1>
                    <p>Aquí encontrarás 7 diferentes tipos de búsquedas disponibles:</p>
                    <ol>
                        <li>Película con mayor duración según año, plataforma y tipo de duración.<br>
                            (<a href="https://mlops-render.onrender.com/get_max_duration/2020/netflix/min"> Ejemplo1</a>)</li>
                        <li>Cantidad de películas según plataforma, con un puntaje mayor a XX en determinado año.<br>
                             (<a href="https://mlops-render.onrender.com/get_score_count/disney/3/2020"> Ejemplo2</a>)</li>
                        <li>Cantidad de películas según plataforma.<br>
                             (<a href="https://mlops-render.onrender.com/get_count_platform/amazon"> Ejemplo3</a>)</li>
                        <li>Actor que más se repite según plataforma y año.<br>
                             (<a href="https://mlops-render.onrender.com/get_actor/disney/1999"> Ejemplo4</a>)</li>
                        <li>La cantidad de contenidos que se publicó por país y año.<br>
                           (<a href="https://mlops-render.onrender.com/prod_per_country/movie/argentina/2020"> Ejemplo5</a>)</li>
                        <li>La cantidad total de contenidos según el rating de audiencia dado.<br>
                             (<a href="https://mlops-render.onrender.com/get_contents/13+"> Ejemplo6</a>)</li>
                        <li>Modelo de recomendación de películas.<br>
                            (<a href="https://mlops-render.onrender.com/get_recomendation/finding nemo"> Ejemplo7</a>)</li>
                    </ol>
                    <p>En el archivo README.md del repositorio de GitHub (<a href="https://github.com/MatyTrova/PI-MLOps">https://github.com/MatyTrova/PI-MLOps</a>), encontrarás información detallada sobre el formato de búsqueda que debes seguir para cada una de las consultas disponibles.</p>
                     <p>Proyecto desarrollado por: Matias Trovatto</p>
                </div>
                <img src="https://raw.githubusercontent.com/MatyTrova/PI-MLOps/main/imgs/michael.jpg" width="420" height="315">
            </div>
        </div>
    </body>
"""
    return HTMLResponse(content=output)

# Se desarrollan las consultas que fueron solicitadas por el cliente:

# Consulta 1: Película (sólo película, no serie, ni documentales, etc) con mayor duración según año, plataforma y tipo de duración. 
# La función debe llamarse get_max_duration(year, platform, duration_type) y debe devolver sólo el string del nombre de la película.

@app.get('/get_genero/{anio}')
def genero(Anio: str):
    
    # Filtrar el DataFrame por el año proporcionado
    df_filtered = df[df['anio'] == int(Anio)]

    # Unir todas las listas de géneros en una sola lista y eliminar valores nulos
    all_genres = [genre for sublist in df_filtered['genres'].dropna() for genre in sublist]

    # Contar la cantidad de veces que aparece cada género, incluyendo los nulos
    genre_counts = pd.Series(all_genres).value_counts()

    # Obtener los 5 géneros más vendidos
    top_5_genres = genre_counts.index[:5].tolist()

    return top_5_genres