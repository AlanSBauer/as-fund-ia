import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.modelos import comparar_modelos
from src.avaliacao import avaliar_modelo, avaliar_casos_dificeis

os.makedirs("resultados", exist_ok=True)

df = pd.read_csv("data/Avaliacoes.csv")

print(len(df))

categorias = df["sentimento"].value_counts()
print(categorias)

avaliacao = df["texto"].str.split().str.len()

df["quantidade_palavras"] = avaliacao

media = df.groupby("sentimento")["quantidade_palavras"].mean()
print(media)

positivos = df[df["sentimento"] == "positivo"]
print(positivos.head(3))

negativos = df[df["sentimento"] == "negativo"]
print(negativos.head(3))

with open("resultados/exploracao.txt", "w", encoding="utf-8") as arquivo:
    arquivo.write(f"Total de avaliações: {len(df)}\n\n")
    arquivo.write(f"Distribuição das classes:\n{categorias}\n\n")
    arquivo.write(f"Média de palavras por sentimento:\n{media}\n\n")
    arquivo.write("Exemplos positivos:\n" + "\n".join(positivos["texto"].head(3)) + "\n\n")
    arquivo.write("Exemplos negativos:\n" + "\n".join(negativos["texto"].head(3)) + "\n\n")
    
x = df["texto"]
y = df["sentimento"]

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, test_size=0.25, stratify=y, random_state=42
)

melhor_modelo, melhor_config = comparar_modelos(x_treino, y_treino)

print("Melhor configuração:", melhor_config)

avaliar_modelo(melhor_modelo, x_teste, y_teste)
avaliar_casos_dificeis(melhor_modelo)