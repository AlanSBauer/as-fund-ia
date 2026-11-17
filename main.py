import pandas as pd;

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
    arquivo.write(f"Exemplos positivos:\n{positivos['texto'].head(3).to_string(index=False)}\n\n")
    arquivo.write(f"Exemplos negativos:\n{negativos['texto'].head(3).to_string(index=False)}\n\n")
    