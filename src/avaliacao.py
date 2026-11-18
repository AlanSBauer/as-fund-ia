import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def avaliar_modelo(modelo, x_teste, y_teste):
    previstos = modelo.predict(x_teste)
    probabilidades = modelo.predict_proba(x_teste)

    classes = ["negativo", "positivo"]
    matriz = confusion_matrix(y_teste, previstos, labels=classes)

    pd.DataFrame(matriz, index=classes, columns=classes).to_csv("resultados/matriz_confusao.csv")

    relatorio = classification_report(y_teste, previstos, zero_division=0)

    with open("resultados/relatorio_classificacao.txt", "w", encoding="utf-8") as arquivo:
        arquivo.write(relatorio)

    resultados = pd.DataFrame({
        "texto": list(x_teste),
        "real": list(y_teste),
        "previsto": previstos,
        "probabilidade": probabilidades.max(axis=1)
    })

    erros = resultados[resultados["real"] != resultados["previsto"]]
    erros.to_csv("resultados/erros.csv", index=False)

    vetorizador = modelo.named_steps["vetorizador"]
    classificador = modelo.named_steps["classificador"]

    palavras = pd.DataFrame({
        "palavra": vetorizador.get_feature_names_out(),
        "coeficiente": classificador.coef_[0]
    })

    positivas = palavras.nlargest(10, "coeficiente")
    negativas = palavras.nsmallest(10, "coeficiente")

    positivas["sentimento"] = "positivo"
    negativas["sentimento"] = "negativo"

    pd.concat([positivas, negativas]).to_csv("resultados/palavras_influentes.csv", index=False)


def avaliar_casos_dificeis(modelo):
    df = pd.read_csv("data/casos_dificeis.csv")

    previstos = modelo.predict(df["texto"])
    probabilidades = modelo.predict_proba(df["texto"])
    indice = list(modelo.classes_).index("positivo")

    resultados = pd.DataFrame({
        "texto": df["texto"],
        "tipo": df["tipo"],
        "esperado": df["sentimento_esperado"],
        "previsto": previstos,
        "probabilidade_positivo": probabilidades[:, indice]
    })

    resultados.to_csv("resultados/casos_dificeis_predicoes.csv", index=False)