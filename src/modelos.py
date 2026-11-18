import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from src.processamento import tratar_texto

def manter_stopwords(texto):
    return tratar_texto(texto, False)

def remover_stopwords(texto):
    return tratar_texto(texto, True)

def comparar_modelos(x_treino, y_treino):
    configuracoes = [
        ("A", "mantidas", "BoW", CountVectorizer(preprocessor=manter_stopwords, token_pattern=r"(?u)\b\w+\b")),
        ("B", "mantidas", "TF-IDF", TfidfVectorizer(preprocessor=manter_stopwords, token_pattern=r"(?u)\b\w+\b")),
        ("C", "removidas", "BoW", CountVectorizer(preprocessor=remover_stopwords, token_pattern=r"(?u)\b\w+\b")),
        ("D", "removidas", "TF-IDF", TfidfVectorizer(preprocessor=remover_stopwords, token_pattern=r"(?u)\b\w+\b"))
    ]

    resultados = []
    maior_f1 = -1
    melhor_modelo = None
    melhor_config = None

    validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for config, stopwords, tipo, vetorizador in configuracoes:
        modelo = Pipeline([
            ("vetorizador", vetorizador),
            ("classificador", LogisticRegression(max_iter=1000, random_state=42))
        ])

        metricas = cross_validate(modelo, x_treino, y_treino, cv=validacao, scoring=["accuracy", "f1_macro"])

        modelo.fit(x_treino, y_treino)

        media_f1 = metricas["test_f1_macro"].mean()

        resultados.append({
            "config": config,
            "stopwords": stopwords,
            "vetorizador": tipo,
            "vocabulario": len(vetorizador.vocabulary_),
            "acc_cv_media": metricas["test_accuracy"].mean(),
            "acc_cv_desvio": metricas["test_accuracy"].std(),
            "f1_cv_media": media_f1,
            "f1_cv_desvio": metricas["test_f1_macro"].std()
        })

        if media_f1 > maior_f1:
            maior_f1 = media_f1
            melhor_modelo = modelo
            melhor_config = config

    comparacao = pd.DataFrame(resultados)
    comparacao.to_csv("resultados/comparacao.csv", index=False)

    return melhor_modelo, melhor_config