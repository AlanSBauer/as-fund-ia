import re

stopwords = ["a", "o", "de", "da", "os", "as", "das", "dos", "um", "uma", "com", "no", "na", "em", "para"]

def tratar_texto(texto, remover_stopwords):
    texto = texto.lower()
    palavras = re.findall(r"[a-záàâãéêíóôõúçü]+", texto)

    if remover_stopwords:
        palavras_filtradas = []

        for palavra in palavras:
            if palavra not in stopwords:
                palavras_filtradas.append(palavra)

        palavras = palavras_filtradas

    return " ".join(palavras)