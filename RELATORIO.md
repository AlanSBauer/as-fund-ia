# RELATORIO — Análise de Sentimento com PLN

**Nome:** Alan dos Santos Bauer
**Repositório:** as-fund-ia
**Data:** 17/11/2026

> Responda cada pergunta com **4 a 12 linhas**, com suas palavras e usando números dos arquivos de `resultados/`. Apague as instruções em itálico ao entregar.

## Tabela-resumo (preencha com os seus resultados)

| Config | Stopwords | Vetorizador | Vocabulário | Acurácia CV (média ± desvio) | F1 CV (média ± desvio) |
|---|---|---|---|---|---|
| A | mantidas | BoW | | | |
| B | mantidas | TF-IDF | | | |
| C | removidas | BoW | | | |
| D | removidas | TF-IDF | | | |

**Configuração escolhida:** ____ **Acurácia no teste:** ____ **F1 no teste:** ____

---

## 1. Pré-processamento
*Quais decisões você tomou (minúsculas, pontuação, números, stopwords)? Qual o tamanho do vocabulário com e sem remoção de stopwords? O que você fez com a palavra "não" e por quê?*

No meu código foi feito a conversão de todas as palavras para minúsculas, isso evita problemas com palavras com alguma letra maiúscula. Também foi removido números e pontuações, mantendo os acentos, e também foi criado uma lista na parte de processamento, onde foi definido 15 stopwords como "o", "a", "os", "as"... Nos resultados de comparação, houve 412 palavras sem stopwords e 425 com stopwords. Na comparação é feito tanto com stopword quanto sem. A palavra "não" foi mantida pois faz total diferença em uma frase, pois por exemplo se for tirado o não em "não gosto disso", muda totalmente o sentido da frase.

## 2. BoW vs. TF-IDF
*Compare as quatro configurações levando em conta o desvio padrão. As diferenças são relevantes? Qual você escolheu e com base em quê? Explique a diferença conceitual entre as duas representações.*

Na comparação das quatro configurações a "C" foi a escolhida porque apresentou o maior F1 médio na validação cruzada que foi o valor de 84,57%, a segunda melhor foi a A, depois D e B. A diferença entre a C e a A foi bem pequena pois a A foi de 84,01%, mas as diferenças foram pequenas em relação aos desvios. A que apresentou o menor desvio padrão foi a "A" com 7,17%. As diferenças entre Bow e TF-IDF são que Bow conta quantas vezes cada palavra aparece no texto enquanto TF-IDF, atribui pesos nas palavras, porem nas palavras que aparecem mais nas avaliações ele atribui um peso menor.

## 3. Avaliação no teste
*Mostre a matriz de confusão e interprete precisão, recall e F1 de cada classe. Para uma loja, qual erro é mais grave: classificar uma reclamação como elogio ou o contrário? Por que a acurácia é menos enganosa aqui do que no exemplo de fraude da Aula 8?*

Aqui está a matriz:

| Real / Previsto | Negativo | Positivo |
|---|---|---|
| Negativo | 27 | 6 |
| Positivo | 3 | 29 |

Na matriz de confusão, o modelo acertou 27 avaliações negativas e 29 positivas, mas classificou 6 reclamações como elogios e 3 elogios como reclamações. Nos sentimentos negativos tiveram 90% de precisão, 82% de recall e 86% de F1, enquanto nos positivos tiveram 83% de precisão, 91% de recall e 87% de F1. Os negativos só tiveram a precisão maior do que os positivos. Para uma loja acredito que classificar uma reclamação como elogio é um erro grave, pois com isso a loja não teria como saber da reclamação e não mudaria o comportamento por exemplo quanto aquela reclamação... Nesse caso, o nosso dataset está bem balanceado com 130 avaliações positivas e 130 negativas, enquanto em sistema de fraudes, a grande maioria das transações são legítimas, dificultando identificação de fraude.

## 4. Validação cruzada vs. divisão única
*Por que a validação cruzada é mais confiável do que um único conjunto de teste para comparar configurações? Por que a escolha não pode ser feita olhando o teste? O que seria vazamento de dados neste trabalho e como você o evitou?*

Porque na validação cruzada avalia cada configuração 5 vezes, enquanto divisão única 1 vez. Os dados de um teste servem apenas para verificar se ele funciona com dados que ainda não conhece, por isso não deve ser feito olhando o teste. E o vazamento de dados foi evitado utilizando a pipeline, e basicamente é utilizar os dados de teste durante o treinamento. 

## 5. Análise de erros e palavras influentes
*Escolha 3 textos de `erros.csv` e explique por que o modelo pode ter errado. Observe as 10 palavras mais influentes de cada classe: há alguma que não carrega sentimento por si só (ex.: palavra funcional ou de domínio)? O que isso diz sobre o que o modelo realmente aprendeu?*

- "Esperava mais, a qualidade é bem inferior ao que prometeram"
- "Ficou feio na sala, a cor é bem diferente da foto"
- "Cheguei animado, mas me arrependi, tudo estava sujo"

Escolhi essas 3 frases, nas 2 primeiras possui uma palavra "bem", acredito que essa palavra possa ter feito o modelo errar, pois é uma palavra que pode trazer um sentido de "bom" por exemplo. E na terceira a palavra animado acredito que também pode trazer essa mesma ideia de ser bom, por estar animado, isso pode ter ajudado o modelo a errar nesse caso. A palavra "bem" e a palavra "tudo" que tem na terceira frase, também aparecem com um coeficiente positivo nas palavras influentes, "bem" com 0,96 e "tudo" com 0,71... O modelo pode ter pego essas palavras mais influentes e com isso acabou errando nessas situações! Pelo que eu percebi a maioria das palavras influentes são adjetivos positivos ou negativos, porem dinheiro, tudo e bem, não se encaixam, porém o modelo utilizou elas como palavras influentes...

## 6. Casos difíceis e limitações
*Com base em `casos_dificeis_predicoes.csv`, como o modelo se saiu em negação, ironia e frases mistas? Explique por que BoW/TF-IDF têm essa limitação, proponha duas melhorias (cite técnicas vistas na Aula 11) e discuta duas limitações do dataset e um risco de usar este modelo em produção.*

Nos casos de ironias por exemplo, se torna mais dificil, pois tem uma que começa com "Que ótimo" mas termina com "o produto chegou quebrado", nesse caso o modelo identifica a palavra ótimo como positiva e acaba errando pelo fato de ser uma ironia. Em negação o modelo acertou 2 de 4 casos, em ironias 2 de 3 e as duas mistas que tiveram foram classificadas como positivas. A limitação do Bow e TF-IDF não consideram totalmente a ordem das palavras, e para melhorar isso poderia ser usado n-gramas e melhorar o tratamento das negações. A limitação do dataset é que são apenas 260 avaliações. Um risco seria a loja deixar de ver alguma avaliação importante por causa do modelo em produção.

---

## Declaração de uso de IA

Marque uma opção e complete:

- [ ] Não usei IA generativa em nenhuma parte deste trabalho.
- [x] Usei IA generativa **somente** para: Eu vou ser bem sincero, eu usei de forma com que eu fosse fazendo, tirando dúvidas, pedindo exemplos, e tive diversas dificuldades para fazer esse trabalho... Sem isso eu não teria terminado essa avaliação a tempo, e acredito que não teria conseguido fazer! As respostas do relátorio eu fiz tudo com minhas palavras, mas também tive dificuldades e levei bastante tempo para fazer!
