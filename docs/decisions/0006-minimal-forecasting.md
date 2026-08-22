# 0006 — Forecasting de incremento com pooling causal e ridge compartilhado

Status: **Accepted**
Data: 2026-08-22

## Contexto

Raw values, diferenças e VectorChain produzem sequências com comprimentos e dimensões diferentes.
Uma regressão linear sobre arrays achatados exigiria padding ou truncamento desigual, enquanto
prever o próximo vetor faria cada representação usar um horizonte temporal diferente. O primeiro
benchmark precisa manter alvo, split e algoritmo downstream iguais sem introduzir uma rede neural.

## Decisão

- Usar janelas de contexto com o mesmo intervalo de tempo bruto e prever o incremento futuro
  `x[target] - x[origin]` para todas as representações.
- Transformar cada janela separadamente pelo caminho causal; nenhum ponto após `origin` participa
  do input.
- Converter a sequência variável em um vetor fixo aplicando, por coluna, as mesmas estatísticas
  ordenadas: último valor, média e desvio padrão populacional.
- Ajustar média e escala dos inputs exclusivamente nos exemplos de treino.
- Usar a mesma regressão ridge, mesmo `alpha`, mesmo alvo e nenhum tuning por representação.
- Separar temporalmente treino, validação e teste pelo índice do alvo. Validação e teste são
  rolling-origin sem refit e podem usar observações anteriores já disponíveis.
- Reportar passos, valores escalares e bytes da representação antes do pooling, além da dimensão e
  memória do modelo, porque um vetor VectorChain possui várias features.
- Comparar também com persistência (`prediction = x[origin]`) apenas como referência sem treino;
  ela não é tratada como uma quarta representação equivalente.

## Consequências

- O horizonte no domínio original é idêntico e diretamente comparável.
- O modelo downstream e seu hiperparâmetro são iguais, mas o número de coeficientes varia com o
  número de features da representação; essa diferença fica explícita no relatório.
- O pooling é simples e auditável, mas pode remover ordem interna relevante. O resultado mede o
  conjunto representação + pooling registrado, não todo uso possível do VectorChain.
- Finalizar cada janela no `origin` usa somente o prefixo observado, porém os vetores terminais de
  janelas sobrepostas são representações provisórias diferentes; não são segmentos históricos
  imutáveis de um único stream contínuo.
- Prever incremento permite reconstruir o valor com o último valor observado sem dar ao modelo de
  diferenças ou VectorChain acesso privilegiado a offset absoluto.

## Alternativas consideradas

- **Autoregressão do próximo token:** rejeitada porque tokens VectorChain cobrem durações variáveis.
- **Padding e flatten:** rejeitada por favorecer comprimentos específicos e aumentar capacidade de
  maneira oculta.
- **Reamostragem dos vetores na grade bruta:** rejeitada porque elimina a redução de sequência que o
  experimento pretende medir.
- **GRU/LSTM:** adiada; adicionaria capacidade e tuning antes de estabelecer uma baseline linear.
