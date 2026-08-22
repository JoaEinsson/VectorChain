# Protocolo de forecasting mínimo

Status: **pré-especificado antes da primeira execução**.

## Pergunta

Com o mesmo contexto temporal e o mesmo regressor downstream, VectorChain mantém erro de previsão
próximo ao raw usando menos passos e menos valores escalares de entrada?

## Dataset e alvo

As sete dinâmicas sintéticas canônicas são geradas uma vez com seeds derivadas e parâmetros
versionados. Para cada origem, o input contém somente a janela inclusiva
`[origin - context_length + 1, origin]`. O alvo comum é:

```text
delta_target = x[origin + horizon] - x[origin]
prediction = x[origin] + predicted_delta
```

O benchmark inicial usa horizonte de um ponto e stride explícito. Não há normalização calculada
sobre a série completa.

## Split sem leakage

O split é definido pelo índice do alvo em cada sinal:

1. treino: primeira fração temporal;
2. validação: fração seguinte, apenas reportada;
3. teste: fração terminal.

O modelo e o scaler são ajustados somente no treino agregado dos sete sinais. Validação não escolhe
hiperparâmetros nesta baseline. Em validação e teste, cada previsão rolling-origin pode usar valores
reais anteriores ao `origin`, mas nunca o alvo atual ou pontos posteriores. O conjunto de IDs,
origens e alvos é idêntico para todas as representações.

## Representações

- `raw`: valores da janela em uma coluna;
- `first_difference`: primeiras diferenças da mesma janela;
- `vectorchain`: transformação causal da janela com tolerância `0.03` transferida da baseline de
  reconstrução e features default configuráveis.

Cada matriz variável é resumida, por coluna e nesta ordem, por `last`, `mean` e `std`. O mesmo
pooling é aplicado a todas. O scaler por coluna do vetor resumido aprende média e desvio padrão
somente no treino; escala constante vira `1`.

## Modelo downstream

Regressão ridge fechada em NumPy, com intercepto não regularizado e `alpha=0.001`. O algoritmo,
alpha, repetições de timing, warm-up e alvo são idênticos. Não há busca de hiperparâmetro, seleção
de features ou ajuste específico por representação.

## Métricas e artefatos

- MAE e RMSE no valor original, globais e por sinal;
- persistência como referência;
- média de passos, valores escalares e bytes do input antes do pooling;
- dimensão pooled, parâmetros e bytes do estado do modelo;
- bytes do design matrix de treino;
- runtime de representação, treino e inferência, com mediana e quartis para os dois últimos;
- todas as previsões, índices temporais, comprimentos e falhas preservados.

## Critério pré-especificado

Separar três perguntas, sem trocar a definição após observar o teste:

- **paridade preditiva:** RMSE VectorChain no teste menor ou igual a `1.10 × RMSE raw`;
- **redução estrutural:** passos médios VectorChain menores ou iguais a 50% dos passos raw;
- **redução de payload:** valores escalares médios VectorChain menores ou iguais aos valores raw.

O sucesso conjunto exige os três critérios. Ganhar em passos e perder em payload, ou manter
payload e degradar previsão, será reportado como tradeoff, não como sucesso.

## Hipóteses e limitações

- F1: o pipeline preservará exatamente os mesmos alvos e splits entre representações.
- F2: diferenças podem ser competitivas porque o alvo também é um incremento.
- F3: VectorChain pode reduzir passos, mas suas múltiplas features podem anular a redução de payload.
- F4: nenhuma representação é presumida superior; persistência pode vencer modelos aprendidos.

Esta rodada tem uma realização por sinal, sinais sintéticos, um horizonte, uma tolerância, um
contexto, um pooling e um alpha. Não há intervalo de confiança nem alegação de generalização.
