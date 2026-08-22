# 0007 — Robustez de forecasting por seed e grade fatorial

Status: **Accepted**
Data: 2026-08-22

## Contexto

A baseline mínima observou paridade VectorChain no teste por margem de apenas 0,02 ponto
percentual, enquanto a validação falhou. Janelas rolling-origin se sobrepõem e não podem ser
contadas como réplicas independentes. Também não sabemos se a conclusão depende da seed, horizonte,
contexto ou tolerância.

## Decisão

- Executar uma grade fatorial de cinco seeds, três horizontes, três contextos e três tolerâncias
  VectorChain.
- Manter dataset, sinais, split, pooling, ridge, alpha, features e critérios da baseline; somente os
  eixos registrados na grade podem variar.
- Usar a seed completa do conjunto de sete sinais como unidade de réplica. Janelas servem para
  calcular a métrica de cada seed, não para inflar o número de réplicas.
- Avaliar raw e primeira diferença uma vez por `(seed, horizon, context)`, sem repetir artificialmente
  seus resultados para cada tolerância VectorChain.
- Declarar uma célula `(horizon, context, tolerance)` robusta quando ao menos 80% das seeds — quatro
  de cinco — satisfizerem simultaneamente paridade preditiva, redução estrutural e redução de
  payload no teste.
- Preservar resultados seed a seed e por sinal. O relatório de referência pode omitir previsões
  individuais da grade para não versionar dezenas de megabytes, desde que configuração, condições,
  agregações e falhas permaneçam auditáveis.

## Consequências

- A inferência descritiva usa cinco réplicas independentes de ruído, não milhares de janelas
  correlacionadas.
- A taxa de 80% é operacional e possui resolução de 20 pontos percentuais; não substitui intervalo
  de confiança nem teste estatístico.
- O fatorial permite observar interações, mas não autoriza escolher retrospectivamente a melhor
  célula como novo default.
- A grade exige 225 avaliações de representação: 90 baselines e 135 VectorChain.
- O stride passa a 4 para limitar custo e correlação entre janelas; isso torna a robustez uma
  condição separada da baseline de stride 2.

## Alternativas consideradas

- **Tratar janelas como réplicas:** rejeitada por pseudorreplicação.
- **Variar um eixo por vez:** mais barato, mas não mede interações entre horizonte, contexto e
  tolerância.
- **Selecionar pelo melhor teste:** rejeitada por leakage de análise e viés de seleção.
- **Trinta seeds em uma única configuração:** útil depois, mas não testa sensibilidade aos três
  parâmetros que motivaram esta etapa.
