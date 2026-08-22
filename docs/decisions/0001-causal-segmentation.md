# 0001 — Segmentação causal por corda e resíduo máximo

Status: **Accepted**
Data: 2026-08-22

## Contexto

O MVP precisa representar cada segmento pelos próprios endpoints e reconstruí-lo por interpolação
linear. Uma regressão OLS pode minimizar erro contra uma linha diferente daquela definida pelo vetor,
fazendo a tolerância de segmentação perder relação direta com a reconstrução.

Também é necessário preservar continuidade quando um novo ponto viola a tolerância.

## Decisão

- Ajustar o segmento candidato pela corda entre primeiro e último ponto.
- Usar o maior resíduo absoluto como erro de decisão.
- Ao receber uma violação em `t`, emitir o segmento terminado em `t - 1`.
- Iniciar o próximo segmento em `(t - 1, t)`, compartilhando a articulação.
- Distinguir segmento emitido, segmento aberto e fechamento terminal.
- Construir o modo batch sobre a mesma transição online.

## Consequências

- A regra é simples, causal e alinhada à reconstrução.
- `tolerance` possui interpretação direta no domínio vertical do sinal.
- Outliers isolados podem criar segmentos curtos.
- A escala do sinal afeta o resultado e deve ser explícita nos experimentos.
- `min_segment_length > 2` pode forçar aceitação acima da tolerância.

## Alternativas consideradas

- **OLS incremental:** minimiza SSE, mas a linha ajustada não coincide necessariamente com o vetor de
  endpoints.
- **RMSE da corda:** menos sensível a outliers, porém permite erros locais grandes.
- **Segmentação offline ótima:** útil como baseline futura, mas inadequada ao caminho causal principal.
