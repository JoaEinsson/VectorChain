# 0005 — Similaridade sequencial com padronização da gallery e DTW normalizado

Status: **Accepted**
Data: 2026-08-22

## Contexto

Raw values, diferenças, segmentação fixa e VectorChain produzem sequências com comprimentos e
números de features diferentes. Achatá-las ou completar com zeros favoreceria algumas
representações por construção. Também é necessário evitar que escalas como `dt` e `r` dominem o
custo local sem relação com utilidade para retrieval.

## Decisão

- Representar toda entrada como matriz `(n_steps, n_features)` em `float64`.
- Ajustar média e desvio padrão por coluna usando somente a gallery de cada representação.
- Aplicar a mesma transformação às queries; colunas constantes usam escala `1`.
- Comparar sequências por DTW com banda Sakoe–Chiba explícita.
- Usar RMS entre features como custo local e dividir o custo acumulado pelo comprimento do caminho.
- Resolver empates de vizinhos pelo índice original da gallery para determinismo.
- Usar exatamente o mesmo split de gallery/query para todas as representações e ablations.

O DTW normalizado não é tratado como métrica matemática universal. Ele é uma regra operacional
única e auditável para este primeiro benchmark de retrieval.

## Consequências

- Sequências adaptativas podem ser comparadas sem padding ou truncamento.
- A gallery define a escala sem consultar queries, evitando leakage de pré-processamento.
- DTW em Python possui custo computacional quadrático na largura efetiva da banda; a primeira
  baseline usa séries e conjuntos pequenos.
- Diferenças no número de features não alteram automaticamente a magnitude do custo local.
- Resultados continuam dependentes da banda, tolerância e composição da gallery.

## Alternativas consideradas

- **Flatten + Euclidean:** simples, mas exige comprimentos iguais e ignora alinhamento.
- **Padding:** introduz artefatos e favorece sequências curtas.
- **Resumo estatístico global:** barato, porém remove a ordem que a representação pretende preservar.
- **Modelo aprendido:** prematuro e confunde qualidade da representação com capacidade do modelo.
