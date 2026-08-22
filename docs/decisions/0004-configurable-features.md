# 0004 — Projeção configurável de features após a segmentação

Status: **Accepted**
Data: 2026-08-22

## Contexto

As ablations precisam comparar subconjuntos de propriedades geométricas sem duplicar ou modificar a
lógica causal. Se a seleção de features participar da segmentação, duas ablations podem produzir
fronteiras diferentes e deixar de medir apenas o valor adicional de cada feature.

Também é necessário manter `inverse_transform` viável no MVP, o que requer preservar duração e
deslocamento vertical de cada vetor.

## Decisão

- Separar o pipeline em segmentação causal, cálculo geométrico e projeção de colunas.
- Aceitar `features` como sequência explícita no construtor de `VectorChain`.
- Usar `("dt", "dy", "theta", "r", "delta_theta")` como default imutável.
- Suportar `dt`, `dy`, `theta`, `r`, `delta_theta` e `delta_r`.
- Exigir `dt` e `dy` em toda seleção durante o primeiro MVP.
- Rejeitar nomes desconhecidos, duplicados ou uma string isolada.
- Preservar a ordem solicitada em `vectors_` e expô-la por `feature_names_`.
- Definir `delta_theta = 0.0` e `delta_r = 0.0` para o primeiro vetor.
- Garantir por teste que seleções diferentes produzem as mesmas fronteiras.

## Consequências

- Ablations alteram somente a matriz downstream.
- O número de colunas de `vectors_` passa a depender da configuração.
- A reconstrução continua possível porque `dt` e `dy` são obrigatórios.
- Features derivadas podem ser adicionadas sem tocar na máquina de estado, desde que recebam ADR e
  testes.
- Uma API futura poderá reconstruir a partir de estado estrutural separado e então relaxar a
  obrigatoriedade de `dt` e `dy`.

## Alternativas consideradas

- **Sempre calcular e retornar todas as features:** simples, mas obriga experimentos a conhecer e
  recortar colunas internamente.
- **Permitir qualquer subconjunto:** flexível, porém permite matrizes que não sustentam a API de
  reconstrução planejada.
- **Feature selection dentro do segmentador:** mistura representação com critério de corte e invalida
  comparações controladas.
