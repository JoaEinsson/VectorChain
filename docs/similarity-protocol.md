# Protocolo de similaridade, ablations e retrieval

Status: **pré-especificado antes da primeira execução**.

## Objetivo

Testar se representações de instâncias da mesma dinâmica permanecem próximas sob mudanças de
amplitude, offset e ruído, e se isso melhora retrieval nearest-neighbor em relação a transformações
simples.

## Representações

Todas retornam matrizes `(n_steps, n_features)`:

1. `raw`: valores em uma coluna.
2. `normalized_raw`: z-score por série; série constante vira zeros.
3. `first_difference`: primeira diferença em uma coluna.
4. `first_second_difference`: primeira e segunda diferenças alinhadas a partir do terceiro ponto.
5. `fixed_linear`: vetores de segmentos articulados com duração fixa.
6. VectorChain nas cinco ablations já definidas, de `(dt, dy)` até a inclusão de `delta_r`.

A segmentação fixa compartilha endpoints e usa exatamente as mesmas fórmulas de features do núcleo,
mas suas fronteiras são predeterminadas e não dependem dos valores futuros.

## Comparação

Para cada representação, média e escala por coluna são ajustadas somente na gallery. Query e gallery
são então comparadas pela distância definida no ADR 0005. A banda DTW, tolerância VectorChain,
duração fixa e todas as variantes dos sinais pertencem à configuração versionada.

## Dataset e split

As sete dinâmicas sintéticas formam as classes. Gallery e query usam seeds e combinações distintas
de amplitude, offset e ruído, mas preservam os parâmetros que definem a classe dinâmica. Não há
instância idêntica nos dois lados.

A tolerância inicial de `0.03` vem do compromisso uniforme observado no experimento de reconstrução,
sem consultar os rótulos deste retrieval. Isso deve ser declarado como transferência de uma decisão
anterior, não como tuning independente.

## Métricas

- top-1 accuracy;
- top-3 accuracy;
- mean reciprocal rank (MRR) do primeiro item da classe correta;
- distância média query–gallery dentro da classe;
- distância média entre classes;
- razão de separação `between / within`;
- comprimento médio das sequências;
- runtime total de representação e matriz de distâncias.

Empates são resolvidos de maneira estável pelo índice da gallery. O ranking completo é preservado.

## Hipóteses

- S1: `normalized_raw` deve remover amplitude e offset melhor que `raw`.
- S2: representações baseadas em diferenças devem ser invariantes a offset, mas não necessariamente a
  amplitude.
- S3: ao menos uma ablation VectorChain deve superar `(dt, dy)` se propriedades geométricas
  adicionarem informação útil.
- S4: nenhuma representação é presumida superior; empates e resultados negativos serão reportados.

## Limitações planejadas

- Primeiro conjunto pequeno e inteiramente sintético.
- Sem variação temporal explícita nesta rodada.
- Uma configuração de DTW e uma tolerância VectorChain.
- Sem tuning por representação e sem modelo aprendido.
