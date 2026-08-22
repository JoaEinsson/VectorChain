# Roadmap

O roadmap é orientado por gates científicos. Concluir código não basta; cada etapa precisa produzir
evidência verificável antes da próxima.

## Etapa 0 — Fundação

- [x] Repositório e remoto Git
- [x] Layout Python em `src/`
- [x] Contrato matemático inicial
- [x] Contrato explícito de causalidade
- [x] Protocolo e política de artefatos
- [x] Configuração local de qualidade e CI
- [x] Lockfile e gates locais validados
- [x] CI executada no GitHub

## Etapa 1 — Núcleo causal

- [x] Estado online com `update`, `finalize` e `reset`
- [x] Wrapper `fit_transform`
- [x] Features configuráveis, canônicas e fronteiras
- [x] Testes de propriedades causais
- [x] Casos constantes, rampas, ruído e entradas inválidas

Gate: batch e stream equivalentes; alteração arbitrária de futuro não modifica segmentos emitidos.

## Etapa 2 — Reconstrução e sintéticos

- [x] `inverse_transform`
- [x] Métricas MAE, RMSE e compressão
- [x] Sete geradores sintéticos determinísticos
- [x] Plot da cadeia articulada

Gate: reconstrução preserva shape, endpoints e continuidade; geradores são reprodutíveis.

## Etapa 3 — Primeiro experimento

- [x] Runner configurável de compressão × reconstrução
- [x] Manifesto de ambiente e resultados brutos
- [x] Curvas e relatório de referência
- [x] Análise de condições de sucesso e falha

Gate: reprodução documentada a partir de clone limpo.

## Etapa 4 — Similaridade e ablations

- [ ] Baselines raw, normalizado, diferenças e segmentação fixa
- [ ] Ablations de features geométricas
- [ ] Invariância e retrieval nearest-neighbor

Gate: mesma divisão de dados e protocolo para todas as representações.

## Etapa 5 — Forecasting mínimo

- [ ] Modelo downstream idêntico para raw, diferenças e VectorChain
- [ ] Separação temporal sem leakage
- [ ] Métricas, comprimento, runtime e memória aproximada

Gate: conclusões atribuíveis à representação, não à diferença de modelo ou tuning.
