# Changelog

Todas as mudanças relevantes deste projeto serão registradas neste arquivo.

O formato segue [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) e o projeto adotará
[Semantic Versioning](https://semver.org/) a partir das primeiras versões públicas. Durante a fase
`0.x`, mudanças incompatíveis serão documentadas explicitamente.

## [Unreleased]

### Added

- Layout instalável em `src/`, metadados do pacote e lockfile reproduzível com `uv`.
- Contratos de especificação matemática, causalidade e protocolo experimental.
- ADRs para segmentação, artefatos e tooling.
- Gates locais com Ruff, mypy, pytest, coverage e build.
- CI multiplataforma, templates de contribuição e atualização automática de dependências.
- Licença Apache-2.0 e metadados de citação do autor.
- Núcleo de segmentação causal com API online e wrapper batch.
- Projeção configurável de `dt`, `dy`, `theta`, `r`, `delta_theta` e `delta_r`.
- Testes unitários e de propriedades para causalidade, articulação e equivalência batch/stream.
- Reconstrução linear stateful com suporte a ablações e ordem arbitrária de features.
- Métricas públicas MAE, RMSE, fator de compressão e fração retida.
- Atributos pós-ajuste de compressão e erro de reconstrução.
- Sete geradores sintéticos determinísticos com seed ou `numpy.random.Generator` explícito.
- Plot Matplotlib opcional de original, reconstrução, segmentos e pontos de articulação.
- Runner configurável do benchmark compressão × reconstrução com artefatos auditáveis.
- Relatório de referência da baseline causal com métricas brutas, timings, ambiente e figuras.
- Baselines raw, normalizada, diferenças e segmentação linear fixa com features configuráveis.
- Padronização aprendida somente na gallery, DTW normalizado e retrieval nearest-neighbor estável.
- Runner pré-especificado de similaridade com split compartilhado, cinco ablations e ranking completo.
- Relatório de referência de retrieval com réplica determinística e resultados negativos preservados.
