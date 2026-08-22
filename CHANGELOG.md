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
