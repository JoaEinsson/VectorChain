# 0003 — Layout src, pyproject e ambiente com uv

Status: **Accepted**
Data: 2026-08-22

## Contexto

O projeto precisa ser instalável, reproduzível e testado contra o pacote real, evitando imports
acidentais da raiz e listas divergentes de dependências.

## Decisão

- Usar `src/vectorchain` e testes externos em `tests/`.
- Centralizar metadados e configurações em `pyproject.toml`.
- Usar Hatchling como backend de build.
- Usar `uv.lock` versionado para resolução exata.
- Fixar Python local em 3.12 e declarar suporte inicial a Python 3.11–3.13.
- Usar Ruff, mypy, pytest, Hypothesis e coverage como gates.

## Consequências

- Testes exigem instalação do pacote, feita automaticamente por `uv sync` e `uv run`.
- O ambiente é reprodutível entre plataformas suportadas pelo lockfile.
- Contribuidores precisam instalar `uv`.
- Alterações de dependências sempre geram um diff explícito no lockfile.

## Alternativas consideradas

- **Layout flat:** menor árvore, mas mais sujeito a imports que não representam a instalação real.
- **requirements.txt manual:** amplamente reconhecido, porém separa metadados e não expressa o grafo
  universal com a mesma fidelidade.
- **Poetry/PDM:** adequados, mas `uv` já está disponível no ambiente e cobre o fluxo necessário.
