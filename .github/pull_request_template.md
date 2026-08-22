## Objetivo

Descreva o problema e a menor mudança proposta.

## Evidência

Inclua testes, métricas ou reprodução que sustentem a mudança. Para resultados experimentais,
informe configuração, seeds e run id.

## Impacto científico

- [ ] Não altera definições matemáticas, causalidade ou métricas.
- [ ] Alterações científicas estão registradas em um ADR.
- [ ] Não há uso de informação futura no caminho causal.
- [ ] Resultados negativos ou limitações relevantes foram preservados.

## Qualidade

- [ ] `uv lock --check`
- [ ] `uv run ruff format --check .`
- [ ] `uv run ruff check .`
- [ ] `uv run mypy src/vectorchain`
- [ ] `uv run pytest --cov=vectorchain --cov-report=term-missing`
- [ ] `uv build`
- [ ] Documentação e changelog foram atualizados quando necessário.

## Risco e reversão

Descreva comportamento incompatível, risco numérico e como reverter a mudança.
