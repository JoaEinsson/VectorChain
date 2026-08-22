# Instruções para agentes de desenvolvimento

Estas regras se aplicam a todo o repositório.

## Fontes de verdade

1. `prompt_inicial.md` define objetivo, escopo e prioridades do projeto.
2. `docs/specification.md` define a semântica matemática implementável.
3. `docs/causality-contract.md` define o contrato causal verificável.
4. ADRs aceitos em `docs/decisions/` registram escolhas e justificativas.
5. Código e testes devem implementar essas definições, não substituí-las silenciosamente.

Se duas fontes entrarem em conflito, interrompa a mudança, identifique o conflito e proponha um
ADR. Nunca edite `prompt_inicial.md`; ele é o registro histórico do charter inicial.

## Invariantes

- O caminho principal de segmentação é estritamente online e causal.
- Um ponto futuro nunca pode mudar um segmento já emitido.
- O segmento aberto é estado provisório e deve ser distinguido dos segmentos finalizados.
- Processamento batch deve ser apenas uma conveniência sobre a mesma transição de estado online.
- Reconstrução deve preservar o número e a ordem das amostras.
- Seeds, configurações e definições de métricas nunca podem ficar implícitas em experimentos.
- Resultados negativos devem ser preservados e reportados.

## Regras de implementação

- Priorize correção, causalidade, clareza, reprodutibilidade, simplicidade e performance, nessa
  ordem.
- Use type hints e docstrings nas APIs públicas.
- Não adicione abstrações, dependências ou otimizações sem necessidade demonstrada.
- Não mude defaults, métricas, convenções de fronteira ou unidades sem ADR e testes de regressão.
- Reutilize o estado causal; não mantenha implementações batch e online matematicamente distintas.
- Mantenha NumPy como única dependência obrigatória enquanto for suficiente.
- Não transforme notebooks em fonte canônica de experimentos.

## Verificação

Antes de declarar uma mudança concluída, execute:

```powershell
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src/vectorchain
uv run pytest --cov=vectorchain --cov-report=term-missing
uv build
```

Mudanças causais também exigem testes que alterem arbitrariamente o sufixo de um sinal e comparem
somente os segmentos que já haviam sido emitidos antes do corte.

## Git e artefatos

- Preserve mudanças do usuário e mantenha mudanças não relacionadas fora do diff.
- Faça commits somente quando a tarefa ativa autorizar; nunca faça push automaticamente.
- Não versione `.venv`, caches ou conteúdo gerado sob `artifacts/`.
- Um resultado promovido para `reports/reference/` deve incluir configuração, commit e ambiente.
