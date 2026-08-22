# VectorChain

VectorChain é um MVP científico para investigar se séries temporais escalares podem ser
representadas de forma útil como uma cadeia causal e adaptativa de vetores.

O projeto está em fase **pré-alpha**. A infraestrutura de governança e reprodutibilidade está
sendo estabelecida antes da implementação do algoritmo. Não há, neste momento, resultados que
sustentem superioridade sobre representações tradicionais.

## Pergunta de pesquisa

> Uma representação causal e adaptativa de séries temporais como vetores pode preservar dinâmica
> relevante, reduzir o comprimento efetivo da sequência e ajudar em reconstrução, similaridade,
> forecasting e detecção de mudança?

O escopo original está preservado em [`prompt_inicial.md`](prompt_inicial.md). As definições
executáveis do método serão mantidas em [`docs/specification.md`](docs/specification.md), e a
causalidade em [`docs/causality-contract.md`](docs/causality-contract.md).

## Estado do projeto

- [x] Charter e escopo inicial
- [x] Estrutura de governança e reprodutibilidade
- [ ] Segmentação causal adaptativa
- [ ] Reconstrução e sinais sintéticos
- [ ] Experimento compressão × reconstrução
- [ ] Similaridade, retrieval e forecasting

## Ambiente de desenvolvimento

O projeto usa Python 3.11 ou posterior e `uv` para ambiente e dependências. A versão local
recomendada é Python 3.12.

```powershell
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/vectorchain
```

Consulte [`CONTRIBUTING.md`](CONTRIBUTING.md) para o fluxo completo e
[`docs/reproducibility.md`](docs/reproducibility.md) para reprodução de experimentos.

As proteções que precisam ser habilitadas na interface do GitHub estão listadas em
[`docs/repository-settings.md`](docs/repository-settings.md).

## Licença

VectorChain é distribuído sob a [Apache License 2.0](LICENSE).
