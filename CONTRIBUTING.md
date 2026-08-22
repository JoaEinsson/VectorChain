# Contribuindo com o VectorChain

VectorChain é um projeto científico. Correção, causalidade e rastreabilidade têm precedência sobre
novidade, performance ou volume de funcionalidades.

## Preparação do ambiente

Instale `uv` e execute, na raiz do repositório:

```powershell
uv sync
uv run pre-commit install
```

O `uv.lock` faz parte do repositório. Depois de gerado, a reprodução de um ambiente existente deve
usar `uv sync --locked`.

## Fluxo de mudança

1. Abra uma issue quando a mudança alterar matemática, API pública ou protocolo experimental.
2. Para uma escolha técnica relevante, registre um ADR em `docs/decisions/`.
3. Implemente a menor mudança que responda à questão proposta.
4. Adicione testes de exemplo e, quando aplicável, testes de propriedade.
5. Execute todos os gates locais.
6. Descreva no pull request o efeito científico, limitações e evidências.

Commits devem ser pequenos e usar preferencialmente os prefixos:

- `feat:` funcionalidade;
- `fix:` correção;
- `test:` testes;
- `docs:` documentação;
- `exp:` protocolo ou execução experimental;
- `refactor:` mudança interna sem alteração intencional de comportamento;
- `chore:` infraestrutura e manutenção.

Não misture alteração da definição matemática com refatoração ampla ou novos experimentos no
mesmo commit.

## Gates obrigatórios

```powershell
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src/vectorchain
uv run pytest --cov=vectorchain --cov-report=term-missing
uv build
```

## Mudanças científicas

Uma mudança científica deve declarar:

- hipótese ou problema que a motivou;
- comportamento anterior e proposto;
- impacto esperado nas métricas;
- novos parâmetros e respectivos defaults;
- evidência de causalidade quando tocar na segmentação;
- riscos, limitações e resultado negativo, se houver.

Não altere fixtures ou resultados de referência apenas para fazer um teste passar. A causa da
mudança deve estar documentada e revisável.

## Dependências

Dependências de runtime só devem ser adicionadas quando forem necessárias ao pacote principal.
Ferramentas de testes e experimentação pertencem ao grupo `dev`; funcionalidades opcionais devem
usar extras. Toda mudança de dependência deve atualizar e versionar o `uv.lock`.

## Artefatos

Saídas de experimentos pertencem a `artifacts/` e não são versionadas. Resultados pequenos,
canônicos e revisados podem ser promovidos para `reports/reference/` junto com configuração,
ambiente e commit de origem.
