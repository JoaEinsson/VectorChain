# Reprodutibilidade

## Ambiente

O projeto declara dependências amplas no `pyproject.toml` e versões resolvidas no `uv.lock`. O
lockfile é versionado e não deve ser editado manualmente.

Preparação reproduzível:

```powershell
uv sync --locked
uv run pytest
```

Para desenvolvimento de dependências, execute `uv lock` explicitamente e revise o diff antes de
versioná-lo.

## Configurações

Configurações canônicas ficam em `configs/` e são imutáveis para um resultado publicado. Se uma
configuração precisar mudar depois de usada, crie uma nova configuração ou registre claramente uma
nova versão experimental.

Defaults do código não substituem o registro da configuração efetiva. O experimento deve serializar
todos os valores efetivamente usados em `config.json`, inclusive defaults.

## Identidade de uma execução

O run id recomendado é:

```text
<UTC timestamp>_<config-hash-8>_<git-sha-7>
```

Cada `environment.json` deve conter:

- run id;
- horário UTC;
- commit e indicação de worktree suja;
- versão do Python;
- plataforma e arquitetura;
- versões de VectorChain, NumPy e Matplotlib;
- configuração integral;
- seeds;
- comando executado.

O primeiro benchmark é reproduzido, a partir da raiz do clone, com:

```powershell
uv run python experiments/01_reconstruction.py --config configs/reconstruction/baseline.toml
```

O runner também grava `timings.csv` com cada repetição e `manifest.json` com tamanho e SHA-256 de
cada arquivo do run. `metrics.csv` contém mediana e quartis sem tratar repetições de runtime como
novas observações estatísticas do sinal.

Se a árvore estiver suja, o experimento pode rodar, mas deve registrar esse fato e não pode ser
promovido a resultado de referência sem preservar o diff correspondente.

## Aleatoriedade

- Use `numpy.random.Generator`, criado com seed explícita.
- Não use `numpy.random.seed` como estado global em código de biblioteca.
- Derive seeds de repetições de maneira determinística e registre a lista produzida.
- Não reutilize a mesma realização de ruído como se fosse uma repetição estatística independente.

## Performance

- Use `time.perf_counter`.
- Faça um warm-up quando bibliotecas ou caches puderem afetar a primeira execução.
- Registre repetições individuais e reporte mediana e intervalo interquartil.
- Não compare timings obtidos em máquinas diferentes como se fossem equivalentes.

## Promoção de resultados

Dados brutos são gravados em `artifacts/`, que é ignorado pelo Git. Uma execução pode ser promovida
para `reports/reference/` somente se:

1. usar um commit identificável e uma configuração versionada;
2. passar todos os testes e gates;
3. conter ambiente e seeds;
4. reproduzir dentro das tolerâncias documentadas;
5. incluir resultados negativos e falhas relevantes.
