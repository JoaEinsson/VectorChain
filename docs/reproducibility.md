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

O benchmark de similaridade e ablations usa um split único e configuração independente:

```powershell
uv run python experiments/02_similarity.py --config configs/similarity/baseline.toml
```

O forecasting mínimo reproduzível é executado com:

```powershell
uv run python experiments/04_forecasting.py --config configs/forecasting/baseline.toml
```

A grade pré-especificada de robustez usa a baseline acima como configuração base:

```powershell
uv run python experiments/05_forecasting_robustness.py --config configs/forecasting/robustness.toml
```

A ablation cinemática pré-especificada reutiliza a baseline, fixa a tolerância exploratória e
compara as features sobre os mesmos exemplos:

```powershell
uv run python experiments/06_forecasting_feature_ablation.py --config configs/forecasting/feature_ablation.toml
```

O runner também grava `timings.csv` com cada repetição e `manifest.json` com tamanho e SHA-256 de
cada arquivo do run. `metrics.csv` contém mediana e quartis sem tratar repetições de runtime como
novas observações estatísticas do sinal.

No benchmark de similaridade, `samples.csv` é a fonte auditável do split e das seeds. Cada
representação precisa conter exatamente esses mesmos IDs em `sequences.csv`; o scaler por coluna é
ajustado somente nas sequências da gallery. `neighbors.csv` preserva todos os ranks, inclusive
empates e erros de classificação, em vez de armazenar somente os acertos.

No forecasting, `examples.csv` registra `context_start`, `origin` e `target_index`. O input termina
em `origin`, o alvo ocorre em `origin + horizon` e o split usa exclusivamente o índice do alvo. Os
IDs em `inputs.csv` devem coincidir entre representações. Scaler e ridge são ajustados apenas nas
linhas marcadas como treino; validação e teste não provocam refit.

Na robustez, a seed do conjunto completo de sinais é a unidade de réplica. `conditions.csv` contém
as métricas pareadas seed a seed e `summary.csv` agrega exatamente essas seeds. Contagens de
janelas não devem ser apresentadas como tamanho amostral da análise de robustez.

Na ablation cinemática, `step_audit.csv` deve ter a mesma assinatura por
`(seed, contexto, horizonte)` em todas as variantes. `summary_by_seed.csv` contém a unidade de
réplica usada pelo gate; `gate.json` separa a decisão científica do status de execução registrado
em `environment.json`.

## Programa pós-MVP

Experimentos destinados a sustentar claims seguem
[`post-mvp-claim-protocol.md`](post-mvp-claim-protocol.md). Além das regras anteriores, cada etapa
deve registrar:

- nível de claim pretendido e hipótese que pode ser falsificada;
- contraste primário e análises secundárias;
- origem de cada escolha, distinguindo configuração herdada, exploratória e confirmatória;
- orçamento de tuning e quais partições podem influenciá-lo;
- parâmetros e payload downstream para cada representação;
- unidade experimental independente e método de incerteza;
- condição de parada ou redução do claim.

`tolerance=0.1` foi identificada na grade de robustez e deve carregar essa proveniência em toda nova
análise. Repetir a mesma grade não a transforma em configuração confirmatória. Uma confirmação
precisa de unidades não usadas para escolhê-la e não pode alterar seus critérios depois de abrir o
teste.

Uma segunda execução no mesmo commit e ambiente é reprodução computacional. Replicação
independente requer, no mínimo, execução por outro operador ou implementação/ambiente separado e,
preferencialmente, novos dados. Relatórios devem usar esses termos sem intercambiá-los.

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

Resultados pós-MVP também devem indicar qual gate do roadmap foi satisfeito e qual formulação de
claim permanece proibida.
