# Protocolo da ablation cinemática no forecasting

Status: **pré-especificado antes da primeira execução da Etapa 7**.

Implementação registrada em `experiments/06_forecasting_feature_ablation.py` e configuração
congelada em `configs/forecasting/feature_ablation.toml`. A execução canônica deve partir de commit
limpo com:

```powershell
uv run python experiments/06_forecasting_feature_ablation.py --config configs/forecasting/feature_ablation.toml
```

## Pergunta

Adicionar relações entre segmentos melhora o forecasting além da geometria de cada segmento quando
fronteiras, exemplos, alvo, split, pooling e algoritmo downstream permanecem iguais?

Esta etapa isola mecanismo dentro do benchmark sintético já observado. Ela não é confirmação
externa e não pode, sozinha, sustentar generalização.

## Base congelada

O experimento herda de `configs/forecasting/baseline.toml`:

- sete sinais de 1.024 pontos e `noise_std=0.01`;
- split temporal 60%/20%/20%;
- alvo de incremento e avaliação no domínio raw;
- pooling `(last, mean, std)`;
- ridge com `alpha=0.001`;
- segmentação causal com `min_segment_length=2`.

A grade reutiliza as cinco seeds, contextos `32/64/128`, horizontes `1/4/16` e stride 4 do estudo
de robustez. `tolerance=0.1` é fixada porque foi a região exploratória mais favorável; sua seleção
anterior é registrada e impede tratar este estudo como confirmação da tolerância.

## Variantes

| Variante | Features | Features pooled | Parâmetros ridge | Papel |
|---|---|---:|---:|---|
| `segment` | `dt, dy` | 6 | 7 | cadeia adaptativa básica |
| `absolute_geometry` | `dt, dy, theta, r` | 12 | 13 | referência sem relações |
| `turning_matched` | `dt, dy, theta, delta_theta` | 12 | 13 | controle de capacidade substituindo `r` |
| `turning` | `dt, dy, theta, r, delta_theta` | 15 | 16 | contraste primário e default atual |
| `full_relational` | todas as anteriores + `delta_r` | 18 | 19 | análise secundária |

Raw e primeira diferença são avaliados uma vez por seed/célula para contexto, payload e comparação
com os resultados anteriores. Todas as variantes VectorChain usam a mesma tolerância. A
independência de fronteiras em relação às features é uma propriedade normativa e testada; o runner
também exige igualdade da quantidade de segmentos em cada exemplo entre todas as variantes.

## Contrastes e split decisório

O contraste primário é:

```text
RMSE(turning) / RMSE(absolute_geometry)
```

O controle de capacidade é:

```text
RMSE(turning_matched) / RMSE(absolute_geometry)
```

Somente `validation` decide o gate. `test` é reportado integralmente para verificar direção e
heterogeneidade, mas o dataset e seus resultados raw já foram observados nas etapas anteriores.
Nenhum resultado desta rodada será chamado de confirmação independente.

## Unidade de análise

Cada seed gera uma realização conjunta dos sete sinais e é a unidade de réplica. As nove combinações
de contexto/horizonte dentro da seed são medidas repetidas, não nove réplicas independentes.

Para cada seed e variante, o runner calcula a média geométrica das nove razões de RMSE contra
`absolute_geometry`. Por célula, preserva as cinco razões seed a seed.

## Margens pré-especificadas

- melhoria prática primária por condição:
  `RMSE(turning) / RMSE(absolute_geometry) <= 0.99`;
- não degradação do controle pareado por seed:
  média geométrica `RMSE(turning_matched) / RMSE(absolute_geometry) <= 1.00`;
- robustez: critério satisfeito em pelo menos quatro de cinco seeds;
- cobertura mínima: pelo menos metade das nove células, isto é, cinco células robustas.

## Gate da Etapa 7

O gate passa somente se, em validação:

1. pelo menos quatro de cinco seeds tiverem média geométrica primária `<= 0.99`;
2. pelo menos cinco de nove células forem robustas para a margem `<= 0.99`;
3. pelo menos quatro de cinco seeds no controle `turning_matched` tiverem média geométrica `<= 1.00`.

O terceiro item impede atribuir a melhora apenas aos três parâmetros adicionais introduzidos pelo
pooling de `delta_theta`. O gate é operacional e exploratório; n=5 não sustenta intervalo de
confiança estreito.

## Métricas e artefatos

- MAE/RMSE globais e por sinal;
- razões pareadas contra raw e `absolute_geometry`;
- passos, features, elementos escalares, bytes e parâmetros;
- sucesso de paridade/payload contra raw;
- melhoria prática contra a referência;
- resumo por célula, seed e sinal;
- assinatura dos passos por exemplo para auditar invariância estrutural;
- decisão de gate serializada separadamente das tabelas;
- ambiente, configuração efetiva, falhas, hashes e figuras pré-especificadas.

## Interpretação permitida

Se o gate passar, a conclusão é apenas que `delta_theta` merece enfrentar controles externos da
Etapa 8. Se falhar, K2 não avança sob este pooling/ridge/benchmark. `full_relational` permanece
secundária e não pode substituir retrospectivamente `turning` como contraste primário.
