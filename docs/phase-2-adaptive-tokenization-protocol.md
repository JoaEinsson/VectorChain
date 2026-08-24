# Protocolo pré-especificado da Fase II — tokenização causal adaptativa

Status: **pré-especificado antes da implementação e da primeira execução da Etapa 11-A**.

Data de congelamento: 2026-08-23.

Este protocolo é uma reformulação independente motivada pela
[`síntese da Fase I`](phase-1-scientific-synthesis.md). Ele não confirma nem reabre K2, K3, K4 ou
K5. O commit que introduzir este documento funciona como registro temporal interno; não equivale a
registro em um serviço externo de pre-registration.

## 1. Pergunta e hipótese

Pergunta primária:

> Sob um orçamento fixo de comprimento de sequência, uma tokenização causal adaptativa em
> `(dt, dy)` preserva contexto útil para previsão direta melhor que tokenizações fixas, smoothing e
> raw comparáveis?

Hipótese K6:

> Uma sequência ordenada de tokens cartesianos adaptativos obtém menor erro que controles com o
> mesmo número de tokens e permanece não inferior a raw com o mesmo payload escalar, usando no
> máximo metade dos passos de entrada.

K6 trata **token** como unidade operacional de comprimento de sequência. Ela não presume redução
em bytes, informação, parâmetros ou tempo. Todas essas quantidades serão registradas separadamente.

## 2. Escopo e não objetivos

### Incluído na Etapa 11-A

- séries escalares, uniformemente amostradas e finitas;
- transformação online do pacote sem `finalize()` nas origens internas;
- estado cartesiano `(dt, dy)`, sem `theta`, `r` ou relações;
- ordem dos tokens preservada por concatenação, sem pooling;
- previsão direta de incrementos raw em três horizontes;
- ridge linear compartilhado para isolar a representação;
- dados sintéticos novos e seeds novas;
- pareamento por origem, alvo, dimensão, parâmetros, tokens e escalares.

### Fora da Etapa 11-A

- rollout da própria cadeia, que pertence a K5 e já falhou;
- `delta_theta`, `delta_r`, cinemática inversa ou claims relacionais;
- LSTM, GRU, Transformer, foundation model ou busca de arquitetura;
- quantização, símbolos ou ABBA-LSTM;
- timestamps irregulares, missing values e entradas multivariadas;
- datasets reais;
- detecção de mudança como alvo;
- escolha de uma narrativa ou controle depois de abrir o teste.

Modelos neurais e dados externos só podem entrar numa Etapa 11-B se K6 passar integralmente e se um
novo protocolo congelar datasets, versões, exclusões e orçamento antes de inspecionar os alvos de
teste. A passagem de K6 não autoriza automaticamente 11-B.

K7, documentada em [`phase-2-revisable-chain-protocol.md`](phase-2-revisable-chain-protocol.md), é
uma hipótese sintética independente sobre revisão de juntas. Ela usa seeds próprias, não entra no
gate K6 e não pode substituir um resultado negativo desta etapa.

## 3. Unidade causal na origem

Depois de observar `x[t]`, a candidata usa:

```text
H - 1 segmentos mais recentes realmente emitidos
+ 1 prefixo do segmento atualmente aberto em t
= H tokens causais
```

Cada token contém, nesta ordem:

```text
(log1p(dt), dy)
```

`log1p(dt)` é pré-processamento numérico, não uma nova feature geométrica. O último token é sempre o
estado provisório aberto; essa posição torna uma flag adicional desnecessária. Um token aberto deve
ter `dt >= 1`. Origens sem `H - 1` segmentos emitidos anteriores são inelegíveis para todas as
representações, não apenas para a candidata.

Segmentos terminais que só existiriam após `finalize()` são proibidos. Alterar qualquer sufixo
posterior a `t` não pode mudar tokens, elegibilidade ou features na origem `t`.

## 4. Tarefa preditiva

Para cada origem elegível, o alvo multioutput é:

```text
y_h(t) = x[t + h] - x[t]
h in {16, 64, 256}
```

O modelo produz os três horizontes diretamente. Nenhuma previsão é realimentada como input; assim,
a Etapa 11-A testa representação de contexto e não recorrência do estado.

O valor corrente `x[t]` é anexado como âncora escalar compartilhada a toda representação treinada.
Ele não conta como token, mas conta em payload, bytes e parâmetros. Isso evita transformar a
ausência de nível absoluto em desvantagem artificial em processos de reversão à média.

Origens candidatas ocorrem a cada 16 amostras. Uma origem pertence a um split somente quando a
origem e o endpoint de todos os seus alvos pertencem ao mesmo bloco. Contextos podem usar o passado
anterior ao início de validação ou teste, porque esse passado já estaria disponível online.

## 5. Dados novos

Cada série possui `n_points = 16384`. O relógio normalizado é `u=t/(n_points-1)`. A condição
primária adiciona ruído observacional gaussiano com `noise_std=0.01`; `0.0` e `0.05` são análises de
sensibilidade e não entram no gate K6.

As oito famílias abaixo não participam dos relatórios de referência da Fase I:

| ID | Definição congelada antes do ruído observacional |
|---|---|
| `amplitude_modulated_sine` | `(1 + 0.35 sin(2π·3u + 0.2)) sin(2π·48u + 0.4)` |
| `beating_multisine` | `0.65 sin(2π·45u + 0.2) + 0.35 sin(2π·49u - 0.3)` |
| `triangle_chirp` | `(2/π) asin(sin(2π·(12u + 36u²)))` |
| `damped_impulse_train` | `x[0]=0`; `x[t]=0.985x[t-1]+a[t]`; impulsos alternados `±1` a cada 512 pontos com jitter uniforme inteiro `[-64,64]` derivado da seed |
| `seasonal_ar` | `x[t]=0.65x[t-1]-0.20x[t-2]+0.25x[t-64]+ε[t]`, termos com índice negativo iguais a zero, `ε~N(0,0.03²)` |
| `switching_ar` | `x[0]=0`; `x[t]=φ[t]x[t-1]+ε[t]`, blocos de 2048 pontos com `φ=(0.85,-0.55,0.35,0.70,-0.30,0.90,0.20,0.65)`, `ε~N(0,0.03²)` |
| `smoothed_telegraph` | `x[0]=0`; níveis em `{-1,-0.5,0.5,1}` e durações inteiras uniformes `[128,512]`; `x[t]=0.97x[t-1]+0.03level[t]` |
| `trend_volatility` | `x[0]=0`; `x[t]=x[t-1]+slope[t]+ε[t]`, slopes em `{-0.002,-0.001,0.001,0.002}` por blocos uniformes `[256,1024]`, `ε~N(0,σ²)` e `σ` alternando entre `0.005` e `0.03` |

Escolhas aleatórias internas usam streams filhos derivados por SHA-256 de
`<master-seed>|<signal-id>|<noise-label>|<component>`. Não se usa o RNG global.

### Seeds primárias

As cinco seeds foram derivadas dos rótulos `vectorchain-phase2-stage11a-seed-0` a
`vectorchain-phase2-stage11a-seed-4`: SHA-256 do UTF-8, primeiros 64 bits little-endian e remoção do
bit de sinal.

| Rótulo | Seed | SHA-256 completo |
|---|---:|---|
| `seed-0` | `6986215303638781975` | `1790c9d9cd05f4e0cc6a23209466c45cf36f7e5cd7678bbfba41f553f771c6ba` |
| `seed-1` | `8048897723814196844` | `6c3665bdd56db3ef5e57432144b6b675c1a169e42822a6f3c727b90c42e29c97` |
| `seed-2` | `2557661753387433198` | `eebc424ad0a37ea3fb57187177423d2ed3671fc10bfecc16e9868b3737f802d5` |
| `seed-3` | `2033170254424922923` | `2b5be3fe9245379c90ea21244a95c8553884d0f2567b50cd9b2e232da43df0cb` |
| `seed-4` | `3256513826814980976` | `7073745d127631ad2f4e6ff634bc75a444cbb0aba3a425cd542b055286f9f519` |

Os digests completos devem ser copiados para `environment.json`. Essas seeds não podem ser usadas
em calibração manual, snapshots de documentação ou testes de implementação antes da execução
canônica. Testes automatizados usam somente fixtures pequenas e seeds `11` e `22`.

Bootstrap usa seed `7956559162843589467`, derivada de
`vectorchain-phase2-stage11a-bootstrap` pela mesma regra.

## 6. Splits e seleção

O eixo temporal é dividido pelos endpoints dos alvos:

- treino externo: primeiros 50%;
- validação externa: 20% seguintes;
- teste fechado: 30% finais.

Os últimos 20% do bloco de treino formam a validação interna para escolhas permitidas. Nenhuma
estatística de validação externa ou teste participa de tuning, padronização ou pareamento de spans.

### Escolhas permitidas no treino interno

- uma tolerância global por `H`, escolhida entre `0.01`, `0.03` e `0.1`;
- janela do controle de média móvel entre `2`, `4`, `8` e `16`, por família e `H`;
- duração fixa e span PAA derivados do contexto adaptativo de treino, sem otimização no alvo.

Uma tolerância é elegível quando, no treino interno:

1. pelo menos 95% das origens candidatas possuem `H` tokens;
2. a mediana do span raw dos `H` tokens é pelo menos `2H`.

Entre tolerâncias elegíveis, vence a de menor NRMSE multi-horizonte da candidata, agregada por média
geométrica sobre famílias e seeds. Empate numérico até `1e-12` favorece a menor tolerância. Se
nenhuma for elegível, a condição é registrada e o gate estrutural falha; não se amplia a grade.

Depois das escolhas, o ridge é reajustado em todo o treino externo. Validação externa é reportada,
mas não altera configuração. O teste é aberto uma única vez após uma execução de seleção completa,
uma auditoria estrutural aprovada e um commit limpo contendo `selection.json` e seu SHA-256.
Uma vez satisfeita a auditoria estrutural, o teste deve ser aberto mesmo que o resultado preditivo
de validação externa seja desfavorável; isso evita parada opcional baseada na direção do efeito.

## 7. Orçamentos primários

Os contextos avaliados são `H in {8,16,32}`; `H=16` é primário. Todas as representações treinadas
têm exatamente `2H+1` escalares incluindo a âncora `x[t]` e, com três outputs e intercepto,
`3(2H+2)` parâmetros preditivos.

| Representação | Passos | Escalares | Construção |
|---|---:|---:|---|
| `adaptive_cartesian` | `H` | `2H+1` | candidata, tokens `(log1p(dt),dy)` + âncora |
| `fixed_cartesian` | `H` | `2H+1` | blocos fixos globais, mesma semântica cartesiana + âncora |
| `causal_paa` | `H` | `2H+1` | média e deslocamento em bins causais de span pareado + âncora |
| `moving_average_step_matched` | `H` | `2H+1` | primeiro e segundo incremento do sinal suavizado trailing + âncora |
| `raw_step_matched` | `H` | `2H+1` | primeiro e segundo incremento raw por passo + âncora |
| `raw_scalar_matched` | `2H` | `2H+1` | últimos `2H` primeiros incrementos raw + âncora |
| `persistence` | 1 | 1 | prevê incremento zero em todos os horizontes |

Para `fixed_cartesian`, a duração por família e `H` é a mediana inteira de `dt` dos segmentos
adaptativos finalizados no treino após a seleção da tolerância, limitada a pelo menos 1. Para
`causal_paa`, o span é a mediana inteira do span raw total da candidata no treino; ele é dividido
em `H` bins contíguos terminando na origem, com diferenças de duração de no máximo uma amostra.

Os blocos de `fixed_cartesian` são ancorados em `t=0` e têm essa duração congelada. Em uma origem
interna, o bloco corrente é truncado em `t`; se `t` coincide com uma fronteira, o bloco recém
completado é o token mais recente. Em ambos os casos são usados os `H` tokens mais recentes com ao
menos um intervalo observado.

Todas as entradas são achatadas preservando ordem. Cada coluna é padronizada com média e desvio do
treino; desvio zero usa escala 1 e é registrado. O alvo é padronizado por horizonte somente com o
treino e revertido antes das métricas. O ridge usa `alpha=0.001`, sem nova busca.

`fixed_cartesian`, `causal_paa` e os controles raw podem receber informação mais favorável que a
candidata em detalhes específicos; isso é deliberadamente conservador. Rank efetivo, passos raw
necessários, escalares, bytes, parâmetros, runtime e memória aproximada são obrigatórios.

## 8. Métricas

Métrica primária por série e horizonte:

```text
RMSE no incremento raw original
ratio(candidate, control) = RMSE(adaptive_cartesian) / RMSE(control)
```

Também registrar:

- MAE e NRMSE pelo desvio dos alvos de treino;
- erro por família, seed, horizonte, contexto e ruído;
- span raw mínimo, mediano, médio e máximo de cada input;
- quantidade de tokens, escalares, bytes e parâmetros;
- rank da design matrix;
- runtime de representação, fit e inferência;
- pico aproximado de memória quando disponível sem dependência adicional;
- taxa de origens elegíveis e motivos de exclusão;
- escolhas internas e todos os candidatos derrotados.

Janelas e origens não são réplicas independentes. A unidade primária é a série completa
`família × seed`. Intervalos descritivos usam bootstrap pareado de 10.000 reamostragens dessas 40
unidades, preservando todos os horizontes da unidade.

## 9. Gate K6

O split decisório é teste, `noise_std=0.01`, `H=16`. Todos os controles abaixo são obrigatórios.

### Margens preditivas

- margem de superioridade `ratio <= 0.99` contra `fixed_cartesian`, `causal_paa`,
  `moving_average_step_matched`, `raw_step_matched` e `persistence`;
- margem de não inferioridade `ratio <= 1.05` contra `raw_scalar_matched`.

Para cada controle, a margem correspondente deve ocorrer simultaneamente em:

1. pelo menos 4/5 seeds, usando a média geométrica sobre famílias e horizontes;
2. pelo menos 6/8 famílias, usando a média geométrica sobre seeds e horizontes;
3. pelo menos 2/3 horizontes, sendo cada horizonte robusto em 4/5 seeds após agregar famílias.

### Condições estruturais

- a candidata usa exatamente `H` passos e `2H+1` escalares em toda origem;
- `raw_scalar_matched` usa exatamente `2H` passos, `2H+1` escalares e o mesmo número de parâmetros;
- a fração de passos candidata/raw scalar-matched é `<= 0.5`;
- a mediana do span raw da candidata é `>= 2H` em 4/5 seeds e 6/8 famílias;
- `fixed_cartesian` e `causal_paa` usam os spans derivados somente do treino;
- 100% das features, previsões e métricas pós-transformação são finitas;
- 100% das condições planejadas terminam ou aparecem explicitamente como falha no artefato;
- testes de invariância de prefixo e equivalência batch/stream passam para a visão de tokens;
- nenhum segmento de `finalize()` aparece numa origem interna.

K6 passa somente se **todas** as margens preditivas, unidades robustas e condições estruturais
passarem. Validade de execução não substitui o gate científico.

## 10. Análises secundárias congeladas

- repetir o quadro em `H=8` e `H=32`;
- repetir em ruído `0.0` e `0.05`;
- heterogeneidade por família e horizonte;
- fronteira erro × tokens × escalares × span × runtime;
- ablation descritiva do prefixo aberto, removendo somente o último token provisório;
- comparação descritiva com `tolerance=0.03` fixo, independentemente da seleção de treino.

Essas análises não podem substituir o contexto, ruído, candidata ou controles primários se K6
falhar. Nenhuma correção por seleção posterior promove uma análise secundária ao gate.

## 11. Sequência operacional e cegamento

1. Versionar este protocolo e o ADR antes de qualquer runner da Etapa 11.
2. Implementar geradores, token views, controles e testes usando apenas fixtures/seeds não canônicas.
3. Criar config que transcreva literalmente este documento; divergências exigem novo ADR antes de
   executar dados canônicos.
4. Executar os gates de engenharia e congelar código/config em commit limpo.
5. Rodar seleção e validação externa sem materializar métricas de teste.
6. Auditar causalidade, completude, pareamentos e `selection.json`; versionar seu hash.
7. Abrir o teste uma vez, gerar `gate.json` e preservar toda a grade, inclusive falhas.
8. Repetir no mesmo commit para verificar determinismo computacional.
9. Promover relatório e tabelas para `reports/reference/` com manifesto Git-normalizado.

Logs de desenvolvimento não podem usar as cinco seeds primárias. Se qualquer seed primária for
executada antes do congelamento do código, o incidente é registrado e um novo conjunto de seeds é
derivado antes do teste.

## 12. Claims condicionais e condição de parada

Se K6 passar, o claim máximo permitido será:

> No benchmark sintético pré-especificado da Etapa 11-A, tokens cartesianos adaptativos tiveram
> menor erro que tokenizações causais de mesmo comprimento e permaneceram não inferiores a um
> contexto raw de mesmo payload usando metade dos passos.

Mesmo após passagem, são proibidos claims de novidade, mecanismo cinemático, generalização real,
compressão em bytes ou superioridade autoregressiva.

Se K6 falhar:

- preservar o resultado negativo;
- não trocar a candidata, o orçamento ou o modelo no mesmo teste;
- não abrir 11-B, modelos neurais ou datasets externos por esta sequência;
- encerrar a hipótese de tokenização adaptativa linear sob o orçamento definido.

## 13. Artefatos obrigatórios

- `config.json`, `environment.json`, `selection.json` e `gate.json`;
- `origins.csv` com limites causais e split;
- `tokens.csv` ou arquivo comprimido integral por origem/representação;
- `predictions.csv` integral;
- `conditions.csv` e `conditions_by_signal.csv`;
- `summary.csv`, `summary_by_seed.csv` e `summary_by_signal.csv`;
- `budget_audit.csv`, `causality_audit.csv` e `failures.csv`;
- figuras derivadas das tabelas, nunca fonte exclusiva de métricas;
- manifesto com tamanho e SHA-256;
- reprodução no mesmo commit com comparação excluindo apenas campos de runtime.
