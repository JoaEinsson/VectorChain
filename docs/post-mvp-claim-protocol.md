# Programa experimental pós-MVP para claims

Status: **programa vigente; Etapas 7 e 8 executadas com gates negativos**.

## Objetivo

Converter o resultado do pacote VectorChain em claims progressivamente mais específicos, sem usar o
teste para selecionar a narrativa. O programa começa isolando o mecanismo e termina, se os gates
anteriores passarem, avaliando a cadeia como espaço de estado autoregressivo.

Este documento não modifica resultados anteriores, o default do pacote ou a especificação do MVP.
A tolerância `0.1` é uma candidata descoberta na grade de robustez; seu desempenho passado é
exploratório para qualquer claim novo.

## Evidência de partida

- Causalidade, equivalência batch/stream e reconstrução estão cobertas por testes.
- Retrieval sintético não mostrou superioridade e permanece resultado negativo.
- O forecasting mínimo foi limítrofe em uma seed.
- A grade com cinco seeds encontrou 14/27 células robustas no teste e 12/27 na validação.
- Em `tolerance=0.1`, 41/45 condições de teste passaram o critério conjunto, mas a configuração foi
  identificada olhando a própria grade.
- O ridge VectorChain possui mais entradas e parâmetros que raw; essa diferença de capacidade não
  pode ser confundida com mecanismo da representação.
- A Etapa 7 executada não sustentou K2: `turning` passou em 0/5 seeds e 1/9 células de validação
  contra `absolute_geometry`; o pacote absoluto tornou-se a candidata exploratória para K1/K3.
- A Etapa 8 não sustentou K3: média móvel trailing e segmentação fixa bloquearam o gate, embora a
  candidata tenha vencido geometria local e EWMA.

## Regras comuns

1. Cada etapa recebe ADR/protocolo, configuração, métrica primária e critério de conclusão antes da
   primeira execução relevante.
2. Escolhas usam treino e validação; o teste confirmatório não pode ser reaberto para retuning.
3. Fronteiras devem ser idênticas entre ablations de features.
4. Reportar duas comparações quando necessário: pipeline idêntico e capacidade downstream pareada.
5. Passos, elementos escalares, bytes, parâmetros e runtime são métricas separadas.
6. Janelas sobrepostas nunca contam como réplicas independentes.
7. Resultados negativos encerram ou redirecionam a hipótese correspondente; não autorizam trocar o
   contraste primário depois do teste.
8. Uma réplica no mesmo commit mede reprodutibilidade computacional, não replicação independente.

## Etapa 7 — Isolamento do mecanismo cinemático

### Pergunta primária

Relações entre segmentos acrescentam informação preditiva além das propriedades de cada segmento,
mantendo fronteiras, exemplos, alvo, split e algoritmo downstream?

### Ablations mínimas

| Nome | Features | Papel |
|---|---|---|
| `segment` | `dt, dy` | cadeia adaptativa básica |
| `absolute_geometry` | `dt, dy, theta, r` | propriedades do elo sem relações |
| `turning` | `dt, dy, theta, r, delta_theta` | default atual e rotação articular |
| `full_relational` | `dt, dy, theta, r, delta_theta, delta_r` | relações angulares e de extensão |

O contraste primário inicial é `turning` versus `absolute_geometry`; `full_relational` é
secundário até um protocolo decidir se `delta_r` merece promoção. Features de segunda ordem ficam
fora desta primeira ablation.

### Controles de capacidade

- registrar dimensão pooled, rank, parâmetros, design matrix e estado do modelo;
- executar uma análise com capacidade comparável entre representações;
- incluir raw com features locais simples quando isso for necessário para separar geometria de
  engenharia de features;
- não interpretar ganho de um modelo maior como ganho causado pela compressão.

### Gate

K2 avança somente se a variante relacional mostrar efeito incremental consistente em unidades
independentes e o resultado não depender de uma única célula escolhida no teste. A direção, medida
primária, margem operacional e método de incerteza serão congelados na configuração da etapa.

Resultado: **K2 não avançou** no protocolo executado. A Etapa 8 pode diagnosticar a utilidade de
`absolute_geometry`, mas não pode rebatizar esse caminho como confirmação relacional.

## Etapa 8 — Controles pareados e anterioridade operacional

### Baselines mínimas

- ABBA e/ou fABBA, preservando tanto descritores contínuos quanto a versão simbólica quando
  aplicável;
- PLA causal por endpoints com regra de erro documentada;
- segmentação fixa com orçamento semelhante;
- downsampling causal pareado por passos ou payload;
- média móvel e suavização exponencial causais;
- raw/diferenças com features locais e capacidade downstream comparável.

QABBA, LLM-ABBA ou HSQP entram quando a tarefa e o modelo tornarem a comparação metodologicamente
válida; não devem ser adicionados apenas para aumentar a quantidade de baselines.

### Pareamentos obrigatórios

- mesmos contextos, origens, alvos e splits;
- orçamento de payload explicitamente comparável;
- capacidade downstream idêntica ou segunda análise pareada;
- tuning com o mesmo orçamento e somente em treino/validação;
- causalidade rotulada: métodos offline não podem ser apresentados como equivalentes online.

### Gate

K3 avança somente se a região VectorChain permanecer na fronteira de Pareto depois dos controles.
Se smoothing, PLA simples ou capacidade explicarem o resultado, o claim deve ser reduzido para o
mecanismo correspondente.

Resultado: **K3 não avançou**. `absolute_geometry` passou contra geometria local e EWMA, mas não
contra média móvel trailing nem segmentação fixa nas células robustas. O resultado permite um
claim descritivo de trade-off no benchmark versionado, não um claim de que adaptação ou relações
cinemáticas causaram o ganho. A Etapa 9 fica fechada como confirmação de K2/K3 até existir uma
hipótese genuinamente reformulada e pré-especificada.

## Etapa 9 — Confirmação externa congelada

### Seleção antes do teste

A configuração candidata será escolhida usando apenas as etapas 7 e 8. `tolerance=0.1` pode ser
congelada como candidata, mas não recebe status de default. Seeds, datasets, exclusões, métricas,
margens e análises serão publicados antes da abertura do teste confirmatório.

### Domínios necessários

- novas seeds e realizações de ruído;
- diferentes intensidades e estruturas de ruído;
- novas dinâmicas sintéticas e mudanças de escala/amplitude;
- datasets públicos reais de mais de um domínio;
- casos com informação de alta frequência, para medir onde a compressão falha.

### Inferência

- usar série, seed completa ou dataset como unidade experimental, conforme o desenho;
- reportar efeitos pareados e intervalos de confiança por blocos independentes;
- preservar heterogeneidade por dinâmica/dataset, sem esconder falhas numa média global;
- separar confirmação estatística de benchmark de performance computacional;
- executar ao menos uma reprodução em ambiente limpo e buscar uma replicação externa quando
  possível.

### Gate

K4 passa somente se o efeito relacional e o Pareto sobreviverem nos dados inéditos com magnitude e
incerteza compatíveis com o critério pré-especificado. Só então o claim de mecanismo pode deixar de
ser descrito como hipótese do benchmark sintético original.

## Etapa 10 — Estado autoregressivo da cadeia

Esta etapa testa o claim mais forte e não deve ser confundida com o forecasting pooled atual.

O protocolo operacional está em
[`vector-state-rollout-protocol.md`](vector-state-rollout-protocol.md). Como o elo anterior só é
emitido depois da primeira observação do elo aberto seguinte, o estado causal passa a ser
`E_i = (S_i, open_dy_(i+1))`; os alvos excluem explicitamente essa parte já conhecida.

### Tarefa

```text
V_1, V_2, ..., V_t -> distribuição ou estimativa de V_(t+1)
V_(t+1) -> V_(t+2) -> ... -> trajetória no horizonte solicitado
```

O experimento deve comparar pelo menos:

- previsão absoluta de `(dt, dy)`;
- previsão de `(r, theta)` com restrições válidas;
- previsão relacional de `delta_theta/delta_r` condicionada ao último elo;
- baselines raw recursivas, persistência e um modelo segmentado/simbólico próximo, como ABBA-LSTM.

### Semântica causal e temporal

- o caminho primário usa somente segmentos já emitidos na origem;
- qualquer uso do segmento aberto deve ser uma condição separada e rotulada como estado provisório;
- `dt` previsto deve ser inteiro e positivo, com projeção ou distribuição explicitamente definida;
- o decoder deve definir como tratar um elo que cruza o horizonte raw;
- a origem vertical e a articulação inicial devem ser explícitas;
- nenhum endpoint futuro real pode ser usado para corrigir rollouts.

### Métricas

- erro por horizonte no domínio raw reconstruído;
- erro de duração, deslocamento, orientação e relações no domínio vetorial;
- drift acumulado e continuidade das articulações;
- taxa de estados inválidos e falhas de término;
- passos, payload, parâmetros, runtime e memória;
- calibração, caso o modelo produza distribuições;
- estabilidade fora do horizonte usado no treino.

### Gate do claim mais forte

K5 passa somente se o rollout vetorial for causal, válido, estável, reconstruível e competitivo com
baselines pareadas em dados inéditos. O claim permitido passa então a ser:

> Uma cadeia cinemática causal pode funcionar como espaço de estado autoregressivo compacto para
> séries temporais nas condições avaliadas.

Falha no gate não invalida os resultados de compressão ou forecasting pooled; apenas impede afirmar
que a cadeia é um espaço de estado autoregressivo útil.

Resultado: **K5-A não avançou**. A cadeia completou rollouts válidos e compactos, mas perdeu para
AR raw em todas as cinco seeds e nos três horizontes. Isso impede C4 nesta formulação e aciona a
condição de parada antes de ABBA-LSTM, dados externos ou aumento de capacidade.
