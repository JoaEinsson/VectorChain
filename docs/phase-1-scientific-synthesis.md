# Síntese científica da Fase I

Status: **fase encerrada e congelada no commit `5297e54`**.

## Objetivo

Este documento consolida a evidência produzida das Etapas 0 a 10-A. Ele não substitui os
protocolos nem os relatórios de referência; funciona como índice de interpretação e impede que um
resultado posterior seja usado para reescrever a pergunta que cada experimento realmente testou.

A pergunta original era deliberadamente ampla:

> Uma cadeia causal e adaptativa de vetores preserva dinâmica relevante, reduz o comprimento
> efetivo e ajuda em reconstrução, similaridade ou forecasting?

A Fase I separou essa pergunta em propriedades de engenharia, utilidade empírica, mecanismo
relacional, exclusão de explicações simples, generalização e previsão autoregressiva da própria
cadeia.

## Evidência acumulada

| Etapa | Pergunta efetivamente testada | Resultado | Consequência |
|---|---|---|---|
| 0–2 | A cadeia pode ser emitida causalmente e reconstruída? | sim, nos contratos e testes versionados | garantia de engenharia, não superioridade empírica |
| 3 | Existe compromisso compressão estrutural × reconstrução? | sim, dependente da tolerância e da dinâmica | redução de passos não implica redução de bytes |
| 4 | A representação melhora invariância/retrieval sintético? | não; baselines simples venceram no top-1 | similaridade permanece resultado negativo |
| 5 | O pacote completo mantém forecasting com menos elementos? | paridade limítrofe no teste e falha na validação | resultado frágil, sem confirmação |
| 6 | A região de forecasting persiste em cinco seeds? | 14/27 células de teste e 12/27 de validação foram robustas | utilidade localizada; `tolerance=0.1` ficou pós-selecionada |
| 7 | Relações entre elos acrescentam valor à geometria absoluta? | não; `turning` passou em 0/5 seeds e 1/9 células | K2 rejeitada nesta formulação |
| 8 | O efeito resiste a smoothing, capacidade e fronteiras fixas? | não contra média móvel trailing e segmentação fixa | K3 não avançou |
| 9 | O mecanismo generaliza para dados inéditos? | não executada porque os gates anteriores falharam | K4 permanece bloqueada, não confirmada nem refutada |
| 10-A | A cadeia funciona como estado autoregressivo causal? | rollout válido, porém pior que AR raw em 5/5 seeds e 3/3 horizontes | K5-A rejeitada; 10-B fechada |

Os resultados completos, configurações e identidades de execução estão em
[`reports/reference/`](../reports/reference/).

## Registro final K1–K5

| ID | Estado ao encerrar a fase | Leitura permitida |
|---|---|---|
| K1 — geometria individual | exploratória, não estabelecida como informação incremental | `theta/r` podem melhorar o condicionamento de um ridge pooled, embora sejam transformações de `dt/dy` |
| K2 — relações cinemáticas | rejeitada na formulação avaliada | `delta_theta/delta_r` não mostraram contribuição consistente |
| K3 — mecanismo além de smoothing/capacidade | gate negativo | média móvel e fronteiras fixas permanecem explicações concorrentes |
| K4 — generalização externa | não aberta | nenhum claim além dos sintéticos versionados |
| K5 — espaço de estado autoregressivo | rejeitada em K5-A | a recorrência vetorial linear não foi competitiva com AR raw |

`Não aberta` não significa evidência negativa. Significa que a pergunta confirmatória não foi
consumida porque suas premissas falharam.

## O que foi estabelecido

### Engenharia e reprodutibilidade

- uma única máquina de estado implementa batch e stream;
- segmentos emitidos são imutáveis e invariantes a alterações futuras;
- a seleção de features não muda fronteiras;
- reconstrução, métricas e sinais sintéticos são determinísticos sob configuração e seed;
- experimentos preservam configuração, commit, ambiente, tabelas brutas e resultados negativos;
- os resultados promovidos possuem reprodução computacional no mesmo commit e ambiente.

Essas afirmações descrevem a implementação e os testes registrados. Não são prova formal para todo
programa possível nem equivalem a replicação independente.

### Observações empíricas delimitadas

- existe uma fronteira mensurável entre quantidade de segmentos e erro de reconstrução;
- o pacote VectorChain encontrou regiões de forecasting pooled com menos passos e payload no
  benchmark sintético original;
- a geometria absoluta foi mais útil ao ridge pooled que as variantes relacionais avaliadas;
- no estado de emissão, a cadeia gerou 100% de rollouts válidos e foi mais estável que transições
  multioutput raw/fixas de mesma dimensão;
- oito estados da cadeia cobriram em média 98,23 intervalos raw, contra 56 passos do controle raw
  de payload pareado na Etapa 10-A.

O último item é redução no número de passos da sequência, não redução automática de escalares,
bytes, parâmetros ou erro.

## O que não foi estabelecido

- superioridade geral de VectorChain;
- novidade de PLA, duração/incremento, ângulos, turning points ou tokenização segmentada;
- informação nova em `theta/r`, que são funções determinísticas de `dt/dy`;
- contribuição preditiva de `delta_theta/delta_r`;
- vantagem causada especificamente por fronteiras adaptativas;
- generalização para datasets reais, timestamps irregulares ou séries multivariadas;
- utilidade da cadeia como estado autoregressivo competitivo;
- benefício de redes maiores, quantização ou modelos simbólicos.

## Escada final de claims

| Nível | Estado | Formulação |
|---|---|---|
| C0 — engenharia | permitido | a implementação satisfaz o contrato causal e reproduz os benchmarks registrados |
| C1 — evidência empírica | permitido com escopo | no benchmark sintético versionado, houve condições de redução de passos/payload com forecasting pooled competitivo |
| C2 — mecanismo cinemático | proibido | K2 e K3 não passaram |
| C3 — generalização | proibido | K4 não foi aberta |
| C4 — estado autoregressivo | proibido | K5-A não passou |

Formulação curta recomendada para a Fase I:

> VectorChain é uma implementação causal e reproduzível de segmentação vetorial que apresentou
> trade-offs úteis em partes de um benchmark sintético, mas não demonstrou vantagem relacional,
> superioridade sobre controles simples nem competitividade como estado autoregressivo.

## Interpretação científica

A contribuição mais sólida da Fase I é a separação experimental de hipóteses que poderiam ter sido
confundidas: compressão estrutural, geometria redundante, relações entre elos, smoothing, fronteiras
adaptativas e recorrência do estado. Os gates negativos estreitaram a explicação em vez de serem
tratados como falhas de execução.

O resultado de estabilidade da Etapa 10-A não autoriza promover K5: AR raw foi simultaneamente
simples, estável e mais preciso. Entretanto, a diferença entre **passos de sequência** e **payload
escalar** permaneceu mensurável. Essa observação motiva uma pergunta nova sobre orçamento de
contexto, desde que ela não seja apresentada como confirmação das hipóteses cinemáticas.

## Transição para a Fase II

A Fase II começa uma escada independente com K6:

> Sob um orçamento fixo de comprimento de sequência, tokens cartesianos adaptativos preservam
> contexto preditivo melhor que tokenizações causais fixas e raw comparáveis?

K6 usa somente `(dt, dy)` como candidata principal, previsão direta no domínio raw e controles sob
orçamento de tokens e de escalares. Ela não reabre K2, K3, K4 ou K5 e não herda como confirmação
nenhuma região selecionada na Fase I.

O desenho está congelado em
[`phase-2-adaptive-tokenization-protocol.md`](phase-2-adaptive-tokenization-protocol.md). Nenhum novo
runner, config ou dado canônico deve ser produzido antes de esse protocolo estar versionado.
