# Protocolo do estado autoregressivo causal da cadeia

Status: **pré-especificado antes da implementação e da primeira execução da Etapa 10-A**.

## Pergunta

Uma sequência causal de estados de emissão VectorChain permite prever e reconstruir trajetórias
futuras com menos passos de contexto, payload e capacidade explicitamente pareados, quando
comparada com estado raw, segmentos fixos, AR raw e persistência?

Esta é uma hipótese nova. Os gates negativos de K2/K3 continuam válidos: a Etapa 10 não pode ser
descrita como confirmação de que relações cinemáticas ou fronteiras adaptativas explicaram os
resultados pooled anteriores.

## Por que o estado não é apenas o elo

Um segmento com endpoint `e` só é emitido quando a amostra `e + 1` viola a corda candidata. No
instante causal de emissão, a primeira diferença do próximo segmento já foi observada. Usar o elo
em `e` como se estivesse disponível antes de `x[e + 1]` introduziria um passo de look-ahead.

O estado causal registrado no evento `i` será, portanto:

```text
E_i = (S_i, open_dy_(i+1))
```

onde `S_i` é o segmento imutável emitido e:

```text
open_dy_(i+1) = x[emitted_at_i] - x[emitted_at_i - 1]
```

O alvo seguinte contém somente quantidades ainda não observadas no evento `i`:

```text
remaining_dt_(i+1) = dt_(i+1) - 1
remaining_dy_(i+1) = dy_(i+1) - open_dy_(i+1)
next_open_dy       = x[emitted_at_(i+1)] - x[emitted_at_(i+1) - 1]
```

O modelo prevê `log1p(remaining_dt)`, `remaining_dy` e `next_open_dy`. Assim, quando
`dt_(i+1) = 1`, a parte ainda não observada tem duração e deslocamento iguais a zero. Segmentos
terminais emitidos apenas por `finalize()` não formam estados nem alvos, porque não possuem um
próximo incremento causal.

## Transição e cinemática direta

Depois da previsão, a duração é projetada explicitamente:

```text
remaining_dt_hat = clip(rint(expm1(z_hat)), 0, 256)
dt_hat           = 1 + remaining_dt_hat
dy_hat           = open_dy_current + remaining_dy_hat
```

`rint` usa desempate half-to-even. Antes da projeção, `NaN`, infinitos e valores negativos são
contados como inválidos; para manter o rollout auditável, `NaN/-inf` projetam para zero e `+inf`
para 256. A saída bruta nunca é substituída nos artefatos.

Se `remaining_dt_hat = 0`, o deslocamento restante efetivo é forçado a zero. Permitir um valor não
zero criaria um salto vertical sem avanço no tempo. O deslocamento bruto e a ocorrência dessa
projeção permanecem registrados separadamente.

O trecho ainda não observado do segmento é interpolado da observação corrente até o endpoint
previsto em `remaining_dt_hat` intervalos. Em seguida, `next_open_dy_hat` cria a primeira amostra
do próximo segmento aberto. A transição avança exatamente `dt_hat` amostras e produz `E_(i+1)`;
ela pode então ser aplicada recursivamente sem consultar endpoints futuros reais.

Um rollout termina quando cobre o maior horizonte raw ou quando atinge 256 eventos previstos. A
trajetória é truncada exatamente em cada horizonte raw `16/64/128`; um elo que cruza o horizonte é
interpolado e cortado, nunca descartado nem corrigido com dados futuros.

## Dados ainda não abertos

O estudo usa os sete geradores canônicos com 2.048 pontos, ruído `0.01` e os parâmetros da baseline.
As cinco seeds foram derivadas antes da execução por SHA-256 dos rótulos
`vectorchain-stage10-seed-0` a `vectorchain-stage10-seed-4`, lendo os primeiros 64 bits little
endian e removendo o bit de sinal:

```text
5334086151564077557
9076691527785929283
1488991049205644303
2506524397075762905
228973438714350052
```

Nenhuma dessas seeds participa dos resultados de referência anteriores. A calibração de
viabilidade utilizou somente as seeds antigas e mostrou que `tolerance=0.1` produz poucos eventos
em rampas e respostas. Por isso o estudo usa o default `tolerance=0.03`, não a região selecionada
na grade observada.

Em cada sinal, o split é determinado pelo instante de emissão do alvo: primeiros 60% dos índices
raw para treino, 20% seguintes para validação e 20% finais para teste. O teste é o split primário e
não pode influenciar modelo, duração fixa, projeções ou critérios. Cada sinal precisa fornecer ao
menos cinco exemplos por split e comprimento de histórico.

## Representações e pareamento

Cada condição usa os últimos `4/8/16` estados completos, sem padding. A ordem é preservada por
concatenação, não por pooling.

O gate usa exclusivamente histórico 8, fixado antes da implementação. Históricos 4 e 16 são
análises de sensibilidade e não podem substituir o contraste primário depois de abrir o teste.
Ao projetar uma janela de estados, `delta_theta/delta_r` do primeiro passo são zero; nenhuma relação
com um estado fora do payload declarado é carregada implicitamente.

| Nome | Conteúdo por passo | Papel |
|---|---|---|
| `vectorchain_cartesian` | `dt,dy,open_dy` | ablation da cadeia básica |
| `vectorchain_absolute` | `dt,dy,theta,r,open_dy` | ablation sem relações |
| `vectorchain_relational` | anteriores + `delta_theta,delta_r` | candidata pré-especificada |
| `raw_matched` | últimas `7 × history` diferenças raw | controle primário de payload/capacidade |
| `fixed_relational` | estados de segmentos fixos com sete campos | controle de fronteiras adaptativas |
| `raw_ar` | AR ridge de uma diferença, recursivo | baseline raw no relógio amostral |
| `persistence` | mantém o último valor | baseline sem treino |

O comprimento fixo é a mediana inteira de `dt` calculada somente nos segmentos de treino da seed;
ele não é ajustado em validação/teste. `vectorchain_relational`, `raw_matched` e
`fixed_relational` têm exatamente `7 × history` entradas e usam o mesmo ridge multioutput. Seus
parâmetros preditivos são `3 × (7 × history + 1)`. Médias e escalas também são aprendidas apenas no
treino e registradas separadamente.

`raw_ar` recebe as mesmas `7 × history` diferenças, mas prevê uma única diferença por passo e a
aplica recursivamente. Ele possui menos parâmetros e muito mais alvos de treino raw; esse viés a
favor do baseline é preservado. As variantes cartesianas/absolutas têm menos payload e parâmetros
e são descritivas: K5 não depende de relações já rejeitadas em K2.

## Métricas

No próximo evento, preservar por exemplo e por sinal:

- MAE de duração restante antes e depois da projeção;
- RMSE de deslocamento restante e do próximo incremento aberto;
- MAE de `dt`, `dy` e `theta` do elo completo reconstruído;
- taxa de duração bruta negativa/não finita;
- validade e continuidade depois da projeção.

No rollout, em horizontes raw comuns:

- RMSE da trajetória completa e erro absoluto no endpoint;
- eventos previstos, duração total e cobertura do horizonte;
- estados inválidos antes/depois da projeção e falhas de término;
- número de passos, escalares, bytes, parâmetros, runtime e memória aproximada;
- span raw efetivamente coberto pelo contexto de entrada.

A seed da realização conjunta dos sete sinais é a unidade de réplica. Origens e horizontes dentro
da seed são medidas repetidas, não réplicas independentes.

## Gate K5-A

O gate usa somente teste, histórico 8 e a razão:

```text
RMSE_rollout(vectorchain_relational) / RMSE_rollout(controle)
```

Para cada um de `raw_matched`, `fixed_relational`, `raw_ar` e `persistence`, a candidata deve:

1. atingir média geométrica da razão `<= 0.99` nos três horizontes em pelo menos 4/5 seeds;
2. atingir a mesma margem em ao menos 2/3 horizontes, de forma robusta em 4/5 seeds;
3. completar 100% dos rollouts com 100% de estados válidos após projeção.

Além disso, contra `raw_matched`, a candidata deve usar no máximo `1/7` dos passos, no máximo o
mesmo número de escalares e exatamente a mesma quantidade de parâmetros preditivos. A grade de
execução precisa estar completa e sem falhas.

Falhar contra qualquer controle impede K5-A. Vencer raw mas perder para segmentos fixos impede
atribuir o resultado à adaptação. Vencer apenas persistência não sustenta utilidade do espaço de
estado. A variante absoluta pode superar a relacional sem invalidar a cadeia, mas nesse caso o
resultado reforça a rejeição prévia do mecanismo relacional.

## Claims e condição de parada

Mesmo se K5-A passar, o claim permitido fica restrito aos sete sinais sintéticos e às novas seeds:

> Uma cadeia causal de estados de emissão sustentou rollout autoregressivo compacto nas condições
> sintéticas pré-especificadas.

Não será permitido afirmar generalização para séries reais, superioridade geral, mecanismo
cinemático ou novidade frente a toda literatura. Uma Etapa 10-B com dados reais/novas dinâmicas e
um controle simbólico próximo, como ABBA-LSTM, só será aberta se K5-A passar. Se K5-A falhar, os
artefatos negativos serão promovidos e a hipótese de estado será encerrada ou reformulada antes de
qualquer modelo mais complexo.

Uma reprodução no mesmo commit verifica determinismo computacional; não substitui replicação
independente.
