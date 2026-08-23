# Protocolo de controles da geometria absoluta no forecasting

Status: **executado e reproduzido; gate científico da Etapa 8 não satisfeito**.

## Pergunta

O ganho exploratório de `absolute_geometry = (dt, dy, theta, r)` permanece quando comparado com
engenharia geométrica local, suavização causal e segmentação fixa sob o mesmo pooling, ridge e
número de parâmetros?

O gate relacional da Etapa 7 falhou. Este protocolo testa K1/K3 e não pode ser descrito como
confirmação de `delta_theta`, `delta_r` ou K2.

## Base congelada

O experimento reutiliza os sete sinais, cinco seeds, contextos `32/64/128`, horizontes `1/4/16`,
stride 4, splits, pooling `(last, mean, std)`, ridge `alpha=0.001` e `tolerance=0.1` dos estudos
anteriores. A tolerância continua exploratória e não é retunada.

Cada representação geométrica produz quatro canais e, após pooling, 12 entradas e 13 parâmetros
ridge. Raw e primeira diferença são mantidos como contexto descritivo, não como controles de
capacidade pareada.

## Representações

| Nome | Construção | Canais | Escopo causal | Papel |
|---|---|---:|---|---|
| `absolute_geometry` | PLA VectorChain com `dt,dy,theta,r` | 4 | online estrito | candidata |
| `local_geometry` | geometria de cada incremento amostral, `dt=1` | 4 | online estrito | controle primário de features/capacidade |
| `moving_average_geometry` | média móvel trailing seguida de geometria local | 4 | online estrito | controle de smoothing FIR |
| `ewma_geometry` | EWMA recursiva seguida de geometria local | 4 | online estrito | controle de smoothing IIR |
| `fixed_geometry` | segmentos de duração fixa ancorados no início da janela | 4 | forecast-causal, fronteiras fixas | controle de PLA/downsampling |
| `abba_geometry` | peças contínuas do `fABBA.compress` oficial | 4 | offline dentro da janela | anterioridade operacional descritiva |

`abba_geometry` usa somente duração e incremento das peças, antes da quantização simbólica. Assim,
o experimento compara o estágio de compressão adaptativa ABBA, não o método simbólico fABBA
completo. Como seus segmentos anteriores podem mudar quando a janela cresce, ele não participa do
gate de causalidade online.

## Tuning sem abrir validação/teste

Cada família com hiperparâmetro recebe exatamente três candidatos:

- média móvel: janelas `2/4/8`;
- EWMA: `alpha = 0.2/0.5/0.8`, multiplicando a observação atual;
- segmentação fixa: `8/16/32` intervalos;
- ABBA: tolerâncias `0.03/0.1/0.3`.

Para cada seed/contexto/horizonte, somente exemplos originalmente marcados como treino são usados.
Dentro de cada sinal, os primeiros 80% desses exemplos formam treino interno e os 20% finais,
validação interna bloqueada. Cada candidato ajusta seu próprio scaler/ridge no treino interno; o
menor RMSE conjunto escolhe o hiperparâmetro, com a ordem acima como desempate determinístico.
Depois, o ridge é reajustado no treino externo completo e avaliado em validação/teste. A candidata e
`local_geometry` não recebem tuning.

Esse desenho concede mais orçamento aos controles que à candidata e, portanto, é conservador para
K3. Resultados de todos os candidatos e escolhas ficam em `tuning.csv`.

## Unidade e contrastes

A seed da realização conjunta dos sete sinais é a unidade de réplica. As nove combinações de
contexto/horizonte são medidas repetidas.

Para cada controle causal, o contraste é:

```text
RMSE(absolute_geometry) / RMSE(controle)
```

Uma seed demonstra vantagem prática quando a média geométrica das nove razões é `<= 0.99`. Uma
célula é robusta quando ao menos quatro das cinco seeds atingem a mesma margem.

O contraste de Pareto por condição exige simultaneamente:

```text
RMSE(candidate) / RMSE(control) <= 1.01
payload(candidate) <= payload(control)
parameters(candidate) <= parameters(control)
```

## Gate K3 desta etapa

Somente validação decide. O gate passa se, para **cada** controle causal:

1. pelo menos quatro de cinco seeds mostram vantagem prática de 1% da candidata;
2. pelo menos cinco de nove células são robustas para essa vantagem;
3. pelo menos cinco de nove células têm Pareto robusto em quatro de cinco seeds.

Além disso, a grade deve estar completa, sem falhas, e todas as condições geométricas devem ter 12
entradas pooled e 13 parâmetros. `abba_geometry` é reportado separadamente e não pode salvar nem
derrubar o gate causal.

Se o gate falhar porque smoothing ou geometria local iguala a candidata, K3 é rejeitada e o claim
é reduzido ao controle explicativo. Se apenas a segmentação fixa falhar, a evidência pode apontar
para compactação adaptativa, mas continua exploratória.

## Reprodutibilidade e limites

O runner deve gravar condições por seed/sinal, tuning interno, payload, parâmetros, ratios,
decisões de Pareto, configuração, dependências, ambiente, falhas, figuras e manifestos. Uma réplica
no mesmo commit mede determinismo computacional, não confirmação externa.

O pacote oficial `fABBA==1.5.2` pertence ao grupo experimental `abba`; SciPy, scikit-learn e pandas
não se tornam dependências do VectorChain. A implementação oficial descreve ABBA/fABBA como
aproximação poligonal adaptativa seguida de agregação simbólica; este protocolo avalia apenas a
primeira etapa contínua.

## Resultado observado

As duas execuções canônicas completaram 360 avaliações e 540 decisões de tuning, sem falhas. A
candidata passou contra `local_geometry` (5/5 seeds, 8/9 células preditivas e Pareto) e
`ewma_geometry` (4/5, 5/9 e 5/9), mas falhou contra `moving_average_geometry` (1/5, 3/9 e 3/9) e
`fixed_geometry` (4/5, 3/9 e 3/9). Consequentemente, K3 não avança nesta formulação.

A média móvel trailing escolheu janela 8 em 37/45 células e teve RMSE agregado aproximadamente 3%
menor que a candidata em validação. O efeito se concentrou em seno, chirp e mudança de regime. Isso
torna smoothing FIR uma explicação concorrente suficiente para bloquear o claim de mecanismo, mas
não apaga a vantagem observada da candidata sobre geometria local e EWMA. Resultados completos e a
auditoria da reprodução estão no
[`relatório de referência`](../reports/reference/forecasting-absolute-geometry-controls/).
