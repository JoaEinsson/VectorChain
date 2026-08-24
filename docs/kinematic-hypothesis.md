# Hipótese cinemática pós-MVP

Status: **enquadramento científico; não altera a especificação do MVP**.

## Origem conceitual

A intuição que originou o VectorChain foi transportar para o gráfico de uma série temporal uma
leitura inspirada em cinemática inversa. Os pontos `(t, x[t])` formam uma trajetória planar
restrita; a segmentação extrai elos adaptativos e as relações entre elos formam articulações.

| Cadeia robótica | VectorChain |
|---|---|
| elo | segmento adaptativo |
| deslocamento cartesiano | `(dt, dy)` |
| comprimento e orientação | `(r, theta)` |
| rotação entre elos | `delta_theta` |
| mudança de extensão | `delta_r` |
| pose cartesiana | pontos `(t, x[t])` |
| extração de coordenadas articulares | série → cadeia |
| cinemática direta | cadeia → trajetória reconstruída |

Essa correspondência é uma inspiração matemática, não uma afirmação de que o problema seja
cinemática inversa clássica. O eixo temporal é monotônico, `dt > 0`, `theta` fica restrito a
`(-pi/2, pi/2)` e `r` mistura unidades de tempo e amplitude. A cadeia não possui os graus de
liberdade de um manipulador físico.

## Duas coordenadas do mesmo elo

Para cada segmento emitido `i`, o MVP registra:

```text
v_i = (dt_i, dy_i)
q_i = (r_i, theta_i)
```

com:

```text
r_i     = sqrt(dt_i**2 + dy_i**2)
theta_i = atan2(dy_i, dt_i)
```

`v_i` e `q_i` não são informações independentes; são duas parametrizações do mesmo deslocamento.
As variáveis relacionais começam em:

```text
delta_theta_i = theta_i - theta_(i-1)
delta_r_i     = r_i - r_(i-1)
```

Uma extensão futura poderá avaliar `delta2_theta` e `delta2_r`, mas essas features não pertencem à
API atual e exigirão ADR, definição de fronteira e testes antes de serem implementadas.

## Codificador e decodificador

O codificador causal pode ser representado por:

```text
(x_0, ..., x_t) -> fronteiras emitidas -> (v_1, ..., v_m) -> features relacionais
```

Um segmento só entra na cadeia final quando a amostra seguinte demonstra a violação da tolerância,
ou quando `finalize()` declara o término do stream. O segmento aberto continua provisório. Toda
pesquisa pós-MVP preserva o contrato de causalidade e não pode transformar informação futura em
estado articular passado.

Dados um valor inicial e uma sequência válida de deslocamentos, a operação análoga à cinemática
direta acumula:

```text
t_k = t_0 + sum(dt_i, i=1..k)
x_k = x_0 + sum(dy_i, i=1..k)
```

e interpola cada elo no relógio amostral. Prever `(r, theta)` também requer convertê-los para
`(dt, dy)` respeitando duração inteira positiva e o horizonte solicitado. Como `r*cos(theta)` não
é necessariamente inteiro, um modelo futuro terá de declarar como representa, distribui ou projeta
`dt`; arredondamento silencioso não é aceitável.

## O que o MVP demonstrou

O MVP estabeleceu, por implementação e testes:

- uma máquina de estado causal com equivalência batch/stream;
- imutabilidade dos segmentos emitidos e invariância a sufixos futuros;
- fronteiras independentes da projeção de features;
- reconstrução contínua e determinística;
- benchmarks reproduzíveis de reconstrução, similaridade e forecasting.

Na grade de forecasting, o pacote de features
`(dt, dy, theta, r, delta_theta)` encontrou uma região favorável em `tolerance=0.1`: 41/45
condições de teste satisfizeram simultaneamente paridade preditiva, redução estrutural e redução de
payload; oito de nove células foram robustas em pelo menos quatro de cinco seeds. Isso é evidência
sobre o pacote completo neste benchmark sintético, não sobre `delta_theta` isoladamente.

O forecasting atual também não prevê o próximo vetor. Ele aplica o mesmo pooling a cada
representação, ajusta um ridge e prevê um incremento no domínio original:

```text
janela raw -> VectorChain -> pooling -> delta_target raw
```

Portanto, os resultados existentes não sustentam ainda o claim de que a cadeia seja um espaço de
estado autoregressivo próprio.

Na ablation posterior, `turning` não superou `absolute_geometry` de forma consistente: 0/5 seeds e
1/9 células de validação atingiram a margem pré-especificada. O controle `turning_matched` passou em
2/5 seeds e `full_relational` piorou todas as médias geométricas por seed. K2, portanto, não avança
sob este pooling/ridge/benchmark. Em contrapartida, o pacote absoluto superou `segment` e raw de
forma exploratória; como `theta/r` são transformações determinísticas de `dt/dy`, isso é evidência
de utilidade para o modelo pooled, não de informação adicional em sentido informacional.

Nos controles posteriores, `absolute_geometry` passou o gate individual contra geometria local e
EWMA, mas não contra média móvel trailing nem segmentação fixa. A janela trailing 8 foi selecionada
em 37/45 células e teve erro agregado menor em validação, principalmente em seno, chirp e mudança
de regime. K3, portanto, não avança: o efeito observado continua compatível com smoothing FIR e não
pode ser atribuído de forma exclusiva à adaptação das fronteiras.

## Registro de hipóteses

| ID | Hipótese | Evidência necessária | Condição de rejeição |
|---|---|---|---|
| K1 | Geometria individual acrescenta sinal além de `(dt, dy)` | `theta/r` vencem a variante básica com fronteiras idênticas | efeito ausente ou instável fora da análise exploratória |
| K2 | Relações cinemáticas acrescentam sinal além da geometria individual | `delta_theta/delta_r` vencem a variante sem relações sob capacidade pareada | variante relacional não melhora ou piora consistentemente |
| K3 | O ganho não é somente suavização ou capacidade downstream | VectorChain vence controles de smoothing, PLA e parâmetros pareados | controle mais simples explica o mesmo Pareto |
| K4 | A região útil generaliza | confirmação congelada em novas seeds, ruídos, dinâmicas e datasets reais | efeito desaparece ou muda de sinal fora dos sintéticos originais |
| K5 | A cadeia é um espaço de estado preditivo | rollout `V_1:t -> V_t+1` compacto, estável e reconstruível | drift, duração inválida ou erro não competitivo em rollout |

Estado após a Etapa 8: K2 foi rejeitada para a formulação avaliada e K3 também não avançou. A
geometria absoluta venceu controles locais e EWMA, mas não permaneceu robustamente superior à
média móvel trailing e à segmentação fixa. K1 permanece apenas como utilidade exploratória de
features para o ridge pooled, não como mecanismo informacional ou cinemático estabelecido.

Estado após a Etapa 10-A: K5 também não avançou na formulação linear pré-especificada. O estado de
emissão produziu rollouts válidos e compactos, mas teve RMSE maior que AR raw em 5/5 seeds e 3/3
horizontes. O estado cartesiano foi ligeiramente melhor e muito menor que o relacional, reforçando
que as relações cinemáticas não trouxeram benefício observável nesta tarefa.

Hipóteses rejeitadas permanecem resultados científicos. Nenhuma etapa deve ser redesenhada depois
de abrir o teste apenas para preservar a narrativa cinemática.

## Escada de claims

### C0 — Garantias de engenharia

É permitido afirmar que a implementação satisfaz o contrato causal e é reproduzível dentro dos
testes e ambientes registrados. Isso não é uma prova formal universal do algoritmo.

### C1 — Resultado empírico atual

É permitido afirmar que o pacote VectorChain encontrou uma região de melhoria conjunta no
benchmark sintético versionado. O domínio, o modelo, as seeds e os critérios devem acompanhar o
claim.

### C2 — Mecanismo cinemático

Só será permitido afirmar que relações entre segmentos carregam informação incremental depois de
K2 e K3 passarem em novas formulações pré-especificadas. Os testes atuais falharam ambos; portanto,
C2 permanece explicitamente proibido.

### C3 — Generalização confirmatória

Só será permitido ampliar o claim além dos sete sinais atuais depois de K4 passar em unidades
experimentais inéditas e com incerteza estatística adequada.

### C4 — Espaço de estado autoregressivo

O claim mais forte exige K5: previsão recursiva diretamente na cadeia, decodificação cinemática e
comparação com modelos raw e segmentados sob orçamento pareado.

K5-A falhou no teste sintético pré-especificado; portanto, C4 permanece proibido nesta formulação.

A escada cinemática da Fase I está encerrada. A síntese final está em
[`phase-1-scientific-synthesis.md`](phase-1-scientific-synthesis.md). K6, na Fase II, testa
tokenização cartesiana sob orçamento e não deve ser citada como continuação ou confirmação de
K1–K5.

K7 formula uma variável que a Fase I não possuía: a atualização temporal da mesma junta dentro de
uma cauda provisória, `theta_i^(t)-theta_i^(t-1)` e `r_i^(t)-r_i^(t-1)`. Isso é diferente das
relações espaciais entre elos avaliadas por K2. O protocolo está em
[`phase-2-revisable-chain-protocol.md`](phase-2-revisable-chain-protocol.md); qualquer resultado
deve preservar essa distinção e não rebatizar K2 como inconclusiva.

## Formulações controladas

Formulação atual defensável:

> No benchmark sintético pré-especificado, uma representação VectorChain estritamente causal
> encontrou condições em que reduziu passos e payload mantendo ou melhorando a previsão.

Formulação pós-ablation também defensável:

> Neste benchmark sintético e modelo ridge com pooling, o pacote de geometria absoluta foi mais
> parcimonioso e teve menor erro agregado que as variantes relacionais avaliadas.

Formulação-alvo do mecanismo:

> Relações cinemáticas entre segmentos adaptativos contêm informação preditiva além da geometria
> dos segmentos individuais.

Formulação-alvo mais forte:

> Uma cadeia cinemática causal pode funcionar como espaço de estado autoregressivo compacto para
> séries temporais.

Não usar `primeiro`, `novo`, `estado da arte`, `prova de superioridade` ou equivalentes sem revisão
de anterioridade e evidência correspondente ao nível do claim.
