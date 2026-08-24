# Literatura próxima e limites de novidade

Status: **levantamento inicial, não revisão sistemática**.

Data de corte: 2026-08-23.

## Objetivo

Este documento impede que componentes conhecidos sejam apresentados como novidade do VectorChain.
Ele registra trabalhos próximos encontrados depois da conclusão do MVP e delimita a hipótese ainda
aberta. A ausência de um item nesta lista não demonstra ausência na literatura.

## Linhagem mais próxima

### Segmentação e representação online

- Fuchs et al., *Online segmentation of time series based on polynomial least-squares
  approximations* (IEEE TPAMI, 2010), DOI
  [`10.1109/TPAMI.2010.44`](https://doi.org/10.1109/TPAMI.2010.44), apresenta segmentação online e
  interpreta coeficientes locais como inclinação, curvatura e mudança de curvatura.
- Piecewise Linear Approximation e segmentação de streams possuem uma literatura anterior extensa.
  Portanto, segmentação adaptativa, online ou baseada em slope não é isoladamente uma contribuição
  segura de novidade.

### Família ABBA

- Elsworth e Güttel, *ABBA: adaptive Brownian bridge-based symbolic aggregation of time series*
  (Data Mining and Knowledge Discovery, 2020), DOI
  [`10.1007/s10618-020-00689-6`](https://doi.org/10.1007/s10618-020-00689-6), aproxima a série por
  uma cadeia poligonal adaptativa, representa cada trecho por duração e incremento e depois agrupa
  esses pares em símbolos.
- Chen e Güttel, *An efficient aggregation method for the symbolic representation of temporal
  data* (ACM TKDD, 2023), DOI
  [`10.1145/3532622`](https://doi.org/10.1145/3532622), introduz fABBA e acelera a agregação por um
  procedimento baseado em ordenação. A implementação pública está no
  [`nla-group/fABBA`](https://github.com/nla-group/fABBA).
- Para a comparação operacional da Etapa 8, a versão oficial `fABBA==1.5.2` é fixada como extra
  experimental. O runner usa `compress` e somente suas peças contínuas; não representa essa etapa
  como equivalência ao pipeline simbólico fABBA completo.
- Elsworth e Güttel, *Time Series Forecasting Using LSTM Networks: A Symbolic Approach* (2020),
  [`arXiv:2003.05672`](https://arxiv.org/abs/2003.05672), já trata forecasting como geração de uma
  sequência simbólica ABBA e reconstrói a previsão numérica.
- Carson, Chen e Kang, *Quantized symbolic time series approximation* (QABBA),
  [`arXiv:2411.15209`](https://arxiv.org/abs/2411.15209), combina ABBA e quantização e inclui
  aplicações de regressão com LLMs.
- Carson, Chen e Kang, *LLM-ABBA: Understanding time series via symbolic approximation*,
  [`arXiv:2411.18506`](https://arxiv.org/abs/2411.18506), aplica a representação simbólica a
  classificação, regressão e forecasting com LLMs.
- Abdullahi et al., *HSQP: A Plug-and-Play Symbolic–Quantized Framework for Time-Series
  Tokenization in Large Language Models* (IEEE Access, 2026), DOI
  [`10.1109/ACCESS.2026.3674765`](https://doi.org/10.1109/ACCESS.2026.3674765), integra patching,
  ABBA e quantização afim em tokens para modelos de séries temporais.

Consequência: não atribuir novidade a `série -> segmentos adaptativos -> duração/incremento`, à
tokenização desses segmentos ou ao forecasting de uma sequência segmentada em sentido amplo.

### Ângulos, viradas e relações locais

- Han e Gao, *A method for representing stock time series features based on trend and inclination
  angle turning points* (Computer Science and Information Systems, 2026), DOI
  [`10.2298/CSIS250710074H`](https://doi.org/10.2298/CSIS250710074H), combina mudanças de ângulo de
  inclinação e seleção de turning points para representar séries financeiras.

Consequência: `theta`, diferença angular e turning points também não constituem, isoladamente, um
claim seguro de novidade. Curvatura e mudanças de slope aparecem em outras literaturas de
segmentação, detecção de mudança e análise de trajetórias.

### Revisão incremental e cinemática inversa temporal

- Elmeleegy et al., *Online Piece-wise Linear Approximation of Numerical Streams with Precision
  Guarantees* (PVLDB, 2009), DOI
  [`10.14778/1687627.1687645`](https://doi.org/10.14778/1687627.1687645), mantém aproximações
  lineares de streams sob limite de erro. Revisão de um estado linear corrente não é, portanto,
  novidade isolada.
- Kaess et al., *iSAM2: Incremental Smoothing and Mapping with Fluid Relinearization and
  Incremental Variable Reordering* (ICRA, 2011),
  [página e artigo dos autores](https://people.csail.mit.edu/kaess/pub/Kaess11icra.html), mostra o
  princípio geral de novas medições revisarem variáveis latentes afetadas sem refazer todo o batch.
- Murooka et al., *Optimization Computation of Time-Series Inverse Kinematics Considering
  Time-Varying and Time-Invariant Configurations and Adjacency* (2019), DOI
  [`10.9746/sicetr.55.664`](https://doi.org/10.9746/sicetr.55.664), formula cinemática inversa de
  movimento robótico com regularização entre configurações adjacentes.

Esses trabalhos não estabelecem a formulação exata de K7, mas impedem reivindicar como novas a
revisão incremental, a regularização fixed-lag ou a otimização temporal de juntas. A pergunta do
VectorChain é empírica e mais estreita: se correções temporais de uma cauda poligonal causal em um
gráfico escalar acrescentam sinal preditivo sob controles pareados.

## Comparação conceitual

| Propriedade | Precedente identificado | Situação no VectorChain |
|---|---|---|
| aproximação linear adaptativa | ABBA, PLA | implementada |
| segmentação online | SwiftSeg e outros métodos de stream | contrato causal mais estrito, mas componente conhecido |
| duração e incremento por trecho | ABBA/fABBA | `dt/dy` implementados |
| ângulo ou mudança angular | representação por turning points e curvatura | `theta/delta_theta` implementados |
| revisão incremental de variáveis recentes | PLA online, iSAM2/fixed-lag smoothing | núcleo estrutural K7 implementado, ainda sem resultado |
| IK temporal regularizada | otimização de movimento robótico | inspiração para K7, não equivalência de domínio |
| atualização temporal da mesma junta como feature preditiva | não estabelecida por este levantamento | hipótese K7, sem resultado |
| sequência simbólica prevista | ABBA-LSTM, LLM-ABBA | não é o caminho atual |
| quantização/tokenização | QABBA, HSQP | fora da Fase I; não planejada para 11-A |
| ablação causal de relações mantendo fronteiras | não estabelecida por este levantamento | executada; gate relacional negativo |
| cadeia geométrica contínua como estado autoregressivo | não estabelecida por este levantamento | executada em 10-A; pior que AR raw |
| Pareto predição × passos × payload com controles pareados | não estabelecido por este levantamento | controles executados; gates K3/K5-A negativos |

`Não estabelecida` significa somente que o levantamento inicial não localizou uma correspondência
direta. Não significa inexistência de anterioridade.

## Fronteira de contribuição candidata — K7

K7 não declara novidade sobre segmentação poligonal, revisão incremental, fixed-lag smoothing,
cinemática inversa temporal, ângulo ou comprimento. Esses ingredientes possuem precedentes próximos.
A fronteira candidata, ainda sujeita a busca sistemática e resultado experimental, é a combinação
operacional de:

1. uma cadeia poligonal escalar online com prefixo comprometido imutável e cauda revisável limitada;
2. identidades persistentes de elos, permitindo medir a mudança temporal da **mesma** articulação;
3. `update_theta/update_r` usados como variáveis preditivas, distintos de diferenças espaciais entre
   elos simultâneos;
4. ablations causais pareadas que preservam fronteiras, origens, alvos, payload e capacidade do
   modelo;
5. teste isolado contra geometria imutável, geometria revisada, relações espaciais, raw pareado e
   persistência.

Por enquanto isso deve aparecer apenas como **fronteira de contribuição candidata** ou **hipótese
K7**, nunca como invenção ou novidade provada. Se K7 passar e a busca ampliada não encontrar uma
formulação equivalente, a eventual contribuição poderá ser declarada na seção de contribuições de
um artigo como o método e o protocolo pareado; o resultado preditivo será um claim empírico separado.
Se K7 falhar, a contribuição restante será metodológica/diagnóstica e o resultado negativo deverá
ser preservado.

## Lacuna de pesquisa provisória

A lacuna que motivou a Fase I estava na combinação verificável de:

1. emissão estritamente causal e irreversível de uma cadeia contínua;
2. separação entre geometria do elo e relações entre elos;
3. ablation das relações sem alterar as fronteiras;
4. contabilização explícita de passos, payload e capacidade downstream;
5. previsão futura no relógio de eventos e decodificação cinemática recursiva.

Os testes posteriores não demonstraram valor incremental das relações nem competitividade do estado
recursivo. A combinação ainda pode ser descrita como enquadramento e protocolo experimental, mas
não como novo mecanismo preditivo estabelecido.

A área não deve ser descrita como abandonada: os trabalhos de 2024–2026 mostram atividade recente
em simbolização, quantização, LLMs e turning points. Na Fase I, a oportunidade provisória foi
formulada na interseção dessas linhas com relações intersegmentos causais; os gates posteriores não
confirmaram essa diferenciação.

A Fase II testa eficiência sob orçamento de comprimento, mas tokenização adaptativa, patching e
representações por segmentos já possuem anterioridade próxima. K6 é uma pergunta empírica do
VectorChain, não um claim de invenção. Antes de qualquer publicação da Fase II, a busca deve ser
expandida especificamente para `adaptive tokenization`, `time-series patching`, `dynamic patching`,
`event-based sampling` e comparações sob `token budget` ou `context budget`.

## Busca exigida antes de publicação

Antes de qualquer claim de novidade:

- pesquisar IEEE Xplore, ACM Digital Library, Scopus/Web of Science, Google Scholar e arXiv;
- cobrir `online/streaming PLA`, `adaptive polygonal chain`, `segment transition`, `turning angle`,
  `discrete curvature`, `event-based forecasting`, `symbolic forecasting`, `fixed-lag smoothing`,
  `incremental smoothing`, `time-series inverse kinematics` e `ABBA`;
- fazer backward e forward snowballing dos trabalhos acima;
- registrar consulta, data, base, resultados incluídos/excluídos e justificativa;
- comparar formalmente algoritmo, causalidade, estado, decoder, tarefa e protocolo experimental;
- substituir este levantamento por uma seção de related work revisável e bibliografia versionada.
