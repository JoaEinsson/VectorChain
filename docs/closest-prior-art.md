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

## Comparação conceitual

| Propriedade | Precedente identificado | Situação no VectorChain |
|---|---|---|
| aproximação linear adaptativa | ABBA, PLA | implementada |
| segmentação online | SwiftSeg e outros métodos de stream | contrato causal mais estrito, mas componente conhecido |
| duração e incremento por trecho | ABBA/fABBA | `dt/dy` implementados |
| ângulo ou mudança angular | representação por turning points e curvatura | `theta/delta_theta` implementados |
| sequência simbólica prevista | ABBA-LSTM, LLM-ABBA | não é o caminho atual |
| quantização/tokenização | QABBA, HSQP | fora do MVP |
| ablação causal de relações mantendo fronteiras | não estabelecida por este levantamento | planejada |
| cadeia geométrica contínua como estado autoregressivo | não estabelecida por este levantamento | planejada, ainda inexistente |
| Pareto predição × passos × payload com controles pareados | não estabelecido por este levantamento | evidência inicial sem todos os controles |

`Não estabelecida` significa somente que o levantamento inicial não localizou uma correspondência
direta. Não significa inexistência de anterioridade.

## Lacuna de pesquisa provisória

A hipótese de diferenciação não depende de uma feature isolada. Ela está na combinação verificável
de:

1. emissão estritamente causal e irreversível de uma cadeia contínua;
2. separação entre geometria do elo e relações entre elos;
3. ablation das relações sem alterar as fronteiras;
4. contabilização explícita de passos, payload e capacidade downstream;
5. previsão futura no relógio de eventos e decodificação cinemática recursiva.

O projeto deve demonstrar valor incremental de cada item. Uma combinação pode ser publicável mesmo
quando seus componentes são conhecidos, mas a redação precisa dizer `combinação`, `enquadramento`
ou `evidência`, e não reivindicar invenção de PLA, ângulo, curvatura ou previsão simbólica.

A área não deve ser descrita como abandonada: os trabalhos de 2024–2026 mostram atividade recente
em simbolização, quantização, LLMs e turning points. A oportunidade provisória parece resultar da
fragmentação entre essas linhas e a modelagem causal contínua de relações intersegmentos, não da
ausência geral de pesquisadores.

## Busca exigida antes de publicação

Antes de qualquer claim de novidade:

- pesquisar IEEE Xplore, ACM Digital Library, Scopus/Web of Science, Google Scholar e arXiv;
- cobrir `online/streaming PLA`, `adaptive polygonal chain`, `segment transition`, `turning angle`,
  `discrete curvature`, `event-based forecasting`, `symbolic forecasting` e `ABBA`;
- fazer backward e forward snowballing dos trabalhos acima;
- registrar consulta, data, base, resultados incluídos/excluídos e justificativa;
- comparar formalmente algoritmo, causalidade, estado, decoder, tarefa e protocolo experimental;
- substituir este levantamento por uma seção de related work revisável e bibliografia versionada.
