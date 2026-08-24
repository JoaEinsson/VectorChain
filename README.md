# VectorChain

VectorChain é um projeto científico para investigar se séries temporais escalares podem ser
representadas de forma útil como uma cadeia causal e adaptativa de vetores.

O projeto está em fase **pré-alpha**, com desenvolvimento científico ativo encerrado. A Fase I e a
cadeia revisável K7 foram concluídas com resultados reproduzíveis. K7 confirmou somente que revisar
a geometria provisória ajuda a própria baseline geométrica; suas atualizações temporais falharam.
O teste eliminatório posterior da trajetória completa de revisões também foi negativo. A evidência
atual sustenta resultados diagnósticos delimitados no benchmark sintético, não novidade de nível
médio, superioridade geral ou utilidade prática.

## Pergunta de pesquisa

> Uma representação causal e adaptativa de séries temporais como vetores pode preservar dinâmica
> relevante, reduzir o comprimento efetivo da sequência e ajudar em reconstrução, similaridade,
> forecasting e detecção de mudança?

A hipótese pós-MVP deriva da inspiração em cinemática inversa aplicada ao gráfico da série:

> Relações cinemáticas entre segmentos adaptativos contêm informação preditiva além da geometria
> dos segmentos individuais?

O claim mais forte da Fase I perguntou se a própria cadeia poderia funcionar como espaço de estado
autoregressivo causal e compacto. O teste direto foi válido, mas perdeu para AR raw e encerrou essa
formulação.

O escopo original está preservado em [`prompt_inicial.md`](prompt_inicial.md). As definições
executáveis do método serão mantidas em [`docs/specification.md`](docs/specification.md), e a
causalidade em [`docs/causality-contract.md`](docs/causality-contract.md).

O enquadramento pós-MVP está em
[`docs/kinematic-hypothesis.md`](docs/kinematic-hypothesis.md), o programa experimental em
[`docs/post-mvp-claim-protocol.md`](docs/post-mvp-claim-protocol.md) e a literatura mais próxima em
[`docs/closest-prior-art.md`](docs/closest-prior-art.md).

A interpretação final da Fase I está em
[`docs/phase-1-scientific-synthesis.md`](docs/phase-1-scientific-synthesis.md). A nova hipótese K6,
sobre tokenização causal sob orçamento de sequência, foi congelada em
[`docs/phase-2-adaptive-tokenization-protocol.md`](docs/phase-2-adaptive-tokenization-protocol.md).
Uma hipótese separada, K7, testa atualizações temporais numa cauda articulada revisável e está em
[`docs/phase-2-revisable-chain-protocol.md`](docs/phase-2-revisable-chain-protocol.md).

## Estado do projeto

- [x] Charter e escopo inicial
- [x] Estrutura de governança e reprodutibilidade
- [x] Núcleo de segmentação causal adaptativa
- [x] Reconstrução e métricas fundamentais
- [x] Sinais sintéticos e visualização científica
- [x] Experimento compressão × reconstrução
- [x] Similaridade, retrieval e ablations
- [x] Forecasting mínimo
- [x] Grade fatorial de robustez do forecasting
- [x] MVP científico inicial concluído
- [x] Isolamento do mecanismo cinemático, com gate negativo preservado
- [x] Controles ABBA/PLA/smoothing e capacidade pareada, com gate negativo preservado
- [ ] Confirmação em dados inéditos e reais, bloqueada pelos gates da Fase I
- [x] Forecasting autoregressivo diretamente na cadeia, com gate K5-A negativo preservado
- [x] Síntese científica K1–K5 e encerramento da Fase I
- [x] Protocolo da Fase II congelado antes de código ou execução
- [ ] Implementação e execução da Etapa 11-A
- [x] Protocolo mínimo da cadeia revisável K7 congelado
- [x] Núcleo estrutural causal da cauda revisável K7
- [x] Três sinais e seis representações pareadas K7, testados somente com seeds 11/22
- [x] Runner K7 separado de treino/validação, sem acesso ao teste fechado
- [x] Seleção canônica K7 congelada antes da abertura do teste
- [x] Implementação e execução da Etapa 12-A, com K7 completo negativo preservado
- [x] Teste eliminatório da trajetória completa de revisões, negativo e reproduzido
- [x] Condição de parada final documentada

## Direção pós-MVP

O resultado inicial pertenceu ao pacote `(dt, dy, theta, r, delta_theta)`. As ablations posteriores
mostraram que `delta_theta/delta_r` não causaram melhora consistente, e os controles simples
mostraram que smoothing FIR concorre com a geometria absoluta. A interpretação foi reduzida de
acordo com esses gates, sem alterar retrospectivamente o benchmark original.

A tolerância `0.1` continua candidata pós-análise, não novo default. Os controles pareados foram
executados: média móvel trailing e segmentação fixa impediram atribuir o resultado à adaptação ou a
um mecanismo cinemático exclusivo. O forecasting vetorial recursivo também foi executado como nova
pergunta exploratória: produziu rollouts válidos e compactos, mas perdeu consistentemente para AR
raw e não sustentou o claim de espaço de estado.

A continuação não retuna essas perguntas. A Fase II usa apenas `(dt, dy)` e testa se tokens
adaptativos ordenados preservam mais contexto sob um orçamento fixo de comprimento de sequência.
O gate exige controles de mesmo número de tokens e raw de mesmo payload escalar; dados externos e
modelos maiores permanecem condicionados ao teste linear sintético.

K7 trata outra variável: não `delta_theta/delta_r` entre elos num snapshot, mas
`update_theta/update_r` da mesma junta entre duas observações. Somente quatro elos provisórios podem
ser rearticulados; o prefixo comprometido continua imutável. O primeiro teste usa três modulações
oscilatórias isoladas — frequência, baseline e assimetria da crista — sem sinal combinado ou grade
desnecessária.

O teste canônico de K7 confirmou a vantagem da geometria revisada sobre a imutável, mas rejeitou o
valor das atualizações como features e perdeu para raw. O último teste preservou a geometria atual e
acrescentou toda a linhagem recursiva das revisões. A trajetória real venceu sua versão embaralhada,
mas piorou a geometria isolada e teve aproximadamente o dobro do RMSE do raw pareado. Isso encerra
a hipótese sem nova rodada de representação ou modelo.

## Ambiente de desenvolvimento

O projeto usa Python 3.11 ou posterior e `uv` para ambiente e dependências. A versão local
recomendada é Python 3.12.

```powershell
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/vectorchain
```

## Uso do núcleo causal

```python
from vectorchain import VectorChain

vc = VectorChain(
    tolerance=0.03,
    causal=True,
    features=("dt", "dy", "theta", "r", "delta_theta"),
)

vectors = vc.fit_transform([0.0, 0.2, 0.4, 0.41, 0.42])
reconstructed = vc.inverse_transform(vectors)

print(vectors)
print(reconstructed)
print(vc.segment_boundaries_)
print(vc.compression_factor_, vc.reconstruction_error_)
```

Para processamento online, use a mesma máquina de estado consumida pelo wrapper batch:

```python
vc.reset()
segments = []
for value in stream:
    segments.extend(vc.update(value))
segments.extend(vc.finalize())
```

O experimento K7 usa um componente separado; ele não muda segmentos emitidos por `VectorChain`:

```python
from vectorchain import RevisableVectorChain

tail = RevisableVectorChain(
    tolerance=0.03,
    lambda_revision=0.1,
    lambda_bend=0.1,
)

for value in stream:
    version = tail.update(value)
    current_links = version.links

immutable_prefix = tail.committed_
version_history = tail.versions_
audit_events = tail.events_
```

Cada observação aceita cria uma `WorkingVersion` imutável. A cauda contém no máximo quatro elos e
256 intervalos raw; o elo completo mais antigo passa para `committed_` antes de exceder o limite.
`update_theta/update_r` comparam apenas o mesmo `link_id`, e elos novos recebem zero. O wrapper
`fit_transform` apenas repete `update` e não compromete artificialmente a cauda ao terminar a
entrada. Esse núcleo ainda não constitui um resultado científico de K7.

As features disponíveis são `dt`, `dy`, `theta`, `r`, `delta_theta` e `delta_r`. `dt` e `dy` são
obrigatórias no primeiro MVP. Alterar a seleção ou a ordem das features não muda as fronteiras.

`inverse_transform` reconstrói uma amostra por índice por meio de interpolação linear. Ele aceita
qualquer ordem configurada de features e permite alterar `dy`, desde que a quantidade de vetores e
os valores de `dt` continuem compatíveis com as fronteiras do ajuste. `compression_factor_` mede
`n_points / n_vectors`; é redução estrutural do comprimento da sequência, não redução em bytes.

Os sete sinais canônicos exigem uma seed ou um `numpy.random.Generator` explícito:

```python
from vectorchain import generate_chirp

signal = generate_chirp(rng=1729, n_points=1000, noise_std=0.01)
```

Para visualizar original, reconstrução, segmentos e articulações com a dependência opcional:

```python
from vectorchain import VectorChain
from vectorchain.plotting import plot_vector_chain

vc = VectorChain(tolerance=0.03)
vc.fit_transform(signal)
axis = plot_vector_chain(signal, vc, title="chirp | seed=1729")
axis.figure.savefig("vectorchain.png", dpi=150)
```

As fórmulas e unidades estão registradas em
[`docs/synthetic-signals.md`](docs/synthetic-signals.md).

O primeiro benchmark reproduzível e sua análise estão em
[`reports/reference/reconstruction-baseline/`](reports/reference/reconstruction-baseline/). O
resultado identifica um compromisso útil na condição nominal, mas também registra pouca compressão
em tolerâncias próximas ao ruído e forte dependência da geometria do sinal.

A comparação pré-especificada de similaridade está em
[`reports/reference/similarity-retrieval-baseline/`](reports/reference/similarity-retrieval-baseline/).
Nesta primeira tarefa sintética, baselines raw, normalizada, diferenças e segmentação fixa superaram
as ablations VectorChain no top-1. O resultado negativo é preservado como orientação para a próxima
iteração, não ocultado por seleção de configuração.

O benchmark de forecasting está em
[`reports/reference/minimal-forecasting-baseline/`](reports/reference/minimal-forecasting-baseline/).
VectorChain usou 10,58× menos passos e 2,12× menos valores escalares, com RMSE de teste 9,98% maior
que raw. O limite de paridade de 10% foi satisfeito por margem mínima no teste e falhou na validação;
o relatório preserva essa fragilidade e o custo de um modelo pooled maior.

A continuação com cinco seeds, três horizontes, três contextos e três tolerâncias está em
[`reports/reference/forecasting-robustness-grid/`](reports/reference/forecasting-robustness-grid/).
Quatorze de 27 células de teste foram robustas em pelo menos quatro de cinco seeds. A tolerância
`0.1` foi a região mais favorável nesta grade sintética, mas é mantida como hipótese para confirmação
independente, não como novo padrão escolhido sobre os próprios resultados.

O roadmap completo até o claim autoregressivo está em [`docs/roadmap.md`](docs/roadmap.md). A
conclusão do MVP e o início desse programa foram registrados no
[`ADR 0008`](docs/decisions/0008-post-mvp-kinematic-claim-program.md).

A primeira ablation cinemática pós-MVP está em
[`reports/reference/forecasting-kinematic-feature-ablation/`](reports/reference/forecasting-kinematic-feature-ablation/).
O gate relacional não passou: `delta_theta/delta_r` não acrescentaram efeito consistente sobre a
geometria absoluta no ridge pooled. O resultado negativo redireciona os próximos controles para a
utilidade de `theta/r`, sem sustentar informação relacional incremental.

Os controles resultantes estão em
[`reports/reference/forecasting-absolute-geometry-controls/`](reports/reference/forecasting-absolute-geometry-controls/).
`absolute_geometry` superou geometria local e EWMA, mas o gate global falhou contra média móvel
trailing e segmentação fixa. O claim atual é, portanto, um trade-off empírico delimitado no
benchmark sintético; não é evidência de um mecanismo adaptativo ou relacional exclusivo.

O teste direto da cadeia como estado está em
[`reports/reference/forecasting-vector-state-rollout/`](reports/reference/forecasting-vector-state-rollout/).
Sem pooling, o estado causal completou rollouts compactos e venceu controles multioutput pareados,
mas teve RMSE maior que AR raw nas cinco seeds e nos três horizontes. K5-A não avançou; modelos
maiores e confirmação externa permanecem fechados por essa condição de parada.

O balanço da Fase I está na
[`síntese científica`](docs/phase-1-scientific-synthesis.md). O protocolo K6 permanece preservado
como hipótese não executada, mas não é uma próxima etapa ativa após a decisão de parada.

A cadeia revisável possui protocolo próprio em
[`docs/phase-2-revisable-chain-protocol.md`](docs/phase-2-revisable-chain-protocol.md), ADR próprio e
seeds diferentes. Ela não muda a API causal atual nem pode ser usada para reinterpretar o resultado
negativo das relações espaciais da Fase I.

O último teste, sua réplica byte a byte e a decisão final estão em
[`reports/reference/revision-path-kill-test/`](reports/reference/revision-path-kill-test/). A
trajetória completa venceu somente o controle embaralhado (`0,9556`), perdeu para geometria
(`1,0409`) e última atualização (`1,0629`) e ficou muito atrás de raw (`2,0462`).

Consulte [`CONTRIBUTING.md`](CONTRIBUTING.md) para o fluxo completo e
[`docs/reproducibility.md`](docs/reproducibility.md) para reprodução de experimentos.

As proteções que precisam ser habilitadas na interface do GitHub estão listadas em
[`docs/repository-settings.md`](docs/repository-settings.md).

## Licença

VectorChain é distribuído sob a [Apache License 2.0](LICENSE).
