# VectorChain

VectorChain é um projeto científico para investigar se séries temporais escalares podem ser
representadas de forma útil como uma cadeia causal e adaptativa de vetores.

O projeto está em fase **pré-alpha**. O MVP científico inicial foi concluído: núcleo causal,
reconstrução, similaridade, forecasting e robustez possuem resultados reproduzíveis. A evidência
atual sustenta somente um resultado delimitado no benchmark sintético; não sustenta superioridade
geral nem identifica ainda o mecanismo responsável pelo ganho observado.

## Pergunta de pesquisa

> Uma representação causal e adaptativa de séries temporais como vetores pode preservar dinâmica
> relevante, reduzir o comprimento efetivo da sequência e ajudar em reconstrução, similaridade,
> forecasting e detecção de mudança?

A hipótese pós-MVP deriva da inspiração em cinemática inversa aplicada ao gráfico da série:

> Relações cinemáticas entre segmentos adaptativos contêm informação preditiva além da geometria
> dos segmentos individuais?

O claim mais forte, reservado para uma etapa posterior, pergunta se a própria cadeia pode funcionar
como espaço de estado autoregressivo causal e compacto.

O escopo original está preservado em [`prompt_inicial.md`](prompt_inicial.md). As definições
executáveis do método serão mantidas em [`docs/specification.md`](docs/specification.md), e a
causalidade em [`docs/causality-contract.md`](docs/causality-contract.md).

O enquadramento pós-MVP está em
[`docs/kinematic-hypothesis.md`](docs/kinematic-hypothesis.md), o programa experimental em
[`docs/post-mvp-claim-protocol.md`](docs/post-mvp-claim-protocol.md) e a literatura mais próxima em
[`docs/closest-prior-art.md`](docs/closest-prior-art.md).

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
- [ ] Isolamento do mecanismo cinemático
- [ ] Controles ABBA/PLA/smoothing e capacidade pareada
- [ ] Confirmação em dados inéditos e reais
- [ ] Forecasting autoregressivo diretamente na cadeia

## Direção pós-MVP

O resultado atual pertence ao pacote `(dt, dy, theta, r, delta_theta)`. Ele não demonstra que
`delta_theta` causou a melhora: o ridge VectorChain possui mais entradas que raw e uma tolerância
alta também pode atuar como suavizador. A próxima etapa preserva as fronteiras e compara
progressivamente segmento básico, geometria absoluta e relações cinemáticas.

A tolerância `0.1` continua candidata pós-análise, não novo default. Antes de qualquer claim de
mecanismo, VectorChain será comparado com ABBA/fABBA, PLA causal, downsampling e smoothing sob
payload e capacidade downstream pareados. O forecasting vetorial recursivo somente começa depois
de esses controles e de uma confirmação externa passarem seus gates.

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

Consulte [`CONTRIBUTING.md`](CONTRIBUTING.md) para o fluxo completo e
[`docs/reproducibility.md`](docs/reproducibility.md) para reprodução de experimentos.

As proteções que precisam ser habilitadas na interface do GitHub estão listadas em
[`docs/repository-settings.md`](docs/repository-settings.md).

## Licença

VectorChain é distribuído sob a [Apache License 2.0](LICENSE).
