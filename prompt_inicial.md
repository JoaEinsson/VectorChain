Quero construir um MVP científico em Python chamado provisoriamente **VectorChain** para testar a hipótese de que séries temporais podem ser representadas de forma útil como uma cadeia adaptativa de vetores, em vez de apenas como pontos amostrados uniformemente no tempo.

O objetivo desta primeira versão NÃO é criar um framework complexo nem buscar state of the art. Quero a implementação mínima, limpa, reproduzível e cientificamente defensável para responder à pergunta:

> Uma representação causal e adaptativa de séries temporais como vetores pode preservar dinâmica relevante, reduzir o comprimento efetivo da sequência e ajudar em tarefas como reconstrução, similaridade, forecasting e detecção de mudança?

## Conceito central

Dada uma série temporal escalar:

[
x_1,x_2,\ldots,x_T
]

quero segmentá-la causalmente em trechos aproximadamente lineares.

Cada trecho deve ser representado por um vetor:

[
V_i=(\Delta t_i,\Delta y_i,\theta_i,r_i,\Delta\theta_i)
]

onde:

[
\Delta t_i=t_{fim}-t_{inicio}
]

[
\Delta y_i=x_{fim}-x_{inicio}
]

[
\theta_i=\operatorname{atan2}(\Delta y_i,\Delta t_i)
]

[
r_i=\sqrt{\Delta t_i^2+\Delta y_i^2}
]

[
\Delta\theta_i=\theta_i-\theta_{i-1}
]

Se necessário, inclua posteriormente outras features simples como:

* (\Delta r_i)
* erro de reconstrução do segmento
* variância/resíduo dentro do segmento

Mas não adicione complexidade sem necessidade.

## Restrição mais importante: causalidade

A segmentação usada para forecasting deve ser **online/causal**.

Ao decidir se o ponto (x_t) pertence ao segmento atual ou inicia um novo vetor, o algoritmo não pode utilizar nenhum ponto futuro.

Não quero leakage causado por segmentação offline.

Pode existir uma implementação offline apenas como comparação opcional, mas o modo principal deve ser:

```python
VectorChain(causal=True)
```

## Segmentação inicial

Implemente primeiro uma estratégia simples baseada em erro de aproximação linear.

Ideia:

1. Começar um segmento em um ponto.
2. Adicionar pontos sequencialmente.
3. Ajustar ou atualizar uma linha usando somente os pontos vistos até aquele momento.
4. Enquanto o erro permanecer abaixo de uma tolerância, continuar o segmento.
5. Quando ultrapassar o limiar, fechar o vetor anterior e iniciar um novo.

Quero uma solução simples e compreensível antes de qualquer algoritmo sofisticado.

Parâmetros iniciais possíveis:

```python
VectorChain(
    tolerance=0.03,
    causal=True,
    min_segment_length=2
)
```

Evite dependências desnecessárias.

## API desejada

Algo aproximadamente assim:

```python
vc = VectorChain(
    tolerance=0.03,
    causal=True
)

Z = vc.fit_transform(x)

x_hat = vc.inverse_transform(Z)

print(Z)
```

Cada vetor deve poder ser retornado em formato estruturado, por exemplo DataFrame ou ndarray.

Quero acesso a:

```python
vc.vectors_
vc.segment_boundaries_
vc.compression_ratio_
vc.reconstruction_error_
```

A reconstrução não precisa ser perfeita. Ela deve interpolar os segmentos vetoriais adequadamente.

## Experimentos prioritários

Não comece com Transformer.

Quero primeiro testar se a própria representação possui propriedades úteis.

### Experimento 1 — Reconstrução e compressão

Para cada série:

* número original de pontos
* número de vetores
* compression ratio
* MAE
* RMSE de reconstrução
* runtime

Produza curvas de:

[
\text{compression ratio} \times \text{reconstruction error}
]

variando `tolerance`.

O objetivo é observar se aparece um bom compromisso entre compressão e preservação da forma.

### Experimento 2 — Invariância / generalização geométrica

Gere sinais sintéticos da mesma dinâmica com:

* amplitudes diferentes
* offsets diferentes
* algum ruído
* eventualmente escalas temporais diferentes

Exemplos:

* seno
* seno com frequência variável
* chirp
* rampas
* piecewise linear
* resposta de primeira ordem
* resposta subamortecida de segunda ordem
* sinais com mudança de regime

Compare a distância entre séries usando:

1. valores crus
2. first differences
3. representação VectorChain
4. DTW, se for simples adicionar

Teste se versões da mesma dinâmica ficam próximas no espaço vetorial apesar de diferenças de escala/offset.

Não assuma que vai funcionar: reporte resultado negativo normalmente.

### Experimento 3 — Similarity retrieval

Escolha um trecho de sinal.

Procure os (k) trechos mais semelhantes usando:

* série crua
* diferenças
* VectorChain

Veja se VectorChain recupera padrões de dinâmica semelhantes.

Comece com nearest-neighbor simples.

Não use neural network aqui.

### Experimento 4 — Forecasting mínimo

Somente depois dos anteriores.

Compare exatamente o mesmo modelo downstream com inputs diferentes:

A:

[
x_t
]

B:

[
\Delta x_t
]

C:

[
V_i
]

Pode começar com modelos simples:

* linear regression / autoregression
* MLP pequeno
* GRU ou LSTM pequeno

Não quero arquitetura sofisticada no primeiro benchmark.

O objetivo é testar a representação, não provar que uma rede complexa consegue aprender.

Registre:

* MAE
* RMSE
* tamanho da sequência de entrada
* tempo de treino/inferência
* memória aproximada se for fácil medir

## Ablations

Quero conseguir testar facilmente estas representações:

```text
(dt, dy)

(dt, dy, theta)

(dt, dy, theta, r)

(dt, dy, theta, r, delta_theta)

(dt, dy, theta, r, delta_theta, delta_r)
```

Isso é importante para descobrir se alguma feature realmente adiciona informação.

Em particular, quero testar se `delta_theta`, interpretado como mudança de direção/curvatura local, adiciona valor além de simples differencing.

## Baselines obrigatórios

Inclua pelo menos:

* raw values
* normalized raw values
* first difference
* first + second difference
* fixed-size linear segmentation
* adaptive VectorChain

Se houver uma implementação simples de piecewise linear approximation, pode incluir também.

O objetivo é impedir que VectorChain pareça novo apenas porque não foi comparado contra transformações equivalentes.

## Dados sintéticos

Crie um módulo para geração reprodutível de sinais.

Use seeds fixas.

Inclua pelo menos:

```python
generate_sine()
generate_chirp()
generate_ramp()
generate_piecewise_linear()
generate_first_order_response()
generate_second_order_response()
generate_regime_change()
```

Permita variar:

* amplitude
* offset
* frequência
* ruído
* número de pontos

## Visualizações

Quero plots simples e científicos usando matplotlib.

Não faça dashboards.

Inclua:

1. série original
2. reconstrução VectorChain
3. vetores/segmentos sobrepostos ao sinal
4. erro por tolerance
5. compression ratio por tolerance
6. comparação de retrieval quando aplicável

Uma visualização importante:

mostrar os vetores consecutivos como segmentos com seus pontos inicial e final, de modo que seja visualmente evidente a ideia de “cadeia articulada”.

## Estrutura sugerida do projeto

```text
vectorchain/
    __init__.py
    core.py
    features.py
    reconstruction.py
    metrics.py
    synthetic.py
    similarity.py

experiments/
    01_reconstruction.py
    02_invariance.py
    03_retrieval.py
    04_forecasting.py
    05_ablations.py

tests/
    test_core.py
    test_causality.py
    test_reconstruction.py
    test_features.py

README.md
requirements.txt
```

Pode adaptar se houver uma estrutura melhor, mas mantenha simples.

## Teste de causalidade obrigatório

Quero um teste explícito garantindo que alterar dados futuros não modifica vetores que já deveriam ter sido produzidos no passado.

Por exemplo:

```python
prefix = signal[:100]

a = vectorize(prefix)

modified = full_signal.copy()
modified[100:] = valores_completamente_diferentes

b = vectorize(modified)

assert vectors_up_to_t_100(a) == vectors_up_to_t_100(b)
```

A implementação exata pode variar, mas a propriedade deve ser testada.

## Princípios de implementação

Prioridades, nesta ordem:

1. correção
2. causalidade
3. clareza
4. reprodutibilidade
5. simplicidade
6. performance

Não faça premature optimization.

Não crie classes abstratas ou arquiteturas enormes sem necessidade.

Use type hints e docstrings.

Quando uma escolha matemática for arbitrária, documente-a.

Quando houver tradeoff, exponha como parâmetro em vez de escondê-lo.

## Critério de sucesso inicial

O MVP já será útil se conseguirmos responder claramente:

1. Quanto uma série pode ser reduzida em número de elementos?
2. Qual erro de reconstrução essa redução produz?
3. A representação agrupa dinâmicas semelhantes melhor que raw/differencing?
4. `delta_theta` ou outras propriedades geométricas adicionam informação?
5. Forecasting mantém desempenho semelhante usando significativamente menos elementos?
6. A representação continua válida em modo estritamente causal?

Não tente provar que VectorChain é superior.

Quero descobrir honestamente em quais condições ela é útil e em quais falha.

## Entregável inicial

Implemente primeiro somente:

* `VectorChain`
* transformação
* reconstrução
* geração de sinais sintéticos
* métricas
* testes de causalidade
* experimento de compression × reconstruction
* visualização dos segmentos

Depois rode os testes e o primeiro experimento.

Somente após essa base funcionar, avance para similarity/retrieval e forecasting.

Ao terminar cada etapa, faça commit lógico ou mantenha as mudanças claramente separadas para que seja fácil revisar e reverter.
