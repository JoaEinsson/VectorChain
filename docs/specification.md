# Especificação matemática do MVP

Status: **proposta aceita para a primeira implementação**.

Este documento transforma o charter de `prompt_inicial.md` em um contrato implementável. Mudanças
na semântica abaixo exigem um ADR, testes de regressão e atualização do protocolo experimental.

## 1. Entrada

A entrada inicial é uma série escalar unidimensional e uniformemente amostrada:

```text
x = (x[0], x[1], ..., x[n-1])
```

Regras do MVP:

- `n >= 2`;
- valores devem ser reais e finitos;
- o tempo implícito é `t[i] = i`;
- `tolerance >= 0`;
- `min_segment_length` é inteiro e `>= 2`;
- entradas inválidas produzem `ValueError`, sem imputação silenciosa.

Suporte a timestamps irregulares e valores ausentes fica fora do primeiro MVP.

## 2. Fronteiras e articulação

Uma fronteira é o par inclusivo `(start, end)`, com `0 <= start < end < n`. Segmentos consecutivos
compartilham seu ponto de articulação:

```text
start[i] = end[i - 1]
```

Assim, não há lacunas e a cadeia permanece conectada. O ponto compartilhado não representa uma
amostra nova; ele é apenas o final de um vetor e o início do próximo.

## 3. Regra de segmentação inicial

Para um segmento candidato `(s, e)`, a aproximação é a corda que liga os valores observados nos
endpoints:

```text
y_hat[j] = x[s] + (j - s) * (x[e] - x[s]) / (e - s),  s <= j <= e
```

O erro de decisão é o maior resíduo absoluto no segmento candidato:

```text
segment_error(s, e) = max(abs(x[j] - y_hat[j]))
```

A corda foi escolhida porque coincide com a reconstrução definida pelo vetor e torna a tolerância
diretamente interpretável. Não é uma regressão por mínimos quadrados.

Processamento online:

1. Inicie com os pontos `0` e `1`.
2. Receba o próximo ponto `e`, usando somente o prefixo observado até `e`.
3. Se o tamanho mínimo ainda não tiver sido atingido, aceite o ponto.
4. Caso contrário, aceite se `segment_error(s, e) <= tolerance`.
5. Se houver violação, emita `(s, e - 1)` e inicie o novo segmento articulado `(e - 1, e)`.
6. Ao terminar explicitamente o stream, emita o segmento aberto.

`min_segment_length` é uma restrição operacional sobre fechamentos normais. O segmento terminal pode
ser menor quando o stream é finalizado. Para valores maiores que `2`, a aceitação forçada até o
mínimo também pode fazer o erro ultrapassar a tolerância; esse comportamento será reportado, não
ocultado.

## 4. Vetores e features

Para cada fronteira `(start, end)`:

```text
dt          = end - start
dy          = x[end] - x[start]
theta       = atan2(dy, dt)
r           = sqrt(dt**2 + dy**2)
delta_theta = theta - previous_theta
```

Como `dt > 0`, `theta` fica no intervalo `(-pi/2, pi/2)`. Não é necessário circular
`delta_theta`. Para o primeiro vetor, `delta_theta = 0.0`; essa escolha é convencional e deve ser
incluída nas ablations.

A ordem canônica das colunas é:

```text
(dt, dy, theta, r, delta_theta)
```

Todas as colunas são `float64` no primeiro MVP. `segment_boundaries_` usa inteiros com shape
`(n_vectors, 2)`.

`r` combina unidades de tempo discreto e amplitude. Portanto, não deve ser interpretado como
distância física sem normalização explícita. Isso é uma limitação da feature, não um detalhe de
implementação.

## 5. Reconstrução

A reconstrução interpola linearmente cada vetor entre seus endpoints. Pontos de articulação são
escritos uma única vez e segmentos adjacentes devem produzir exatamente o mesmo valor nesse ponto.

`inverse_transform(Z)` no objeto ajustado usa `initial_value_`, `n_samples_` e as fronteiras
armazenadas. Um array contendo apenas `(dt, dy, ...)` não determina o offset vertical; reconstrução
independente exige fornecer o valor inicial explicitamente em uma API futura.

## 6. Atributos públicos planejados

Após `fit_transform`:

- `vectors_`: array `(n_vectors, 5)` na ordem canônica;
- `segment_boundaries_`: array inteiro `(n_vectors, 2)`;
- `compression_ratio_`: alias documentado do fator estrutural `n_points / n_vectors`;
- `compression_factor_`: `n_points / n_vectors`;
- `retention_fraction_`: `n_vectors / n_points`;
- `reconstruction_error_`: RMSE da reconstrução no domínio original;
- `initial_value_`: primeiro valor da série;
- `n_samples_`: número original de amostras.

O fator mede redução no comprimento da sequência, não redução em bytes: um vetor possui várias
features. Relatórios devem mostrar `n_points` e `n_vectors`, nunca apenas o fator agregado.

## 7. Métricas

Para resíduos `e[i] = x[i] - x_hat[i]`:

```text
MAE  = mean(abs(e))
RMSE = sqrt(mean(e**2))
compression_factor = n_points / n_vectors
retention_fraction = n_vectors / n_points
```

Runtime usa relógio monotônico de alta resolução, inclui somente transformação ou reconstrução
explicitamente identificada e é reportado como mediana e intervalo interquartil de repetições.

## 8. Não objetivos do MVP

- Garantir invariância a amplitude ou escala temporal.
- Tratar `r` como medida física universal.
- Otimizar incrementalmente a avaliação de erro antes de medir necessidade.
- Aceitar dados multivariados ou irregulares.
- Oferecer compatibilidade estável de serialização.
- Alegar vantagem sobre baselines antes dos experimentos.
