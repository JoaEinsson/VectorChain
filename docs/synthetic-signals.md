# Sinais sintéticos canônicos

Este documento registra as escolhas matemáticas dos sete geradores usados na baseline. Elas são
deliberadamente simples e não pretendem cobrir toda classe de sistemas dinâmicos.

## Convenções comuns

Para `n_points = n`, todos os sinais usam tempo normalizado e inclusivo:

```text
t[i] = i / (n - 1),  i = 0, ..., n - 1
```

O sinal retornado é `offset + amplitude * shape(t) + noise`, em `float64`. Quando `noise_std > 0`,
`noise` é gaussiano independente com média zero e desvio padrão `noise_std`, medido nas unidades do
sinal final.

Toda função exige `rng`, que pode ser uma seed inteira ou `numpy.random.Generator`. Uma seed cria um
gerador local reproduzível; um objeto `Generator` é consumido somente quando há ruído. O estado
aleatório global do NumPy nunca é usado.

Frequências são ciclos por intervalo normalizado. Fase é medida em radianos. Parâmetros físicos não
devem ser inferidos dessa escala sem uma conversão explícita.

## Formas

- `generate_sine`: `sin(2*pi*frequency*t + phase)`.
- `generate_chirp`: chirp linear cuja fase em ciclos é
  `start_frequency*t + (end_frequency-start_frequency)*t**2/2`.
- `generate_ramp`: `shape(t) = t`; `amplitude` é a elevação total.
- `generate_piecewise_linear`: interpolação entre tempos `(0, .2, .45, .7, 1)` e valores
  `(0, .8, -.4, 1, .2)`.
- `generate_first_order_response`: resposta ao degrau `1 - exp(-t/time_constant)`.
- `generate_second_order_response`: resposta ao degrau padrão de um sistema de segunda ordem com
  `0 < damping_ratio < 1`; `natural_frequency` é convertida para radianos por tempo normalizado.
- `generate_regime_change`: seno com fase contínua, troca de frequência em `change_fraction` e
  adição de `level_shift` depois da mudança. O shift é multiplicado por `amplitude` como parte da
  forma canônica.

## Parâmetros default

Os defaults definem apenas formas nominais. A baseline científica continua definindo `n_points`,
`noise_std` e seed em `configs/reconstruction/baseline.toml`; o runner experimental deve sempre
registrar os valores efetivamente usados.
