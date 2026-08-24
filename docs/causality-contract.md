# Contrato de causalidade

Status: **normativo para todo caminho marcado como causal**.

## Definição

Depois de consumir a amostra de índice `t`, o estado e qualquer saída emitida podem depender somente
de `x[0:t+1]`, parâmetros explícitos e estado derivado desse mesmo prefixo.

Formalmente, para dois sinais `x` e `y` com prefixos iguais até `t`, executar a mesma sequência de
operações online até `t` deve produzir estados observáveis e segmentos emitidos iguais:

```text
x[0:t+1] == y[0:t+1]  =>  emitted_x(t) == emitted_y(t)
```

## Segmento emitido versus segmento aberto

Um segmento pode terminar em `t - 1` e ser emitido somente quando a amostra `t` revela que a
tolerância foi ultrapassada. Alterar `x[t]` pode, portanto, mudar a decisão tomada em `t` sem violar
causalidade. O instante de emissão, e não apenas o endpoint, determina o prefixo permitido.

O segmento atualmente aberto é provisório. `finalize()` transforma esse estado provisório em saída
porque o chamador declarou o fim do stream. Essa operação terminal não deve ser incluída em uma
comparação de prefixos com um stream que continua.

## API e estado esperados

A implementação deve ter uma única transição causal de estado, usada pelo modo batch:

```text
reset() -> estado inicial
update(x_t) -> zero ou mais segmentos finalizados
finalize() -> segmento terminal aberto
fit_transform(x) -> reset; update para cada ponto; finalize
```

Não deve existir um algoritmo offline separado por trás de `fit_transform(causal=True)`.

## Propriedades obrigatórias

1. **Invariância de prefixo:** sufixos arbitrariamente diferentes não alteram saídas já emitidas.
2. **Equivalência batch/stream:** o wrapper batch coincide com `update` repetido e `finalize`.
3. **Imutabilidade:** um segmento emitido nunca é editado posteriormente.
4. **Isolamento de reset:** processar uma nova série após `reset` não depende da série anterior.
5. **Determinismo:** mesmos dados, parâmetros e versão produzem as mesmas fronteiras e features.
6. **Sem relógio oculto:** decisões usam índices ou timestamps fornecidos, nunca horário de execução.
7. **Independência de features:** mudar as colunas projetadas não altera nenhuma fronteira.

## Teste de alteração futura

O teste recomendado deve operar diretamente sobre o stream:

```python
left = VectorChain(...)
right = VectorChain(...)

for value in prefix:
    emitted_left.extend(left.update(value))
    emitted_right.extend(right.update(value))

for value in original_suffix:
    left.update(value)

for value in modified_suffix:
    right.update(value)

assert emitted_left == emitted_right
```

Testes baseados em propriedades devem variar prefixo, sufixo, tolerância, ruído e posição de corte.
O segmento aberto no corte deve ser comparado como estado provisório somente se a API tornar essa
representação pública e documentada.

## Práticas proibidas no modo causal

- Ajustar tolerância usando estatísticas calculadas sobre a série completa.
- Normalizar usando média, variância, mínimo ou máximo futuros.
- Recalcular segmentos passados depois que forem emitidos.
- Usar detecção de changepoint que examine os dois lados do corte.
- Preencher um valor ausente usando interpolação com o próximo ponto.
- Comparar um prefixo finalizado artificialmente com um stream ainda aberto e chamar a diferença de
  leakage.

## Extensão causal revisável

O [`ADR 0013`](decisions/0013-bounded-revisable-tail.md) autoriza pesquisar uma extensão separada
com mais de um elo provisório. Ela não altera a API nem a semântica do `VectorChain` atual.

A implementação estrutural dessa extensão é `RevisableVectorChain`. Seu `update(x_t)` acrescenta
uma `WorkingVersion` imutável e seu `fit_transform(x)` é somente a repetição dessa transição; o fim
de um array não compromete artificialmente a cauda. `versions_` e `events_` são logs append-only,
enquanto `committed_` contém snapshots que nunca voltam ao estado de trabalho.

Nessa extensão:

- `committed` é o único equivalente a emitido e continua imutável;
- `working` é uma cauda causal, provisória, limitada e explicitamente versionada;
- uma nova amostra pode produzir uma nova versão de `working`, nunca editar uma versão já registrada;
- o mesmo prefixo observado, parâmetros e estado anterior devem produzir a mesma nova versão;
- um sufixo futuro pode mudar versões futuras da cauda, mas jamais o prefixo comprometido;
- toda revisão usa somente amostras disponíveis no instante declarado;
- compromisso, revisão e criação de elo são eventos distintos no log.

Os testes de alteração futura passam a verificar duas propriedades separadas:

```text
prefixos iguais até t => committed(t) e working_version(t) iguais
sufixos diferentes depois de t => todo committed_at_or_before(t) permanece igual
```

Chamar um elo revisável de emitido, sobrescrever seu histórico ou omitir a latência de compromisso é
violação do contrato.
