# 0011 — Estado causal de emissão para rollout autoregressivo

Status: **Accepted**
Data: 2026-08-23

## Contexto

O forecasting anterior finaliza uma cadeia dentro de cada janela, faz pooling e prevê um incremento
raw. Ele não testa a cadeia como sequência ordenada nem como estado recursivo. Ao mesmo tempo, um
segmento online só se torna observável quando a primeira amostra do próximo segmento provoca sua
emissão. Posicionar o elo no próprio endpoint usaria informação disponível apenas um passo depois.

As Etapas 7 e 8 também rejeitaram, respectivamente, o efeito relacional incremental e a exclusão
de smoothing/fronteiras fixas como explicações. A nova tarefa precisa preservar esses resultados e
não os transformar em confirmação retrospectiva.

## Decisão

- Definir o estado causal `E_i` como o segmento emitido mais a primeira diferença já observada do
  segmento aberto seguinte.
- Prever somente duração e deslocamento restantes, além da primeira diferença do estado seguinte.
- Projetar duração para inteiro não negativo por uma regra explícita e registrar valores antes da
  projeção.
- Forçar deslocamento restante zero quando a duração restante projetada for zero, preservando a
  saída bruta e a contagem dessa correção de validade.
- Reconstruir por cinemática direta, avançar no relógio raw e reaplicar a mesma transição em rollout.
- Usar o default `tolerance=0.03`; `0.1` não tem densidade de eventos suficiente em alguns sinais e
  foi selecionada em resultados anteriores.
- Comparar estados cartesiano, absoluto e relacional, mas manter o relacional como candidato
  pré-especificado para um teste severo da formulação original.
- Parear inputs e parâmetros com raw e segmentos fixos, além de comparar AR raw e persistência.
- Usar cinco seeds novas e teste temporal fechado como split decisório.
- Tratar esta etapa como K5-A sintética. Dados externos e ABBA-LSTM ficam condicionados à passagem
  do gate causal básico.

## Consequências

- O primeiro incremento aberto deixa de ser leakage oculto e passa a fazer parte explícita do estado.
- O modelo pode ser reaplicado recursivamente sem usar uma fronteira ou duração futura verdadeira.
- `dt=1` possui alvo restante exatamente zero e não exige caso temporal fictício.
- A projeção garante estados utilizáveis, mas a taxa de saídas brutas inválidas continua uma métrica.
- Ridge e concatenação ordenada isolam o valor do estado antes de introduzir redes recorrentes ou
  Transformers.
- Um gate negativo encerra a escalada de capacidade; não autoriza experimentar modelos maiores no
  mesmo teste.

## Alternativas consideradas

- **Disponibilizar o segmento no endpoint:** rejeitada; a emissão causal ocorre um passo depois.
- **Finalizar o segmento aberto em cada origem:** rejeitada no caminho primário; cria estados que não
  existiriam no stream contínuo.
- **Prever `dt/dy` completos sem registrar o incremento conhecido:** rejeitada; mascara que parte do
  alvo já foi observada.
- **Usar `tolerance=0.1`:** rejeitada para esta tarefa; a cadeia pode ter apenas um a quatro elos em
  sinais completos já observados.
- **Começar por LSTM/Transformer:** rejeitada; capacidade e representação ficariam confundidas antes
  de uma baseline linear forte.
- **Executar ABBA-LSTM agora:** adiada; é offline/simbólica e não deve consumir a etapa se o estado
  causal básico já falhar.
