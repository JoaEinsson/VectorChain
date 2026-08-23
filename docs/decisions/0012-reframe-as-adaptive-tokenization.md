# 0012 — Reformular a Fase II como tokenização causal adaptativa

Status: **Accepted**
Data: 2026-08-23

## Contexto

A Fase I encerrou três explicações fortes. Relações `delta_theta/delta_r` não acrescentaram valor
consistente (K2), smoothing e fronteiras fixas impediram atribuir o efeito pooled à adaptação (K3)
e o estado vetorial recursivo perdeu para AR raw em todas as seeds e horizontes primários (K5-A).
K4 e a confirmação externa permaneceram fechadas porque seus gates antecedentes não passaram.

Apesar disso, a Etapa 10-A preservou uma observação operacional: uma quantidade menor de passos
vetoriais cobriu um histórico raw mais longo e produziu estados válidos. O pareamento também mostrou
que número de passos, escalares e parâmetros são orçamentos diferentes. Essa observação não foi
isolada como pergunta primária na Fase I.

## Decisão

- Encerrar formalmente a escada cinemática K1–K5 na Fase I, preservando seus resultados.
- Abrir K6 como hipótese independente sobre comprimento de sequência, não sobre cinemática ou
  recorrência de estado.
- Usar somente tokens cartesianos `(dt, dy)` como candidata principal.
- Fornecer `x[t]` como âncora escalar idêntica a todas as representações treinadas.
- Prever alvos raw diretamente com concatenação ordenada e ridge linear, sem pooling nem rollout.
- Comparar sob dois pareamentos explícitos: mesmo número de tokens e mesmo payload escalar.
- Incluir obrigatoriamente segmentos fixos, PAA causal, média móvel, raw step-matched, raw
  scalar-matched e persistência.
- Usar somente famílias sintéticas e seeds novas na Etapa 11-A.
- Condicionar qualquer Etapa 11-B externa à passagem integral de K6 e a outro protocolo anterior à
  inspeção dos datasets de teste.
- Não implementar runner ou config canônica antes de versionar o protocolo da Fase II.

## Consequências

- A Fase II não restaura C2, C3 ou C4 e não transforma estabilidade em acurácia.
- `theta`, `r`, `delta_theta` e `delta_r` permanecem disponíveis no pacote, mas fora da candidata.
- O termo `compacto` deve indicar qual orçamento diminuiu; redução de tokens não pode ser descrita
  como redução de bytes sem medição correspondente.
- Um resultado positivo de K6 ficará restrito ao benchmark, ridge, budgets e dados pré-especificados.
- Um resultado negativo encerrará a formulação linear de tokenização sob esse orçamento antes de
  modelos maiores ou dados externos.
- A API pública e o default `tolerance=0.03` não mudam por esta decisão.

## Alternativas consideradas

- **Retunar K5 ou testar LSTM/Transformer:** rejeitada; usaria capacidade para contornar um gate já
  aberto e negativo.
- **Executar K4/Etapa 9 com o claim relacional original:** rejeitada; K2 e K3 falharam.
- **Promover estabilidade dos rollouts a claim de utilidade:** rejeitada; AR raw foi mais preciso e
  estável.
- **Usar a geometria relacional como token principal:** rejeitada; as ablations favoreceram a
  representação cartesiana menor.
- **Encerrar todo trabalho experimental:** não escolhida; K6 isola uma dimensão de orçamento ainda
  não testada como pergunta primária, sem negar que a publicação da Fase I também é um resultado
  completo e válido.
