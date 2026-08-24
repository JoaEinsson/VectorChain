# 0013 — Cauda articulada revisável com compromisso limitado

Status: **Accepted**
Data: 2026-08-23

## Contexto

As features relacionais da Fase I foram calculadas entre elos já emitidos e imutáveis. Nesse estado,
`theta/r` são transformações de `dt/dy` e `delta_theta/delta_r` espaciais também são derivados do
mesmo snapshot. O gate negativo de K2 não responde se a **mudança temporal** das juntas durante uma
revisão causal contém informação.

Permitir que qualquer vetor histórico mude indefinidamente violaria a imutabilidade das emissões,
aumentaria custo sem limite e confundiria smoothing offline com estado online. A alternativa mínima
é manter uma cauda provisória explícita e comprometer o prefixo com atraso limitado.

## Decisão

- Manter `VectorChain` e seu modo causal atual inalterados.
- Criar um componente separado, sem reutilizar o termo `emitido` para elos revisáveis.
- Dividir o estado em prefixo comprometido imutável e cauda de trabalho versionada.
- Limitar a cauda a quatro elos e 256 intervalos raw; o limite que ocorrer primeiro força o
  compromisso do elo completo mais antigo.
- Preservar as fronteiras temporais propostas pelo segmentador causal e revisar somente as
  ordenadas das juntas.
- Ancorar exatamente a última junta comprometida e o ponto corrente.
- Resolver a revisão por mínimos quadrados quadráticos com penalidades de mudança temporal e
  curvatura de slope, usando somente NumPy.
- Registrar separadamente relações espaciais e atualizações temporais de `theta/r`.
- Testar K7 com três mecanismos sintéticos isolados antes de combinar efeitos ou mover fronteiras.

## Consequências

- Vários elos podem permanecer provisórios, mas nenhum elo comprometido pode mudar.
- A saída pública precisa distinguir `committed`, `working` e cada versão da cauda.
- O instante de compromisso, a latência e o histórico de correções tornam-se parte do artefato.
- A causalidade continua baseada no prefixo observado; a imutabilidade aplica-se ao prefixo
  comprometido, enquanto revisões da cauda são eventos novos e não reescrita de logs anteriores.
- Como os tempos ficam fixos, `dt` permanece inteiro positivo e o primeiro solver é determinístico e
  convexo.
- K7 não restaura K2: `update_theta/update_r` são variáveis temporais diferentes dos deltas espaciais
  rejeitados anteriormente.
- Uma revisão futura de fronteiras exigirá outro ADR e só poderá ser considerada se K7 passar.

## Alternativas consideradas

- **Revisar toda a cadeia:** rejeitada; custo e latência crescem sem limite e o prefixo perde
  identidade.
- **Alterar elos já emitidos:** rejeitada; contradiz o contrato vigente e torna resultados históricos
  mutáveis.
- **Mover tempos e amplitudes das juntas no primeiro teste:** rejeitada; introduz otimização
  não convexa, projeção inteira e uma nova explicação concorrente.
- **Usar um solver genérico de cinemática inversa:** rejeitada; o problema escalar de tempos fixos
  admite uma solução quadrática menor e auditável.
- **Testar um único sinal com frequência, média e cristas variando juntas:** rejeitada; um resultado
  não identificaria qual mecanismo acionou as correções.
- **Começar por Kalman, LSTM ou Transformer:** rejeitada nesta etapa; K7 precisa primeiro demonstrar
  efeito incremental das correções sob o mesmo ridge e as mesmas fronteiras.
