# 0008 — Programa pós-MVP centrado na hipótese cinemática

Status: **Accepted**
Data: 2026-08-23

## Contexto

O MVP implementou e testou uma cadeia causal de segmentos com features absolutas e relacionais. A
grade de forecasting encontrou uma região favorável em `tolerance=0.1`, mas o resultado mede o
pacote completo, usa sinais sintéticos e possui confundidores de suavização e capacidade do ridge.

A motivação original veio de uma analogia com cinemática inversa: o gráfico da série é interpretado
como uma cadeia planar articulada, enquanto `delta_theta` e `delta_r` descrevem transições entre
movimentos. O levantamento inicial de literatura mostrou anterioridade próxima em PLA online,
ABBA/fABBA, ângulos/turning points e forecasting simbólico. Assim, a novidade não pode ser atribuída
a nenhum desses componentes isoladamente.

## Decisão

- Preservar `prompt_inicial.md`, a especificação, o contrato causal e os defaults do MVP.
- Registrar a cinemática como enquadramento científico pós-MVP, não como nova semântica da API.
- Separar explicitamente quatro níveis futuros: mecanismo, controles, confirmação externa e estado
  autoregressivo.
- Tratar `tolerance=0.1` como candidata descoberta pós-análise, nunca como default confirmado.
- Isolar primeiro `delta_theta/delta_r` por ablation com fronteiras idênticas.
- Exigir controles de smoothing, PLA/ABBA, payload e capacidade antes de atribuir o ganho às relações.
- Exigir dados inéditos antes de generalizar além do benchmark sintético atual.
- Reservar o claim de espaço de estado para um experimento que preveja diretamente vetores futuros e
  os decodifique recursivamente.
- Manter resultados negativos e reduzir o claim quando uma explicação mais simples vencer.

O programa detalhado fica em `docs/post-mvp-claim-protocol.md`; o enquadramento e a escada de claims
ficam em `docs/kinematic-hypothesis.md`; a fotografia inicial de anterioridade fica em
`docs/closest-prior-art.md`.

## Consequências

- A conclusão do MVP não autoriza alterar o default nem anunciar superioridade geral.
- O próximo código experimental deverá responder a um contraste causal específico, não ampliar o
  modelo autoregressivo prematuramente.
- Features de segunda ordem exigem ADR próprio e não entram silenciosamente no pacote.
- A comparação com ABBA/fABBA passa a ser requisito científico, mas não dependência obrigatória da
  biblioteca principal.
- Cada gate pode encerrar uma hipótese sem apagar o valor das etapas anteriores.
- README, relatórios e eventual artigo deverão indicar claramente o nível de claim alcançado.

## Alternativas consideradas

- **Ir diretamente para um Transformer de vetores:** rejeitada porque confundiria valor da
  representação, capacidade do modelo e estabilidade autoregressiva.
- **Declarar `0.1` como novo default:** rejeitada por seleção sobre a própria grade de teste.
- **Tratar `delta_theta` como novidade suficiente:** rejeitada pela anterioridade em mudanças de
  ângulo, curvatura e turning points.
- **Abandonar o projeto diante de ABBA:** rejeitada; a pergunta relacional e o protocolo causal ainda
  são hipóteses testáveis, desde que comparadas diretamente com a literatura próxima.
