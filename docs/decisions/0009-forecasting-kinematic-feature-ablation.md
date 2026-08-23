# 0009 — Ablation cinemática no forecasting antes dos controles externos

Status: **Accepted**
Data: 2026-08-23

## Contexto

O pacote `(dt, dy, theta, r, delta_theta)` apresentou uma região favorável na grade de robustez,
mas o experimento não isolou a contribuição de relações entre segmentos. A variante possui três
entradas pooled e três coeficientes ridge a mais que `(dt, dy, theta, r)`, de modo que capacidade é
um confundidor imediato.

O teste do benchmark já foi observado. Usá-lo novamente pode localizar mecanismo dentro da tarefa,
mas não produz confirmação externa.

## Decisão

- Fixar `tolerance=0.1` e reutilizar a grade de seeds/contextos/horizontes para comparação pareada.
- Avaliar cinco variantes de features com fronteiras invariantes.
- Usar `turning` versus `absolute_geometry` como contraste primário.
- Adicionar `turning_matched`, com o mesmo número de features da referência, como controle de
  capacidade por substituição de `r` por `delta_theta`.
- Usar validação, não teste, para decidir o gate operacional.
- Exigir margem primária de 1%, robustez em quatro de cinco seeds e pelo menos cinco de nove células.
- Exigir não degradação do controle pareado em quatro de cinco seeds.
- Tratar `full_relational` e `segment` como análises secundárias, sem promoção retrospectiva.
- Registrar igualdade de passos por exemplo entre variantes e depender dos testes do núcleo para
  igualdade exata de fronteiras.

## Consequências

- O experimento mede se a hipótese relacional merece avançar, não se ela está confirmada.
- A comparação pareada reduz variação entre sinais, mas as nove células continuam medidas repetidas.
- Substituir `r` por `delta_theta` não é equivalência perfeita de conteúdo, apenas de dimensão do
  ridge; ambos os contrastes precisam ser interpretados juntos.
- Um gate negativo será publicado e impedirá avançar K2 sem nova hipótese registrada.
- ABBA/fABBA, smoothing e confirmação real continuam fora desta etapa e permanecem gates futuros.

## Alternativas consideradas

- **Comparar somente o pacote completo com raw:** rejeitada porque não isola mecanismo.
- **Usar test como split decisório:** rejeitada porque seus resultados já orientaram a escolha de
  `tolerance=0.1`.
- **Projetar todas as variantes por PCA para a mesma dimensão:** adiada; introduziria outro estágio
  aprendido e uma nova fonte de tuning.
- **Adicionar `delta2_theta` agora:** rejeitada por ampliar a hipótese antes de testar a relação de
  primeira ordem já implementada.
