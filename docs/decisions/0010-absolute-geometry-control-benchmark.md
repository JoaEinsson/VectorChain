# 0010 — Controles pareados para a geometria absoluta

Status: **Accepted**
Data: 2026-08-23

## Contexto

A Etapa 7 rejeitou efeito incremental consistente de `delta_theta/delta_r`, mas
`absolute_geometry` teve menor erro agregado e menor payload que as variantes relacionais. Como
`theta/r` são funções de `dt/dy`, o resultado pode ser explicado por engenharia não linear de
features, suavização forte, segmentação ou capacidade downstream.

## Decisão

- Reenquadrar a Etapa 8 como teste de K1/K3, não continuação de K2.
- Manter `absolute_geometry` e a configuração exploratória congeladas.
- Comparar controles geométricos locais, média móvel trailing, EWMA e segmentos fixos com quatro
  canais e 13 parâmetros ridge.
- Fazer tuning somente em uma divisão bloqueada dentro do treino, com três candidatos por família.
- Exigir vantagem preditiva por seed/célula e Pareto de erro, payload e parâmetros.
- Integrar `fABBA.compress` 1.5.2 em grupo experimental separado e rotulá-lo como offline dentro da
  janela.
- Não incluir quantização simbólica ABBA/fABBA nesta tarefa pooled.

## Consequências

- O controle local separa utilidade da geometria adaptativa de utilidade da transformação não
  linear no nível amostral.
- Os suavizadores verificam se a região `tolerance=0.1` é explicada por low-pass causal simples.
- Segmentação fixa testa adaptação contra orçamento de passos próximo.
- O controle ABBA melhora a comparação com anterioridade, mas não satisfaz o contrato online.
- Dependências pesadas ficam fora do pacote principal e da matriz padrão de CI.
- Um gate negativo reduz explicitamente o claim; não autoriza retuning da candidata.

## Alternativas consideradas

- **Retunar VectorChain junto com os controles:** rejeitada porque reabriria uma configuração
  escolhida em resultados já observados.
- **Escolher controles no teste externo:** rejeitada por vazamento de seleção.
- **Usar símbolos fABBA locais como números:** rejeitada porque rótulos de clusters não são
  coordenadas comparáveis entre janelas.
- **Adicionar SciPy/scikit-learn ao runtime:** rejeitada; são necessários apenas ao controle
  externo.
