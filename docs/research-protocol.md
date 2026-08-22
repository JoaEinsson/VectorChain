# Protocolo científico inicial

Status: **pré-especificado antes da primeira execução experimental**.

## Objetivo primário

Medir o compromisso entre redução do comprimento da sequência e erro de reconstrução produzido pela
segmentação causal adaptativa.

O experimento inicial não testa superioridade geral, forecasting ou similaridade.

## Hipóteses

- H1: aumentar a tolerância reduz, em média, o número de vetores.
- H2: aumentar a tolerância aumenta ou mantém o erro de reconstrução.
- H3: sinais localmente lineares apresentam melhor compromisso que sinais ruidosos ou de curvatura
  rápida.
- H4: não existe uma tolerância única necessariamente adequada a sinais com escalas diferentes.

Resultados contrários ou não monotônicos devem ser reportados sem remoção seletiva.

## Dados sintéticos

O conjunto inicial contém:

- seno;
- chirp;
- rampa;
- piecewise linear;
- resposta de primeira ordem;
- resposta subamortecida de segunda ordem;
- mudança de regime.

Parâmetros canônicos estão em `configs/reconstruction/baseline.toml`. Geradores recebem um objeto de
RNG ou seed explicitamente; não devem depender do estado aleatório global.

As fórmulas, unidades normalizadas e parâmetros próprios de cada forma estão registrados em
`docs/synthetic-signals.md`.

## Condições

A primeira execução usa sinais em escala nominal comparável e tolerância absoluta. Experimentos com
normalização, amplitude variável ou tolerância relativa devem ser rotulados como condições
separadas; não podem ser combinados silenciosamente com a baseline.

Para cada sinal e tolerância, registrar:

- número original de pontos;
- número de vetores;
- fator de compressão estrutural;
- fração retida;
- MAE;
- RMSE;
- runtime da transformação;
- runtime da reconstrução.

Runtime é repetido pelo menos cinco vezes e resumido por mediana e intervalo interquartil. Métricas
determinísticas não devem ser tratadas como cinco observações independentes apenas porque o runtime
foi repetido.

O runtime de transformação mede exclusivamente `reset`, um `update` por amostra e `finalize` sobre
o array sintético já gerado; exclui construção do objeto, geração do sinal, reconstrução e cálculo
de métricas. O runtime de reconstrução mede somente `inverse_transform`. Essa separação evita que o
cálculo automático de `reconstruction_error_` no wrapper `fit_transform` seja contado duas vezes.

## Visualizações previstas

1. Original e reconstrução no mesmo eixo.
2. Segmentos e pontos de articulação sobrepostos ao sinal.
3. RMSE por tolerância, separado por sinal.
4. Fator de compressão por tolerância, separado por sinal.
5. Fronteira de compromisso compressão × erro, sem reduzir todos os sinais a uma única média.

Toda figura deve informar seed, tolerância, unidades, tamanho do sinal e commit ou run id.

## Integridade da análise

- A grade completa de tolerâncias é preservada, mesmo quando produz resultados ruins.
- Falhas e parâmetros inválidos são registrados, não convertidos em zero ou descartados.
- Arredondamento é apenas de apresentação; CSV mantém precisão suficiente para reprodução.
- Alteração do protocolo após observar resultados recebe novo nome/configuração e justificativa.
- Conclusões distinguem redução de elementos de redução de memória.

## Critério de conclusão da etapa

A etapa termina quando um comando documentado recria tabelas e figuras a partir de um clone limpo,
os testes causais passam e o relatório identifica claramente condições úteis e falhas observadas.
