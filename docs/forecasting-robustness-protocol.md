# Protocolo de robustez do forecasting

Status: **pré-especificado antes da primeira execução da grade**.

## Objetivo

Testar se a observação limítrofe da baseline mínima permanece sob novas realizações de ruído e
mudanças explícitas de horizonte, contexto e tolerância, sem retunar o modelo downstream.

## Base congelada

O runner carrega `configs/forecasting/baseline.toml` como fonte do dataset, geradores, parâmetros de
sinal, split temporal, representações, features, pooling e ridge. A grade não pode alterar:

- as sete dinâmicas ou seus parâmetros nominais;
- `noise_std=0.01` e `n_points=1024`;
- split 60%/20%/20%;
- pooling `(last, mean, std)`;
- ridge com `alpha=0.001`;
- features VectorChain `(dt, dy, theta, r, delta_theta)`;
- alvo de incremento e avaliação no valor original.

## Grade registrada

- seeds: `1729`, `2718`, `31415`, `104729`, `8675309`;
- horizontes: `1`, `4`, `16` pontos;
- contextos: `32`, `64`, `128` pontos;
- tolerâncias: `0.01`, `0.03`, `0.1`;
- stride: `4`.

Isso produz 45 condições raw, 45 condições de primeira diferença e 135 condições VectorChain,
cada uma com linhas separadas para validação e teste.

## Comparações pareadas

Para cada seed, horizonte e contexto, raw é a referência pareada. Cada tolerância VectorChain usa
exatamente os mesmos IDs e alvos daquela condição. São calculados:

- `rmse_vectorchain / rmse_raw`;
- `steps_raw / steps_vectorchain`;
- `scalar_elements_raw / scalar_elements_vectorchain`;
- os três critérios da baseline e seu `joint_success`.

Primeira diferença também recebe razões pareadas, mas não participa do critério conjunto
VectorChain.

## Unidade de réplica e agregação

A unidade de réplica é uma seed, que gera uma nova realização para os sete sinais. Para cada célula
e split, o relatório preserva as cinco métricas seed a seed e agrega média, mediana, quartis, mínimo
e máximo. Nenhum erro-padrão baseado em janelas sobrepostas será calculado.

Uma célula é `robust_cell=true` somente quando `joint_success_rate >= 0.8` no teste. Validação é
reportada separadamente; uma célula que passa no teste e falha sistematicamente na validação será
tratada como instável.

## Hipóteses

- R1: redução estrutural deve permanecer frequente em toda a grade.
- R2: tolerância maior deve reduzir passos e payload, mas pode degradar paridade preditiva.
- R3: horizontes maiores devem tornar a paridade mais difícil, sobretudo em chirp e mudança de
  regime.
- R4: contexto maior não é presumido melhor; o pooling pode diluir informação local.
- R5: nenhuma célula será promovida a default apenas por obter a maior taxa de sucesso.

## Artefatos e limitações

O run grava condições seed a seed, métricas por sinal, resumo por célula, ambiente, hashes e figuras
pré-especificadas. Previsões individuais permanecem somente nos artefatos brutos ignorados pelo Git
e não são necessárias para a referência agregada.

Cinco seeds dão resolução grosseira, os sinais continuam sintéticos e a grade não cobre amplitude,
escala temporal contínua, outras features, outros modelos ou multi-step recursivo. O estudo é de
sensibilidade descritiva, não confirmação estatística definitiva.
