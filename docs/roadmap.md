# Roadmap

O roadmap é orientado por gates científicos. Concluir código não basta; cada etapa precisa produzir
evidência verificável antes da próxima.

## Etapa 0 — Fundação

- [x] Repositório e remoto Git
- [x] Layout Python em `src/`
- [x] Contrato matemático inicial
- [x] Contrato explícito de causalidade
- [x] Protocolo e política de artefatos
- [x] Configuração local de qualidade e CI
- [x] Lockfile e gates locais validados
- [x] CI executada no GitHub

## Etapa 1 — Núcleo causal

- [x] Estado online com `update`, `finalize` e `reset`
- [x] Wrapper `fit_transform`
- [x] Features configuráveis, canônicas e fronteiras
- [x] Testes de propriedades causais
- [x] Casos constantes, rampas, ruído e entradas inválidas

Gate: batch e stream equivalentes; alteração arbitrária de futuro não modifica segmentos emitidos.

## Etapa 2 — Reconstrução e sintéticos

- [x] `inverse_transform`
- [x] Métricas MAE, RMSE e compressão
- [x] Sete geradores sintéticos determinísticos
- [x] Plot da cadeia articulada

Gate: reconstrução preserva shape, endpoints e continuidade; geradores são reprodutíveis.

## Etapa 3 — Primeiro experimento

- [x] Runner configurável de compressão × reconstrução
- [x] Manifesto de ambiente e resultados brutos
- [x] Curvas e relatório de referência
- [x] Análise de condições de sucesso e falha

Gate: reprodução documentada a partir de clone limpo.

## Etapa 4 — Similaridade e ablations

- [x] Baselines raw, normalizado, diferenças e segmentação fixa
- [x] Ablations de features geométricas
- [x] Invariância e retrieval nearest-neighbor

Gate concluído: mesma divisão de dados e protocolo para todas as representações, com ranking e
matrizes de distância preservados no resultado de referência.

## Etapa 5 — Forecasting mínimo

- [x] Modelo downstream idêntico para raw, diferenças e VectorChain
- [x] Separação temporal sem leakage
- [x] Métricas, comprimento, runtime e memória aproximada

Gate concluído: alvo, pooling, scaler de treino, ridge, alpha e split são compartilhados; dimensão do
modelo decorrente das features é registrada explicitamente. A paridade VectorChain passou no teste
por margem mínima, falhou na validação e não é tratada como evidência robusta.

## Etapa 6 — Robustez do forecasting

- [x] Cinco seeds independentes como unidade de replicação
- [x] Grade pré-especificada de horizontes, contextos e tolerâncias
- [x] Critérios pareados de paridade, redução estrutural e payload
- [x] Réplica completa e relatório de referência

Gate concluído: 225 avaliações sem falhas em cada execução; todos os campos científicos foram
idênticos na réplica. Quatorze de 27 células de teste e 12 de 27 de validação foram robustas em pelo
menos quatro de cinco seeds. A região observada em tolerância `0.1` permanece candidata para
confirmação independente, não configuração padrão.

## Marco — MVP científico inicial concluído

As etapas 0–6 encerram o MVP definido pelo charter original. O resultado autoriza um claim empírico
limitado sobre o pacote completo no benchmark sintético, mas ainda não identifica por que ele
funcionou. Em particular:

- o efeito de `delta_theta` não foi isolado no forecasting;
- `delta_r` não participou da grade de robustez;
- suavização e maior capacidade downstream continuam explicações concorrentes;
- `tolerance=0.1` foi identificada olhando a própria grade;
- o modelo atual prevê um incremento raw após pooling, não o próximo vetor da cadeia.

O programa pós-MVP é normatizado em
[`post-mvp-claim-protocol.md`](post-mvp-claim-protocol.md). A interpretação cinemática e os limites
de cada claim estão em [`kinematic-hypothesis.md`](kinematic-hypothesis.md); a anterioridade próxima
está em [`closest-prior-art.md`](closest-prior-art.md).

## Etapa 7 — Isolamento do mecanismo cinemático

- [x] Pré-especificar métrica primária, margem, seeds e unidade de análise
- [x] Comparar `dt/dy` com geometria absoluta, `delta_theta` e `delta_r`
- [x] Garantir fronteiras, exemplos, alvos e splits idênticos entre ablations
- [x] Executar análise com capacidade downstream pareada
- [x] Preservar efeitos por seed, célula e dinâmica

Gate: a variante relacional precisa acrescentar efeito preditivo consistente sobre
`(dt, dy, theta, r)` em unidades independentes. Se não acrescentar, rejeitar o mecanismo relacional
sem reescrever o resultado do pacote completo.

Resultado: **gate não satisfeito**. `turning` passou em 0/5 seeds e 1/9 células de validação;
`turning_matched` passou em 2/5 seeds. K2 não avança sob o pooling/ridge atual. A evidência
exploratória migra para uma pergunta mais estreita: se a utilidade do pacote de geometria absoluta
resiste a controles de smoothing, features locais e capacidade. Consulte o
[`relatório de referência`](../reports/reference/forecasting-kinematic-feature-ablation/).

## Etapa 8 — Controles pareados e literatura operacional

Após o gate negativo da Etapa 7, esta etapa testa K1/K3 usando `absolute_geometry` como candidata;
ela não é confirmação do mecanismo relacional.

Protocolo congelado em
[`forecasting-controls-protocol.md`](forecasting-controls-protocol.md); o estágio contínuo oficial
`fABBA.compress==1.5.2` é controle descritivo `window_offline`, fora do gate causal.

- [x] Integrar a compressão contínua fABBA como controle experimental fora da dependência principal
- [x] Comparar PLA adaptativa com segmentação fixa sob o mesmo modelo downstream
- [x] Adicionar média móvel e suavização exponencial estritamente causais
- [x] Comparar geometria local com features e número de parâmetros pareados
- [x] Restringir o tuning a uma divisão bloqueada do treino e registrar o orçamento conservador
- [x] Reportar fronteira predição × passos × payload × capacidade × runtime

Gate: o VectorChain relacional precisa permanecer na fronteira de Pareto depois dos controles. Se
um suavizador, PLA simples ou capacidade adicional explicar o resultado, reduzir o claim ao
mecanismo efetivamente observado.

Resultado: **gate não satisfeito**. A candidata passou contra geometria local e EWMA, mas falhou
contra média móvel trailing (1/5 seeds e 3/9 células robustas) e segmentação fixa (4/5 seeds, porém
apenas 3/9 células robustas). K3 não avança; smoothing FIR e fronteiras fixas permanecem
explicações concorrentes. Consulte o
[`relatório de referência`](../reports/reference/forecasting-absolute-geometry-controls/).

## Etapa 9 — Confirmação externa congelada

Esta etapa fica **bloqueada para confirmação de K2/K3** pelos gates negativos das Etapas 7 e 8. Só
pode ser reaberta para uma hipótese reformulada e pré-especificada, sem tratar a geometria
relacional/adaptativa como mecanismo já demonstrado.

- [ ] Congelar configuração candidata sem alterar o default do pacote
- [ ] Registrar datasets, exclusões, métricas e critérios antes de abrir o teste
- [ ] Usar novas seeds, ruídos, escalas e dinâmicas sintéticas
- [ ] Avaliar datasets públicos reais de mais de um domínio
- [ ] Incluir condições em que detalhe de alta frequência seja importante
- [ ] Calcular efeitos pareados e incerteza por série/seed/dataset independente
- [ ] Reproduzir em ambiente limpo e preparar replicação por terceiro

Gate: a contribuição relacional e o Pareto precisam sobreviver nos dados inéditos. Só então é
permitido sustentar que relações cinemáticas carregam informação preditiva além da geometria dos
elos nas condições avaliadas.

## Etapa 10 — Estado autoregressivo da cadeia

Protocolo causal pré-especificado em
[`vector-state-rollout-protocol.md`](vector-state-rollout-protocol.md). A Etapa 10-A testa primeiro
o estado de emissão com seeds sintéticas novas; uma Etapa 10-B externa só é aberta se esse gate
básico passar.

- [x] Definir tarefa causal `E_1:t -> E_(t+1)` no relógio real de emissão
- [ ] Comparar previsão de `(dt, dy)`, `(r, theta)` e incrementos articulares
- [ ] Garantir duração positiva, articulação contínua e estado inicial explícito
- [ ] Definir truncamento do elo que cruza o horizonte raw
- [ ] Implementar rollout multi-vetor sem usar endpoints futuros reais
- [ ] Reconstruir a trajetória por cinemática direta e medir drift
- [ ] Comparar com raw recursivo, persistência e ABBA-LSTM ou equivalente próximo
- [ ] Medir validade, estabilidade, erro, payload, parâmetros, runtime e memória
- [ ] Confirmar o rollout em dados não usados para seleção

Gate do claim mais forte: o rollout vetorial precisa ser causal, válido, estável, compacto,
reconstruível e competitivo sob comparação pareada. Somente então o projeto poderá afirmar que uma
cadeia cinemática causal funciona como espaço de estado autoregressivo compacto nas condições
avaliadas.
