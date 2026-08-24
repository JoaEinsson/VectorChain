# Experimentos

Os scripts deste diretório são pontos de entrada reprodutíveis, não bibliotecas auxiliares nem
notebooks exploratórios.

Um experimento deve:

1. receber sua configuração explicitamente;
2. fixar e registrar todas as seeds;
3. reutilizar funções públicas ou internas do pacote em `src/vectorchain`;
4. gravar dados em `artifacts/<run-id>/`;
5. produzir métricas tabulares antes das figuras;
6. registrar ambiente e commit Git;
7. nunca alterar um resultado anterior em silêncio.

O benchmark de reconstrução e compressão é executado na raiz do repositório com:

```powershell
uv run python experiments/01_reconstruction.py --config configs/reconstruction/baseline.toml
```

Cada execução cria um diretório imutável em `artifacts/<run-id>/` com configuração efetiva,
ambiente, seeds derivadas por sinal, métricas agregadas, timings individuais, manifesto de hashes,
vetores opcionais e figuras. O processo termina com código diferente de zero quando qualquer
condição falha, mas preserva no CSV tudo que conseguiu executar.

O benchmark pré-especificado de similaridade, ablations e retrieval é executado com:

```powershell
uv run python experiments/02_similarity.py --config configs/similarity/baseline.toml
```

Ele gera uma única divisão gallery/query compartilhada por todas as representações, ajusta a
padronização exclusivamente na gallery e grava `samples.csv`, `sequences.csv`, `metrics.csv`, o
ranking completo em `neighbors.csv` e, quando habilitado, as matrizes em `distances.npz`. A banda
DTW resolvida e os runtimes de representação e distância são registrados por condição.

O benchmark mínimo de forecasting é executado com:

```powershell
uv run python experiments/04_forecasting.py --config configs/forecasting/baseline.toml
```

O runner cria uma única lista de exemplos rolling-origin, separada temporalmente pelo índice do
alvo e compartilhada por raw, primeira diferença e VectorChain. O scaler e a regressão ridge usam
somente treino. `examples.csv` prova os limites temporais; `inputs.csv` registra passos, features e
bytes antes do pooling; `predictions.csv` preserva cada previsão de validação e teste.

A grade fatorial de robustez do forecasting é executada com:

```powershell
uv run python experiments/05_forecasting_robustness.py --config configs/forecasting/robustness.toml
```

Ela mantém a baseline congelada e varia somente seeds, horizontes, contextos, tolerâncias e stride
declarados. `conditions.csv` preserva cada seed; `summary.csv` agrega por célula sem tratar janelas
sobrepostas como réplicas independentes. Raw e diferenças são avaliados uma vez por
`(seed, horizon, context)`, não uma vez por tolerância VectorChain.

## Experimentos pós-MVP

O isolamento pré-especificado do mecanismo cinemático é executado com:

```powershell
uv run python experiments/06_forecasting_feature_ablation.py --config configs/forecasting/feature_ablation.toml
```

Ele compara cinco subconjuntos de features sobre fronteiras e exemplos idênticos, preserva razões
pareadas por seed/célula/sinal, registra parâmetros e payload, audita a contagem de passos de cada
exemplo e grava a decisão científica em `gate.json`. O gate usa validação; o teste é apenas
descritivo porque esse benchmark já foi observado.

O run revisado e sua réplica estão em
[`reports/reference/forecasting-kinematic-feature-ablation/`](../reports/reference/forecasting-kinematic-feature-ablation/).
O gate não passou; futuros controles devem tratar `absolute_geometry` como candidata exploratória,
não promover retrospectivamente outra variante relacional.

Os controles pareados da geometria absoluta são executados com o grupo experimental ABBA:

```powershell
uv run --group abba python experiments/07_forecasting_controls.py --config configs/forecasting/controls.toml
```

O runner seleciona hiperparâmetros exclusivamente numa divisão bloqueada dentro do treino,
mantém 12 entradas pooled e 13 parâmetros para candidata/controles e separa controles online do
estágio de compressão `fABBA.compress`, que é offline dentro de cada janela. O protocolo congelado
está em [`docs/forecasting-controls-protocol.md`](../docs/forecasting-controls-protocol.md).

O run promovido e sua reprodução computacional estão em
[`reports/reference/forecasting-absolute-geometry-controls/`](../reports/reference/forecasting-absolute-geometry-controls/).
O gate não passou: geometria local e EWMA não explicaram o efeito, mas média móvel trailing e
segmentação fixa impediram atribuí-lo à adaptação da cadeia.

O primeiro teste direto do estado autoregressivo da cadeia é executado com:

```powershell
uv run python experiments/08_vector_state_rollout.py --config configs/forecasting/vector_state_rollout.toml
```

Ele usa somente segmentos realmente emitidos, inclui explicitamente o primeiro incremento do elo
aberto e prevê apenas duração/deslocamento ainda desconhecidos. A ordem dos estados é concatenada,
sem pooling, e o rollout é reconstruído no relógio raw. O protocolo pré-especificado está em
[`docs/vector-state-rollout-protocol.md`](../docs/vector-state-rollout-protocol.md).

O run promovido e sua reprodução estão em
[`reports/reference/forecasting-vector-state-rollout/`](../reports/reference/forecasting-vector-state-rollout/).
O gate K5-A não passou: o estado foi válido e compacto, mas perdeu para AR raw em todas as seeds e
horizontes.

A escada experimental da Fase I terminou com a condição de parada de K5-A:

1. preservar os resultados negativos de K2, K3 e K5-A;
2. não abrir confirmação externa ou modelos maiores sem uma hipótese nova e justificativa própria;
3. tratar qualquer continuação como reformulação, não como promoção do claim autoregressivo.

Os requisitos e gates estão em
[`docs/post-mvp-claim-protocol.md`](../docs/post-mvp-claim-protocol.md). Novos números de scripts e
configs só serão atribuídos quando o protocolo específico da etapa correspondente estiver
congelado.

## Fase II pré-especificada

A Etapa 11-A testará tokens cartesianos adaptativos sob orçamento de sequência. O protocolo já está
congelado em
[`docs/phase-2-adaptive-tokenization-protocol.md`](../docs/phase-2-adaptive-tokenization-protocol.md),
mas ainda não existe runner, config nem resultado canônico.

Antes de criar `09_*`, a implementação deve:

1. usar somente seeds `11` e `22` em testes e desenvolvimento;
2. materializar o prefixo aberto sem chamar `finalize()` em origens internas;
3. manter `2H+1` escalares, incluindo `x[t]`, e parâmetros iguais nas seis representações treinadas;
4. separar o comando de seleção/validação do comando que abre o teste;
5. transcrever os cinco inteiros canônicos e os digests do protocolo sem regenerá-los.

Qualquer divergência exige atualizar protocolo e ADR antes de executar uma seed canônica. A Etapa
11-B externa permanece fechada mesmo depois da implementação; ela depende da passagem integral de
K6 e de outro protocolo anterior aos dados reais.

### Cadeia revisável K7

A Etapa 12-A possui protocolo independente em
[`docs/phase-2-revisable-chain-protocol.md`](../docs/phase-2-revisable-chain-protocol.md). O módulo
`revisable_chain.py` materializa os três sinais, as origens causais comuns e as seis matrizes de
design. O runner de desenvolvimento é executado com:

```powershell
uv run python experiments/09_revisable_chain_validation.py --config configs/forecasting/revisable_chain_development.toml
```

Esse comando aceita somente seeds `11/22`, gera cada sinal apenas até o fim da validação, seleciona
uma única dupla de regularizadores no treino interno e ajusta um ridge multioutput por série e
representação. `selection.json`, previsões de validação, modelos, estados versionados e auditorias
entram no manifesto; não existe modo de teste nem `gate.json` nesse runner. Ainda não há dado
canônico executado ou promovido.

Depois que o desenvolvimento passar integralmente pelos testes e o código estiver em commit limpo,
o mesmo runner pode receber o escopo canônico **somente de seleção**:

```powershell
uv run python experiments/09_revisable_chain_validation.py --config configs/forecasting/revisable_chain_selection.toml
```

Esse escopo exige exatamente as cinco seeds pré-registradas, recusa worktree sujo e continua sem
implementar ou materializar o teste. A futura abertura do teste será outro comando e deverá consumir
o `selection.json` congelado por hash.

O runner deve permanecer limitado a:

1. uma cauda de quatro elos e 256 intervalos, com fronteiras fixas;
2. um solver quadrático NumPy e log versionado de cada revisão/compromisso;
3. três modulações isoladas: frequência, baseline e assimetria da crista;
4. quatro ablations vetoriais, raw pareado e persistência;
5. seeds de desenvolvimento `11/22`, nunca as seeds canônicas do protocolo.

Não implementar sinal combinado, fronteiras móveis, Kalman ou rede neural antes da decisão K7.
Durante o desenvolvimento, uma origem comum também precisa ter 16 incrementos anteriores para
`raw_matched`; não se usa padding nem se criam origens diferentes por representação.
