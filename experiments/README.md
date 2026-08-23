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

A ordem restante, condicionada aos gates, é:

1. executar e interpretar ABBA/fABBA, PLA, smoothing, payload e capacidade pareados;
2. confirmação em dados inéditos apenas para hipóteses que sobreviverem;
3. somente então `V_1:t -> V_(t+1)` e rollout no relógio de eventos.

Os requisitos e gates estão em
[`docs/post-mvp-claim-protocol.md`](../docs/post-mvp-claim-protocol.md). Novos números de scripts e
configs só serão atribuídos quando o protocolo específico da etapa correspondente estiver
congelado.
