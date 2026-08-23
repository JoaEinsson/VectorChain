# Resultados de referência

Este diretório contém somente resultados pequenos, canônicos e revisados. Não armazene aqui saídas
exploratórias nem arquivos grandes.

Cada resultado de referência deve identificar:

- commit do código;
- configuração integral do experimento;
- seed ou lista de seeds;
- versões de Python e dependências;
- sistema operacional;
- métricas brutas, sem arredondamento destrutivo;
- script e comando usados para reprodução.

Resultados disponíveis:

- [`reconstruction-baseline/`](reconstruction-baseline/): primeira baseline causal de compressão ×
  reconstrução, com sete sinais sintéticos e cinco tolerâncias.
- [`similarity-retrieval-baseline/`](similarity-retrieval-baseline/): comparação pré-especificada de
  dez representações no mesmo split, incluindo baselines, segmentação fixa e cinco ablations.
- [`minimal-forecasting-baseline/`](minimal-forecasting-baseline/): ridge compartilhado com split
  temporal auditável para raw, diferenças e VectorChain, incluindo payload e persistência.
- [`forecasting-robustness-grid/`](forecasting-robustness-grid/): grade fatorial com cinco seeds,
  três horizontes, três contextos e três tolerâncias, incluindo réplica determinística.
- [`forecasting-kinematic-feature-ablation/`](forecasting-kinematic-feature-ablation/): isolamento
  das features relacionais com capacidade pareada e gate negativo preservado.
- [`forecasting-absolute-geometry-controls/`](forecasting-absolute-geometry-controls/): geometria
  local, smoothing causal, segmentos fixos e fABBA contínuo; K3 não avançou.
