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
