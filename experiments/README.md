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
