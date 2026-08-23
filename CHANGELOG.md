# Changelog

Todas as mudanças relevantes deste projeto serão registradas neste arquivo.

O formato segue [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) e o projeto adotará
[Semantic Versioning](https://semver.org/) a partir das primeiras versões públicas. Durante a fase
`0.x`, mudanças incompatíveis serão documentadas explicitamente.

## [Unreleased]

### Added

- Layout instalável em `src/`, metadados do pacote e lockfile reproduzível com `uv`.
- Contratos de especificação matemática, causalidade e protocolo experimental.
- ADRs para segmentação, artefatos e tooling.
- Gates locais com Ruff, mypy, pytest, coverage e build.
- CI multiplataforma, templates de contribuição e atualização automática de dependências.
- Licença Apache-2.0 e metadados de citação do autor.
- Núcleo de segmentação causal com API online e wrapper batch.
- Projeção configurável de `dt`, `dy`, `theta`, `r`, `delta_theta` e `delta_r`.
- Testes unitários e de propriedades para causalidade, articulação e equivalência batch/stream.
- Reconstrução linear stateful com suporte a ablações e ordem arbitrária de features.
- Métricas públicas MAE, RMSE, fator de compressão e fração retida.
- Atributos pós-ajuste de compressão e erro de reconstrução.
- Sete geradores sintéticos determinísticos com seed ou `numpy.random.Generator` explícito.
- Plot Matplotlib opcional de original, reconstrução, segmentos e pontos de articulação.
- Runner configurável do benchmark compressão × reconstrução com artefatos auditáveis.
- Relatório de referência da baseline causal com métricas brutas, timings, ambiente e figuras.
- Baselines raw, normalizada, diferenças e segmentação linear fixa com features configuráveis.
- Padronização aprendida somente na gallery, DTW normalizado e retrieval nearest-neighbor estável.
- Runner pré-especificado de similaridade com split compartilhado, cinco ablations e ranking completo.
- Relatório de referência de retrieval com réplica determinística e resultados negativos preservados.
- Protocolo causal de forecasting de incremento com pooling compartilhado e split temporal auditável.
- Runner ridge para raw, diferenças e VectorChain com persistência, payload, memória e timings.
- Relatório de forecasting replicado, incluindo a paridade limítrofe e falha correspondente na validação.
- Runner fatorial de robustez do forecasting com cinco seeds, três horizontes, três contextos e três
  tolerâncias.
- Relatório replicado da grade de robustez, com critérios pareados, resultados por sinal e região de
  sucesso preservada como hipótese para confirmação independente.
- Enquadramento pós-MVP inspirado em cinemática inversa, com distinção entre elos, articulações e
  previsão autoregressiva no relógio de eventos.
- Escada de claims e protocolo experimental até isolamento de mecanismo, controles pareados,
  confirmação externa e rollout direto da cadeia vetorial.
- Levantamento inicial de anterioridade cobrindo PLA online, ABBA/fABBA, forecasting simbólico,
  quantização e representações baseadas em mudança angular.
- ADR do programa científico pós-MVP e roadmap expandido até o claim de espaço de estado
  autoregressivo compacto.
- Orientação para que novos runners isolem o mecanismo antes de implementar autoregressão vetorial.
- Runner pré-especificado da ablation cinemática com cinco variantes, controle de capacidade,
  auditoria estrutural por exemplo e gate científico separado do status de execução.
- Relatório replicado da ablation cinemática preservando o gate negativo de `delta_theta/delta_r`
  e redirecionando os controles para a utilidade exploratória da geometria absoluta.
- Protocolo e runner da Etapa 8 com geometria local, smoothing trailing/EWMA, segmentação fixa,
  compressão ABBA opcional, tuning bloqueado no treino e gate predição × payload × capacidade.
- Relatório reproduzido dos controles da Etapa 8, preservando a falha contra média móvel trailing e
  segmentação fixa e reduzindo explicitamente o claim de mecanismo.
- Protocolo e ADR da Etapa 10-A para estado causal de emissão, alvo não observado, duração projetada
  e rollout autoregressivo pareado contra raw, segmentos fixos, AR e persistência.
