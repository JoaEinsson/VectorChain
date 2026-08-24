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
- Runner da Etapa 10-A com ridge multioutput, concatenação ordenada de estados, cinemática direta,
  AR raw recursivo, artefatos por evento/origem e gate K5-A separado do status de execução.
- Relatório reproduzido da Etapa 10-A, preservando rollouts válidos/compactos e a rejeição de K5-A
  diante da superioridade consistente do AR raw.
- Síntese científica da Fase I com matriz K1–K5, escada final de claims e encerramento explícito das
  hipóteses cinemática e autoregressiva avaliadas.
- Protocolo e ADR da Fase II para K6: tokenização cartesiana causal sob budgets pareados, oito
  famílias sintéticas novas, seeds inéditas, teste fechado e condição de parada anterior ao código.
- Protocolo K7 e ADR para uma cauda articulada revisável com compromisso limitado, distinguindo
  relações espaciais de atualizações temporais de junta em três mecanismos sintéticos isolados.
- Extensão documental do contrato causal para prefixo `committed` imutável e cauda `working`
  versionada, sem alterar a API atual.
- Núcleo `RevisableVectorChain` separado, com fronteiras propostas pelo segmentador causal, solver
  quadrático NumPy, IDs estáveis, histórico append-only e compromisso limitado a quatro elos ou
  256 intervalos raw.
- Três modulações K7 isoladas com coordenadas latentes analíticas e seis matrizes causais pareadas de
  17 escalares, mantendo quatro passos vetoriais contra 16 incrementos raw e persistência separada.
- Gerador K7 por prefixo observável e runner separado de treino/validação com seleção global dos
  regularizadores, ridge multioutput por série, artefatos estruturais e barreira explícita contra o
  teste fechado.
- Escopo canônico de seleção K7 limitado aos primeiros 70% das cinco seeds pré-registradas, com
  recusa de worktree sujo e identidade do commit/config incorporada ao arquivo de seleção.
- Lock canônico K7 em `(lambda_revision=0.1, lambda_bend=1.0)` e referência pré-teste auditada,
  preservando a melhora de K7-R e os sinais negativos de K7-D/K7-U na validação.
