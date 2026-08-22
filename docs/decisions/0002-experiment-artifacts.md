# 0002 — Artefatos locais e resultados canônicos versionados

Status: **Accepted**
Data: 2026-08-22

## Contexto

Versionar toda execução causa crescimento do repositório e mistura exploração com evidência. Ignorar
tudo, por outro lado, impede auditoria dos resultados citados.

## Decisão

- Gravar execuções completas em `artifacts/<run-id>/`, ignoradas pelo Git.
- Registrar configuração efetiva, ambiente, commit, seeds, métricas e figuras.
- Promover somente resultados pequenos e revisados para `reports/reference/`.
- Exigir que resultados promovidos apontem para configuração e commit reproduzíveis.
- Adiar DVC ou tracking server até que volume ou colaboração tornem arquivos locais insuficientes.

## Consequências

- O repositório permanece pequeno.
- Resultados citados continuam auditáveis.
- Artefatos exploratórios locais podem ser perdidos e devem ser copiados externamente quando forem
  importantes.
- A promoção exige uma etapa deliberada de revisão.

## Alternativas consideradas

- **Versionar todos os artefatos:** simples no início, mas degrada rapidamente o histórico Git.
- **MLflow ou serviço equivalente:** útil em maior escala, porém excessivo para o primeiro MVP.
- **DVC desde o início:** defensável para datasets grandes, que ainda não existem neste projeto.
