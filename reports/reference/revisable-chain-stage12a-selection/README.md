# K7 Stage 12-A — seleção e validação pré-teste

Status: **seleção congelada; teste fechado não materializado**.

Esta referência preserva a execução canônica de treino, seleção interna e validação externa de K7.
Ela não contém `gate.json`, não decide K7 e não acessou os 30% finais das séries.

## Proveniência

- run: `20260824T141458527105Z-badcd65998`;
- commit científico limpo: `742d51f215d3699aed75b2c26129801b09e4417e`;
- configuração: [`config.json`](config.json);
- ambiente: [`environment.json`](environment.json);
- manifesto integral dos artefatos locais: [`source-artifact-manifest.json`](source-artifact-manifest.json);
- SHA-256 do manifesto de origem: `b2513e607d43f7bffe8a8915cc4f8c5c1c62ffe18b1e77d29ba7088ddd7c60cf`;
- tamanho total dos artefatos de origem: `33.762.057` bytes;
- falhas: zero;
- auditorias causais: 15/15;
- estados estruturais válidos: 43.005/43.005;
- `test_materialized=false` no ambiente e na seleção.

Os artefatos integrais permanecem em
`artifacts/revisable-chain-stage12a-selection/20260824T141458527105Z-badcd65998/`. O manifesto
preservado registra o hash de cada tabela, modelo e arquivo comprimido, mesmo quando o arquivo
volumoso não é duplicado nesta referência Git.

## Seleção congelada

Uma única dupla global foi escolhida pelo NRMSE de `revisable_absolute` no treino interno:

```text
lambda_revision = 0.1
lambda_bend     = 1.0
global_nrmse    = 0.5230412893176433
```

O lock que o futuro comando de teste deverá consumir está em
[`configs/forecasting/revisable_chain_selection.lock.json`](../../../configs/forecasting/revisable_chain_selection.lock.json).
Seu SHA-256 canônico com finais de linha LF é
`d4e3e4ba8b03e4e8e3ca638cc2061790f84e81475a44cdc5713fb331525f58b2`. O arquivo de origem gerado
no Windows tinha finais CRLF e SHA-256
`c2c83339a786d4a99192cff50aaa7c39a0d3f1da4e214cb4a166462c0c6bab5c`; o conteúdo JSON é idêntico.

## Resultado de validação, não decisório

| Comparação | Geomean da razão de RMSE | Células no limiar |
|---|---:|---:|
| revisável absoluta / imutável absoluta | 0.829554 | 45/45 (`<=0.99`) |
| temporal / revisável absoluta | 1.257053 | 0/45 (`<=0.99`) |
| temporal / revisável espacial | 1.033058 | 4/45 (`<=0.99`) |
| temporal / raw pareado | 1.450430 | 0/45 (`<=1.05`) |
| energia changing / stationary | 0.961470 | 3/15 (`>=1.25`) |

K7-R apresentou sinal consistente na validação. K7-D e K7-U foram desfavoráveis: a revisão da
cauda ajudou a geometria estática, mas `update_theta/update_r` não acrescentaram previsão sobre a
geometria revisada e não se aproximaram do raw pareado. Isso é evidência pré-teste contra o claim
completo, não uma decisão científica final.

Conforme o protocolo, o resultado não autoriza retuning, troca de features ou cancelamento opcional
do teste. A próxima implementação deve apenas consumir o lock, abrir o teste uma vez e preservar o
resultado independentemente do sinal.
