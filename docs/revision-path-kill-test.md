# Teste eliminatório da trajetória de revisões

Status: **congelado antes da primeira execução integral**.

## Pergunta

K7 representou cada elo por sua última correção `update_theta/update_r` e substituiu parte da
geometria por essas correções. Isso não testou se a história completa das correções acrescenta
informação condicional à geometria atual.

Este experimento pergunta:

> Conhecendo a geometria revisada atual, a trajetória causal completa pela qual os quatro elos
> atuais chegaram a essa geometria melhora a previsão do sinal?

Em notação informacional, a candidata exige evidência compatível com:

```text
I(futuro ; trajetória_de_revisão | geometria_atual) > 0
```

O teste é eliminatório. Um resultado negativo encerra essa extensão; não autoriza outra
representação do mesmo histórico, uma rede maior ou novos geradores sintéticos.

## Representações

Todas as representações compartilham origens, alvos `H={1,8,32}` e splits por endpoint.

| Nome | Conteúdo | Escalares |
|---|---|---:|
| `geometry` | quatro elos `dt,dy,theta,r` e âncora atual | 17 |
| `geometry_last_update` | geometria + última correção de `theta/r` dos quatro elos | 25 |
| `geometry_revision_path` | geometria + descritor da linhagem completa | 121 |
| `geometry_sham_path` | geometria + o mesmo descritor embaralhado dentro de cada split | 121 |
| `raw_matched` | 120 incrementos raw anteriores e âncora atual | 121 |

Para cada origem, a linhagem contém todas as correções observadas até aquele instante para as
identidades dos quatro elos atuais. O descritor possui:

- assinatura de caminho de níveis 1 e 2 sobre oito coordenadas `theta/r`, preservando ordem e
  interações entre elos;
- variação total, energia e número de reversões por coordenada;
- idade e quantidade de revisões por elo.

Assim, o candidato inclui acumulação, recursão, reversão, persistência e propagação entre elos. O
controle `last_update` isola o ganho do histórico sobre a diferença de primeira ordem; `sham_path`
controla dimensão e distribuições marginais; `raw_matched` controla utilidade diante dos dados.

## Dados, ajuste e separação

- três mecanismos K7 isolados: modulação de frequência, baseline e assimetria de crista;
- cinco seeds novas: `1103, 2207, 3301, 4409, 5501`;
- 4096 amostras por série e ruído `0.02`;
- solver K7 já congelado: `lambda_revision=0.1`, `lambda_bend=1.0`;
- treino `[0,2048)`, validação `[2048,2867)` e teste intocado `[2867,4096)`;
- ridge multioutput separado por série e representação;
- `alpha` global por representação escolhido na validação entre
  `{0.01,0.1,1,10,100}`; após a escolha, reajuste em treino+validação e uma abertura do teste.

O histórico embaralhado recebe permutações determinísticas e independentes em treino, validação e
teste. Nenhum valor-alvo participa das permutações.

## Gate e parada

A razão é `RMSE(geometry_revision_path) / RMSE(comparador)`, agregada por média geométrica. A
candidata precisa, simultaneamente:

- razão `<=0.98` contra `geometry`, `geometry_last_update` e `geometry_sham_path`;
- razão `<=1.05` contra `raw_matched`;
- para cada comparação, passar em pelo menos 4/5 seeds, 2/3 mecanismos e 2/3 horizontes.

O teste só passa se as quatro comparações passarem integralmente. Caso contrário, o resultado
permitido é negativo: a linhagem implementada não sustenta uma continuação científica do
VectorChain. Mesmo um resultado positivo ainda seria evidência sintética, não um claim de novidade
ou de utilidade real.
