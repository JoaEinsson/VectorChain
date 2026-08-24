# Protocolo pré-especificado da Fase II — cadeia articulada revisável

Status: **pré-especificado antes da implementação e da primeira execução da Etapa 12-A**.

Data de congelamento: 2026-08-23.

Este protocolo testa K7, uma hipótese diferente de K2 e K6. K2 avaliou relações espaciais entre
elos imutáveis; K6 avalia tokenização imutável sob orçamento. K7 permite que uma cauda provisória
seja corrigida recursivamente quando cada nova amostra chega e pergunta se a própria correção das
articulações contém sinal preditivo.

O termo `inspirada em cinemática inversa` descreve a formulação geométrica. O problema reduzido
abaixo mantém os tempos das juntas fixos e resolve suas ordenadas por mínimos quadrados
regularizados; ele não deve ser apresentado como um solver genérico de robótica.

## 1. Pergunta e hipótese

Pergunta primária:

> Quando uma cauda causal de vetores pode se rearticular dentro de um atraso limitado, as correções
> temporais de ângulo e comprimento carregam informação preditiva além da geometria revisada e das
> relações espaciais estáticas?

Separação necessária:

```text
delta_theta_space[i,t] = theta[i,t] - theta[i-1,t]
delta_r_space[i,t]     = r[i,t]     - r[i-1,t]

update_theta[i,t] = theta[i,t] - theta[i,t-1]
update_r[i,t]     = r[i,t]     - r[i,t-1]
```

Os primeiros pares descrevem a forma da cadeia num instante. Os segundos descrevem como a mesma
junta mudou após uma nova observação. Somente `update_theta/update_r` constituem a candidata K7.

K7 possui três subgates:

- **K7-R:** revisar a cauda melhora a geometria estática imutável;
- **K7-D:** a dinâmica de correção melhora geometria revisada e relações espaciais revisadas;
- **K7-U:** o resultado permanece útil diante de raw com payload e parâmetros pareados.

## 2. Estado comprometido e estado provisório

O `VectorChain` atual e seu contrato de emissão não mudam. A extensão mantém dois estados:

```text
C_t = prefixo comprometido, público e imutável
W_t = cauda de trabalho, causal, versionada e revisável
```

`W_t` contém no máximo quatro elos e no máximo 256 intervalos raw. Quando um quinto elo entra ou o
span excederia 256, o elo completo mais antigo é comprometido e nunca mais pode mudar. Se apenas o
elo aberto exceder 256 intervalos, a condição falha explicitamente; não se inventa uma fronteira.

Os tempos das fronteiras são propostos pelo segmentador causal atual com `tolerance=0.03` e não são
movidos por K7. A revisão altera somente as ordenadas das juntas provisórias. Isso isola a pergunta
sobre rearticulação antes de introduzir busca combinatória de fronteiras.

Cada junta e elo recebe identidade estável enquanto permanece em `W_t`. Um elo recém-criado recebe
`update_theta=update_r=0`; atualizações posteriores usam a diferença para a mesma identidade.

## 3. Ajuste inverso da cauda

Considere fronteiras provisórias fixas:

```text
tau_0 < tau_1 < ... < tau_m = t,  1 <= m <= 4
```

e ordenadas de junta `y_0,...,y_m`. `y_0` é a última junta comprometida e `y_m=x[t]`; ambas são
âncoras exatas. As ordenadas internas são obtidas por:

```text
argmin_y
    mean_j (x[j] - PL(j; tau, y))^2
  + lambda_revision * mean_k (y[k] - y_previous[k])^2
  + lambda_bend * mean_k (slope[k] - slope[k-1])^2
```

onde `PL` é a interpolação linear conectada e:

```text
slope[k] = (y[k] - y[k-1]) / (tau[k] - tau[k-1])
```

Termos sem elementos são zero. O problema é quadrático e deve ser resolvido deterministicamente
com NumPy; nenhuma dependência de otimização é autorizada nesta etapa. Depois do ajuste:

```text
dt[i]    = tau_i - tau_(i-1)
dy[i]    = y_i - y_(i-1)
theta[i] = atan2(dy[i], dt[i])
r[i]     = sqrt(dt[i]^2 + dy[i]^2)
```

As âncoras garantem conexão com o prefixo e passagem pelo ponto corrente. `dt` continua inteiro e
positivo porque as fronteiras não mudam.

A penalidade de revisão inclui somente juntas internas cuja identidade existia na versão anterior.
Juntas novas são inicializadas no valor raw da fronteira e não recebem um passado fictício.

### Seleção mínima do regularizador

Uma única dupla global é escolhida no treino interno:

```text
lambda_revision in {0.01, 0.1, 1.0}
lambda_bend     in {0.01, 0.1, 1.0}
```

A escolha minimiza o NRMSE multi-horizonte de `revisable_absolute`, que não contém as features
temporais candidatas. Empates até `1e-12` favorecem maior `lambda_revision` e depois maior
`lambda_bend`. A mesma dupla é usada em sinais, seeds e todas as representações revisáveis. Nenhuma
outra tolerância, lag, quantidade de elos ou regularizador é testado na Etapa 12-A.

## 4. Conjunto sintético mínimo identificável

Não se usa uma senoide única combinando frequência, média e forma: um resultado assim não
identificaria qual mecanismo produziu as correções. A Etapa 12-A usa três condições isoladas de uma
mesma família oscilatória.

Com `n_points=4096`, `u=t/(n_points-1)` e ruído observacional
`epsilon[t] ~ N(0,0.02^2)`, defina:

```text
g(phi, kappa) = (sin(phi) + kappa*sin(2*phi + pi/4)) / (1 + abs(kappa))
```

| Condição | Coordenada que varia | Definição antes do ruído |
|---|---|---|
| `frequency_modulation` | frequência instantânea | `f(u)=20-12*cos(2*pi*3*u)`; `phi[t]=phi[t-1]+2*pi*f(u[t])/(n_points-1)`; `mu=0`, `kappa=0` |
| `baseline_modulation` | baseline/média local | `phi=2*pi*16*u`; `mu(u)=1-cos(2*pi*3*u)`; `kappa=0` |
| `crest_asymmetry_modulation` | assimetria harmônica | `phi=2*pi*16*u`; `mu=0`; `kappa(u)=0.225*(1-cos(2*pi*3*u))` |

`baseline_modulation` varia de zero a duas amplitudes do componente fundamental. Essa escala é mais
diagnóstica que um deslocamento arbitrário `0 -> 12`, que dominaria a tolerância absoluta sem
isolar melhor o mecanismo.

As três modulações completam três ciclos e, portanto, aparecem em treino, validação e teste. Não há
condição combinada, grade de ruído, amplitude extra ou nova família nesta etapa. Uma composição só
poderá ser pré-especificada depois que os três mecanismos isolados passarem.

## 5. Seeds inéditas

As seeds resultam do SHA-256 dos rótulos `vectorchain-phase2-stage12a-seed-0` a
`vectorchain-phase2-stage12a-seed-4`, lendo os primeiros 64 bits little-endian e removendo o bit de
sinal.

| Rótulo | Seed | SHA-256 completo |
|---|---:|---|
| `seed-0` | `2652130430004669680` | `f0ac6e129342cea49e25c9ee081eebeaa9d5ae7cfcfaf07556dc369d5b6de629` |
| `seed-1` | `2132228189405173304` | `38d691c13d32979d97040e026e9895a4e274226247001c2b437cf1c86ada5bd6` |
| `seed-2` | `7118215795047038510` | `2e7e5a3a8dfbc8e2ff468f6ed9cc9f86893d60d17d9adaee35980b7a6d4e54b4` |
| `seed-3` | `7649379155735444565` | `557ce361d00d28ea49412bef27370c8aac6bcc132e5f609b601b5807da61965f` |
| `seed-4` | `1017387535040708910` | `2e0103a1927c1e8eed0bb0bbc58fbf97bd03450b57f46af79918d33a16843016` |

Bootstrap descritivo usa `7817596609602243496`, derivada de
`vectorchain-phase2-stage12a-bootstrap`. Desenvolvimento e testes usam somente seeds `11` e `22`.
Qualquer execução prematura de uma seed canônica invalida o conjunto antes do teste.

## 6. Forecasting e ablations necessárias

Origens ocorrem a cada quatro amostras e exigem quatro elos simultaneamente presentes em `W_t`,
incluindo o aberto. Origens inelegíveis são removidas de todas as representações. O alvo direto é:

```text
y_h(t) = x[t+h] - x[t],  h in {1, 8, 32}
```

Splits pelos endpoints dos alvos:

- treino externo: primeiros 50%;
- validação externa: 20% seguintes;
- teste fechado: 30% finais;
- últimos 20% do treino externo: seleção interna dos dois regularizadores.

Todas as representações treinadas recebem `x[t]` como âncora, têm 17 escalares, três outputs e
`3*(17+1)=54` parâmetros incluindo interceptos. Features são padronizadas somente no treino. O
ridge usa `alpha=0.001`.

| Representação | Quatro features por elo | Papel |
|---|---|---|
| `immutable_absolute` | `dt,dy,theta,r` | controla a revisão |
| `revisable_absolute` | `dt,dy,theta,r` | controla as features temporais |
| `revisable_spatial` | `dt,dy,delta_theta_space,delta_r_space` | testa relações espaciais após revisão |
| `revisable_temporal` | `dt,dy,update_theta,update_r` | candidata K7 |
| `raw_matched` | últimos 16 primeiros incrementos raw | mesmo payload e parâmetros, 16 passos |
| `persistence` | nenhum contexto treinado | prevê incremento zero |

Essas são todas as representações autorizadas. Não entram K6, média móvel, Transformer, Kalman,
ABBA, fronteiras móveis ou modelos probabilísticos. O objetivo é decidir primeiro se revisão e
correções temporais acrescentam sinal no caso linear mínimo.

## 7. Métricas de mecanismo e previsão

### Correção articular

Por elo revisável:

```text
correction_energy = sqrt(update_theta^2 + (update_r / median_r_train)^2)
```

Para cada coordenada latente conhecida (`f`, `mu` ou `kappa`), amostras no quartil superior de sua
derivada absoluta formam a região `changing`; o quartil inferior forma `stationary`. Os quartis são
calculados analiticamente sobre a função geradora, não sobre o sinal ruidoso.

Registrar a razão entre medianas de energia
`median(changing)/max(median(stationary),1e-12)`, além de distribuições completas, identidades dos
elos, idade, instante de criação e instante de compromisso.

### Reconstrução e estabilidade

- RMSE causal da cauda antes e depois da revisão;
- erro do endpoint corrente;
- quantidade e magnitude das revisões por elo;
- atraso raw e quantidade de elos até compromisso;
- edições tentadas ou observadas no prefixo comprometido;
- condição numérica e falhas do sistema linear;
- runtime e memória da atualização.

### Previsão

- RMSE e MAE raw por mecanismo, seed e horizonte;
- razão pareada `RMSE(candidata)/RMSE(controle)`;
- passos, escalares, bytes, parâmetros e rank;
- 10.000 bootstraps pareados das 15 séries completas, apenas como intervalo descritivo.

Origens não são réplicas independentes. A unidade é a série completa `mecanismo × seed`.

## 8. Gate K7

O teste é o único split decisório. As escolhas são congeladas no treino e o teste é aberto uma vez
após auditoria estrutural e commit limpo.

Depois da auditoria estrutural, o teste deve ser aberto mesmo se a validação externa for
desfavorável; validação não funciona como regra de parada opcional.

### K7-R — valor da revisão

`RMSE(revisable_absolute) / RMSE(immutable_absolute) <= 0.99` em:

- pelo menos 4/5 seeds, agregando mecanismos e horizontes;
- pelo menos 2/3 mecanismos, agregando seeds e horizontes;
- pelo menos 2/3 horizontes, robustos em 4/5 seeds.

### K7-D — valor das correções temporais

`revisable_temporal` deve atingir razão `<= 0.99` separadamente contra
`revisable_absolute` e `revisable_spatial`, sob os mesmos três critérios de robustez de K7-R.

Além disso, `correction_energy(changing) / correction_energy(stationary) >= 1.25` deve ocorrer em
4/5 seeds e pelo menos 2/3 mecanismos.

### K7-U — utilidade diante de raw

`RMSE(revisable_temporal) / RMSE(raw_matched) <= 1.05` em 4/5 seeds, 2/3 mecanismos e 2/3
horizontes. A candidata deve usar quatro passos contra 16 de raw, mantendo os mesmos 17 escalares e
54 parâmetros.

### Condições estruturais obrigatórias

- zero alterações no prefixo comprometido;
- cauda com no máximo quatro elos e 256 intervalos;
- endpoints inicial comprometido e corrente preservados até `1e-12`;
- `dt` inteiro positivo, features e soluções finitas em 100% das condições;
- mesma origem, alvo, split e fronteiras entre todas as ablations vetoriais;
- equivalência batch/stream e determinismo da revisão;
- alteração arbitrária do futuro não muda o estado comprometido nem a versão provisória já
  observada no mesmo instante;
- toda falha aparece em `failures.csv`; validade de execução não substitui o gate científico.

K7 passa somente se K7-R, K7-D, K7-U e todas as condições estruturais passarem.

## 9. Sequência operacional

1. Versionar este protocolo, a extensão causal e o ADR 0013.
2. Implementar somente a cauda revisável, solver quadrático, identidades e testes com seeds 11/22.
3. Implementar as três condições sintéticas e as seis representações, sem gerar seeds canônicas.
4. Congelar código/config em commit limpo.
5. Rodar treino, seleção e validação sem materializar teste; congelar `selection.json` e seu hash.
6. Abrir teste uma vez, preservar todos os resultados e repetir no mesmo commit.
7. Promover tabelas, relatório, ambiente e manifestos para `reports/reference/`.

Nenhuma composição de mecanismos ou modelo adicional é executado antes da decisão K7.

## 10. Claims e condição de parada

Se K7 passar integralmente, o claim máximo será:

> Nos três mecanismos sintéticos pré-especificados, correções temporais de uma cauda vetorial
> revisável acrescentaram sinal preditivo além da geometria revisada e das relações espaciais, com
> forecasting não inferior a raw de mesmo payload usando um quarto dos passos.

Esse claim não implica cinemática inversa clássica, novidade, generalização real ou benefício de
fronteiras móveis.

Se qualquer subgate falhar:

- preservar o resultado;
- não substituir o mecanismo, sinal, lag, regularizador ou modelo no teste;
- não executar a condição combinada;
- não abrir revisão de fronteiras, redes neurais ou dados reais por esta sequência.

## 11. Artefatos obrigatórios

- config, ambiente, `selection.json`, `gate.json` e manifesto;
- juntas e elos versionados por instante em tabela integral comprimida;
- `commit_audit.csv`, `causality_audit.csv` e `solver_audit.csv`;
- origens, inputs, previsões, métricas por condição e `failures.csv`;
- resumos por seed, mecanismo e horizonte;
- plots de cadeia antes/depois, energia de correção e erro pareado derivados das tabelas;
- reprodução no mesmo commit, excluindo somente campos de runtime da comparação científica.
