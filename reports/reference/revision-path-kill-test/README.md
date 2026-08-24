# Teste eliminatório da trajetória de revisões

Status: **resultado negativo, reproduzido; condição de parada satisfeita**.

O experimento verificou se a trajetória completa pela qual os quatro elos revisáveis chegaram à
geometria atual acrescenta informação preditiva. Diferentemente de K7-D, a candidata preservou
`dt,dy,theta,r` e acrescentou toda a linhagem das identidades atuais: assinatura temporal de ordem
2, variação total, energia, reversões, idade e quantidade de revisões.

## Resultado decisório

Cada razão abaixo é `RMSE(geometry_revision_path) / RMSE(comparador)`, agregada por média
geométrica sobre três mecanismos, cinco seeds novas e horizontes `1/8/32`.

| Comparação | Razão global | Limiar | Seeds | Mecanismos | Horizontes | Resultado |
|---|---:|---:|---:|---:|---:|---|
| geometria atual | 1,040939 | 0,980 | 0/5 | 0/3 | 0/3 | não passou |
| geometria + última atualização | 1,062916 | 0,980 | 0/5 | 0/3 | 0/3 | não passou |
| trajetória embaralhada | 0,955571 | 0,980 | 5/5 | 3/3 | 3/3 | passou |
| raw pareado | 2,046248 | 1,050 | 0/5 | 0/3 | 0/3 | não passou |

A estrutura temporal real da trajetória venceu seu controle embaralhado em aproximadamente 4,4%,
de forma consistente. Isso mostra que o descritor não era apenas ruído intercambiável. Contudo,
essa informação foi redundante ou insuficiente para a tarefa: acrescentá-la piorou a geometria
isolada em 4,1%, perdeu para a atualização mais recente em 6,3% e teve pouco mais que o dobro do
RMSE do raw pareado.

O ridge da candidata e do sham escolheu `alpha=100` exclusivamente na validação, enquanto
geometria, última atualização e raw escolheram `0.01`, `0.01` e `1.0`, respectivamente. A forte
regularização selecionada é evidência adicional de que a linhagem de 104 escalares não forneceu
ganho utilizável suficiente para justificar sua complexidade.

## Análise secundária obrigatoriamente descritiva

`geometry_last_update / geometry` teve razão global `0,979323`, equivalente a melhora média de
2,07%. Esse contraste não era o gate principal e não foi robusto: atingiu `<=0.98` em somente 2/5
seeds, 1/3 mecanismos e 2/3 horizontes. A melhora concentrou-se em modulação de frequência e nos
horizontes 8/32; não constitui resgate de K7 nem resultado publicável isolado. Essa variante também
permaneceu aproximadamente 92,5% pior que raw pareado, por derivação das razões primárias.

## Identidade e reprodução

- commit científico limpo: `2bca8ccd5e4892721b8e4b96cd102d774a744a6f`;
- configuração SHA-256:
  `28173b876f61056f5b297ab1cc2f216e3f33296d24d9c7eb30ad32dc85af2450`;
- execução primária: `20260824T163213710896Z-28173b876f`, 27,85 s;
- reprodução: `20260824T163320936986Z-28173b876f`, 29,31 s;
- ambiente: Windows 11, CPython 3.12.12 e NumPy 2.5.2;
- `config`, seleção, métricas, razões, gate e relatório foram idênticos byte a byte nas duas
  execuções.

Comando:

```powershell
uv run python experiments/11_revision_path_kill_test.py --config configs/forecasting/revision_path_kill_test.toml
```

## Decisão científica

O claim permitido é apenas:

> Nos três mecanismos sintéticos e cinco seeds novas, a ordem e as interações da trajetória de
> revisão continham associação acima de um controle embaralhado, mas não acrescentaram utilidade
> preditiva à geometria atual e ficaram muito atrás do histórico raw pareado.

Isso não sustenta novidade de nível médio, superioridade prática nem continuação com outra
codificação do mesmo histórico. Conforme o protocolo congelado, a extensão de trajetória de
revisões e o desenvolvimento científico ativo do VectorChain são encerrados neste resultado.
