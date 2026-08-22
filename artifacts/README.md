# Artefatos locais

Esta pasta recebe saídas brutas e reproduzíveis de experimentos. Todo o conteúdo, exceto este
arquivo, é ignorado pelo Git.

Cada execução deve criar uma pasta própria contendo, no mínimo:

```text
artifacts/<run-id>/
├── config.json
├── environment.json
├── metrics.csv
└── plots/
```

Um artefato só pode ser promovido para `reports/reference/` depois de revisado e acompanhado do
commit e da configuração que o produziram.
