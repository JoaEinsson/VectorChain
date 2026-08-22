# Configuração recomendada do repositório GitHub

Algumas regras não podem ser impostas por arquivos versionados. Depois que a primeira CI passar,
configure no GitHub:

## Branch `main`

- Exigir pull request antes de merge para mudanças colaborativas.
- Exigir os checks `Quality gates` e todos os jobs `Tests`.
- Exigir que a branch esteja atualizada antes do merge.
- Bloquear force push e deleção.
- Exigir resolução de todas as conversas.
- Manter bypass apenas para recuperação administrativa.

Para trabalho individual inicial, a exigência de uma aprovação externa pode ficar desativada; os
checks automatizados continuam obrigatórios.

## Segurança

- Habilitar Dependabot alerts e security updates.
- Habilitar private vulnerability reporting.
- Habilitar secret scanning e push protection quando disponíveis para o repositório.
- Revisar permissões de Actions e manter `GITHUB_TOKEN` somente leitura por padrão.

## Colaboração

- Habilitar Discussions se o link do template de issues for mantido.
- Criar labels `experiment` e `dependencies`; `bug` normalmente já existe.
- Usar squash merge para preservar um histórico principal legível.
- Desabilitar merge commits se o time optar por squash como política única.

## Releases

- Não publicar pacote antes do primeiro benchmark reproduzível.
- Usar tags assinadas `v0.x.y` quando possível.
- Anexar changelog, commit, lockfile e limitações conhecidas a cada release.
- Definir uma licença antes da primeira distribuição pública.
