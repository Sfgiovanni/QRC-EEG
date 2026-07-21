#!/usr/bin/env bash
# Verify, commit and push the complete local QRC-EEG version to the fixed
# master branch. The GitHub token is entered silently and is never stored in
# the remote URL, a regular file, Git config, commit, or command argument.
set -euo pipefail

readonly REPOSITORY_URL="https://github.com/Sfgiovanni/QRC-EEG.git"
readonly REPOSITORY_PAGE="https://github.com/Sfgiovanni/QRC-EEG/tree/master"
readonly REQUIRED_BRANCH="master"
readonly MAX_FILE_BYTES=$((95 * 1024 * 1024))

die() {
    printf 'ERRO: %s\n' "$*" >&2
    exit 1
}

command -v git >/dev/null 2>&1 || die "git não foi encontrado"
[[ -t 0 ]] || die "execute o script em um terminal interativo"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_DIR"

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "diretório não é um repositório Git"
branch=$(git branch --show-current)
[[ "$branch" == "$REQUIRED_BRANCH" ]] || die "branch atual '$branch'; esperado '$REQUIRED_BRANCH'"

origin=$(git remote get-url origin 2>/dev/null || true)
[[ "$origin" == "$REPOSITORY_URL" ]] || die "origin aponta para '$origin'; esperado '$REPOSITORY_URL'"

# Refresh only repository-derived indexes and run the fast release audit.
# No EEG/synthetic/shot simulation is started here.
if [[ -x .venv/bin/python && -f scripts/verify_repository_release.py ]]; then
    printf 'Atualizando o índice derivado e executando verificação rápida...\n'
    .venv/bin/python scripts/build_repository_release.py index
    .venv/bin/python scripts/verify_repository_release.py --quick
else
    printf 'AVISO: ambiente/verificador de release indisponível; executando apenas verificações Git.\n' >&2
fi

printf '\nAlterações locais que serão incluídas:\n'
git --no-pager status --short
git add -A

# Abort before commit if a likely credential was accidentally staged.
while IFS= read -r staged_path; do
    basename=${staged_path##*/}
    [[ "$staged_path" == "scripts/push_master_with_token.sh" ]] && continue
    case "$basename" in
        .env|.env.*|*credentials*|*credential*|*secret*|*token*|*.pem|*.key)
            die "arquivo potencialmente sensível staged: $staged_path"
            ;;
    esac
done < <(git diff --cached --name-only --diff-filter=ACMR)

# GitHub rejects individual Git blobs above 100 MiB. Keep a safety margin.
while IFS= read -r -d '' staged_path; do
    [[ -f "$staged_path" ]] || continue
    size=$(wc -c <"$staged_path")
    (( size <= MAX_FILE_BYTES )) || die "arquivo excede 95 MiB: $staged_path ($size bytes)"
done < <(git diff --cached --name-only --diff-filter=ACMR -z)

# Generated CSVs intentionally use CRLF and would create false whitespace
# warnings. Source code and documentation remain checked.
git diff --cached --check -- . ':(exclude)**/*.csv' || \
    die "o diff staged contém whitespace inválido ou marcador de conflito"

if git diff --cached --quiet; then
    printf '\nNada novo para commit; será enviado o HEAD atual.\n'
else
    printf '\nResumo staged:\n'
    git --no-pager diff --cached --stat --compact-summary
    readonly DEFAULT_COMMIT_MESSAGE="Add Gate 1B post-gate robustness extension of the effective-kernel mechanism"
    printf '\nMensagem do commit [%s]: ' "$DEFAULT_COMMIT_MESSAGE"
    IFS= read -r commit_message
    commit_message=${commit_message:-"$DEFAULT_COMMIT_MESSAGE"}
    git commit -m "$commit_message" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
fi

printf '\nUse um fine-grained GitHub PAT para Sfgiovanni/QRC-EEG com Contents: Read and write.\n'
IFS= read -rsp 'GitHub token (entrada oculta): ' github_token
printf '\n'
[[ -n "$github_token" ]] || die "token vazio"

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/qrc-eeg-push.XXXXXX")
chmod 700 "$tmp_dir"
writer_pid=""

cleanup() {
    stty echo 2>/dev/null || true
    if [[ -n "$writer_pid" ]]; then
        kill "$writer_pid" 2>/dev/null || true
    fi
    unset github_token 2>/dev/null || true
    rm -rf -- "$tmp_dir"
}
trap cleanup EXIT HUP INT TERM

token_fifo="$tmp_dir/token.fifo"
mkfifo -m 600 "$token_fifo"

# This temporary program contains no token; it reads the secret once from the
# protected FIFO only when Git requests the HTTPS password.
printf '%s\n' \
    '#!/usr/bin/env bash' \
    'case "$1" in' \
    '  *sername*) printf '\''%s\n'\'' "x-access-token" ;;' \
    '  *assword*) IFS= read -r secret <"$GITHUB_TOKEN_FIFO"; printf '\''%s\n'\'' "$secret" ;;' \
    '  *) exit 1 ;;' \
    'esac' >"$tmp_dir/askpass.sh"
chmod 700 "$tmp_dir/askpass.sh"

(umask 077; printf '%s\n' "$github_token" >"$token_fifo") &
writer_pid=$!
unset github_token

printf '\nEnviando HEAD para origin/master...\n'
GITHUB_TOKEN_FIFO="$token_fifo" \
GIT_ASKPASS="$tmp_dir/askpass.sh" \
GIT_TERMINAL_PROMPT=0 \
git -c credential.helper= push --set-upstream origin HEAD:"$REQUIRED_BRANCH"

wait "$writer_pid"
writer_pid=""
printf '\nPush concluído: %s\n' "$REPOSITORY_PAGE"
printf 'O token não foi armazenado. Você pode revogá-lo nas configurações do GitHub.\n'
