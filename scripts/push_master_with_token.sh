#!/usr/bin/env bash
# Commit the current QRC-EEG phase-1 working tree and push master to the fixed GitHub
# repository. The GitHub token is read without echo and passed to Git through
# a restricted FIFO; it is never stored in a file, remote URL, or Git config.
set -euo pipefail

readonly EXPECTED_REMOTE="https://github.com/Sfgiovanni/QRC-EEG.git"
readonly EXPECTED_BRANCH="master"

die() {
  printf 'ERRO: %s\n' "$*" >&2
  exit 1
}

command -v git >/dev/null 2>&1 || die "git não foi encontrado"
[[ -t 0 ]] || die "execute este script em um terminal interativo"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_DIR"

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "não é um repositório Git"

branch=$(git branch --show-current)
[[ "$branch" == "$EXPECTED_BRANCH" ]] || die "branch atual é '$branch'; esperado '$EXPECTED_BRANCH'"

remote=$(git remote get-url origin 2>/dev/null || true)
[[ "$remote" == "$EXPECTED_REMOTE" ]] || die "origin aponta para '$remote'; esperado '$EXPECTED_REMOTE'"

printf '\nAlterações locais que serão incluídas:\n'
git status --short

# Include the complete current version, including intentional deletions. Files
# ignored by .gitignore (raw EEG, virtualenv, caches) remain excluded.
git add -A
# Python's csv writer intentionally emits CRLF records. Git's --check reports
# every CR as "trailing whitespace", producing thousands of false positives;
# keep the check for source/docs while excluding generated CSV artifacts.
git diff --cached --check -- . ':(exclude)**/*.csv' || \
  die "o diff staged contém erros de whitespace/conflito fora dos CSVs gerados"

if git diff --cached --quiet; then
  printf '\nNenhuma alteração nova para commit; será enviado o HEAD atual.\n'
else
  printf '\nResumo do commit:\n'
  git --no-pager diff --cached --shortstat
  printf '\nMensagem do commit [Fix segment-blocked ridge selection and rerun EEG]: '
  IFS= read -r commit_message
  commit_message=${commit_message:-"Fix segment-blocked ridge selection and rerun EEG"}
  git commit -m "$commit_message"
fi

printf '\nUse um fine-grained PAT com acesso ao repositório Sfgiovanni/QRC-EEG'
printf ' e permissão Contents: Read and write.\n'
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

cat >"$tmp_dir/askpass.sh" <<'ASKPASS'
#!/usr/bin/env bash
case "$1" in
  *sername*) printf '%s\n' "Sfgiovanni" ;;
  *assword*) IFS= read -r token <"$GITHUB_TOKEN_FIFO"; printf '%s\n' "$token" ;;
  *) exit 1 ;;
esac
ASKPASS
chmod 700 "$tmp_dir/askpass.sh"

# The background writer holds the secret only until Git asks for a password.
(umask 077; printf '%s\n' "$github_token" >"$token_fifo") &
writer_pid=$!
unset github_token

printf '\nEnviando master para %s ...\n' "$EXPECTED_REMOTE"
GITHUB_TOKEN_FIFO="$token_fifo" \
GIT_ASKPASS="$tmp_dir/askpass.sh" \
GIT_TERMINAL_PROMPT=0 \
git -c credential.helper= push --set-upstream origin "$EXPECTED_BRANCH"

wait "$writer_pid"
writer_pid=""

printf '\nPush concluído: https://github.com/Sfgiovanni/QRC-EEG/tree/master\n'
printf 'O token não foi armazenado.\n'
