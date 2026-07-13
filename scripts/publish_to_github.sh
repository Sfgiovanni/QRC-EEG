#!/bin/sh
# Safely create (if needed) and push this repository to GitHub using a PAT.
# The token is entered hidden, piped only to github_api.py and to a
# restricted FIFO for git's askpass -- it is never written to disk, the
# remote URL, git config, or the command line.
set -eu

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

command -v git >/dev/null 2>&1 || die "git is required"
command -v python3 >/dev/null 2>&1 || die "python3 is required"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_DIR"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not inside a Git repository"

printf 'GitHub repository name: '
IFS= read -r repo_name
case "$repo_name" in
    ""|*/*|*[!A-Za-z0-9._-]*) die "use only letters, numbers, dot, underscore, or hyphen" ;;
esac

printf 'Visibility [private/public] (Enter = private): '
read -r visibility
visibility=${visibility:-private}
case "$visibility" in
    private|Private|PRIVATE|privado|Privado|PRIVADO|pr|PR|pvt) private_json=true; visibility=private ;;
    public|Public|PUBLIC|publico|Publico|PUBLICO|público|Público|PÚBLICO|pub|PUB|p|P) private_json=false; visibility=public ;;
    *) die "visibility must be private/privado or public/publico (or press Enter for private)" ;;
esac

printf 'GitHub token (input hidden): ' >&2
stty -echo 2>/dev/null || true
IFS= read -r token
stty echo 2>/dev/null || true
printf '\n' >&2
[ -n "$token" ] || die "empty token"

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/qrc-eeg-github-publish.XXXXXX")
chmod 700 "$tmp_dir"
cleanup() {
    stty echo 2>/dev/null || true
    [ -z "${writer_pid:-}" ] || kill "$writer_pid" 2>/dev/null || true
    rm -rf "$tmp_dir"
    unset token 2>/dev/null || true
}
trap cleanup EXIT HUP INT TERM

api_request() {
    method=$1
    url=$2
    data=${3:-}
    printf '%s\n%s' "$token" "$data" | python3 "$SCRIPT_DIR/github_api.py" "$method" "$url" "$tmp_dir/response.json"
}

status=$(api_request GET "https://api.github.com/user")
[ "$status" = 200 ] || die "GitHub authentication failed (HTTP $status)"
owner=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["login"])' "$tmp_dir/response.json")
[ -n "$owner" ] || die "could not determine GitHub username"

status=$(api_request GET "https://api.github.com/repos/$owner/$repo_name")
case "$status" in
    200) printf 'Remote repository already exists: %s/%s\n' "$owner" "$repo_name" ;;
    404)
        payload=$(printf '{"name":"%s","private":%s,"description":"Exponential state-memory kernel vs QRC alternatives, case study on Bonn EEG data"}' "$repo_name" "$private_json")
        status=$(api_request POST "https://api.github.com/user/repos" "$payload")
        [ "$status" = 201 ] || die "repository creation failed (HTTP $status); the token may lack repository-creation permission"
        printf 'Created %s repository: %s/%s\n' "$visibility" "$owner" "$repo_name"
        ;;
    *) die "repository lookup failed (HTTP $status)" ;;
esac

remote_url="https://github.com/$owner/$repo_name.git"
if git remote get-url origin >/dev/null 2>&1; then
    current=$(git remote get-url origin)
    [ "$current" = "$remote_url" ] || die "origin already points to $current; refusing to replace it"
else
    git remote add origin "$remote_url"
fi

# GIT_ASKPASS reads the token once from a restricted FIFO. The token is never
# stored in the remote URL, a regular file, the command line, or Git config.
fifo="$tmp_dir/token.fifo"
mkfifo "$fifo"
printf '%s\n' \
    '#!/bin/sh' \
    'case "$1" in' \
    '    *sername*) printf '\''%s\n'\'' "$GITHUB_OWNER" ;;' \
    '    *assword*) IFS= read -r secret <"$GITHUB_TOKEN_FIFO"; printf '\''%s\n'\'' "$secret" ;;' \
    '    *) exit 1 ;;' \
    'esac' >"$tmp_dir/askpass.sh"
chmod 700 "$tmp_dir/askpass.sh"

(printf '%s\n' "$token" >"$fifo") &
writer_pid=$!
unset token

GITHUB_OWNER=$owner \
GITHUB_TOKEN_FIFO=$fifo \
GIT_ASKPASS=$tmp_dir/askpass.sh \
GIT_TERMINAL_PROMPT=0 \
git -c credential.helper= push --set-upstream origin master
wait "$writer_pid"
writer_pid=

printf '\nPublished successfully: https://github.com/%s/%s\n' "$owner" "$repo_name"
printf 'The token was not saved. You may now revoke it in GitHub settings.\n'
