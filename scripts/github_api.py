"""Minimal GitHub API client used by publish_to_github.sh.

The first stdin line is the token; the remaining stdin content is an optional
JSON request body. The token is never accepted as a command-line argument.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: github_api.py METHOD URL OUTPUT")
    method, url, output = sys.argv[1:]
    token = sys.stdin.readline().rstrip("\n")
    body = sys.stdin.read().encode() or None
    if not token:
        raise SystemExit("empty token")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "qrc-eeg-publisher/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status, payload = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, payload = error.code, error.read()
    Path(output).write_bytes(payload)
    print(status, end="")


if __name__ == "__main__":
    main()
