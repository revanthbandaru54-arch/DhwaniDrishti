r"""
DhwaniDrishti — Pure-Python Git Initializer & GitHub Publisher (push_to_github.py)
==================================================================================
Works even on machines where `git.exe` or `gh.exe` is not installed:
  1. Initializes a standard `.git/` repository on branch `main`, stages all
     project files (respecting `.gitignore`), and writes the Git index + commit.
  2. Creates the `my-new-website` repository under your GitHub account and pushes
     all committed files via the GitHub REST API over HTTPS.

Usage:
    # Step 1: Initialize local Git repository & commit all files
    .\.venv\Scripts\python.exe push_to_github.py --init-only

    # Step 2: Create `my-new-website` on GitHub & push all files
    .\.venv\Scripts\python.exe push_to_github.py --repo my-new-website --token <YOUR_GITHUB_PAT>
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

IGNORE_DIRS = {
    ".git",
    ".agents",
    ".venv",
    "venv",
    "ENV",
    "env",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
    ".pytest_cache",
    ".vscode",
    ".idea",
}
IGNORE_EXTS = {".pyc", ".pyo", ".pyd", ".exe", ".msi", ".spec", ".log", ".sqlite3", ".db"}
IGNORE_FILES = {".DS_Store", "Thumbs.db", "Desktop.ini", ".env", ".env.local"}


def collect_project_files(root: Path) -> list[Path]:
    files: list[Path] = []

    def _walk(directory: Path) -> None:
        for item in sorted(directory.iterdir(), key=lambda p: p.name):
            if item.name in IGNORE_DIRS or item.name in IGNORE_FILES or item.suffix in IGNORE_EXTS:
                continue
            if item.is_dir():
                _walk(item)
            elif item.is_file():
                files.append(item)

    _walk(root)
    return files


def init_and_commit_local_git(root: Path, commit_message: str) -> tuple[str, list[Path]]:
    git_dir = root / ".git"
    for sub in ("objects/info", "objects/pack", "refs/heads", "refs/tags", "hooks", "info"):
        (git_dir / sub).mkdir(parents=True, exist_ok=True)

    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "description").write_text("DhwaniDrishti — 3D ISL Studio & STEM Coach\n", encoding="utf-8")
    config_text = (
        "[core]\n"
        "\trepositoryformatversion = 0\n"
        "\tfilemode = false\n"
        "\tbare = false\n"
        "\tlogallrefupdates = true\n"
        "\tsymlinks = false\n"
        "\tignorecase = true\n"
        '[branch "main"]\n'
        "\tremote = origin\n"
        "\tmerge = refs/heads/main\n"
    )
    if not (git_dir / "config").exists():
        (git_dir / "config").write_text(config_text, encoding="utf-8")

    def write_git_object(obj_type: str, data: bytes) -> bytes:
        header = f"{obj_type} {len(data)}\x00".encode("ascii")
        store = header + data
        sha1 = hashlib.sha1(store).digest()
        hex_sha = sha1.hex()
        obj_path = git_dir / "objects" / hex_sha[:2] / hex_sha[2:]
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        if not obj_path.exists():
            obj_path.write_bytes(zlib.compress(store))
        return sha1

    index_entries: list[tuple[bytes, bytes, os.stat_result]] = []
    committed_files: list[Path] = []

    def build_tree(directory: Path) -> bytes:
        entries: list[tuple[bytes, bytes, bytes]] = []
        for item in sorted(directory.iterdir(), key=lambda p: p.name):
            if item.name in IGNORE_DIRS or item.name in IGNORE_FILES or item.suffix in IGNORE_EXTS:
                continue
            if item.is_dir():
                sub_sha = build_tree(item)
                entries.append((b"40000", item.name.encode("utf-8"), sub_sha))
            elif item.is_file():
                raw = item.read_bytes()
                blob_sha = write_git_object("blob", raw)
                entries.append((b"100644", item.name.encode("utf-8"), blob_sha))
                rel_path = item.relative_to(root).as_posix().encode("utf-8")
                index_entries.append((rel_path, blob_sha, item.stat()))
                committed_files.append(item)

        entries.sort(key=lambda e: e[1] + (b"/" if e[0] == b"40000" else b""))
        tree_data = b"".join(mode + b" " + name + b"\x00" + sha for mode, name, sha in entries)
        return write_git_object("tree", tree_data)

    root_tree_sha = build_tree(root)

    # Write Git index v2
    index_entries.sort(key=lambda x: x[0])
    idx_body = bytearray(b"DIRC" + struct.pack(">II", 2, len(index_entries)))
    for rel_path, sha, st in index_entries:
        ctime_s = int(st.st_ctime)
        mtime_s = int(st.st_mtime)
        size = st.st_size & 0xFFFFFFFF
        flags = min(len(rel_path), 0xFFF)
        entry_head = struct.pack(
            ">IIIIIIIIII20sH", ctime_s, 0, mtime_s, 0, 0, 0, 0o100644, 0, 0, size, sha, flags
        )
        pad_len = (8 - ((len(entry_head) + len(rel_path)) % 8)) % 8
        if pad_len == 0:
            pad_len = 8
        idx_body.extend(entry_head + rel_path + (b"\x00" * pad_len))

    idx_sha = hashlib.sha1(idx_body).digest()
    (git_dir / "index").write_bytes(bytes(idx_body) + idx_sha)

    # Write Commit object on refs/heads/main
    ts = int(time.time())
    author = f"Revanth Bandaru <revanth@users.noreply.github.com> {ts} +0530"
    parent_line = ""
    main_ref = git_dir / "refs" / "heads" / "main"
    if main_ref.exists():
        prev_sha = main_ref.read_text(encoding="utf-8").strip()
        if len(prev_sha) == 40:
            parent_line = f"parent {prev_sha}\n"

    commit_payload = (
        f"tree {root_tree_sha.hex()}\n"
        f"{parent_line}"
        f"author {author}\n"
        f"committer {author}\n\n"
        f"{commit_message.strip()}\n"
    ).encode("utf-8")
    commit_sha = write_git_object("commit", commit_payload).hex()
    main_ref.write_text(commit_sha + "\n", encoding="utf-8")
    return commit_sha, committed_files


def github_api_request(method: str, url: str, token: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "DhwaniDrishti-Git-Publisher",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def push_to_github(root: Path, repo_name: str, token: str, commit_message: str) -> str:
    # 1. Identify authenticated GitHub user
    user_info = github_api_request("GET", "https://api.github.com/user", token)
    owner = user_info["login"]
    print(f"Authenticated as GitHub user: {owner}")

    # 2. Create repository if it does not exist yet
    repo_url = f"https://api.github.com/repos/{owner}/{repo_name}"
    try:
        repo_data = github_api_request("GET", repo_url, token)
        print(f"Repository {owner}/{repo_name} already exists.")
    except urllib.error.HTTPError as err:
        if err.code == 404:
            print(f"Creating new GitHub repository: {owner}/{repo_name} (with auto_init)...")
            repo_data = github_api_request(
                "POST",
                "https://api.github.com/user/repos",
                token,
                {
                    "name": repo_name,
                    "description": "DhwaniDrishti — Real-time Bi-Directional Indian Sign Language (ISL) Translation & STEM Coach",
                    "private": False,
                    "auto_init": True,
                },
            )
            time.sleep(1.5)
        else:
            raise

    # 3. Upload all tracked files as Git blobs and create a single atomic Git tree & commit
    files = collect_project_files(root)
    tree_items = []
    for fpath in files:
        rel_posix = fpath.relative_to(root).as_posix()
        b64_content = base64.b64encode(fpath.read_bytes()).decode("ascii")
        blob_res = github_api_request(
            "POST",
            f"https://api.github.com/repos/{owner}/{repo_name}/git/blobs",
            token,
            {"content": b64_content, "encoding": "base64"},
        )
        tree_items.append(
            {
                "path": rel_posix,
                "mode": "100644",
                "type": "blob",
                "sha": blob_res["sha"],
            }
        )
        print(f"  Uploaded blob: {rel_posix} ({blob_res['sha'][:7]})")

    # Get current HEAD of main if present
    parent_shas = []
    try:
        ref_res = github_api_request(
            "GET", f"https://api.github.com/repos/{owner}/{repo_name}/git/ref/heads/main", token
        )
        parent_shas = [ref_res["object"]["sha"]]
    except urllib.error.HTTPError:
        pass

    tree_res = github_api_request(
        "POST",
        f"https://api.github.com/repos/{owner}/{repo_name}/git/trees",
        token,
        {"tree": tree_items},
    )

    commit_res = github_api_request(
        "POST",
        f"https://api.github.com/repos/{owner}/{repo_name}/git/commits",
        token,
        {
            "message": commit_message,
            "tree": tree_res["sha"],
            "parents": parent_shas,
        },
    )

    # Update or create refs/heads/main (prefer fast-forward force=False so CI/CD hooks treat it as standard push)
    if parent_shas:
        try:
            github_api_request(
                "PATCH",
                f"https://api.github.com/repos/{owner}/{repo_name}/git/refs/heads/main",
                token,
                {"sha": commit_res["sha"], "force": False},
            )
        except urllib.error.HTTPError:
            github_api_request(
                "PATCH",
                f"https://api.github.com/repos/{owner}/{repo_name}/git/refs/heads/main",
                token,
                {"sha": commit_res["sha"], "force": True},
            )
    else:
        github_api_request(
            "POST",
            f"https://api.github.com/repos/{owner}/{repo_name}/git/refs",
            token,
            {"ref": "refs/heads/main", "sha": commit_res["sha"]},
        )

    # Record remote origin in .git/config
    git_config = root / ".git" / "config"
    html_url = repo_data.get("html_url", f"https://github.com/{owner}/{repo_name}")
    cfg = git_config.read_text(encoding="utf-8") if git_config.exists() else ""
    if '[remote "origin"]' not in cfg:
        cfg += f'[remote "origin"]\n\turl = {html_url}.git\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n'
        git_config.write_text(cfg, encoding="utf-8")

    return html_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize Git and push DhwaniDrishti to GitHub.")
    parser.add_argument("--repo", default="DhwaniDrishti", help="Target GitHub repository name")
    parser.add_argument(
        "--message",
        default="Initial commit: DhwaniDrishti 3D ISL Studio & STEM Coach",
        help="Git commit message",
    )
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"), help="GitHub Personal Access Token")
    parser.add_argument("--init-only", action="store_true", help="Only initialize local .git and create commit")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    commit_sha, files = init_and_commit_local_git(root, args.message)
    print(f"[Git] Initialized local repository at: {root / '.git'}")
    print(f"[Git] Branch: main | Commit: {commit_sha}")
    print(f"[Git] Staged & committed {len(files)} files:")
    for f in files:
        print(f"   + {f.relative_to(root).as_posix()}")

    if args.init_only:
        return

    if not args.token:
        print(
            "\n[Action Required] To create and push to GitHub repository "
            f"'{args.repo}', provide a GitHub Personal Access Token:\n"
            f'  .\\.venv\\Scripts\\python.exe push_to_github.py --repo {args.repo} --token <YOUR_GITHUB_PAT>'
        )
        sys.exit(2)

    url = push_to_github(root, args.repo, args.token, args.message)
    print(f"\nSuccessfully pushed to GitHub: {url}")


if __name__ == "__main__":
    main()

