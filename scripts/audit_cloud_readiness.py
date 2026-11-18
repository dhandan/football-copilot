"""Read-only local Git security baseline for issue #9.

Reports only locations and types, never detected secret values.
Heuristic tool: not a replacement for GitHub secret scanning.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "private-key-header": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github-token": re.compile(r"\b(?:ghp_|gho_|ghu_|ghs_|ghr_)[A-Za-z0-9_]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "api-token-assignment": re.compile(
        r"""(?i)\b(?:FOOTBALL_DATA_API_KEY|ODDS_API_KEY|OPENAI_API_KEY|GROQ_API_KEY|GEMINI_API_KEY|GOOGLE_API_KEY|AWS_SECRET_ACCESS_KEY)\b\s*[:=]\s*["']?([^\s"'#,;]+)"""
    ),
}
EXAMPLE_VALUES = ("your_", "example", "placeholder", "replace_", "changeme", "dummy", "test_", "none", "null", "<", "os.getenv", "st.secrets")
SUSPECT_BASENAMES = {".env", "secrets.toml", "credentials.json", "service-account.json"}
BINARY_SUFFIXES = {".duckdb", ".db", ".pkl", ".parquet", ".png", ".jpg", ".jpeg", ".webp", ".pdf", ".ico", ".zip", ".pyc"}
MAX_TEXT_FILE = 2_000_000


def classify(line: str):
    """Return finding types only, never the matched value."""
    found = []
    for name, pattern in PATTERNS.items():
        for match in pattern.finditer(line):
            if name == "api-token-assignment" and match.group(1).strip().lower().startswith(EXAMPLE_VALUES):
                continue
            found.append(name)
    return sorted(set(found))


def inspect_tracked(root: Path):
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True).stdout
    paths = [Path(p.decode("utf-8", errors="replace")) for p in out.split(b"\x00") if p]
    findings, skipped = [], []
    for relative in paths:
        name = relative.name.lower()
        path = root / relative
        if name in SUSPECT_BASENAMES or (name.startswith(".env.") and name != ".env.example"):
            findings.append((str(relative), "-", "tracked-sensitive-filename"))
        if not path.is_file() or path.suffix.lower() in BINARY_SUFFIXES or path.stat().st_size > MAX_TEXT_FILE:
            skipped.append(str(relative))
            continue
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                for lineno, line in enumerate(handle, 1):
                    for kind in classify(line):
                        findings.append((str(relative), str(lineno), kind))
        except UnicodeError:
            skipped.append(str(relative))
    return paths, findings, skipped


def inspect_history(root: Path):
    """Scan added diff lines in all locally available Git refs."""
    command = ["git", "log", "--all", "--format=commit:%h", "--patch", "--no-color", "--no-ext-diff", "--"]
    proc = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True,
                            encoding="utf-8", errors="replace")
    commit, path, findings = "unknown", "unknown", []
    try:
        for line in proc.stdout:
            if line.startswith("commit:"):
                commit = line.strip().split(":", 1)[1]
            elif line.startswith("+++ b/"):
                path = line[6:].rstrip("\n")
            elif line.startswith("+") and not line.startswith("+++"):
                for kind in classify(line[1:]):
                    findings.append((commit, path, kind))
    finally:
        if proc.stdout:
            proc.stdout.close()
    if proc.wait() != 0:
        raise RuntimeError("Could not inspect Git history")
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-history", action="store_true",
                        help="Fast current-files check only, not a full audit")
    args = parser.parse_args()
    try:
        paths, current, skipped = inspect_tracked(ROOT)
        print("ISSUE #9 | CLOUD SECURITY READINESS (READ ONLY)")
        print(f"Tracked files: {len(paths)}")
        print(f"Current suspicious findings: {len(current)}")
        for path, line, kind in current[:30]:
            print(f"  {path}:{line} [{kind}]")
        if len(current) > 30:
            print(f"  ... {len(current)-30} more (suppressed)")
        print(f"Binary/large/unreadable files not content-scanned: {len(skipped)}")
        print("  " + ", ".join(skipped[:8]) if skipped else "  none")
        if args.no_history:
            print("History scan SKIPPED: audit incomplete")
            return 1 if current else 0
        shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"],
                                cwd=ROOT, check=True, capture_output=True,
                                text=True).stdout.strip() == "true"
        past = inspect_history(ROOT)
        print(f"Historical suspicious added lines: {len(past)}")
        for commit, path, kind in past[:30]:
            print(f"  {commit} {path} [{kind}]")
        if len(past) > 30:
            print(f"  ... {len(past)-30} more (suppressed)")
        if shallow:
            print("WARNING: shallow clone; incomplete history coverage")
        print("Review locations privately. Revoke and rotate any genuine leaked credentials.")
        print("Heuristics do not certify that a repository or deployed app is secure.")
        return 1 if current or past or shallow else 0
    except (OSError, subprocess.CalledProcessError, RuntimeError) as err:
        print(f"Audit incomplete: {type(err).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
