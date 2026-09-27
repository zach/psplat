"""Check and execute the shell blocks in psplat's documentation.

Every document under ``docs/`` and the README declares its audience on its
first line::

    <!-- psplat-docs: audience=td -->        technical-director facing
    <!-- psplat-docs: audience=dev -->       contributor facing
    <!-- psplat-docs: audience=operator -->  Mac / AWS operator facing

Rules (handoff §5.4, "PowerShell parity"):

- ``td`` and ``dev`` documents give every command in a ```bash block
  immediately followed by the same thing in a ```powershell block;
- ``operator`` documents are bash only;
- ``<!-- doctest: skip -->`` on the line before a bash block marks that pair
  as not executable in CI (a GPU install, an AWS call);
- ``td`` documents are executed: all runnable blocks of one shell are joined
  into one session, run in a fresh directory holding the synthetic fixture
  as ``scene.ply`` and ``cams.json``.

Execution is literal except for one substitution: when ``PSPLAT_DOCS_WHEEL``
is set, the ``psplat`` requirement in ``pip install`` lines is replaced by
that wheel, since the package is tested before it is published.

Usage::

    python scripts/ci/run_doc_blocks.py --check
    python scripts/ci/run_doc_blocks.py --run powershell
    python scripts/ci/run_doc_blocks.py --run bash
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "synthetic"
# Internal documents that are not user instructions and carry no marker.
UNMARKED = {"handoff.md", "plan.md", "lab-notebook.md"}
AUDIENCES = {"td", "dev", "operator"}
MARKER = re.compile(r"<!--\s*psplat-docs:\s*audience=(\w+)\s*-->")
SKIP = re.compile(r"<!--\s*doctest:\s*skip\s*-->")
FENCE = re.compile(r"^```(\w*)\s*$")
PSPLAT_REQ = re.compile(r"""(?<![\w./\\-])(["']?)psplat(\[[^\]]*\])?\1(?![\w./\\-])""")


@dataclass
class Block:
    lang: str
    code: str
    line: int
    skip: bool


@dataclass
class Doc:
    path: Path
    audience: str | None
    blocks: list[Block]


def documents() -> list[Path]:
    return [REPO / "README.md", *sorted((REPO / "docs").glob("*.md"))]


def parse(path: Path) -> Doc:
    lines = path.read_text(encoding="utf-8").splitlines()
    marker = MARKER.search(lines[0]) if lines else None
    audience = marker.group(1) if marker else None
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if m and m.group(1):
            lang, start = m.group(1), i
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            skip = j >= 0 and bool(SKIP.search(lines[j]))
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            blocks.append(Block(lang, "\n".join(body), start + 1, skip))
        elif m:
            # An unlabelled fence: skip to its end so its body is not parsed.
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                i += 1
        i += 1
    return Doc(path, audience, blocks)


def check(doc: Doc) -> list[str]:
    rel = doc.path.relative_to(REPO)
    if doc.audience is None:
        if doc.path.name in UNMARKED:
            return []
        return [f"{rel}: first line must be <!-- psplat-docs: audience=td|dev|operator -->"]
    if doc.audience not in AUDIENCES:
        return [f"{rel}: unknown audience {doc.audience!r}"]
    shell = [
        b for b in doc.blocks if b.lang in ("bash", "powershell", "sh", "shell", "ps1", "pwsh")
    ]
    problems = [
        f"{rel}:{b.line}: use ```bash or ```powershell, not ```{b.lang}"
        for b in shell
        if b.lang not in ("bash", "powershell")
    ]
    if doc.audience == "operator":
        return problems
    k = 0
    while k < len(shell):
        b = shell[k]
        if b.lang == "powershell":
            problems.append(f"{rel}:{b.line}: powershell block without a bash block before it")
            k += 1
            continue
        nxt = shell[k + 1] if k + 1 < len(shell) else None
        if nxt is None or nxt.lang != "powershell":
            problems.append(f"{rel}:{b.line}: bash block needs a powershell block right after it")
            k += 1
            continue
        if nxt.skip and not b.skip:
            problems.append(f"{rel}:{nxt.line}: put doctest: skip before the bash block of a pair")
        nxt.skip = b.skip
        k += 2
    return problems


def substitute(code: str, wheel: str | None) -> str:
    if not wheel:
        return code
    out = []
    for line in code.splitlines():
        if re.search(r"\bpip(3)?(\.exe)?\s+install\b", line):
            line = PSPLAT_REQ.sub(lambda m: f'"{wheel}{m.group(2) or ""}"', line)
        out.append(line)
    return "\n".join(out)


def powershell_script(code: str) -> str:
    """Wrap a PowerShell block so any failing line stops the session.

    Windows PowerShell 5.1 turns a native command's stderr into error records
    when output is redirected, as in CI, so ``$ErrorActionPreference = 'Stop'``
    would fail on harmless warnings. Instead cmdlets are made to stop through
    ``$PSDefaultParameterValues``, and native commands are checked through
    ``$LASTEXITCODE`` after every top-level line.
    """
    head = [
        "$ErrorActionPreference = 'Continue'",
        "$PSDefaultParameterValues['*:ErrorAction'] = 'Stop'",
        'trap { Write-Host "psplat-doctest: failed: $_"; exit 1 }',
    ]
    body: list[str] = []
    depth = 0
    pending: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        pending.append(line)
        depth += stripped.count("{") - stripped.count("}")
        continued = stripped.endswith("`") or stripped.endswith("|") or depth > 0
        if not stripped or stripped.startswith("#") or continued:
            if not stripped or stripped.startswith("#"):
                body.extend(pending)
                pending = []
            continue
        # Escape for a double-quoted PowerShell string: backtick, dollar, quote.
        quoted = stripped.replace("`", "``").replace("$", "`$").replace('"', '`"')
        body.append("$global:LASTEXITCODE = 0")
        body.extend(pending)
        body.append(
            f'if ($LASTEXITCODE -ne 0) {{ Write-Host "psplat-doctest: failed (exit '
            f'$LASTEXITCODE): {quoted}"; exit $LASTEXITCODE }}'
        )
        pending = []
    body.extend(pending)
    return "\n".join([*head, *body]) + "\n"


def run(doc: Doc, shell: str, wheel: str | None) -> bool:
    rel = doc.path.relative_to(REPO)
    blocks = [b for b in doc.blocks if b.lang == shell and not b.skip]
    if not blocks:
        return True
    code = substitute("\n\n".join(b.code for b in blocks), wheel)
    with tempfile.TemporaryDirectory(prefix="psplat-doctest-") as tmp:
        sandbox = Path(tmp)
        for name in ("scene.ply", "cams.json"):
            shutil.copy(FIXTURE / name, sandbox / name)
        if shell == "bash":
            script = sandbox / "doc.sh"
            script.write_text("set -euo pipefail\n" + code + "\n", encoding="utf-8")
            cmd = ["bash", str(script)]
        else:
            script = sandbox / "doc.ps1"
            script.write_text(powershell_script(code), encoding="utf-8-sig")
            exe = os.environ.get("PSPLAT_DOCS_POWERSHELL", "powershell")
            cmd = [exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                   "-File", str(script)]  # fmt: skip
        print(f"--- {rel}: {len(blocks)} {shell} block(s) in {sandbox}", flush=True)
        print(code, flush=True)
        print("---", flush=True)
        result = subprocess.run(cmd, cwd=sandbox, check=False)
    ok = result.returncode == 0
    print(f"=== {rel}: {'ok' if ok else f'FAILED (exit {result.returncode})'}", flush=True)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="check markers and parity only")
    group.add_argument("--run", choices=["bash", "powershell"], help="execute td documents")
    args = parser.parse_args()

    docs = [parse(p) for p in documents()]
    problems = [p for d in docs for p in check(d)]
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    if args.check:
        print(f"{len(docs)} documents checked; markers and parity ok")
        return 0
    wheel = os.environ.get("PSPLAT_DOCS_WHEEL")
    results = [run(d, args.run, wheel) for d in docs if d.audience == "td"]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
