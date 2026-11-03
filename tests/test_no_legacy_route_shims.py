"""The flat routes/<name>.py compatibility shims are gone; every caller
imports the domain package module (routes.email.email_routes, ...).

A new import of a legacy path would fail at runtime only on the code path
that reaches it, so pin it here for the whole tree."""
import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = {
    "admin_wipe_routes", "cleanup_routes", "compare_routes", "contacts_routes",
    "document_helpers", "document_routes", "email_helpers", "email_pollers",
    "email_routes", "gallery_helpers", "gallery_routes", "history_routes",
    "mcp_routes", "memory_routes", "note_routes", "research_routes",
    "search_routes", "task_routes", "vault_routes", "webhook_routes",
}


def _python_files():
    out = subprocess.check_output(["git", "ls-files", "*.py"], cwd=ROOT, text=True)
    return [ROOT / f for f in out.split() if (ROOT / f).is_file()]


def test_legacy_shim_files_are_gone():
    left = sorted(n for n in LEGACY if (ROOT / "routes" / f"{n}.py").exists())
    assert not left


def test_nothing_imports_a_legacy_route_path():
    offenders = []
    for path in _python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module == "routes":
                mods = [f"routes.{a.name}" for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            else:
                continue
            for mod in mods:
                parts = mod.split(".")
                if len(parts) >= 2 and parts[0] == "routes" and parts[1] in LEGACY:
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} {mod}")
    assert not offenders, "\n".join(offenders)
