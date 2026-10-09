"""LightGBM and torch must never share a process (they deadlock). Checked statically."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "gkx"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_module_imports_both_lightgbm_and_torch():
    for path in SRC.glob("*.py"):
        assert not {"lightgbm", "torch"} <= _imports(path), path.name


def test_nn_module_does_not_import_the_tree_module_and_vice_versa():
    assert "gkx.trees" not in (SRC / "nn.py").read_text()
    assert "torch" not in _imports(SRC / "trees.py")
