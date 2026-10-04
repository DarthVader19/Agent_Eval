from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path


_CHILD = (
    "import ast,sys\n"
    "class _OutputLimit:\n"
    " def __init__(self,stream): self.stream=stream; self.remaining=10000\n"
    " def write(self,text):\n"
    "  part=text[:self.remaining]; self.remaining-=len(part); return self.stream.write(part)\n"
    " def flush(self): self.stream.flush()\n"
    "sys.stdout=_OutputLimit(sys.stdout)\n"
    "def _bounded_range(*args):\n"
    " r=range(*args)\n"
    " if len(r)>10000: raise ValueError('range exceeds 10000 items')\n"
    " return r\n"
    "tree=ast.parse(sys.stdin.read(),mode='exec')\n"
    "allowed=(ast.Module,ast.Expr,ast.Assign,ast.AnnAssign,ast.Name,ast.Load,ast.Store,ast.Constant,"
    "ast.BinOp,ast.UnaryOp,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod,ast.Pow,ast.USub,ast.UAdd,"
    "ast.Call,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.For,ast.While,ast.If,ast.Compare,ast.Eq,"
    "ast.NotEq,ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.List,ast.Tuple,ast.Dict,ast.Subscript,ast.Slice,ast.Load,"
    "ast.Store,ast.keyword,ast.Break,ast.Continue,ast.Pass)\n"
    "for n in ast.walk(tree):\n"
    " if not isinstance(n,allowed): raise ValueError('Unsupported Python syntax: '+type(n).__name__)\n"
    " if isinstance(n,ast.Name) and n.id.startswith('__'): raise ValueError('Dunder names are forbidden')\n"
    "safe={'abs':abs,'min':min,'max':max,'sum':sum,'len':len,'range':_bounded_range,'round':round,'int':int,'float':float,'str':str,'list':list,'print':print}\n"
    "exec(compile(tree,'<sandbox>','exec'),{'__builtins__':safe},{})\n"
)

_ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AnnAssign, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Call, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.For,
    ast.While, ast.If, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.List,
    ast.Tuple, ast.Dict, ast.Subscript, ast.Slice, ast.keyword, ast.Break, ast.Continue, ast.Pass,
)


def run_python(code: str, timeout: float = 3.0, max_output: int = 10000) -> str:
    """Run a restricted Python subset in a short-lived isolated interpreter."""
    if len(code) > 50000:
        raise ValueError("Python code exceeds the 50,000 character limit.")
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Unsupported Python syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Dunder names are forbidden.")
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Pow)
            and isinstance(node.right, ast.Constant)
            and isinstance(node.right.value, (int, float))
            and abs(node.right.value) > 100
        ):
            raise ValueError("Exponents are limited to an absolute value of 100.")
    with tempfile.TemporaryDirectory(prefix="agent-eval-") as scratch:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", _CHILD],
            input=code,
            text=True,
            capture_output=True,
            cwd=Path(scratch),
            timeout=timeout,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError((completed.stderr or "Sandbox execution failed.").strip()[:max_output])
    return completed.stdout[-max_output:].strip() or completed.stderr[-max_output:].strip()
