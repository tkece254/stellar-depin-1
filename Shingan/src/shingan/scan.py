from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass

ENTRY_POINT = "run"
GENERATOR_ENTRY_POINT = "generate"

ALLOWED_MODULES = frozenset({"torch", "triton", "math", "functools", "typing", "__future__"})
GENERATOR_MODULES = frozenset({"torch", "numpy", "math", "__future__"})

ALLOWED_TORCH_CALLS = frozenset(
    {
        "torch.empty",
        "torch.empty_like",
        "torch.empty_strided",
        "torch.zeros",
        "torch.zeros_like",
        "torch.device",
        "torch.cuda.current_device",
        "torch.cuda.get_device_properties",
    }
)

FORBIDDEN_GENERATOR_CALLS = (
    "torch.load",
    "torch.save",
    "torch.hub",
    "torch.cuda",
    "torch.distributed",
    "torch.multiprocessing",
    "torch.utils",
    "torch.ops",
    "torch.classes",
    "torch.backends",
    "torch.compile",
    "numpy.load",
    "numpy.save",
    "numpy.savez",
    "numpy.savetxt",
    "numpy.loadtxt",
    "numpy.genfromtxt",
    "numpy.fromfile",
    "numpy.memmap",
    "numpy.ctypeslib",
    "numpy.lib",
)

FORBIDDEN_GENERATOR_METHODS = frozenset({"tofile", "dump", "dumps", "cuda", "pin_memory", "share_memory_"})

ALLOWED_HOST_METHODS = frozenset(
    {
        "contiguous",
        "view",
        "reshape",
        "stride",
        "size",
        "numel",
        "dim",
        "data_ptr",
        "is_contiguous",
        "element_size",
        "flatten",
        "unsqueeze",
        "squeeze",
        "t",
        "transpose",
        "permute",
        "expand",
    }
)

BANNED_BUILTINS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
        "input",
    }
)

FRAME_ATTRIBUTES = frozenset({"f_back", "f_locals", "f_globals", "gi_frame", "tb_frame"})

REVIEWER_TEXT = re.compile(
    r"ignore (all |any )?(previous|prior|above)"
    r"|disregard"
    r"|reviewer|grader"
    r"|as an ai"
    r"|this (kernel|code) is (correct|verified|safe)"
    r"|(mark|treat) (this|it) as (correct|passing|genuine)"
    r"|do not (test|check|verify)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    line: int
    rule: str
    message: str


def scan(source: str) -> list[Finding]:
    tree, findings = _parse(source)
    if tree is None:
        return findings
    scanner = _KernelScanner()
    scanner.visit(tree)
    findings = scanner.findings + _reviewer_text(source)
    if not scanner.jit_functions:
        findings.append(Finding(0, "no-triton", "no @triton.jit kernel defined"))
    if ENTRY_POINT not in scanner.top_level_functions:
        findings.append(Finding(0, "no-entry", f"no top-level {ENTRY_POINT}() function"))
    return sorted(findings, key=lambda f: f.line)


def scan_generator(source: str) -> list[Finding]:
    tree, findings = _parse(source)
    if tree is None:
        return findings
    scanner = _GeneratorScanner()
    scanner.visit(tree)
    findings = scanner.findings
    if GENERATOR_ENTRY_POINT not in scanner.top_level_functions:
        findings.append(Finding(0, "no-entry", f"no top-level {GENERATOR_ENTRY_POINT}() function"))
    return sorted(findings, key=lambda f: f.line)


def _parse(source: str) -> tuple[ast.Module | None, list[Finding]]:
    try:
        return ast.parse(source), []
    except SyntaxError as exc:
        return None, [Finding(exc.lineno or 0, "syntax", exc.msg)]


class _RestrictedScanner(ast.NodeVisitor):
    allowed_modules: frozenset[str] = frozenset()

    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.aliases: dict[str, str] = {}
        self.top_level_functions: set[str] = set()
        self._depth = 0

    def flag(self, node: ast.AST, rule: str, message: str) -> None:
        self.findings.append(Finding(getattr(node, "lineno", 0), rule, message))

    def resolve(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self.resolve(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in self.allowed_modules:
                self.flag(node, "import", f"import of {alias.name} is not allowed")
            if alias.asname:
                self.aliases[alias.asname] = alias.name
            else:
                self.aliases[root] = root

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level or module.split(".")[0] not in self.allowed_modules:
            self.flag(node, "import", f"import from {'.' * node.level}{module} is not allowed")
        for alias in node.names:
            self.aliases[alias.asname or alias.name] = f"{module}.{alias.name}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._depth == 0:
            self.top_level_functions.add(node.name)
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.flag(node, "async", "async functions are not allowed")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in BANNED_BUILTINS:
            self.flag(node, "builtin", f"call to {node.func.id}() is not allowed")
        else:
            self.check_call(node, self.resolve(node.func))
        self.generic_visit(node)

    def check_call(self, node: ast.Call, name: str | None) -> None:
        pass

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if _is_dunder(node.attr) or node.attr in FRAME_ATTRIBUTES:
            self.flag(node, "introspection", f"access to .{node.attr} is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if _is_dunder(node.id) and node.id != "__name__":
            self.flag(node, "introspection", f"access to {node.id} is not allowed")


class _KernelScanner(_RestrictedScanner):
    allowed_modules = ALLOWED_MODULES

    def __init__(self) -> None:
        super().__init__()
        self.jit_functions: set[str] = set()
        self._in_jit = False
        self._entry_params: frozenset[str] = frozenset()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        is_jit = any(self.resolve(_unwrap_call(d)) == "triton.jit" for d in node.decorator_list)
        top_level = self._depth == 0
        if top_level and is_jit:
            self.jit_functions.add(node.name)
        outer_jit, outer_params = self._in_jit, self._entry_params
        self._in_jit = self._in_jit or is_jit
        if top_level and node.name == ENTRY_POINT and not is_jit:
            self._entry_params = frozenset(a.arg for a in node.args.args)
        if top_level and not is_jit:
            self.top_level_functions.add(node.name)
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1
        self._in_jit, self._entry_params = outer_jit, outer_params

    def check_call(self, node: ast.Call, name: str | None) -> None:
        if name is not None and name.startswith("torch.") and name not in ALLOWED_TORCH_CALLS:
            self.flag(node, "torch-call", f"call to {name} computes outside the Triton kernel")
        elif name is None and isinstance(node.func, ast.Attribute) and not self._in_jit:
            if node.func.attr not in ALLOWED_HOST_METHODS:
                self.flag(node, "host-method", f"host-side .{node.func.attr}() is not allowed")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not self._in_jit:
            if isinstance(node.op, ast.MatMult):
                self.flag(node, "host-compute", "matrix multiply outside the Triton kernel")
            elif self._is_entry_tensor(node.left) or self._is_entry_tensor(node.right):
                self.flag(node, "host-compute", "arithmetic on input tensors outside the Triton kernel")
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not self._in_jit and self._is_entry_tensor(node.operand):
            self.flag(node, "host-compute", "arithmetic on input tensors outside the Triton kernel")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if not self._in_jit and self._is_entry_tensor(node.target):
            self.flag(node, "host-compute", "in-place arithmetic on input tensors outside the Triton kernel")
        self.generic_visit(node)

    def _is_entry_tensor(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id in self._entry_params


class _GeneratorScanner(_RestrictedScanner):
    allowed_modules = GENERATOR_MODULES

    def check_call(self, node: ast.Call, name: str | None) -> None:
        if name is not None and any(name == p or name.startswith(f"{p}.") for p in FORBIDDEN_GENERATOR_CALLS):
            self.flag(node, "forbidden-call", f"call to {name} is not allowed in a generator")
        elif isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_GENERATOR_METHODS:
            self.flag(node, "forbidden-call", f".{node.func.attr}() is not allowed in a generator")


def _unwrap_call(node: ast.expr) -> ast.expr:
    return node.func if isinstance(node, ast.Call) else node


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _reviewer_text(source: str) -> list[Finding]:
    findings = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING) and REVIEWER_TEXT.search(token.string):
            findings.append(Finding(token.start[0], "reviewer-text", "text addressed to a reviewer"))
    return findings
