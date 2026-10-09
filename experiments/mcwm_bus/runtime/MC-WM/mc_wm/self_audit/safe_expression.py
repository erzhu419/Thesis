"""AST-validated mathematical expressions for LLM-proposed features/rules."""

from __future__ import annotations

import ast
from types import CodeType
from typing import Any, Iterable

import numpy as np


class UnsafeExpressionError(ValueError):
    pass


_NUMPY_CALLS = {
    "abs", "arctan", "arctan2", "clip", "cos", "cosh", "exp", "expm1",
    "isfinite", "log", "log1p", "maximum", "mean", "minimum", "power",
    "prod", "sign", "sin", "sinh", "sqrt", "square", "sum", "tan",
    "tanh", "where",
}
_NUMPY_CONSTANTS = {"e", "pi"}
_BUILTIN_CALLS = {
    "abs": abs, "all": all, "any": any, "len": len,
    "max": max, "min": min, "range": range,
}
_ALLOWED_NODE_TYPES = {
    ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Attribute,
    ast.Subscript, ast.Slice, ast.Tuple, ast.List,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Call, ast.keyword,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub, ast.Not, ast.Invert,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.BitAnd, ast.BitOr,
}


class _Validator(ast.NodeVisitor):
    def __init__(self, variables: Iterable[str]):
        self.variables = set(variables)

    def generic_visit(self, node):
        if type(node) not in _ALLOWED_NODE_TYPES:
            raise UnsafeExpressionError(
                f"expression node {type(node).__name__} is not allowed")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name):
        allowed = self.variables | {"np", "True", "False", "None"} \
                  | set(_BUILTIN_CALLS)
        if node.id not in allowed:
            raise UnsafeExpressionError(f"name {node.id!r} is not allowed")

    def visit_Attribute(self, node: ast.Attribute):
        if not isinstance(node.value, ast.Name) or node.value.id != "np":
            raise UnsafeExpressionError("only direct np.<math> attributes are allowed")
        if node.attr not in _NUMPY_CALLS | _NUMPY_CONSTANTS:
            raise UnsafeExpressionError(f"NumPy attribute {node.attr!r} is not allowed")

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id not in _BUILTIN_CALLS:
                raise UnsafeExpressionError(f"call {node.func.id!r} is not allowed")
        elif isinstance(node.func, ast.Attribute):
            self.visit_Attribute(node.func)
            if node.func.attr not in _NUMPY_CALLS:
                raise UnsafeExpressionError(
                    f"NumPy constant {node.func.attr!r} is not callable")
        else:
            raise UnsafeExpressionError("indirect calls are not allowed")
        for keyword in node.keywords:
            if keyword.arg is None:
                raise UnsafeExpressionError("expanded keyword arguments are not allowed")
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)


def compile_safe_expression(expr: str, variables: Iterable[str]) -> CodeType:
    if not isinstance(expr, str) or not expr.strip():
        raise UnsafeExpressionError("expression must be a non-empty string")
    if len(expr) > 2_000:
        raise UnsafeExpressionError("expression is too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(str(exc)) from exc
    _Validator(variables).visit(tree)
    return compile(tree, "<safe_math_expression>", "eval")


def evaluate_safe_expression(code: CodeType, variables: dict[str, Any]) -> Any:
    globals_dict = {"__builtins__": {}, "np": np, **_BUILTIN_CALLS}
    return eval(code, globals_dict, variables)


def evaluate_array_expression(expr: str, s: np.ndarray, a: np.ndarray,
                              transpose: bool = False) -> np.ndarray | None:
    try:
        code = compile_safe_expression(expr, ("s", "a"))
        local_s = s.T if transpose else s
        local_a = a.T if transpose else a
        out = evaluate_safe_expression(code, {"s": local_s, "a": local_a})
        arr = np.asarray(out, dtype=np.float64)
    except Exception:
        return None
    if arr.shape != (len(s),) or not np.isfinite(arr).all():
        return None
    return arr
