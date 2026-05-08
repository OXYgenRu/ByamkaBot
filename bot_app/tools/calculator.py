from __future__ import annotations

import ast
import operator as op
from typing import Any

from bot_app.tools.base import ToolRunContext, ToolRunResult, ToolSpec

_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


class SafeCalculatorTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="calculator",
            description="Вычисляет простые арифметические выражения без доступа к Python runtime.",
            input_schema={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            side_effect=False,
        )

    async def run(self, args: dict[str, Any], ctx: ToolRunContext) -> ToolRunResult:
        expression = str(args.get("expression", ""))[:500]
        try:
            value = _eval_expr(expression)
            return ToolRunResult(ok=True, result=f"{expression} = {value}")
        except Exception as exc:
            return ToolRunResult(ok=False, result="", error=str(exc))


def _eval_expr(expr: str) -> float | int:
    node = ast.parse(expr, mode="eval")
    return _eval_node(node.body)


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(float(right)) > 10:
            raise ValueError("power is too large")
        return _ALLOWED_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")
