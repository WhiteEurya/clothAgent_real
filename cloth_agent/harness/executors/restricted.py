"""A bounded pure Python subset for candidate prepare functions.

Never imports or execs candidate modules. The AST interpreter grants only JSON
values and numeric primitives, with a step/size budget on every expression.
"""
from __future__ import annotations

import ast
import math
import operator

from ..policy import PolicyError

BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
           ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b}
FUNCTIONS = {'min': min, 'max': max, 'abs': abs, 'round': round, 'int': int,
             'float': float, 'len': len, 'floor': math.floor, 'ceil': math.ceil}
ALLOWED = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.If,
           ast.Expr, ast.Constant, ast.Name, ast.Load, ast.Store, ast.Dict, ast.List, ast.Tuple,
           ast.Subscript, ast.BinOp, ast.UnaryOp, ast.USub, ast.UAdd, ast.Not,
           ast.BoolOp, ast.And, ast.Or, ast.Compare, ast.IfExp, ast.Call,
           *BINARY, *COMPARE)


def checked(value, depth=0, remaining=None):
    # Bound total traversal too: small aliased lists can expand exponentially
    # when recursively validated or serialized, despite per-container limits.
    if remaining is None:
        remaining = [4096]
    remaining[0] -= 1
    if remaining[0] < 0:
        raise PolicyError('Candidate aggregate value budget exceeded')
    if depth > 16:
        raise PolicyError('Candidate value nesting budget exceeded')
    if value is None or type(value) is bool:
        return value
    if type(value) in (int, float):
        if abs(value) > 1e9 or not math.isfinite(value):
            raise PolicyError('Candidate numeric budget exceeded')
    elif isinstance(value, str):
        if len(value) > 4096:
            raise PolicyError('Candidate string budget exceeded')
    elif isinstance(value, (list, tuple, dict)):
        if len(value) > 128:
            raise PolicyError('Candidate collection budget exceeded')
        for item in (list(value.keys()) + list(value.values()) if isinstance(value, dict) else value):
            checked(item, depth+1, remaining)
    else:
        raise PolicyError('Candidate values must be JSON data')
    return value


class RestrictedProgram:
    def __init__(self, source, *, function_name='prepare', parameters=('request', 'source', 'available'), allow_loops=False):
        if not isinstance(source, str) or len(source) > 16000:
            raise PolicyError('Candidate source size exceeded')
        self.source = source
        try:
            self.tree = ast.parse(source)
            compile(self.tree, '<candidate observation skill>', 'exec')  # syntax only, never executed
        except (SyntaxError, ValueError, RecursionError) as exc:
            raise PolicyError(f'Candidate syntax invalid: {exc}') from exc
        nodes = list(ast.walk(self.tree))
        allowed = ALLOWED + ((ast.For,) if allow_loops else ())
        if len(nodes) > 1200 or any(not isinstance(n, allowed) for n in nodes):
            raise PolicyError('Unsupported Python: imports, attributes, loops and arbitrary code are forbidden')
        if len(self.tree.body) != 1 or not isinstance(self.tree.body[0], ast.FunctionDef):
            raise PolicyError(f'Only def {function_name}{parameters} is allowed')
        function = self.tree.body[0]
        args = function.args
        if (function.name != function_name or function.decorator_list or function.returns
                or [a.arg for a in args.args] != list(parameters)
                or args.defaults or args.kwonlyargs or args.posonlyargs or args.vararg or args.kwarg):
            raise PolicyError('Invalid pure function signature')
        self.parameters = parameters
        for node in nodes:
            if isinstance(node, ast.FunctionDef) and node is not function:
                raise PolicyError('Nested functions are forbidden')
            if isinstance(node, ast.Name) and node.id.startswith('_'):
                raise PolicyError('Private names forbidden')
            if isinstance(node, ast.arg) and node.annotation:
                raise PolicyError('Annotations forbidden')
            if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or node.keywords):
                raise PolicyError('Only numeric primitive calls are allowed')
            if isinstance(node, ast.Assign) and (len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name)):
                raise PolicyError('Only local variable assignment is allowed')
            if isinstance(node, ast.For) and (not isinstance(node.target, ast.Name) or node.orelse):
                raise PolicyError('For requires a local name and no else clause')
            if isinstance(node, ast.Constant):
                checked(node.value)
        self.body = function.body

    def run(self, *arguments):
        if len(arguments) != len(self.parameters):
            raise PolicyError('Incorrect pure function argument count')
        values = checked(dict(zip(self.parameters, arguments)))
        self.remaining = 3000
        def evaluate(node):
            self.remaining -= 1
            if self.remaining < 0:
                raise PolicyError('Candidate instruction budget exhausted')
            if isinstance(node, ast.Constant): value = node.value
            elif isinstance(node, ast.Name): value = values[node.id]
            elif isinstance(node, (ast.List, ast.Tuple)): value = [evaluate(n) for n in node.elts]
            elif isinstance(node, ast.Dict): value = {evaluate(k): evaluate(v) for k, v in zip(node.keys, node.values)}
            elif isinstance(node, ast.Subscript): value = evaluate(node.value)[evaluate(node.slice)]
            elif isinstance(node, ast.BinOp):
                a, b = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Add) and type(a) is list and type(b) is list:
                    if len(a) + len(b) > 128:
                        raise PolicyError('Candidate collection budget exceeded')
                    value = a + b
                elif type(a) not in (int, float) or type(b) not in (int, float):
                    raise PolicyError('Arithmetic operands must be numeric; no sequence multiplication')
                else:
                    value = BINARY[type(node.op)](a, b)
            elif isinstance(node, ast.UnaryOp):
                a = evaluate(node.operand)
                if isinstance(node.op, ast.Not): value = not a
                elif type(a) in (int, float): value = -a if isinstance(node.op, ast.USub) else +a
                else: raise PolicyError('Unary arithmetic must be numeric')
            elif isinstance(node, ast.IfExp): value = evaluate(node.body if evaluate(node.test) else node.orelse)
            elif isinstance(node, ast.BoolOp):
                value = evaluate(node.values[0])
                for part in node.values[1:]:
                    if (isinstance(node.op, ast.And) and not value) or (isinstance(node.op, ast.Or) and value): break
                    value = evaluate(part)
            elif isinstance(node, ast.Compare):
                a, value = evaluate(node.left), True
                for op, expression in zip(node.ops, node.comparators):
                    b = evaluate(expression)
                    if not COMPARE[type(op)](a, b): value = False; break
                    a = b
            elif isinstance(node, ast.Call): value = FUNCTIONS[node.func.id](*[evaluate(a) for a in node.args])
            else: raise PolicyError('Unsupported expression')
            return checked(value)
        def statements(body):
            for node in body:
                self.remaining -= 1
                if self.remaining < 0: raise PolicyError('Candidate instruction budget exhausted')
                if isinstance(node, ast.Return): return True, evaluate(node.value)
                if isinstance(node, ast.Assign): values[node.targets[0].id] = evaluate(node.value)
                elif isinstance(node, ast.If):
                    done, result = statements(node.body if evaluate(node.test) else node.orelse)
                    if done: return done, result
                elif isinstance(node, ast.For):
                    sequence = evaluate(node.iter)
                    if not isinstance(sequence, (list, tuple)):
                        raise PolicyError('For can iterate only a bounded JSON array')
                    for item in sequence:
                        self.remaining -= 1
                        if self.remaining < 0: raise PolicyError('Candidate instruction budget exhausted')
                        values[node.target.id] = item
                        done, result = statements(node.body)
                        if done: return done, result
                elif isinstance(node, ast.Expr): evaluate(node.value)
            return False, None
        try:
            done, result = statements(self.body)
            if not done: raise PolicyError('Candidate did not return a recipe')
            return checked(result)
        except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise PolicyError(f'Candidate prepare failed: {exc}') from exc
