"""Test-only candidate optimizations; never imported by production runners."""

import ast
import inspect
import textwrap


def headless_infer(self, obs, instruction, *, env_id=0):
    if self._needs_refresh(env_id):
        extracted = self._extract_observation(obs, env_id=env_id)
        request = self._pack_request(extracted, instruction)
        response = self._query_server(request)
        chunk = self._postprocess_chunk(self._unpack_response(response))
        self._set_chunk(env_id, chunk)
    return {"action": self._next_action(env_id), "viz": None}


def without_unused_history(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))

    class RemoveHistoryCopies(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(target, ast.Name) and target.id == "per_env_infos"
                   for target in node.targets):
                return None
            return self.generic_visit(node)

        def visit_Expr(self, node):
            value = node.value
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Name)
                    and value.func.value.id == "subtask_status" and value.func.attr == "append"):
                return None
            return self.generic_visit(node)

    tree = ast.fix_missing_locations(RemoveHistoryCopies().visit(tree))
    namespace = dict(function.__globals__)
    exec(compile(tree, "<test-only-no-history>", "exec"), namespace)
    return namespace[function.__name__]
