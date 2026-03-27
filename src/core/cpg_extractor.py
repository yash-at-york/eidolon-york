"""
Eidolon - CPG Extractor
Parses Python source code with Tree-sitter and produces a Code Property Graph:
  nodes: functions and classes with intra-function analysis
  edges: CALLS, DATA_FLOW, IMPORTS - cross-function causal relationships

Compatible with tree-sitter >= 0.25.x (uses direct node traversal, not query captures).

Output JSON:
{
  "nodes": [ { "id": "h_7b9x", "type": "Function", ... } ],
  "edges": [ { "from": "h_1a2b", "to": "h_7b9x", "type": "CALLS", "arg_count": 1 } ]
}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from src.core.mapper import GhostMapper



# Data classes


@dataclass
class CPGNode:
    id: str
    type: str                            # "Function" | "Class"
    is_async: bool = False
    decorators: list[str] = field(default_factory=list)
    docstring: str | None = None
    parameters: list[dict] = field(default_factory=list)
    return_type: str = "Any"
    logic_sequence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "is_async": self.is_async,
            "decorators": self.decorators,
            "docstring": self.docstring,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "logic_sequence": self.logic_sequence,
        }


@dataclass
class CPGEdge:
    from_id: str
    to_id: str
    type: str           # "CALLS" | "DATA_FLOW" | "IMPORTS"
    properties: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "from": self.from_id,
            "to": self.to_id,
            "type": self.type,
            **self.properties,
        }


@dataclass
class CPGPayload:
    service: str
    file_path: str
    nodes: list[CPGNode] = field(default_factory=list)
    edges: list[CPGEdge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "service": self.service,
            "file_path": self.file_path,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }



# CPG Extractor


class CPGExtractor:
    """
    Parses Python source with Tree-sitter (0.25.x compatible) and produces CPG.
    Uses direct AST node traversal instead of the deprecated query-capture API.
    """

    def __init__(self, mapper: GhostMapper, service: str = "default-svc") -> None:
        self.mapper = mapper
        self.service = service
        self._lang = Language(tspython.language())
        self._parser = Parser(self._lang)

    # Public 

    def extract(self, source_code: str, file_path: str = "<unknown>") -> CPGPayload:
        """Parse source_code and return a CPGPayload."""
        if not source_code.strip():
            return CPGPayload(service=self.service, file_path=file_path)

        tree = self._parser.parse(bytes(source_code, "utf-8"))
        payload = CPGPayload(service=self.service, file_path=file_path)

        # Walk the top-level module children
        for node in tree.root_node.children:
            # Handle decorated definitions (wraps function/class)
            target = node
            if node.type == "decorated_definition":
                for child in node.children:
                    if child.type in ("function_definition", "class_definition"):
                        target = child
                        break

            if target.type == "function_definition":
                func_node, edges = self._extract_function(target, decorator_parent=node)
                payload.nodes.append(func_node)
                payload.edges.extend(edges)

            elif target.type == "class_definition":
                class_node = self._extract_class(target, decorator_parent=node)
                payload.nodes.append(class_node)
                # Also extract methods inside class
                body = target.child_by_field_name("body")
                if body:
                    for child in body.children:
                        method_target = child
                        if child.type == "decorated_definition":
                            for gc in child.children:
                                if gc.type == "function_definition":
                                    method_target = gc
                                    break
                        if method_target.type == "function_definition":
                            method_node, method_edges = self._extract_function(
                                method_target, decorator_parent=child
                            )
                            payload.nodes.append(method_node)
                            payload.edges.extend(method_edges)

            elif node.type in ("import_statement", "import_from_statement"):
                payload.edges.extend(self._extract_import(node))

        return payload

    # Function extraction 

    def _extract_function(
        self,
        node,
        decorator_parent=None,
    ) -> tuple[CPGNode, list[CPGEdge]]:
        name_node = node.child_by_field_name("name")
        body_node = node.child_by_field_name("body")
        params_node = node.child_by_field_name("parameters")
        return_type_node = node.child_by_field_name("return_type")

        func_id = self._hash_node(name_node)

        # Detect async: first child is 'async' keyword
        is_async = bool(node.children and node.children[0].type == "async")

        # Parameters
        param_hashes: dict[str, str] = {}
        params = self._extract_parameters(params_node, param_hashes)

        # Decorators
        decorators = []
        if decorator_parent and decorator_parent.type == "decorated_definition":
            for child in decorator_parent.children:
                if child.type == "decorator":
                    decorators.append(self._node_text(child))

        func_node = CPGNode(
            id=func_id,
            type="Function",
            is_async=is_async,
            decorators=decorators,
            docstring=self._extract_docstring(body_node),
            parameters=params,
            return_type=self._node_text(return_type_node) or "Any",
            logic_sequence=self._extract_logic(body_node),
        )

        edges = self._extract_call_edges(func_id, body_node, param_hashes)
        return func_node, edges

    def _extract_class(self, node, decorator_parent=None) -> CPGNode:
        name_node = node.child_by_field_name("name")
        body_node = node.child_by_field_name("body")
        decorators = []
        if decorator_parent and decorator_parent.type == "decorated_definition":
            for child in decorator_parent.children:
                if child.type == "decorator":
                    decorators.append(self._node_text(child))
        return CPGNode(
            id=self._hash_node(name_node),
            type="Class",
            decorators=decorators,
            docstring=self._extract_docstring(body_node),
        )

    def _extract_import(self, node) -> list[CPGEdge]:
        edges = []
        if node.type == "import_statement":
            for child in node.named_children:
                mod_name = self._node_text(child)
                if mod_name:
                    edges.append(CPGEdge(
                        from_id=f"{self.service}::__module__",
                        to_id=self.mapper.hash(mod_name),
                        type="IMPORTS",
                        properties={"module": mod_name},
                    ))
        elif node.type == "import_from_statement":
            mod_node = node.child_by_field_name("module_name")
            mod_name = self._node_text(mod_node) if mod_node else "unknown"
            edges.append(CPGEdge(
                from_id=f"{self.service}::__module__",
                to_id=self.mapper.hash(mod_name),
                type="IMPORTS",
                properties={"from_module": mod_name},
            ))
        return edges

    # Edge extraction 

    def _extract_call_edges(
        self,
        caller_id: str,
        body_node,
        param_hashes: dict[str, str],
    ) -> list[CPGEdge]:
        """Walk body AST and emit CALLS + DATA_FLOW edges."""
        edges: list[CPGEdge] = []
        if not body_node:
            return edges

        def _walk(n) -> None:
            if n.type == "call":
                func_field = n.child_by_field_name("function")
                args_field = n.child_by_field_name("arguments")

                if func_field:
                    callee_text = self._node_text(func_field)
                    if callee_text:
                        callee_hash = self.mapper.hash(callee_text)
                        arg_count = len(args_field.named_children) if args_field else 0
                        edges.append(CPGEdge(
                            from_id=caller_id,
                            to_id=callee_hash,
                            type="CALLS",
                            properties={"arg_count": arg_count},
                        ))
                        # DATA_FLOW: param passed to callee
                        if args_field:
                            for idx, arg in enumerate(args_field.named_children):
                                arg_text = self._node_text(arg)
                                if arg_text and arg_text in param_hashes:
                                    edges.append(CPGEdge(
                                        from_id=param_hashes[arg_text],
                                        to_id=callee_hash,
                                        type="DATA_FLOW",
                                        properties={"arg_position": idx},
                                    ))
            for child in n.children:
                _walk(child)

        _walk(body_node)
        return edges

    # Logic block 

    def _extract_logic(self, node) -> list[dict]:
        steps: list[dict] = []
        if not node:
            return steps

        for child in node.named_children:
            step: dict[str, Any] = {"action": child.type}

            if child.type == "assignment":
                left = child.child_by_field_name("left")
                right = child.child_by_field_name("right")
                if left:
                    step["target"] = self.mapper.hash(self._node_text(left))
                if right:
                    step["value_type"] = right.type

            elif child.type == "if_statement":
                cond = child.child_by_field_name("condition")
                cons = child.child_by_field_name("consequence")
                alt = child.child_by_field_name("alternative")
                step["condition_evaluates"] = cond.type if cond else "unknown"
                if cons:
                    step["then_block"] = self._extract_logic(cons)
                if alt:
                    step["else_block"] = self._extract_logic(alt)

            elif child.type in ("for_statement", "while_statement"):
                body = child.child_by_field_name("body")
                step["loop_body"] = self._extract_logic(body)

            elif child.type == "try_statement":
                body = child.child_by_field_name("body")
                step["try_block"] = self._extract_logic(body)
                step["handlers"] = []
                for handler in child.children:
                    if handler.type == "except_clause":
                        err_type_node = handler.child_by_field_name("value")
                        err_body = handler.child_by_field_name("body")
                        caught = "Exception"
                        if err_type_node:
                            if err_type_node.type == "as_pattern":
                                nn = err_type_node.named_children[0] if err_type_node.named_children else err_type_node
                                caught = self._node_text(nn)
                            else:
                                caught = self._node_text(err_type_node)
                        step["handlers"].append({
                            "catches": caught,
                            "fallback_block": self._extract_logic(err_body),
                        })

            elif child.type == "return_statement":
                ret = child.named_children[0] if child.named_children else None
                step["returns_type"] = ret.type if ret else "None"

            elif child.type == "raise_statement":
                cause = child.named_children[0] if child.named_children else None
                step["raises"] = cause.type if cause else "Unknown"

            elif child.type == "expression_statement":
                expr = child.named_children[0] if child.named_children else None
                if expr and expr.type == "call":
                    func = expr.child_by_field_name("function")
                    step["action"] = "function_call"
                    step["calls"] = self.mapper.hash(self._node_text(func)) if func else "Unknown"

            steps.append(step)
        return steps

    # Parameter extraction 

    def _extract_parameters(
        self,
        param_node,
        param_hashes: dict[str, str],
    ) -> list[dict]:
        params = []
        if not param_node:
            return params

        for p in param_node.named_children:
            name_str: str | None = None
            type_str = "Any"

            if p.type == "typed_parameter":
                name_str = self._node_text(p.named_children[0]) if p.named_children else None
                type_str = self._node_text(p.named_children[1]) if len(p.named_children) > 1 else "Unknown"
            elif p.type in ("typed_default_parameter", "default_parameter"):
                name_node_inner = p.child_by_field_name("name")
                type_node_inner = p.child_by_field_name("type")
                name_str = self._node_text(name_node_inner)
                type_str = self._node_text(type_node_inner) if type_node_inner else "Any"
            elif p.type == "identifier":
                name_str = self._node_text(p)
            elif p.type in ("list_splat_pattern", "dictionary_splat_pattern"):
                inner = p.named_children[0] if p.named_children else p
                prefix = "*" if p.type == "list_splat_pattern" else "**"
                name_str = prefix + self._node_text(inner)

            if name_str:
                hashed = self.mapper.hash(name_str)
                param_hashes[name_str] = hashed
                params.append({"name": hashed, "type": type_str})

        return params

    # Docstring extraction 

    def _extract_docstring(self, body_node) -> str | None:
        if body_node and body_node.named_children:
            first = body_node.named_children[0]
            if first.type == "expression_statement" and first.named_children:
                inner = first.named_children[0]
                if inner.type == "string":
                    return self._node_text(inner)
        return None

    # Helpers 

    def _hash_node(self, node) -> str:
        if node is None:
            return "h_unknown"
        text = node.text
        if text is None:
            return "h_unknown"
        return self.mapper.hash(text)

    def _node_text(self, node) -> str:
        if node is None:
            return ""
        if node.text is None:
            return ""
        return node.text.decode("utf-8") if isinstance(node.text, bytes) else str(node.text)
