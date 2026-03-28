import hashlib
import json
import sqlite3
import tree_sitter_python as tspython
import tree_sitter as ts
from tree_sitter import Language, Parser

class SQLiteMapper:
    def __init__(self, db_path=".ghost_mapper.db"):
        if not hasattr(SQLiteMapper, "_conn") or SQLiteMapper._conn is None:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            self.cursor.execute('CREATE TABLE IF NOT EXISTS hash_map (hash_id TEXT PRIMARY KEY, original_name TEXT)')
            self.conn.commit()
        else:
            self.conn = SQLiteMapper._conn
            self.cursor = SQLiteMapper._cursor

    def save_mapping(self, hash_id, original_name):
        self.cursor.execute('INSERT OR IGNORE INTO hash_map (hash_id, original_name) VALUES (?, ?)', (hash_id, original_name))
        self.conn.commit()

    def close(self):
        """Close the SQLite connection cleanly. Resets the singleton so a new one can be created later if needed."""
        try:
            if hasattr(self, "conn") and self.conn:
                self.conn.close()
        finally:
            # Reset class‑level singleton references
            SQLiteMapper._conn = None
            SQLiteMapper._cursor = None

class DeepGhostWriter:
    def __init__(self, db_mapper, salt="sentry_v3"):
        self.salt = salt
        self.db = db_mapper
        self.PY_LANGUAGE = Language(tspython.language())
        self.parser = Parser()
        self.parser.language = self.PY_LANGUAGE
        
        self.query = ts.Query(self.PY_LANGUAGE, """
        (function_definition name: (identifier) @func.name) @function
        (class_definition name: (identifier) @class.name) @class
        """)

    def _hash(self, text):
        if not text: return None
        clean_text = text.decode('utf8') if isinstance(text, bytes) else text
        raw_hash = hashlib.sha256(f"{clean_text}_{self.salt}".encode()).hexdigest()[:8]
        hashed_name = f"h_{raw_hash}"
        self.db.save_mapping(hashed_name, clean_text)
        return hashed_name

    def _extract_text(self, node):
        return node.text.decode('utf8') if node else None

    def _extract_decorators(self, node):
        decorators = []
        if node.parent and node.parent.type == 'decorated_definition':
            for child in node.parent.named_children:
                if child.type == 'decorator':
                    decorators.append(self._extract_text(child))
        return decorators

    def _extract_docstring(self, body_node):
        if body_node and len(body_node.named_children) > 0:
            first_stmt = body_node.named_children[0]
            if first_stmt.type == 'expression_statement' and len(first_stmt.named_children) > 0:
                inner = first_stmt.named_children[0]
                if inner.type == 'string':
                    return self._extract_text(inner)
        return None

    def _extract_parameters(self, param_node):
        params = []
        if not param_node: return params
        for p in param_node.named_children:
            if p.type == 'typed_parameter':
                name = self._extract_text(p.named_children[0]) if len(p.named_children) > 0 else "Unknown"
                ptype = self._extract_text(p.named_children[1]) if len(p.named_children) > 1 else "Unknown"
                params.append({"name": name, "type": ptype})
            elif p.child_by_field_name('name'):
                p_name_node = p.child_by_field_name('name')
                p_type_node = p.child_by_field_name('type')
                params.append({
                    "name": self._extract_text(p_name_node),
                    "type": self._extract_text(p_type_node) if p_type_node else ("Any" if not p.type.endswith("splat_pattern") else "kwargs/args")
                })
            elif p.type == 'identifier':
                params.append({"name": self._extract_text(p), "type": "Any"})
            elif p.type in ('list_splat_pattern', 'dictionary_splat_pattern'):
                inner = p.named_children[0] if len(p.named_children)>0 else p
                prefix = "*" if p.type == 'list_splat_pattern' else "**"
                params.append({"name": prefix + self._extract_text(inner), "type": "Any"})
            elif p.type == 'default_parameter':
                p_name_node = p.named_children[0] if len(p.named_children) > 0 else p
                params.append({"name": self._extract_text(p_name_node), "type": "Any"})
        return params

    def _extract_logic_block(self, node):
        steps = []
        if not node: return steps

        for child in node.named_children:
            step = {"action": child.type}

            # 1. VARIABLE ASSIGNMENT
            if child.type == "assignment":
                left = child.child_by_field_name("left")
                right = child.child_by_field_name("right")
                if left: step["target"] = self._hash(left.text)
                if right: step["value_type"] = right.type

            # 2. IF STATEMENTS
            elif child.type == "if_statement":
                condition = child.child_by_field_name("condition")
                consequence = child.child_by_field_name("consequence")
                alternative = child.child_by_field_name("alternative")
                if condition: step["condition_evaluates"] = condition.type
                if consequence: step["then_block"] = self._extract_logic_block(consequence)
                if alternative: step["else_block"] = self._extract_logic_block(alternative)

            # 3. LOOPS
            elif child.type in ("for_statement", "while_statement"):
                body = child.child_by_field_name("body")
                step["loop_body"] = self._extract_logic_block(body)

            # 4. TRY / EXCEPT
            elif child.type == "try_statement":
                body = child.child_by_field_name("body")
                step["try_block"] = self._extract_logic_block(body)
                step["handlers"] = []
                for handler in child.children:
                    if handler.type == "except_clause":
                        err_type = handler.child_by_field_name("value") # Tree-sitter uses 'value' field for caught exceptions
                        err_body = handler.child_by_field_name("body")
                        caught_type = "Exception"
                        if err_type:
                            if err_type.type == 'as_pattern':
                                err_node = err_type.named_children[0] if len(err_type.named_children)>0 else err_type
                                caught_type = self._extract_text(err_node)
                            else:
                                caught_type = self._extract_text(err_type)
                        
                        step["handlers"].append({
                            "catches": caught_type,
                            "fallback_block": self._extract_logic_block(err_body)
                        })

            # 5. RETURNS & RAISES
            elif child.type == "return_statement":
                ret_val = child.named_children[0] if len(child.named_children)>0 else None
                step["returns_type"] = ret_val.type if ret_val else "None"
            
            elif child.type == "raise_statement":
                cause = child.named_children[0] if len(child.named_children)>0 else None
                step["raises"] = cause.type if cause else "Unknown"

            # 6. EXPRESSIONS / CALLS
            elif child.type == "expression_statement":
                expr = child.named_children[0] if child.named_children else None
                if expr and expr.type == "call":
                    func = expr.child_by_field_name("function")
                    step["action"] = "function_call"
                    step["calls"] = self._hash(func.text) if func else "Unknown"

            steps.append(step)
        return steps

    def process_code(self, source_code):
        tree = self.parser.parse(bytes(source_code, "utf8"))
        cursor = ts.QueryCursor(self.query)
        captures_dict = cursor.captures(tree.root_node)
        
        captures = []
        for name, nodes in captures_dict.items():
            for node in nodes:
                captures.append((node, name))
        captures.sort(key=lambda x: x[0].start_byte)
        
        ghost_payload = []

        for node, capture_name in captures:
            if capture_name == "function":
                func_name_node = node.child_by_field_name("name")
                body_node = node.child_by_field_name("body")
                params_node = node.child_by_field_name("parameters")
                return_type_node = node.child_by_field_name("return_type")
                
                func_obj = {
                    "node_type": "Function",
                    "identifier": self._hash(func_name_node.text) if func_name_node else "Unknown",
                    "is_async": node.children and node.children[0].type == "async",
                    "decorators": self._extract_decorators(node),
                    "docstring": self._extract_docstring(body_node),
                    "parameters": self._extract_parameters(params_node),
                    "return_type": self._extract_text(return_type_node) if return_type_node else "Any",
                    "logic_sequence": self._extract_logic_block(body_node)
                }
                ghost_payload.append(func_obj)
            
            elif capture_name == "class":
                class_name_node = node.child_by_field_name("name")
                body_node = node.child_by_field_name("body")
                class_obj = {
                    "node_type": "Class",
                    "identifier": self._hash(class_name_node.text) if class_name_node else "Unknown",
                    "decorators": self._extract_decorators(node),
                    "docstring": self._extract_docstring(body_node),
                }
                ghost_payload.append(class_obj)

        return ghost_payload

# --- TEST IT ---
test_code = """
@api.route("/profile")
@require_auth
async def get_user_profile(user_id: int, include_history: bool = False) -> UserProfile:
    \"\"\"Fetches the full user profile including history if requested.\"\"\"
    try:
        if not user_id:
            raise ValueError("Invalid ID")
        
        profile = await db.fetch_profile(user_id)
        
        if include_history:
            for item in profile.history:
                log.audit(item)
                
        return profile
    except DBConnectionError as e:
        log.error("DB failed")
        return None
    except Exception:
        raise HTTPException(500)
"""

if __name__ == "__main__":
    db = SQLiteMapper()
    writer = DeepGhostWriter(db)
    payload = writer.process_code(test_code)
    with open("test_output.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("Wrote output to test_output.json")
