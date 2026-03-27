"""
Eidolon - FalkorDB Graph Store Client
Stores the CPG as a property graph with CALLS, DATA_FLOW, and IMPORTS edges.
Enables N-hop traversal queries: "find all callers of h_7b9x within 2 hops".

Node IDs are namespaced: "{service}::{hash_id}" to enable cross-service federation.
"""
from __future__ import annotations

from falkordb import FalkorDB

from ghost_config import FALKORDB_GRAPH, FALKORDB_HOST, FALKORDB_PORT, DEMO_MODE


class GraphStore:
    """
    FalkorDB graph store for CPG nodes and edges.

    In DEMO_MODE, a minimal in-memory dictionary graph is used as a fallback
    so the demo runs without requiring Docker.
    """

    def __init__(self) -> None:
        self._demo_mode = DEMO_MODE
        if not self._demo_mode:
            self._db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
            self._graph = self._db.select_graph(FALKORDB_GRAPH)
        else:
            # Lightweight in-memory graph for demo
            self._nodes: dict[str, dict] = {}
            self._edges: list[dict] = []

    # Node operations 

    def upsert_node(self, node_id: str, properties: dict) -> None:
        """Create or update a CPG node."""
        if self._demo_mode:
            self._nodes[node_id] = {"node_id": node_id, **properties}
            return

        # Escape properties for Cypher
        props_str = self._props_to_cypher({"node_id": node_id, **properties})
        self._graph.query(f"""
            MERGE (n:GhostNode {{node_id: '{self._esc(node_id)}'}})
            SET n += {{{props_str}}}
        """)

    def upsert_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: str,
        properties: dict | None = None,
    ) -> None:
        """Create or update a directed CPG edge."""
        if self._demo_mode:
            self._edges.append({
                "from": from_id,
                "to": to_id,
                "type": edge_type,
                **(properties or {}),
            })
            return

        props_str = self._props_to_cypher(properties or {})
        self._graph.query(f"""
            MATCH (a:GhostNode {{node_id: '{self._esc(from_id)}'}}),
                  (b:GhostNode {{node_id: '{self._esc(to_id)}'}})
            MERGE (a)-[r:{edge_type}]->(b)
            SET r += {{{props_str}}}
        """)

    # Graph operations 

    def traverse(self, node_id: str, depth: int = 2) -> dict:
        """
        Return the N-hop neighbourhood of a node as a dict:
        {"nodes": [...], "edges": [...]}
        """
        if self._demo_mode:
            # Simple BFS in memory
            visited = set()
            frontier = {node_id}
            result_nodes = []
            result_edges = []
            for _ in range(depth):
                next_frontier = set()
                for fid in frontier:
                    if fid in visited:
                        continue
                    visited.add(fid)
                    if fid in self._nodes:
                        result_nodes.append(self._nodes[fid])
                    for edge in self._edges:
                        if edge["from"] == fid or edge["to"] == fid:
                            result_edges.append(edge)
                            next_frontier.add(edge["to"])
                            next_frontier.add(edge["from"])
                frontier = next_frontier - visited
            return {"nodes": result_nodes, "edges": result_edges}

        result = self._graph.query(f"""
            MATCH path = (start:GhostNode {{node_id: '{self._esc(node_id)}'}})
                         -[*1..{depth}]-> (neighbor:GhostNode)
            RETURN nodes(path) AS ns, relationships(path) AS rs
        """)

        nodes, edges = [], []
        seen_nodes: set[str] = set()
        seen_edges: set[str] = set()

        for row in result.result_set:
            for n in row[0]:
                if n.properties["node_id"] not in seen_nodes:
                    nodes.append(n.properties)
                    seen_nodes.add(n.properties["node_id"])
            for r in row[1]:
                edge_key = f"{r.src_node}→{r.dest_node}"
                if edge_key not in seen_edges:
                    edges.append({
                        "from": r.src_node,
                        "to": r.dest_node,
                        "type": r.type,
                        **r.properties,
                    })
                    seen_edges.add(edge_key)

        return {"nodes": nodes, "edges": edges}

    def get_callers(self, node_id: str) -> list[str]:
        """Return the node IDs of all direct callers of node_id."""
        if self._demo_mode:
            return [e["from"] for e in self._edges if e["to"] == node_id and e["type"] == "CALLS"]

        result = self._graph.query(f"""
            MATCH (caller:GhostNode)-[:CALLS]->(n:GhostNode {{node_id: '{self._esc(node_id)}'}})
            RETURN caller.node_id
        """)
        return [row[0] for row in result.result_set]

    def sync_from_payload(self, cpg_dict: dict, service: str) -> None:
        """
        Bulk load a complete CPG payload into the graph.
        Nodes and edges are namespaced with the service prefix.
        """
        for node in cpg_dict.get("nodes", []):
            namespaced_id = f"{service}::{node['id']}"
            self.upsert_node(namespaced_id, {
                "type": node.get("type", "Unknown"),
                "service": service,
                "is_async": node.get("is_async", False),
                "return_type": node.get("return_type", "Any"),
            })

        for edge in cpg_dict.get("edges", []):
            from_id = f"{service}::{edge['from']}"
            to_id = f"{service}::{edge['to']}"
            edge_props = {k: v for k, v in edge.items() if k not in ("from", "to", "type")}
            self.upsert_edge(from_id, to_id, edge.get("type", "CALLS"), edge_props)

    # Helpers 

    def _esc(self, s: str) -> str:
        """Escape single quotes for Cypher string literals."""
        return s.replace("'", "\\'")

    def _props_to_cypher(self, props: dict) -> str:
        """Convert a dict to Cypher property syntax: key: 'value', ..."""
        parts = []
        for k, v in props.items():
            if isinstance(v, bool):
                parts.append(f"{k}: {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                parts.append(f"{k}: {v}")
            else:
                parts.append(f"{k}: '{self._esc(str(v))}'")
        return ", ".join(parts)
