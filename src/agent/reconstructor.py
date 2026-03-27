"""
Eidolon - Patch Reconstructor
Translates hashed patch diffs back to human-readable code using the local Mapper DB.
This module NEVER sends identifying information to any external service.
"""
from __future__ import annotations

import re
from pathlib import Path

from rich.console import Console
from rich.syntax import Syntax

from src.core.mapper import GhostMapper

console = Console()

# Matches h_XXXXXXXX patterns in patch text
_HASH_RE = re.compile(r"\bh_([0-9a-f]{8})\b")


class PatchReconstructor:
    """
    Reconstructs hashed patch specifications into human-readable diffs.

    The de-hashing happens entirely locally - the Mapper DB is read-only
    and never transmitted anywhere.
    """

    def __init__(self, mapper: GhostMapper | None = None) -> None:
        # Open in read_only mode - reconstructor only does lookups (hash → name), never writes.
        # This allows concurrent access alongside ghost_daemon / sync_worker write connections.
        self._mapper = mapper or GhostMapper(read_only=True)

    def reconstruct_patch(self, hashed_patch: dict) -> str:
        """
        Convert a hashed patch dict into a human-readable diff description.
        Returns a formatted string showing the proposed change.
        """
        if not hashed_patch:
            return "No patch available."

        lines = ["=" * 60, "Eidolon - Reconstructed Patch", "=" * 60, ""]

        target_node = hashed_patch.get("target_node", "")
        change_type = hashed_patch.get("change_type", "UNKNOWN")
        description = hashed_patch.get("description", "")
        insert_before = hashed_patch.get("insert_before")
        new_logic = hashed_patch.get("new_logic", "")

        # De-hash all identifiers
        target_name = self._dehash(target_node) or target_node
        description_dehashed = self._dehash_text(description)
        new_logic_dehashed = self._dehash_text(new_logic)

        lines.append(f"Change Type : {change_type}")
        lines.append(f"Target      : {target_name} ({target_node})")
        lines.append("")

        if description_dehashed:
            lines.append("Description:")
            lines.append(f"  {description_dehashed}")
            lines.append("")

        if insert_before:
            insert_name = self._dehash(insert_before) or insert_before
            lines.append(f"Insert Before: {insert_name} ({insert_before})")
            lines.append("")

        if new_logic_dehashed:
            lines.append("Proposed Logic:")
            for logic_line in new_logic_dehashed.split("\n"):
                lines.append(f"  + {logic_line}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def print_patch(self, hashed_patch: dict) -> None:
        """Print the reconstructed patch with rich formatting."""
        text = self.reconstruct_patch(hashed_patch)
        console.print()
        console.print(Syntax(text, "diff", theme="monokai", line_numbers=False))
        console.print()

    def _dehash(self, hash_id: str) -> str | None:
        """Look up a single hash ID in the Mapper DB."""
        if not hash_id:
            return None
        if not _HASH_RE.match(hash_id):
            return hash_id  # not a hash - return as-is
        return self._mapper.lookup(hash_id)

    def _dehash_text(self, text: str) -> str:
        """Replace all h_XXXXXXXX tokens in a text with their original names."""
        def replace(m: re.Match) -> str:
            hash_id = f"h_{m.group(1)}"
            original = self._mapper.lookup(hash_id)
            return original if original else hash_id

        return _HASH_RE.sub(replace, text)
