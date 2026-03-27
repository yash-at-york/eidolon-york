"""
Eidolon - Code Embedding Service
Uses CodeT5+ 110M to produce 256-dim structural embeddings.
"""
from __future__ import annotations

import json
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from ghost_config import HF_EMBEDDING_MODEL

_model = None
_tokenizer = None

def _get_model():
    global _model, _tokenizer
    if _model is None:
        import transformers
        # Monkey-patch PretrainedConfig to avoid AttributeError for missing is_decoder
        orig_get = transformers.PretrainedConfig.__getattribute__
        def patched_get(self, key):
            if key == "is_decoder":
                try:
                    return orig_get(self, key)
                except AttributeError:
                    return False
            return orig_get(self, key)
        transformers.PretrainedConfig.__getattribute__ = patched_get
        
        _tokenizer = AutoTokenizer.from_pretrained(HF_EMBEDDING_MODEL, trust_remote_code=True)
        _model = AutoModel.from_pretrained(HF_EMBEDDING_MODEL, trust_remote_code=True)
        _model.eval()
    return _model, _tokenizer

def _node_to_text(node: dict) -> str:
    parts = []
    node_type = node.get("type", "Function")
    parts.append(f"{node_type}")

    if node.get("is_async"):
        parts.append("async")

    params = node.get("parameters", [])
    if params:
        param_types = " ".join(p.get("type", "Any") for p in params)
        parts.append(f"params:{param_types}")

    ret = node.get("return_type", "Any")
    parts.append(f"returns:{ret}")

    logic = node.get("logic_sequence", [])
    if logic:
        actions = " ".join(step.get("action", "") for step in logic[:10])
        parts.append(f"logic:{actions}")

    decorators = node.get("decorators", [])
    if decorators:
        parts.append(f"decorators:{len(decorators)}")

    return " ".join(parts)

def _encode_texts(texts: list[str]) -> torch.Tensor:
    model, tokenizer = _get_model()
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        # CodeT5p-embedding natively returns the pooled tensor directly
        embedding = model(**inputs)
        # Some versions return it as a tensor directly, others as a tuple
        if isinstance(embedding, tuple):
            embedding = embedding[0]
        # fallback just in case it returns BaseModelOutput
        elif hasattr(embedding, "last_hidden_state"):
            embedding = embedding.last_hidden_state[:, 0, :]
    return embedding

def embed_node(node: dict) -> list[float]:
    text = _node_to_text(node)
    vec = _encode_texts([text])[0]
    return vec.tolist()

def embed_nodes_batch(nodes: list[dict]) -> list[list[float]]:
    if not nodes:
        return []
    texts = [_node_to_text(n) for n in nodes]
    vecs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = _encode_texts(batch)
        vecs.extend(emb.tolist())
    return vecs

def embed_error_text(error_text: str) -> list[float]:
    vec = _encode_texts([error_text])[0]
    return vec.tolist()

