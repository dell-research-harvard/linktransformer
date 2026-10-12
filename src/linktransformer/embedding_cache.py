"""
On-disk cache of text embeddings, shared by all LinkTransformer functions that embed text.

Embeddings are stored in a SQLite file inside the cache folder, keyed by the model and the exact
text. For local models the key is a fingerprint of the model's current weights (not its name or
path), so retraining a model, or reusing a path for a different model, never returns stale vectors.
API embedding models are keyed by provider and model name.
"""
import hashlib
import os
import sqlite3
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch

CACHE_FILE_NAME = "embeddings.sqlite"
_SQL_BATCH = 500  # keys per SELECT; stays under SQLite's limit on query parameters


def model_fingerprint(model: torch.nn.Module) -> str:
    """
    Identify a model's current weights. Hashes every tensor's name, shape, dtype, sum and leading
    values, which changes whenever the weights change (e.g. after training) and is fast to compute.
    """
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            # on the CPU: MPS has no float64, and CPU float64 sums are deterministic across devices
            flat = tensor.detach().reshape(-1).cpu()
            digest.update(f"{name}|{tuple(tensor.shape)}|{tensor.dtype}".encode())
            digest.update(flat[:256].double().numpy().tobytes())
            digest.update(np.float64(flat.double().sum().item()).tobytes())
    # inputs longer than this are truncated, which changes their embeddings
    digest.update(str(getattr(model, "max_seq_length", "")).encode())
    return digest.hexdigest()[:32]


def _text_key(text) -> bytes:
    return hashlib.sha256(str(text).encode("utf-8")).digest()


class EmbeddingCache:
    """Embeddings stored in `<folder>/embeddings.sqlite`, keyed by (model key, text)."""

    def __init__(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        self.path = os.path.join(folder, CACHE_FILE_NAME)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "model TEXT NOT NULL, text_key BLOB NOT NULL, vector BLOB NOT NULL, "
                "PRIMARY KEY (model, text_key)) WITHOUT ROWID"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=60)

    def get(self, model_key: str, text_keys: Sequence[bytes]) -> Dict[bytes, np.ndarray]:
        found = {}
        with self._connect() as conn:
            for start in range(0, len(text_keys), _SQL_BATCH):
                batch = list(text_keys[start:start + _SQL_BATCH])
                placeholders = ",".join("?" * len(batch))
                rows = conn.execute(
                    f"SELECT text_key, vector FROM embeddings WHERE model = ? AND text_key IN ({placeholders})",
                    [model_key, *batch],
                )
                for text_key, vector in rows:
                    found[text_key] = np.frombuffer(vector, dtype=np.float32)
        return found

    def put(self, model_key: str, text_keys: Sequence[bytes], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (model, text_key, vector) VALUES (?, ?, ?)",
                [(model_key, key, vector.tobytes()) for key, vector in zip(text_keys, vectors)],
            )


def cached_embeddings(strings: Sequence, cache_folder: str, model_key: str,
                      embed: Callable[[List], np.ndarray]) -> np.ndarray:
    """
    Return embeddings for `strings`, computing (with `embed`) and storing only those not yet cached.

    :param strings: Texts to embed, in order.
    :param cache_folder: Folder holding the cache file.
    :param model_key: Identifies the model that produced the embeddings.
    :param embed: Function mapping a list of texts to a 2-d float array of their embeddings.
    :return: float32 array with one row per input string, in input order.
    """
    cache = EmbeddingCache(cache_folder)
    keys = [_text_key(s) for s in strings]
    unique_keys = list(dict.fromkeys(keys))
    found = cache.get(model_key, unique_keys)

    first_text = dict(zip(reversed(keys), reversed(list(strings))))  # key -> one text with that key
    missing = [k for k in unique_keys if k not in found]
    if missing:
        vectors = np.asarray(embed([first_text[k] for k in missing]), dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(missing):
            raise ValueError(f"Expected {len(missing)} embeddings, got an array of shape {vectors.shape}")
        cache.put(model_key, missing, vectors)
        found.update(zip(missing, vectors))

    return np.stack([found[k] for k in keys]).astype(np.float32, copy=False)
