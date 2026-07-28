"""The embedding model, loaded once per process (spec 6.1, 7.13).

BAAI/bge-small-en-v1.5 through fastembed's ONNX runtime. The spec's constraint is
open-source models running on CPU and it names sentence-transformers, BGE and E5
as the options; this takes BGE, but through ONNX rather than torch. Measured on
``python:3.12-slim``: the torch route adds 1,208 MB to the image and this one
adds 194 MB, for the same model and the same constraint satisfied.

**The model is a process-level singleton.** Loading it costs a few hundred
milliseconds and several hundred megabytes of resident memory, so loading it per
request would make the first question of every conversation slow and the tenth
concurrent one fatal. It is also baked into the image (see the Dockerfile), so
this never reaches the network -- a container with no internet access can still
answer questions about data it already holds.

BGE asks for a prefix on *queries* but not on the passages they are matched
against, which is not a detail this module lets a caller get wrong: there are two
functions, and the asymmetry lives inside them.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Must match the width declared on the ``run_chunks.embedding`` column. pgvector
# needs a fixed width to store the type at all, and a mismatch is an error at
# insert rather than a wrong answer at query time -- which is the right way round.
EMBEDDING_DIMENSIONS = 384

# BGE was trained with an instruction prefix on the query side only. Retrieval
# quality drops measurably without it, and applying it to the passages too is the
# more common mistake -- it makes every stored vector point in a slightly
# similar direction, which flattens the distances the search depends on.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None
_lock = threading.Lock()


def get_model():
    """The shared embedding model, loaded on first use.

    Double-checked locking because the worker and the API both serve concurrent
    requests: two threads racing here would otherwise load two copies of the
    model into memory and keep whichever finished last.
    """
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from fastembed import TextEmbedding

                logger.info("Loading the embedding model %s", MODEL_NAME)
                _model = TextEmbedding(MODEL_NAME)
    return _model


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed passages for storage. No prefix -- see ``QUERY_PREFIX``."""
    if not texts:
        return []
    return [vector.tolist() for vector in get_model().embed(texts)]


def embed_query(text: str) -> list[float]:
    """Embed one question, with the prefix BGE expects on the query side."""
    return next(iter(get_model().embed([QUERY_PREFIX + text]))).tolist()
