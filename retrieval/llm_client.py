"""
OpenAI client singleton, embedding, token counting, and LLM call helpers.
"""

from __future__ import annotations

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from .config import EMBED_MODEL, LLM_MODEL

load_dotenv()

# Module-level singletons
client = OpenAI()
_enc = tiktoken.encoding_for_model("gpt-4o-mini")


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts in batches of 100, L2-normalize for cosine via IndexFlatIP."""
    batch_size = 100
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs.extend([item.embedding for item in resp.data])
    arr = np.array(vecs, dtype=np.float32)
    # Explicit L2 normalization so IndexFlatIP == cosine similarity
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _enc.decode(tokens[:max_tokens])


def llm(prompt: str, max_tokens: int = 512) -> str:
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()
