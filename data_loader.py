import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768

_EMBED_BATCH_SIZE = 64   # nomic-embed-text handles 64 comfortably
_EMBED_WORKERS = 4       # concurrent requests to Ollama


def _embed_batch(batch: list[str]) -> list[list[float]]:
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    response = client.embeddings.create(model=EMBED_MODEL, input=batch)
    return [item.embedding for item in response.data]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts in parallel batches via local Ollama nomic-embed-text."""
    if not texts:
        return []

    batches = [texts[i : i + _EMBED_BATCH_SIZE] for i in range(0, len(texts), _EMBED_BATCH_SIZE)]

    # preserve order: map index → result
    results: dict[int, list[list[float]]] = {}
    with ThreadPoolExecutor(max_workers=_EMBED_WORKERS) as pool:
        futures = {pool.submit(_embed_batch, b): idx for idx, b in enumerate(batches)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    ordered: list[list[float]] = []
    for i in range(len(batches)):
        ordered.extend(results[i])
    return ordered
