from sentence_transformers import CrossEncoder

# Lightweight, fast cross-encoder — good balance for local use.
# Swap to "cross-encoder/ms-marco-MiniLM-L-12-v2" for higher accuracy.
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model


_MAX_CANDIDATES = 40  # cap CPU cross-encoder work regardless of upstream count


def rerank(query: str, chunks: list[str], top_k: int) -> list[str]:
    if not chunks:
        return []

    candidates = chunks[:_MAX_CANDIDATES]
    model = _get_model()
    pairs = [(query, chunk) for chunk in candidates]
    scores = model.predict(pairs, batch_size=32, show_progress_bar=False)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_k]]
