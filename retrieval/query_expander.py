import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
# Use a small fast model for expansion — 1.7B is plenty for rephrasing
LLM_MODEL = os.getenv("OLLAMA_EXPAND_MODEL", "qwen3:1.7b")
client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

_SYSTEM = (
    "You are a query expansion assistant for a semantic search system. "
    "Given a user question, produce alternative phrasings that capture the same intent "
    "using different vocabulary, synonyms, and sentence structures. "
    "Do NOT add new topics or assumptions. Return ONLY the questions, one per line, no numbering."
)


def expand_query(question: str, n: int = 3) -> list[str]:
    """
    Returns the original question plus n rephrased variants.
    Falls back to just the original if the API call fails.
    """
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.7,
            max_tokens=256,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Question: {question}\n\nWrite {n} alternative phrasings:"},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        # Strip Qwen3 thinking blocks before parsing
        import re
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        variants = [line.strip() for line in raw.splitlines() if line.strip()][:n]
    except Exception:
        variants = []

    seen, result = set(), [question]
    seen.add(question.lower())
    for v in variants:
        if v.lower() not in seen:
            seen.add(v.lower())
            result.append(v)
    return result
