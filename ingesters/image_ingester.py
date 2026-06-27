import base64
import mimetypes
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen3.6:35b")
client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

_PROMPT = (
    "You are a precise image analyst for a RAG system. "
    "Describe this image comprehensively: objects, text, colors, spatial layout, "
    "context, and any information a user might search for. "
    "Be specific and thorough — your description will be used for semantic search retrieval."
)


def load_and_describe_image(path: str) -> list[str]:
    mime, _ = mimetypes.guess_type(path)
    if not mime or not mime.startswith("image/"):
        raise ValueError(f"Not a recognized image file: {path}")

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": _PROMPT},
                ],
            }
        ],
        max_tokens=1024,
    )

    description = response.choices[0].message.content.strip()
    filename = os.path.basename(path)
    return [f"[Image: {filename}]\n{description}"]
