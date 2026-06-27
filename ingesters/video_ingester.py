import base64
import os
import cv2
import whisper
from openai import OpenAI
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen3.6:35b")
client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
splitter = SentenceSplitter(chunk_size=800, chunk_overlap=150)

_whisper_model = None

_FRAME_CAPTION_PROMPT = (
    "You are analyzing a video frame for a RAG retrieval system. "
    "Describe what is visible: people, objects, text on screen, setting, actions, "
    "and any information relevant to understanding this moment in the video. Be concise but complete."
)

FRAME_INTERVAL_SECONDS = 30


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def _transcribe_audio(video_path: str) -> str:
    model = _get_whisper()
    result = model.transcribe(video_path, fp16=False)
    return result.get("text", "").strip()


def _extract_frames(video_path: str, interval_seconds: int = FRAME_INTERVAL_SECONDS) -> list[tuple[float, str]]:
    """Returns list of (timestamp_seconds, base64_jpeg)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(1, int(fps * interval_seconds))
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            timestamp = frame_idx / fps
            frames.append((timestamp, b64))
        frame_idx += 1

    cap.release()
    return frames


def _caption_frame(b64_jpeg: str, timestamp: float) -> str:
    ts_str = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}"
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"},
                    },
                    {"type": "text", "text": _FRAME_CAPTION_PROMPT},
                ],
            }
        ],
        max_tokens=512,
    )
    caption = response.choices[0].message.content.strip()
    return f"[Video frame at {ts_str}]\n{caption}"


def load_and_chunk_video(path: str) -> list[str]:
    filename = os.path.basename(path)
    chunks: list[str] = []

    transcript = _transcribe_audio(path)
    if transcript:
        transcript_chunks = splitter.split_text(transcript)
        for i, chunk in enumerate(transcript_chunks):
            chunks.append(f"[Video transcript: {filename}, segment {i + 1}]\n{chunk}")

    frames = _extract_frames(path)
    for timestamp, b64 in frames:
        caption = _caption_frame(b64, timestamp)
        chunks.append(caption)

    return chunks
