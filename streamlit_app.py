import asyncio
import os
import time
from collections import Counter
from pathlib import Path

import nest_asyncio
import inngest
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sklearn.decomposition import PCA

import subprocess
import sys

nest_asyncio.apply()
load_dotenv()


def run_async(coro):
    """Run an async coroutine safely regardless of Streamlit's event loop state."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
        return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

st.set_page_config(page_title="Production RAG", page_icon="🔍", layout="wide")

_DOCUMENT_EXTS = ["pdf", "docx", "txt", "md"]
_IMAGE_EXTS = ["jpg", "jpeg", "png", "gif", "webp", "bmp"]
_VIDEO_EXTS = ["mp4", "mov", "avi", "mkv", "webm"]
_ALL_EXTS = _DOCUMENT_EXTS + _IMAGE_EXTS + _VIDEO_EXTS
_MEDIA_ICONS = {"document": "📄", "image": "🖼️", "video": "🎬"}

# Estimated seconds per step — used to animate the progress bar realistically.
# Embedding step is generous because the SPLADE sparse model downloads on first use (~2–5 min).
_STEP_ESTIMATES: dict[str, list[tuple[str, int]]] = {
    "document": [
        ("Loading & chunking document", 10),
        ("Embedding & storing chunks", 60),
    ],
    "image": [
        ("Captioning image with Ollama", 30),
        ("Embedding & storing", 20),
    ],
    "video": [
        ("Transcribing audio + captioning frames", 180),
        ("Embedding & storing chunks", 60),
    ],
}

_DONE_STATUSES = {"completed", "succeeded", "success", "finished"}
_FAIL_STATUSES = {"failed", "cancelled"}


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def _detect_media_type(filename: str) -> str:
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext in _DOCUMENT_EXTS:
        return "document"
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "video"
    raise ValueError(f"Unsupported file type: .{ext}")


def _event_name(media_type: str) -> str:
    return f"rag/ingest_{media_type}"


async def send_ingest_event(file_path: Path, media_type: str, refresh: bool = False) -> str:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name=_event_name(media_type),
            data={
                "file_path": str(file_path.resolve()),
                "source_id": file_path.name,
                "refresh": refresh,
            },
        )
    )
    return result[0]


def save_uploaded_file(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path


async def send_query_event(question: str, top_k: int) -> str:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query",
            data={"question": question, "top_k": top_k},
        )
    )
    return result[0]


def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json().get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 300.0, poll_interval_s: float = 1.0) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = (run.get("status") or "").lower()
            last_status = status or last_status
            if status in _DONE_STATUSES:
                return run.get("output") or {}
            if status in _FAIL_STATUSES:
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out (last status: {last_status})")
        time.sleep(poll_interval_s)


def wait_for_ingestion_with_progress(event_id: str, media_type: str, timeout_s: float = 600.0) -> dict:
    """Poll Inngest run and animate a progress bar based on step time estimates."""
    steps = _STEP_ESTIMATES.get(media_type, [("Processing", 30), ("Storing", 10)])
    total_estimated = sum(t for _, t in steps)

    bar = st.progress(0, text="Waiting for worker to pick up the job…")
    status_label = st.empty()
    start = time.time()

    while True:
        elapsed = time.time() - start

        # Determine current step and progress fraction from elapsed time
        cumulative = 0.0
        current_step_name = steps[-1][0]  # default to last step when time is exceeded
        overall = 0.95
        for step_name, step_duration in steps:
            if elapsed < cumulative + step_duration:
                within_step = (elapsed - cumulative) / step_duration
                overall = (cumulative + within_step * step_duration) / total_estimated
                current_step_name = step_name
                break
            cumulative += step_duration

        pct = min(0.95, overall)
        bar.progress(pct, text=f"⏳ **{current_step_name}**…  ({int(pct * 100)}%)")

        # Poll actual run status
        try:
            runs = fetch_runs(event_id)
        except Exception:
            runs = []

        if runs:
            status = runs[0].get("status", "")
            status_lower = status.lower()
            status_label.caption(f"Inngest status: `{status}`")

            if status_lower in _DONE_STATUSES:
                bar.progress(1.0, text="✅ Ingestion complete! (100%)")
                status_label.empty()
                return runs[0].get("output") or {}

            if status_lower in _FAIL_STATUSES:
                bar.progress(1.0, text=f"❌ Ingestion {status_lower}")
                status_label.empty()
                raise RuntimeError(f"Ingestion {status_lower}")

        if elapsed > timeout_s:
            bar.progress(1.0, text="⏱️ Timed out")
            raise TimeoutError("Ingestion timed out")

        time.sleep(1.5)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

@st.cache_data
def fetch_db_stats() -> dict:
    client = QdrantClient(url="http://localhost:6333", timeout=10)
    collection = "docs"
    if not client.collection_exists(collection):
        return {"total": 0, "by_type": {}, "sources": []}

    info = client.get_collection(collection)
    total = info.points_count or 0

    # Scroll all payloads to aggregate by media_type and source
    all_points, _ = client.scroll(
        collection_name=collection,
        with_payload=True,
        with_vectors=False,
        limit=10_000,
    )
    by_type: Counter = Counter()
    sources: dict[str, str] = {}  # source_id → media_type
    for p in all_points:
        payload = p.payload or {}
        mt = payload.get("media_type", "unknown")
        src = payload.get("source", "")
        by_type[mt] += 1
        if src:
            sources[src] = mt

    return {"total": total, "by_type": dict(by_type), "sources": sources}


# ---------------------------------------------------------------------------
# UI — Tabs
# ---------------------------------------------------------------------------

st.title("Production RAG")
tab_ingest, tab_query, tab_db, tab_ig = st.tabs(["📥 Ingest", "💬 Query", "🗄️ Database", "📸 Instagram"])

def _scan_folder(folder: Path) -> list[Path]:
    """Recursively find all supported files under folder."""
    supported = {f".{e}" for e in _ALL_EXTS}
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in supported
    )


def _ingest_file_list(files: list[Path], refresh: bool) -> None:
    """Ingest a list of absolute file paths with per-file progress."""
    st.markdown(f"**{len(files)} file(s) — processing one by one:**")
    for i, path in enumerate(files):
        st.markdown(f"---\n**{i + 1} / {len(files)} — {path.name}**")
        try:
            media_type = _detect_media_type(path.name)
            icon = _MEDIA_ICONS[media_type]
            event_id = run_async(
                send_ingest_event(path, media_type, refresh=refresh)
            )
            action = "🔄 Refreshing" if refresh else f"{icon} Ingesting"
            st.markdown(f"{action} **{path.name}**")
            try:
                wait_for_ingestion_with_progress(event_id, media_type)
                st.success(f"{icon} **{path.name}** is ready to query!")
            except RuntimeError as e:
                st.error(f"{path.name}: {e}")
            except TimeoutError:
                st.warning(f"{path.name}: still running in background — check http://localhost:8288")
        except ValueError as e:
            st.error(f"{path.name}: {e}")


with tab_ingest:
    # ── shared refresh toggle ──────────────────────────────────────────────
    refresh_mode = st.toggle(
        "Replace existing (refresh DB)",
        value=False,
        help="Delete all previously stored chunks for each file before re-ingesting.",
    )

    st.divider()

    # ── Section 1: File uploader ───────────────────────────────────────────
    st.subheader("Upload files")
    uploaded_files = st.file_uploader(
        "Choose documents, images, or videos",
        type=_ALL_EXTS,
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, TXT, MD · JPG, PNG, GIF, WEBP · MP4, MOV, AVI, MKV",
    )

    if uploaded_files:
        saved_paths = []
        for uploaded in uploaded_files:
            p = save_uploaded_file(uploaded)
            saved_paths.append(p)
        _ingest_file_list(saved_paths, refresh=refresh_mode)

    st.divider()

    # ── Section 2: Folder import ───────────────────────────────────────────
    st.subheader("Import from folders")

    if "selected_folders" not in st.session_state:
        st.session_state.selected_folders = []

    def _pick_folder() -> str | None:
        """Open a native folder dialog in a subprocess to avoid tkinter main-thread restriction."""
        script = (
            "import tkinter as tk; from tkinter import filedialog; "
            "root = tk.Tk(); root.withdraw(); root.wm_attributes('-topmost', 1); "
            "path = filedialog.askdirectory(title='Select a folder to import'); "
            "root.destroy(); print(path)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True
        )
        folder = result.stdout.strip()
        return folder if folder else None

    if st.button("📁 Browse & add folder"):
        picked = _pick_folder()
        if picked and picked not in st.session_state.selected_folders:
            st.session_state.selected_folders.append(picked)
            st.rerun()
        elif picked:
            st.toast("Folder already added.")

    # Show queued folders with remove buttons
    if st.session_state.selected_folders:
        st.markdown(f"**{len(st.session_state.selected_folders)} folder(s) queued:**")
        for i, folder in enumerate(list(st.session_state.selected_folders)):
            col_path, col_rm = st.columns([5, 1])
            col_path.markdown(f"📂 `{folder}`")
            if col_rm.button("✕", key=f"rm_folder_{i}", help="Remove"):
                st.session_state.selected_folders.pop(i)
                st.rerun()

        if st.button("🗑️ Clear all folders", type="secondary"):
            st.session_state.selected_folders.clear()
            st.rerun()

        # Scan all folders and preview
        all_found: list[Path] = []
        for folder in st.session_state.selected_folders:
            fp = Path(folder)
            if fp.is_dir():
                all_found.extend(_scan_folder(fp))
            else:
                st.warning(f"Skipping invalid path: `{folder}`")

        # Deduplicate (same file in overlapping folders)
        seen, unique_files = set(), []
        for p in all_found:
            if str(p) not in seen:
                seen.add(str(p))
                unique_files.append(p)

        if unique_files:
            st.markdown(f"**{len(unique_files)} file(s) found across all folders:**")
            preview_data = []
            for p in unique_files:
                try:
                    mt = _detect_media_type(p.name)
                except ValueError:
                    mt = "?"
                preview_data.append({
                    "File": p.name,
                    "Type": mt,
                    "Folder": str(p.parent),
                    "Size": f"{p.stat().st_size / 1024:.1f} KB",
                })
            st.dataframe(preview_data, use_container_width=True, hide_index=True)

            if st.button("▶ Start folder ingestion", type="primary"):
                _ingest_file_list(unique_files, refresh=refresh_mode)
        else:
            st.warning("No supported files found in the selected folders.")

# ---------------------------------------------------------------------------
# UI — Query
# ---------------------------------------------------------------------------

with tab_query:
    st.subheader("Ask a question")

    with st.form("rag_query_form"):
        question = st.text_input("Your question", placeholder="What does the document say about…?")
        top_k = st.number_input("Chunks to retrieve", min_value=1, max_value=20, value=10, step=1)
        submitted = st.form_submit_button("Ask")

        if submitted and question.strip():
            with st.spinner("Searching and generating answer…"):
                try:
                    event_id = run_async(send_query_event(question.strip(), int(top_k)))
                    output = wait_for_run_output(event_id)
                    answer = output.get("answer", "")
                    sources = output.get("sources", [])
                    num_contexts = output.get("num_contexts", 0)

                    st.subheader("Answer")
                    st.write(answer or "(No answer returned)")

                    if sources:
                        with st.expander(f"Sources ({len(sources)}) · {num_contexts} context chunks used"):
                            for s in sources:
                                st.write(f"- `{s}`")
                except TimeoutError:
                    st.error("Query timed out. Check the Inngest dashboard.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ---------------------------------------------------------------------------
# UI — Database
# ---------------------------------------------------------------------------

with tab_db:
    st.subheader("Database overview")

    if st.button("🔄 Refresh stats"):
        fetch_db_stats.clear()

    try:
        stats = fetch_db_stats()

        if stats["total"] == 0:
            st.info("No data ingested yet. Upload files in the Ingest tab.")
        else:
            # Top metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total chunks", stats["total"])
            col2.metric("Unique sources", len(stats["sources"]))
            col3.metric("Media types", len(stats["by_type"]))

            st.divider()

            # Chunks by media type
            left, right = st.columns(2)
            with left:
                st.markdown("**Chunks by media type**")
                for mt, count in sorted(stats["by_type"].items()):
                    icon = {"document": "📄", "image": "🖼️", "video": "🎬"}.get(mt, "❓")
                    pct = count / stats["total"] * 100
                    st.markdown(f"{icon} **{mt}** — {count} chunks ({pct:.1f}%)")
                    st.markdown(
                        f'<div style="background:#e0e0e0;border-radius:4px;height:10px;width:100%">'
                        f'<div style="background:#4CAF50;border-radius:4px;height:10px;width:{pct:.1f}%"></div>'
                        f'</div><br>',
                        unsafe_allow_html=True,
                    )

            with right:
                st.markdown("**Indexed sources**")
                for source, mt in sorted(stats["sources"].items()):
                    icon = {"document": "📄", "image": "🖼️", "video": "🎬"}.get(mt, "❓")
                    st.markdown(f"{icon} `{source}`")

            st.divider()

            # ── Vector space visualisation ─────────────────────────────────
            st.subheader("Vector space")
            st.caption(
                "Each dot is a stored chunk. Nearby dots are semantically similar. "
                "Edges connect chunks with cosine similarity ≥ the threshold below."
            )

            reduction = st.radio(
                "Dimensionality reduction", ["PCA", "UMAP"], horizontal=True,
                help="UMAP preserves local cluster structure better; PCA is faster."
            )
            sim_threshold = st.slider(
                "Similarity edge threshold", 0.70, 0.99, 0.88, 0.01,
                help="Draw a line between two chunks when their cosine similarity exceeds this value."
            )
            color_by = st.radio("Colour by", ["source", "media_type"], horizontal=True)

            if st.button("🔭 Build visualisation"):
                with st.spinner("Fetching vectors from Qdrant…"):
                    client = QdrantClient(url="http://localhost:6333", timeout=30)
                    all_points, _ = client.scroll(
                        collection_name="docs",
                        with_payload=True,
                        with_vectors=["dense"],
                        limit=10_000,
                    )

                if not all_points:
                    st.warning("No vectors found.")
                else:
                    vecs = np.array([p.vector["dense"] for p in all_points], dtype=np.float32)
                    labels = [
                        p.payload.get(color_by, "unknown") for p in all_points
                    ]
                    texts = [
                        (p.payload.get("text") or "")[:120] + "…" for p in all_points
                    ]
                    sources = [p.payload.get("source", "") for p in all_points]

                    with st.spinner(f"Running {reduction} on {len(vecs)} vectors…"):
                        if reduction == "UMAP":
                            import umap
                            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(vecs) - 1))
                        else:
                            reducer = PCA(n_components=2, random_state=42)
                        coords = reducer.fit_transform(vecs)

                    # Cosine similarity edges
                    norms = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
                    sim_matrix = norms @ norms.T
                    edge_x, edge_y = [], []
                    n = len(coords)
                    for i in range(n):
                        for j in range(i + 1, n):
                            if sim_matrix[i, j] >= sim_threshold:
                                edge_x += [coords[i, 0], coords[j, 0], None]
                                edge_y += [coords[i, 1], coords[j, 1], None]

                    unique_labels = sorted(set(labels))
                    palette = [
                        "#4C72B0","#DD8452","#55A868","#C44E52","#8172B2",
                        "#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD",
                    ]
                    color_map = {lb: palette[i % len(palette)] for i, lb in enumerate(unique_labels)}

                    fig = go.Figure()

                    # Draw edges first (behind dots)
                    if edge_x:
                        fig.add_trace(go.Scatter(
                            x=edge_x, y=edge_y, mode="lines",
                            line=dict(color="rgba(150,150,150,0.25)", width=1),
                            hoverinfo="none", showlegend=False,
                        ))

                    # Draw one scatter trace per label (for legend)
                    for lb in unique_labels:
                        idx = [i for i, l in enumerate(labels) if l == lb]
                        fig.add_trace(go.Scatter(
                            x=coords[idx, 0], y=coords[idx, 1],
                            mode="markers",
                            marker=dict(size=8, color=color_map[lb], opacity=0.85,
                                        line=dict(width=0.5, color="white")),
                            name=lb,
                            text=[f"<b>{sources[i]}</b><br>{texts[i]}" for i in idx],
                            hovertemplate="%{text}<extra></extra>",
                        ))

                    fig.update_layout(
                        height=620,
                        margin=dict(l=0, r=0, t=30, b=0),
                        legend=dict(title=color_by, orientation="h", y=-0.08),
                        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"{len(coords)} chunks plotted · {len(edge_x) // 3} similarity edges drawn")

            st.divider()
            st.caption("Qdrant dashboard → [localhost:6333/dashboard](http://localhost:6333/dashboard)")

    except Exception as e:
        st.error(f"Could not connect to Qdrant: {e}")

# ---------------------------------------------------------------------------
# UI — Instagram Post Generator
# ---------------------------------------------------------------------------

def _openai_client():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "your-openai-api-key-here":
        return None, "OPENAI_API_KEY not set. Add it to your .env file."
    try:
        from openai import OpenAI as _OpenAI
        return _OpenAI(api_key=api_key), None
    except Exception as e:
        return None, str(e)


def _generate_caption(client, topic: str, tone: str, extra_context: str, num_hashtags: int) -> str:
    context_block = f"\n\nAdditional context:\n{extra_context}" if extra_context.strip() else ""
    prompt = (
        f"Write an Instagram caption about: {topic}\n"
        f"Tone: {tone}\n"
        f"Requirements:\n"
        f"- Engaging opening line (hook)\n"
        f"- 2–4 sentences of body copy\n"
        f"- A clear call-to-action at the end\n"
        f"- {num_hashtags} relevant hashtags on a new line at the bottom\n"
        f"- Use line breaks for readability\n"
        f"- NO markdown, NO asterisks, just clean text{context_block}\n\n"
        f"Return ONLY the caption text."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.85,
        max_tokens=512,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional social media copywriter specialising in Instagram. "
                    "You write punchy, authentic captions that drive engagement."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def _generate_image(client, topic: str, style: str, aspect: str) -> str:
    size_map = {"Square (1:1)": "1024x1024", "Portrait (4:5)": "1024x1024", "Story (9:16)": "1024x1792"}
    size = size_map.get(aspect, "1024x1024")
    image_prompt = (
        f"Instagram-worthy photo of: {topic}. "
        f"Style: {style}. "
        f"High quality, visually striking, professional photography, "
        f"warm natural lighting, social media aesthetic. "
        f"No text overlays."
    )
    resp = client.images.generate(
        model="dall-e-3",
        prompt=image_prompt,
        size=size,
        quality="standard",
        n=1,
    )
    return resp.data[0].url


with tab_ig:
    st.subheader("Instagram Post Generator")
    st.caption("Generate a caption + AI image using OpenAI · gpt-4o-mini + DALL-E 3")

    client_ig, key_error = _openai_client()

    if key_error:
        st.error(f"⚠️ {key_error}")
        st.markdown(
            "**To fix:**\n"
            "1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)\n"
            "2. Create a new **Project API key** (starts with `sk-proj-`)\n"
            "3. Paste it into `.env` → `OPENAI_API_KEY=sk-proj-...`\n"
            "4. Restart Streamlit (`Ctrl+C` then `streamlit run streamlit_app.py`)"
        )
        st.info("💡 If your key starts with `sk-svcac`, it is a deprecated service-account key. Create a fresh one.")
    else:
        with st.form("ig_form"):
            col_a, col_b = st.columns(2)

            with col_a:
                topic = st.text_area(
                    "Topic / Subject",
                    placeholder="e.g. our new summer collection of handmade jewellery",
                    height=100,
                )
                tone = st.selectbox(
                    "Caption tone",
                    ["Inspirational", "Casual & friendly", "Professional", "Humorous", "Educational", "Luxurious"],
                )
                num_hashtags = st.slider("Number of hashtags", 5, 30, 15)

            with col_b:
                style = st.selectbox(
                    "Image style",
                    [
                        "Bright & airy lifestyle",
                        "Dark & moody editorial",
                        "Flat lay product shot",
                        "Candid street photography",
                        "Minimalist studio",
                        "Vibrant pop art",
                        "Golden hour outdoors",
                    ],
                )
                aspect = st.selectbox(
                    "Image format",
                    ["Square (1:1)", "Portrait (4:5)", "Story (9:16)"],
                )
                generate_image = st.checkbox("Generate AI image (DALL-E 3)", value=True)

            extra_context = st.text_area(
                "Extra context (optional)",
                placeholder="Paste a product description, brand guidelines, or any notes…",
                height=80,
            )

            submitted_ig = st.form_submit_button("✨ Generate Post", type="primary")

        if submitted_ig:
            if not topic.strip():
                st.warning("Please enter a topic.")
            else:
                col_cap, col_img = st.columns([1, 1])

                with col_cap:
                    with st.spinner("Writing caption…"):
                        try:
                            caption = _generate_caption(client_ig, topic, tone, extra_context, num_hashtags)
                            st.subheader("Caption")
                            st.text_area("Copy this", value=caption, height=320, label_visibility="collapsed")
                            st.caption(f"~{len(caption.split())} words · {len(caption)} characters")
                        except Exception as e:
                            st.error(f"Caption error: {e}")
                            caption = None

                if generate_image:
                    with col_img:
                        with st.spinner("Generating image with DALL-E 3 (may take ~15 s)…"):
                            try:
                                img_url = _generate_image(client_ig, topic, style, aspect)
                                st.subheader("Image")
                                st.image(img_url, use_container_width=True)
                                st.caption("⚠️ URL expires in ~1 hour — right-click → Save image now")
                            except Exception as e:
                                st.error(f"Image error: {e}")
