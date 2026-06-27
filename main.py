import datetime
import logging
import os
import uuid

import inngest
import inngest.fast_api
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI

from custom_types import MediaType, RAGChunkAndSrc, RAGSearchResult, RAGUpsertResult, RAQQueryResult
from data_loader import embed_texts
from ingesters.document_ingester import load_and_chunk_document
from ingesters.image_ingester import load_and_describe_image
from ingesters.video_ingester import load_and_chunk_video
from retrieval.query_expander import expand_query
from retrieval.reranker import rerank
from vector_db import QdrantStorage

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3.6:35b")

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


def _make_ids(source_id: str, n: int) -> list[str]:
    return [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(n)]


def _upsert_chunks(chunks: list[str], source_id: str, media_type: MediaType, refresh: bool = False) -> RAGUpsertResult:
    store = QdrantStorage()
    if refresh:
        store.delete_by_source(source_id)
    vecs = embed_texts(chunks)
    ids = _make_ids(source_id, len(chunks))
    payloads = [{"source": source_id, "text": chunks[i], "media_type": media_type.value} for i in range(len(chunks))]
    store.upsert(ids, vecs, payloads, chunks)
    return RAGUpsertResult(ingested=len(chunks), media_type=media_type)


# ---------------------------------------------------------------------------
# Ingest: Document (PDF / DOCX / TXT / MD)
# ---------------------------------------------------------------------------

@inngest_client.create_function(
    fn_id="RAG: Ingest Document",
    trigger=inngest.TriggerEvent(event="rag/ingest_document"),
    throttle=inngest.Throttle(limit=2, period=datetime.timedelta(minutes=1)),
)
async def rag_ingest_document(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        file_path = ctx.event.data["file_path"]
        source_id = ctx.event.data.get("source_id", file_path)
        chunks = load_and_chunk_document(file_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id, media_type=MediaType.document)

    def _upsert(c: RAGChunkAndSrc) -> RAGUpsertResult:
        refresh = ctx.event.data.get("refresh", False)
        return _upsert_chunks(c.chunks, c.source_id, MediaType.document, refresh=refresh)

    loaded = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    result = await ctx.step.run("embed-and-upsert", lambda: _upsert(loaded), output_type=RAGUpsertResult)
    return result.model_dump()


# ---------------------------------------------------------------------------
# Ingest: Image
# ---------------------------------------------------------------------------

@inngest_client.create_function(
    fn_id="RAG: Ingest Image",
    trigger=inngest.TriggerEvent(event="rag/ingest_image"),
    throttle=inngest.Throttle(limit=5, period=datetime.timedelta(minutes=1)),
)
async def rag_ingest_image(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        file_path = ctx.event.data["file_path"]
        source_id = ctx.event.data.get("source_id", file_path)
        chunks = load_and_describe_image(file_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id, media_type=MediaType.image)

    def _upsert(c: RAGChunkAndSrc) -> RAGUpsertResult:
        refresh = ctx.event.data.get("refresh", False)
        return _upsert_chunks(c.chunks, c.source_id, MediaType.image, refresh=refresh)

    loaded = await ctx.step.run("caption-image", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    result = await ctx.step.run("embed-and-upsert", lambda: _upsert(loaded), output_type=RAGUpsertResult)
    return result.model_dump()


# ---------------------------------------------------------------------------
# Ingest: Video
# ---------------------------------------------------------------------------

@inngest_client.create_function(
    fn_id="RAG: Ingest Video",
    trigger=inngest.TriggerEvent(event="rag/ingest_video"),
    throttle=inngest.Throttle(limit=1, period=datetime.timedelta(minutes=5)),
)
async def rag_ingest_video(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        file_path = ctx.event.data["file_path"]
        source_id = ctx.event.data.get("source_id", file_path)
        chunks = load_and_chunk_video(file_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id, media_type=MediaType.video)

    def _upsert(c: RAGChunkAndSrc) -> RAGUpsertResult:
        refresh = ctx.event.data.get("refresh", False)
        return _upsert_chunks(c.chunks, c.source_id, MediaType.video, refresh=refresh)

    loaded = await ctx.step.run("transcribe-and-caption", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    result = await ctx.step.run("embed-and-upsert", lambda: _upsert(loaded), output_type=RAGUpsertResult)
    return result.model_dump()


# ---------------------------------------------------------------------------
# Query — unified across all media types, with hybrid search + reranking
# ---------------------------------------------------------------------------

@inngest_client.create_function(
    fn_id="RAG: Query",
    trigger=inngest.TriggerEvent(event="rag/query"),
)
async def rag_query(ctx: inngest.Context):
    def _search(ctx: inngest.Context) -> RAGSearchResult:
        question = ctx.event.data["question"]
        top_k = int(ctx.event.data.get("top_k", 10))

        # 1. Expand the query into multiple variants
        variants = expand_query(question, n=3)

        # 2. Embed all variants in one API call
        all_vecs = embed_texts(variants)

        # 3. Search with each variant, merge unique chunks
        store = QdrantStorage()
        seen_texts: set[str] = set()
        merged_contexts: list[str] = []
        merged_sources: set[str] = set()

        for variant, vec in zip(variants, all_vecs):
            found = store.search(query_text=variant, query_dense_vec=vec, top_k=top_k * 2)
            for ctx_text in found["contexts"]:
                if ctx_text not in seen_texts:
                    seen_texts.add(ctx_text)
                    merged_contexts.append(ctx_text)
            merged_sources.update(found["sources"])

        # 4. Rerank all merged candidates against the original question
        reranked = rerank(question, merged_contexts, top_k=top_k)
        return RAGSearchResult(contexts=reranked, sources=list(merged_sources))

    question = ctx.event.data["question"]
    found = await ctx.step.run("hybrid-search-rerank", lambda: _search(ctx), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(found.contexts))
    user_content = (
        "Answer the question using ONLY the numbered context passages below.\n"
        "Rules:\n"
        "- Cite the passage number(s) you used, e.g. [1] or [2][3].\n"
        "- If the answer is not in the context, respond exactly: 'I don't have enough information to answer this.'\n"
        "- Do not add knowledge outside the provided context.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}"
    )

    def _answer() -> str:
        ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        resp = ollama.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=1024,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise retrieval-augmented assistant. "
                        "Answer strictly from the provided context. "
                        "Always cite passage numbers. Never hallucinate."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
        )
        return resp.choices[0].message.content.strip()

    answer = await ctx.step.run("llm-answer", _answer, output_type=str)
    return RAQQueryResult(answer=answer, sources=found.sources, num_contexts=len(found.contexts)).model_dump()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI()

inngest.fast_api.serve(
    app,
    inngest_client,
    [rag_ingest_document, rag_ingest_image, rag_ingest_video, rag_query],
)
