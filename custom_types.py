import pydantic
from enum import Enum


class MediaType(str, Enum):
    document = "document"
    image = "image"
    video = "video"


class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str = None
    media_type: MediaType = MediaType.document


class RAGUpsertResult(pydantic.BaseModel):
    ingested: int
    media_type: MediaType = MediaType.document


class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]


class RAQQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int
