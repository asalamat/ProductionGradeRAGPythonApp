import pathlib
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

_SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


def load_and_chunk_document(path: str) -> list[str]:
    suffix = pathlib.Path(path).suffix.lower()
    if suffix not in _SUPPORTED:
        raise ValueError(f"Unsupported document type: {suffix}")

    if suffix == ".pdf":
        return _chunk_pdf(path)
    if suffix == ".docx":
        return _chunk_docx(path)
    return _chunk_text(path)


def _chunk_pdf(path: str) -> list[str]:
    docs = PDFReader().load_data(file=path)
    chunks = []
    for d in docs:
        text = getattr(d, "text", None)
        if text:
            chunks.extend(splitter.split_text(text))
    return chunks


def _chunk_docx(path: str) -> list[str]:
    import docx  # python-docx
    doc = docx.Document(path)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return splitter.split_text(full_text)


def _chunk_text(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return splitter.split_text(text)
