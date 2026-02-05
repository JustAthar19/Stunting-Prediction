from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings


from dotenv import load_dotenv
load_dotenv()

@dataclass(frozen=True)
class RagConfig:
    docs_dir: Path
    persist_dir: Path
    collection_name: str = "guideline"
    chunk_size: int = 300
    chunk_overlap: int = 80
    top_k: int = 4


def _default_docs_dir(project_root: Path) -> Path:
    p1 = project_root / "data" / "guidelines"
    p2 = project_root / "data" / "guideline"
    if p1.exists():
        return p1
    return p2


def get_rag_config() -> RagConfig:
    """
    Configure via env:
    - RAG_DOCS_DIR: path to folder containing guideline docs
    - RAG_PERSIST_DIR: path to Chroma persistence dir
    - RAG_COLLECTION: collection name
    - RAG_TOP_K: retrieved chunks
    """
    project_root = Path(__file__).resolve().parents[2]
    docs_dir = Path(os.getenv("RAG_DOCS_DIR", str(_default_docs_dir(project_root)))).resolve()
    persist_dir = Path(
        os.getenv("RAG_PERSIST_DIR", str(project_root / "data" / "chroma_guidelines"))
    ).resolve()
    collection_name = os.getenv("RAG_COLLECTION", "guidelines")
    top_k = int(os.getenv("RAG_TOP_K", "4"))
    return RagConfig(docs_dir=docs_dir, persist_dir=persist_dir, collection_name=collection_name, top_k=top_k)


def _has_chroma_index(persist_dir: Path) -> bool:
    if not persist_dir.exists():
        return False
    # Chroma typically writes a sqlite db and/or collections/segments folders.
    if (persist_dir / "chroma.sqlite3").exists():
        return True
    if (persist_dir / "index").exists():
        return True
    if any(persist_dir.glob("**/*.sqlite3")):
        return True
    if any(persist_dir.iterdir()):
        return True
    return False


def _iter_guideline_files(docs_dir: Path) -> Iterable[Path]:
    if not docs_dir.exists():
        return []
    exts = {".pdf", ".txt", ".md", ".docx"}
    return sorted([p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def _load_documents(paths: list[Path]):
    docs = []
    for p in paths:
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(p))
            docs.extend(loader.load())
        elif suffix in {".txt", ".md"}:
            loader = TextLoader(str(p), encoding="utf-8")
            docs.extend(loader.load())
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(p))
            docs.extend(loader.load())
    return docs


def _split_documents(docs, *, chunk_size: int, chunk_overlap: int):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def _doc_id(doc) -> str:
    source = str(doc.metadata.get("source", ""))
    page = str(doc.metadata.get("page", ""))
    content = (doc.page_content or "").strip()
    h = hashlib.sha1()
    h.update(source.encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(page.encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(content.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _get_embeddings(*, gemini_api_key: Optional[str]):
    if not gemini_api_key:
        return None
    return GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=gemini_api_key)


def build_or_update_index(
    *,
    config: Optional[RagConfig] = None,
    gemini_api_key: Optional[str] = None,
    rebuild: bool = False,
) -> dict:
    """
    Ingest all guideline documents into a persistent Chroma collection.

    Returns stats: {docs_dir, persist_dir, files, loaded_docs, chunks, upserted}
    """
    config = config or get_rag_config()

    if rebuild and config.persist_dir.exists():
        shutil.rmtree(config.persist_dir)

    files = list(_iter_guideline_files(config.docs_dir))
    if not files:
        return {
            "docs_dir": str(config.docs_dir),
            "persist_dir": str(config.persist_dir),
            "files": 0,
            "loaded_docs": 0,
            "chunks": 0,
            "upserted": 0,
            "note": "No guideline files found (.pdf/.txt/.md/.docx).",
        }

    embeddings = _get_embeddings(gemini_api_key=gemini_api_key)
    if embeddings is None:
        return {
            "docs_dir": str(config.docs_dir),
            "persist_dir": str(config.persist_dir),
            "files": len(files),
            "loaded_docs": 0,
            "chunks": 0,
            "upserted": 0,
            "note": "GEMINI_API_KEY not set; cannot build vector index (embeddings unavailable).",
        }

    raw_docs = _load_documents(files)
    chunks = _split_documents(raw_docs, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)

    from langchain_chroma import Chroma

    config.persist_dir.mkdir(parents=True, exist_ok=True)
    vs = Chroma(
        collection_name=config.collection_name,
        persist_directory=str(config.persist_dir),
        embedding_function=embeddings,
    )

    ids = [_doc_id(d) for d in chunks]
    vs.add_documents(chunks, ids=ids)

    # Persist for older Chroma wrappers; no-op for newer ones.
    persist = getattr(vs, "persist", None)
    if callable(persist):
        persist()

    return {
        "docs_dir": str(config.docs_dir),
        "persist_dir": str(config.persist_dir),
        "files": len(files),
        "loaded_docs": len(raw_docs),
        "chunks": len(chunks),
        "upserted": len(chunks),
        "collection": config.collection_name,
    }


def get_retriever(*, config: Optional[RagConfig] = None, gemini_api_key: Optional[str] = None):
    config = config or get_rag_config()
    if not _has_chroma_index(config.persist_dir):
        return None

    embeddings = _get_embeddings(gemini_api_key=gemini_api_key)
    if embeddings is None:
        return None

    from langchain_chroma import Chroma

    vs = Chroma(
        collection_name=config.collection_name,
        persist_directory=str(config.persist_dir),
        embedding_function=embeddings,
    )
    return vs.as_retriever(search_kwargs={"k": config.top_k})


def rag_answer(*, question: str, gemini_api_key: Optional[str], model: str,config: Optional[RagConfig] = None) -> Optional[str]:
    config = config or get_rag_config()
    retriever = get_retriever(config=config, gemini_api_key=gemini_api_key)
    if retriever is None:
        return None

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key, temperature=0.4)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Anda adalah asisten edukasi gizi anak. Jawab hanya berdasarkan konteks pedoman yang diberikan.\n"
                "Jika konteks tidak cukup, katakan dengan jujur apa yang tidak diketahui dan beri saran aman untuk konsultasi tenaga kesehatan.\n"
                "Output harus Markdown dan praktis.\n\n"
                "KONTEKS PEDOMAN:\n{context}",
            ),
            ("human", "{input}"),
        ]
    )

    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)

    result = chain.invoke({"input": question})
    answer = result.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    return None


def maybe_auto_build_index(*, config: Optional[RagConfig] = None, gemini_api_key: Optional[str] = None) -> None:
    """
    If RAG_AUTO_BUILD=true and index is missing, build it from the docs dir.
    """
    config = config or get_rag_config()
    auto = (os.getenv("RAG_AUTO_BUILD", "false").strip().lower() in {"1", "true", "yes"})
    if not auto:
        return
    if _has_chroma_index(config.persist_dir):
        return
    build_or_update_index(config=config, gemini_api_key=gemini_api_key, rebuild=False)


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Build/update Chroma index from guideline documents.")
    parser.add_argument("--rebuild", action="store_true", help="Delete existing index and rebuild from scratch.")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    stats = build_or_update_index(gemini_api_key=api_key, rebuild=args.rebuild)
    print(stats)


if __name__ == "__main__":
    _main()
