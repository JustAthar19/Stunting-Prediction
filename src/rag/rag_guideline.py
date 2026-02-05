from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import glob

from dotenv import load_dotenv
load_dotenv()


@dataclass(frozen=True)
class RagConfig:
    docs_dir: Path
    persist_dir: Path
    collection_name: str = "guidelines"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4

def get_rag_config() -> RagConfig:
    project_root = Path(__file__).resolve().parents[2]
    docs_dir = "../data/guideline/"
    persist_dir = Path(os.getenv("RAG_PERSIST_DIR", str(project_root / "data" / "chroma_guidelines"))).resolve()
    collection_name = os.getenv("RAG_COLLECTION", "guidelines")
    top_k = int(os.getenv("RAG_TOP_K", "4"))
    return RagConfig(docs_dir=docs_dir, persist_dir=persist_dir, collection_name=collection_name, top_k=top_k)

# def load_documents(docs_dir: Path) -> list[Document]:
#     from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
#     docs = []
#     for p in docs_dir.glob("*"):
#         if p.is_file():
#             if p.suffix == ".pdf":
#                 loader = PyPDFLoader(str(p))
#                 docs.extend(loader.load())
#             elif p.suffix == ".txt":
#                 loader = TextLoader(str(p), encoding="utf-8")
#                 docs.extend(loader.load())
#             elif p.suffix == ".docx":
#                 loader = Docx2txtLoader(str(p))
#                 docs.extend(loader.load())
#     return docs

print(Path(__file__))