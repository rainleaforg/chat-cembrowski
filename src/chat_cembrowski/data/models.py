from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Paper:
    """
    Represents a research paper.

    source_file is optional: a catalog-only Paper records a work that is known
    to exist but whose PDF has not been acquired yet. Publisher bot protection
    makes that the common case, so the record has to be able to exist before
    the blob does — which is also the shape needed once blobs move to S3.
    Anything that reads source_file must treat "" as "no PDF yet".
    """
    id: str
    source_file: str = ""
    title: Optional[str] = None
    authors: Optional[list[str]] = None
    year: Optional[int] = None
    publication: Optional[str] = None
    first_page_number: Optional[int] = None
    scholar_link: Optional[str] = None   # Google Scholar citation page
    pdf_url: Optional[str] = None        # public PDF, when a lookup found one
    cited_by: Optional[int] = None       # citation count, for sourcing priority
    processed: bool = False
    text: str = ""

    @property
    def has_pdf(self) -> bool:
        """True when a source file has been acquired for this paper."""
        return bool(self.source_file)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_file": self.source_file,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "publication": self.publication,
            "first_page_number": self.first_page_number,
            "scholar_link": self.scholar_link,
            "pdf_url": self.pdf_url,
            "cited_by": self.cited_by,
            "processed": self.processed,
            "text": self.text,
        }
    
@dataclass
class Chunk:
    """Represents a chunk of text from a paper, along with metadata for RAG."""
    id: str
    text: str
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "payload": self.payload,
    }

@dataclass
class Document:
    """
    Represents a miscellaneous context document (txt, docx, code file).

    Unlike papers, documents are expected to be edited in place. content_hash
    is what makes that detectable: doc_ingestion re-extracts when the hash of
    the file's text no longer matches, and clears processed so vectordb
    re-indexes it. Without it, an edited file is indistinguishable from an
    unchanged one and is skipped forever.
    """
    id: str
    title: str
    source_file: str    # filename in data/docs/
    file_type: str      # "txt", "docx", "py", etc.
    text: str = ""
    content_hash: Optional[str] = None  # sha256 of text; detects edits
    processed: bool = False
    kind: str = "document"        # "site" | "document" — see chat-cembrowski's kind tiers
    site_path: Optional[str] = None  # required when kind == "site"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "source_file": self.source_file,
            "file_type": self.file_type,
            "text": self.text,
            "content_hash": self.content_hash,
            "processed": self.processed,
            "kind": self.kind,
            "site_path": self.site_path,
        }


@dataclass
class ImageRecord:
    """Represents an image extracted from a paper, with spatial and caption metadata."""
    id: str
    paper_id: str
    source_file: str                          # filename in data/images/
    page: int                                 # journal page number (offset applied)
    bbox: tuple[float, float, float, float]   # (x0, y0, x1, y1) in PDF points
    caption: Optional[str] = None
    image_type: Optional[str] = None          # "figure", "table", "chart", "image"
    title: Optional[str] = None
    authors: Optional[list[str]] = None
    year: Optional[int] = None
    publication: Optional[str] = None
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "paper_id": self.paper_id,
            "source_file": self.source_file,
            "page": self.page,
            "bbox": list(self.bbox),
            "caption": self.caption,
            "image_type": self.image_type,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "publication": self.publication,
        }
