import json
import logging
from pathlib import Path
from typing import Iterator, Optional

from .models import Paper, ImageRecord, Document

logger = logging.getLogger(__name__)

def save_paper(paper: Paper, output_dir: Optional[str | Path] = None) -> Optional[Path]:
    """
    Save a single Paper object to a JSON file.

    Args:
        paper: Paper object to save
        output_dir: Directory to save the JSON file (default: data/json)

    Returns:
        Path to the saved JSON file or None if saving fails
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[3] / "data" / "json"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    filename = paper.id + ".json"
    filepath = output_dir / filename

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(paper.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save {filepath}: {e}")
        return None

def save_papers_to_json(papers: list[Paper], output_dir: Optional[str | Path] = None) -> Path:
    """
    Save Paper objects to JSON files.

    Args:
        papers: List of Paper objects to save
        output_dir: Directory to save JSON files (default: data/json)

    Returns:
        Path to the output directory
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[3] / "data" / "json"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for paper in papers:
        filename = paper.id + ".json"
        filepath = output_dir / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(paper.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save {filepath}: {e}")

    logger.info(f"Saved {len(papers)} papers to {output_dir}")
    return output_dir

def load_paper(json_file: str | Path) -> Optional[Paper]:
    """
    Load a single Paper object from a JSON file.

    Args:
        json_file: Path to the JSON file
    Returns:
        Paper object or None if loading fails
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        paper = Paper(
            source_file=data.get("source_file", ""),
            id=data["id"],
            title=data.get("title"),
            authors=data.get("authors"),
            year=data.get("year"),
            publication=data.get("publication"),
            first_page_number=data.get("first_page_number"),
            scholar_link=data.get("scholar_link"),
            pdf_url=data.get("pdf_url"),
            cited_by=data.get("cited_by"),
            processed=data.get("processed", False),
            text=data.get("text", ""),
        )
        logger.info(f"Loaded: {Path(json_file).name}")
        return paper
    except Exception as e:
        logger.error(f"Failed to load {json_file}: {e}")
        return None


def load_papers_from_json(json_dir: Optional[str | Path] = None) -> Iterator[Paper]:
    """
    Load Paper objects from JSON files.

    Args:
        json_dir: Directory containing JSON files (default: data/json)

    Returns:
        List of Paper objects
    """
    if json_dir is None:
        json_dir = Path(__file__).resolve().parents[3] / "data" / "json"
    else:
        json_dir = Path(json_dir)

    if not json_dir.exists():
        logger.warning(f"JSON directory not found: {json_dir}")
        return []

    json_files = json_dir.glob("*.json")

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            paper = Paper(
                source_file=data.get("source_file", ""),
                text=data.get("text", ""),
                id=data["id"],
                title=data.get("title"),
                authors=data.get("authors"),
                year=data.get("year"),
                publication=data.get("publication"),
                first_page_number=data.get("first_page_number"),
                scholar_link=data.get("scholar_link"),
                pdf_url=data.get("pdf_url"),
                cited_by=data.get("cited_by"),
                processed=data.get("processed", False),
            )
            logger.info(f"Loaded: {json_file.name}")
            yield paper
        except Exception as e:
            logger.error(f"Failed to load {json_file}: {e}")


def save_document(doc: Document, output_dir: Optional[str | Path] = None) -> Optional[Path]:
    """Save a single Document object to a JSON file in output_dir (default: data/doc_json)."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[3] / "data" / "doc_json"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{doc.id}.json"

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(doc.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save {filepath}: {e}")
        return None


def load_document(json_file: str | Path) -> Optional[Document]:
    """Load a single Document object from a JSON file."""
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Document(
            id=data["id"],
            title=data["title"],
            source_file=data["source_file"],
            file_type=data["file_type"],
            text=data.get("text", ""),
            content_hash=data.get("content_hash"),
            processed=data.get("processed", False),
            kind=data.get("kind", "document"),
            site_path=data.get("site_path"),
        )
    except Exception as e:
        logger.error(f"Failed to load {json_file}: {e}")
        return None


def load_documents_from_json(json_dir: Optional[str | Path] = None) -> Iterator[Document]:
    """Load all Document objects from JSON files in json_dir (default: data/doc_json)."""
    if json_dir is None:
        json_dir = Path(__file__).resolve().parents[3] / "data" / "doc_json"
    else:
        json_dir = Path(json_dir)

    if not json_dir.exists():
        logger.warning(f"Document JSON directory not found: {json_dir}")
        return

    for json_file in json_dir.glob("*.json"):
        doc = load_document(json_file)
        if doc is not None:
            yield doc


def load_image_records_for_paper(
    paper_id: str,
    image_json_dir: Optional[str | Path] = None,
) -> list[ImageRecord]:
    """
    Load all ImageRecord JSONs belonging to paper_id from image_json_dir.

    Args:
        paper_id: ID of the paper whose images to load
        image_json_dir: Directory containing image JSON files (default: data/image_json)

    Returns:
        List of ImageRecord objects for the given paper
    """
    if image_json_dir is None:
        image_json_dir = Path(__file__).resolve().parents[3] / "data" / "image_json"
    else:
        image_json_dir = Path(image_json_dir)

    if not image_json_dir.exists():
        logger.warning(f"Image JSON directory not found: {image_json_dir}")
        return []

    records: list[ImageRecord] = []
    for json_file in image_json_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("paper_id") != paper_id:
                continue
            records.append(
                ImageRecord(
                    id=data["id"],
                    paper_id=data["paper_id"],
                    source_file=data["source_file"],
                    page=data["page"],
                    bbox=tuple(data["bbox"]),
                    caption=data.get("caption"),
                    image_type=data.get("image_type"),
                    title=data.get("title"),
                    authors=data.get("authors"),
                    year=data.get("year"),
                    publication=data.get("publication"),
                )
            )
            logger.info(f"Loaded image record: {json_file.name}")
        except Exception as e:
            logger.warning(f"Failed to load image record {json_file.name}: {e}")
    return records
