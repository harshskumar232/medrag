import json
import csv
import io
from pathlib import Path


def parse_document(file_path: Path, filename: str) -> str:
    """Parse any supported document type and return plain text."""
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        return _parse_pdf(file_path)
    elif ext in ("txt", "md"):
        return _parse_text(file_path)
    elif ext == "csv":
        return _parse_csv(file_path)
    elif ext == "json":
        return _parse_json(file_path)
    else:
        return _parse_text(file_path)


def _parse_pdf(file_path: Path) -> str:
    """Extract text from PDF using PyMuPDF (fitz) — handles scanned + digital PDFs."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(file_path))
        pages = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text")
            if text.strip():
                pages.append(f"[Page {page_num}]\n{text.strip()}")
        doc.close()
        full_text = "\n\n".join(pages)
        if not full_text.strip():
            raise ValueError("No text extracted — PDF may be image-based")
        return full_text
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")
    except Exception as e:
        raise RuntimeError(f"PDF parsing failed: {e}")


def _parse_text(file_path: Path) -> str:
    """Read plain text or markdown files."""
    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            return file_path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode file with any known encoding")


def _parse_csv(file_path: Path) -> str:
    """Convert CSV to readable text format."""
    try:
        import pandas as pd
        df = pd.read_csv(file_path)
        lines = [f"CSV Document: {file_path.name}",
                 f"Rows: {len(df)}, Columns: {len(df.columns)}",
                 f"Columns: {', '.join(df.columns.tolist())}",
                 ""]
        # Convert rows to readable text
        for idx, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if str(val) != "nan"])
            lines.append(f"Row {idx+1}: {row_text}")
        return "\n".join(lines)
    except Exception as e:
        # Fallback: raw CSV read
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return f"CSV Content:\n{content}"


def _parse_json(file_path: Path) -> str:
    """Convert JSON to readable text format."""
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as e:
        return file_path.read_text(encoding="utf-8", errors="replace")


def chunk_text(text: str, doc_name: str, doc_id: str,
               chunk_size: int = 450, overlap: int = 80) -> list[dict]:
    """
    Split text into overlapping chunks for indexing.
    Returns list of chunk dicts with id, text, doc_name, doc_id, chunk_index.
    """
    import hashlib
    chunks = []
    text = text.strip()
    i = 0
    chunk_index = 0

    while i < len(text):
        chunk_text = text[i:i + chunk_size].strip()
        if len(chunk_text) > 30:  # Skip very small chunks
            chunk_id = hashlib.md5(f"{doc_id}_{chunk_index}".encode()).hexdigest()
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "doc_name": doc_name,
                "doc_id": doc_id,
                "chunk_index": chunk_index,
            })
            chunk_index += 1
        i += chunk_size - overlap

    return chunks
