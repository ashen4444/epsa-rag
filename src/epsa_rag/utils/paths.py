from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]

CONFIG_DIR = BASE_DIR / "configs"

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"
BM25_INDEX_DIR = INDEX_DIR / "bm25"
DENSE_INDEX_DIR = INDEX_DIR / "dense"
RESULTS_DIR = DATA_DIR / "results"

OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
TABLES_DIR = OUTPUTS_DIR / "tables"
REPORTS_DIR = OUTPUTS_DIR / "reports"

DOCS_DIR = BASE_DIR / "docs"
ARCHITECTURE_DOCS_DIR = DOCS_DIR / "architecture"
RESEARCH_NOTES_DIR = DOCS_DIR / "research_notes"


def project_path(*parts: str) -> Path:
    """
    Build an absolute path relative to the project root.

    Example:
        project_path("data", "processed", "hotpotqa_paragraph_chunks.jsonl")
    """
    return BASE_DIR.joinpath(*parts)


def ensure_directory(path: Path) -> Path:
    """
    Create a directory if it does not already exist and return the path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path