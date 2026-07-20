from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
AUDIT_DIR = DATA_DIR / "audit"

def ensure_output_directories() -> None:
    for path in (PROCESSED_DIR, AUDIT_DIR, ROOT / "results" / "tables", ROOT / "results" / "figures"):
        path.mkdir(parents=True, exist_ok=True)
