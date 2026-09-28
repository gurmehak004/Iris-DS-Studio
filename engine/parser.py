"""
engine/parser.py
Robust CSV ingestion: handles encoding, delimiters, messy headers.
"""
import chardet
import pandas as pd
import io
import re
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
from typing import Tuple


def detect_encoding(raw_bytes: bytes) -> str:
    result = chardet.detect(raw_bytes)
    return result.get("encoding") or "utf-8"


def detect_delimiter(sample: str) -> str:
    candidates = [",", ";", "\t", "|"]
    counts = {d: sample.count(d) for d in candidates}
    return max(counts, key=counts.get)


def load_csv(file_bytes: bytes) -> Tuple[pd.DataFrame, dict]:
    """
    Load a CSV from raw bytes. Returns (dataframe, meta).
    meta keys: encoding, delimiter, original_shape, cleaned_shape
    """
    encoding = detect_encoding(file_bytes)

    try:
        text = file_bytes.decode(encoding, errors="replace")
    except Exception:
        text = file_bytes.decode("utf-8", errors="replace")

    sample = text[:5000]
    delimiter = detect_delimiter(sample)

    df = pd.read_csv(
        io.StringIO(text),
        sep=delimiter,
        engine="python",
        on_bad_lines="skip",
        dtype_backend="numpy_nullable",
    )

    original_shape = df.shape

    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    # Drop fully empty rows and columns
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    original_rows = df.shape[0]

    # Smart sampling for large datasets (> 50,000 rows) for optimal speed and memory
    is_sampled = False
    if original_rows > 50000:
        df = df.sample(n=50000, random_state=42).reset_index(drop=True)
        is_sampled = True

    meta = {
        "encoding": encoding,
        "delimiter": delimiter,
        "original_rows": original_rows,
        "original_cols": original_shape[1],
        "cleaned_rows": df.shape[0],
        "cleaned_cols": df.shape[1],
        "is_sampled": is_sampled,
    }

    return df, meta


def get_excel_sheet_names(file_bytes: bytes) -> list[str]:
    """Return the worksheet names from an uploaded Excel workbook."""
    with pd.ExcelFile(io.BytesIO(file_bytes)) as workbook:
        return workbook.sheet_names


def load_excel(file_bytes: bytes, sheet_name: str | int = 0) -> Tuple[pd.DataFrame, dict]:
    """Load and normalize one worksheet from an Excel workbook."""
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
    original_shape = df.shape
    df.columns = [str(column).strip() for column in df.columns]
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    original_rows = len(df)
    is_sampled = original_rows > 50000
    if is_sampled:
        df = df.sample(n=50000, random_state=42).reset_index(drop=True)

    return df, {
        "encoding": "Excel workbook",
        "delimiter": None,
        "sheet_name": sheet_name,
        "original_rows": original_rows,
        "original_cols": original_shape[1],
        "cleaned_rows": len(df),
        "cleaned_cols": len(df.columns),
        "is_sampled": is_sampled,
    }


def google_sheet_csv_url(sheet_url: str) -> str:
    """Convert a Google Sheets sharing URL to its CSV export URL."""
    parsed = urlparse(sheet_url.strip())
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", parsed.path)
    if parsed.netloc not in {"docs.google.com", "www.docs.google.com"} or not match:
        raise ValueError("Enter a valid Google Sheets URL.")

    query = parse_qs(parsed.query)
    fragment = parse_qs(parsed.fragment)
    gid = (query.get("gid") or fragment.get("gid") or ["0"])[0]
    return f"https://docs.google.com/spreadsheets/d/{match.group(1)}/export?format=csv&gid={gid}"


def load_google_sheet(sheet_url: str) -> Tuple[pd.DataFrame, dict]:
    """Load a Google Sheet shared for link access through its CSV export endpoint."""
    export_url = google_sheet_csv_url(sheet_url)
    request = Request(export_url, headers={"User-Agent": "IrisDataStudio/1.0"})
    with urlopen(request, timeout=30) as response:
        file_bytes = response.read()

    df, meta = load_csv(file_bytes)
    meta["encoding"] = "Google Sheets"
    meta["source_url"] = sheet_url.strip()
    return df, meta
