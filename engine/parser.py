"""
engine/parser.py
Robust CSV ingestion: handles encoding, delimiters, messy headers.
"""
import chardet
import pandas as pd
import io
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
