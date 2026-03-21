# data_loader.py
"""
DataLoader — loads customer data from multiple file formats.

Supported formats:
  - CSV  (.csv)  — auto-detects delimiter (, ; | tab) and encoding
  - Excel (.xlsx, .xls) — auto-detects sheet, picks the first non-empty one
  - TSV  (.tsv)  — tab-separated
"""

from pathlib import Path
import chardet
import pandas as pd


# Supported extensions → internal format tag
_SUPPORTED = {
    ".csv":  "csv",
    ".tsv":  "tsv",
    ".xlsx": "excel",
    ".xls":  "excel",
}

# Delimiters to probe when sniffing CSV files
_DELIMITERS = [",", ";", "|", "\t"]


class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.data: pd.DataFrame | None = None

        # Populated after load() — useful for debugging and the dashboard
        self.detected_format: str | None = None
        self.detected_encoding: str | None = None
        self.detected_delimiter: str | None = None
        self.detected_sheet: str | None = None
        self.n_rows: int | None = None
        self.n_cols: int | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """
        Auto-detect the file format from the extension and load accordingly.
        Returns a pandas DataFrame and caches it in self.data.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        ext = self.file_path.suffix.lower()
        if ext not in _SUPPORTED:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(_SUPPORTED.keys())}"
            )

        fmt = _SUPPORTED[ext]
        self.detected_format = fmt

        if fmt == "csv":
            df = self._load_csv()
        elif fmt == "tsv":
            df = self._load_tsv()
        else:
            df = self._load_excel()

        self.data = df
        self.n_rows, self.n_cols = df.shape

        print(f"[DataLoader] Loaded {self.n_rows:,} rows x {self.n_cols} columns")
        print(
            f"             format={self.detected_format}"
            + (f"  encoding={self.detected_encoding}" if self.detected_encoding else "")
            + (f"  delimiter='{self.detected_delimiter}'" if self.detected_delimiter else "")
            + (f"  sheet='{self.detected_sheet}'" if self.detected_sheet else "")
        )

        return df

    def summary(self) -> str:
        """Return a human-readable summary of what was detected and loaded."""
        if self.data is None:
            return "No file loaded yet. Call load() first."
        lines = [f"File     : {self.file_path}", f"Format   : {self.detected_format}"]
        if self.detected_encoding:
            lines.append(f"Encoding : {self.detected_encoding}")
        if self.detected_delimiter:
            lines.append(f"Delimiter: '{self.detected_delimiter}'")
        if self.detected_sheet:
            lines.append(f"Sheet    : {self.detected_sheet}")
        lines.append(f"Shape    : {self.n_rows:,} rows x {self.n_cols} columns")
        return "\n".join(lines)

    # ── Private loaders ───────────────────────────────────────────────────────

    def _detect_encoding(self) -> str:
        """
        Read the first 50 KB and use chardet to detect encoding.
        Falls back to utf-8 if detection confidence is below 70%.
        """
        with open(self.file_path, "rb") as f:
            raw = f.read(50_000)
        result = chardet.detect(raw)
        encoding = result.get("encoding") or "utf-8"
        confidence = result.get("confidence", 0)
        if confidence < 0.7:
            encoding = "utf-8"
        self.detected_encoding = encoding
        return encoding

    def _detect_delimiter(self, encoding: str) -> str:
        """
        Read the first 5 lines and score each candidate delimiter by how
        consistently it appears across lines. Highest consistent count wins.
        """
        with open(self.file_path, "r", encoding=encoding, errors="replace") as f:
            lines = [f.readline() for _ in range(5)]
        lines = [l for l in lines if l.strip()]

        best_delim = ","
        best_score = -1

        for delim in _DELIMITERS:
            counts = [line.count(delim) for line in lines]
            if max(counts, default=0) == 0:
                continue
            mean = sum(counts) / len(counts)
            variance = sum((c - mean) ** 2 for c in counts) / len(counts)
            score = mean - variance   # reward consistency, penalise variance
            if score > best_score:
                best_score = score
                best_delim = delim

        self.detected_delimiter = best_delim
        return best_delim

    def _load_csv(self) -> pd.DataFrame:
        encoding = self._detect_encoding()
        delimiter = self._detect_delimiter(encoding)
        try:
            df = pd.read_csv(
                self.file_path,
                sep=delimiter,
                encoding=encoding,
                on_bad_lines="warn",
            )
        except UnicodeDecodeError:
            # latin-1 reads every byte without error — last resort
            self.detected_encoding = "latin-1"
            df = pd.read_csv(
                self.file_path,
                sep=delimiter,
                encoding="latin-1",
                on_bad_lines="warn",
            )
        return df

    def _load_tsv(self) -> pd.DataFrame:
        encoding = self._detect_encoding()
        self.detected_delimiter = "\t"
        return pd.read_csv(
            self.file_path,
            sep="\t",
            encoding=encoding,
            on_bad_lines="warn",
        )

    def _load_excel(self) -> pd.DataFrame:
        xl = pd.ExcelFile(self.file_path)
        # Pick the first sheet that actually has data
        chosen = xl.sheet_names[0]
        for name in xl.sheet_names:
            preview = xl.parse(name, nrows=2)
            if not preview.empty:
                chosen = name
                break
        self.detected_sheet = chosen
        return xl.parse(chosen)