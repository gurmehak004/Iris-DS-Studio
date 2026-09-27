"""
engine/exporter.py
Generates a downloadable PDF report and a cleaned CSV.
Dark-themed, high-contrast, professionally formatted.
"""
import pandas as pd
import io
from fpdf import FPDF
from typing import Dict, Any
import datetime


def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = (s.replace("\u2194", "<->").replace("\u2014", "-").replace("\u2018", "'")
          .replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
          .replace("\u2022", "*").replace("\u2705", "[OK]").replace("\U0001f338", "[Iris]")
          .replace("\u274c", "[X]").replace("\u26a0\ufe0f", "[!]").replace("\U0001f4a1", "[Tip]")
          .replace("\U0001f4ca", "[Chart]").replace("\U0001f50d", "[Search]").replace("\U0001f916", "[ML]"))
    return s.encode("latin-1", "ignore").decode("latin-1")


class IrisPDF(FPDF):
    """Dark-themed, high-contrast PDF report."""

    BG_R, BG_G, BG_B             = 15, 15, 26
    PAPER_R, PAPER_G, PAPER_B    = 26, 26, 46
    PURPLE_R, PURPLE_G, PURPLE_B = 167, 139, 250
    TEAL_R, TEAL_G, TEAL_B       = 45, 212, 191
    TEXT_R, TEXT_G, TEXT_B       = 226, 232, 240
    MUTED_R, MUTED_G, MUTED_B   = 148, 163, 184

    def header(self):
        self.set_fill_color(self.PAPER_R, self.PAPER_G, self.PAPER_B)
        self.rect(0, 0, 210, 20, "F")
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(self.PURPLE_R, self.PURPLE_G, self.PURPLE_B)
        self.set_y(5)
        self.cell(0, 10, "Iris Data Science Studio  -  EDA Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(self.PURPLE_R, self.PURPLE_G, self.PURPLE_B)
        self.line(10, 20, 200, 20)
        self.ln(6)

    def footer(self):
        self.set_y(-14)
        self.set_fill_color(self.PAPER_R, self.PAPER_G, self.PAPER_B)
        self.rect(0, self.get_y() - 2, 210, 20, "F")
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(self.MUTED_R, self.MUTED_G, self.MUTED_B)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def page_bg(self):
        self.set_fill_color(self.BG_R, self.BG_G, self.BG_B)
        self.rect(0, 0, 210, 297, "F")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(self.TEAL_R, self.TEAL_G, self.TEAL_B)
        self.ln(4)
        self.cell(0, 9, _clean_text(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(self.TEAL_R, self.TEAL_G, self.TEAL_B)
        self.line(self.get_x(), self.get_y(), self.get_x() + 180, self.get_y())
        self.ln(3)
        self.set_text_color(self.TEXT_R, self.TEXT_G, self.TEXT_B)

    def sub_title(self, title: str):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(self.PURPLE_R, self.PURPLE_G, self.PURPLE_B)
        self.ln(3)
        self.cell(0, 7, _clean_text(title), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(self.TEXT_R, self.TEXT_G, self.TEXT_B)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(self.TEXT_R, self.TEXT_G, self.TEXT_B)
        self.multi_cell(0, 5.5, _clean_text(text))
        self.ln(1)

    def muted_text(self, text: str):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(self.MUTED_R, self.MUTED_G, self.MUTED_B)
        self.multi_cell(0, 5, _clean_text(text))
        self.ln(1)
        self.set_text_color(self.TEXT_R, self.TEXT_G, self.TEXT_B)

    def kpi_row(self, items: list):
        n = len(items)
        col_w = 170 // max(n, 1)
        self.set_fill_color(self.PAPER_R, self.PAPER_G, self.PAPER_B)
        for label, value in items:
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(self.TEAL_R, self.TEAL_G, self.TEAL_B)
            self.cell(col_w, 9, _clean_text(str(value)), fill=True)
        self.ln()
        for label, value in items:
            self.set_font("Helvetica", "", 7)
            self.set_text_color(self.MUTED_R, self.MUTED_G, self.MUTED_B)
            self.cell(col_w, 5, _clean_text(str(label)))
        self.ln(8)
        self.set_text_color(self.TEXT_R, self.TEXT_G, self.TEXT_B)

    def small_table(self, df: pd.DataFrame, max_rows: int = 20):
        if df is None or len(df) == 0:
            return
        self.set_font("Helvetica", "B", 7)
        self.set_fill_color(self.PAPER_R, self.PAPER_G, self.PAPER_B)
        self.set_text_color(self.PURPLE_R, self.PURPLE_G, self.PURPLE_B)
        col_w = min(170 // max(len(df.columns), 1), 55)
        for col in df.columns:
            self.cell(col_w, 6, _clean_text(str(col)[:16]), border=1, fill=True)
        self.ln()
        self.set_font("Helvetica", "", 7)
        for i, (_, row) in enumerate(df.head(max_rows).iterrows()):
            if i % 2 == 0:
                self.set_fill_color(26, 26, 46)
            else:
                self.set_fill_color(20, 20, 36)
            self.set_text_color(self.TEXT_R, self.TEXT_G, self.TEXT_B)
            for val in row:
                self.cell(col_w, 5, _clean_text(str(val)[:16]), border=0, fill=True)
            self.ln()
        self.ln(3)
        self.set_text_color(self.TEXT_R, self.TEXT_G, self.TEXT_B)


def build_pdf_report(meta: Dict, col_types: Dict, quality: Dict,
                     col_stats: Dict, corr: Dict, ml_results: Dict,
                     iris_says: Dict) -> bytes:
    pdf = IrisPDF()
    pdf.set_auto_page_break(auto=True, margin=18)

    # ── Cover Page ──────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.page_bg()
    pdf.set_y(55)
    pdf.set_font("Helvetica", "B", 30)
    pdf.set_text_color(167, 139, 250)
    pdf.cell(0, 14, "Iris Studio", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(45, 212, 191)
    pdf.cell(0, 8, "Exploratory Data Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(18)

    pdf.set_fill_color(26, 26, 46)
    pdf.set_x(25)
    orig_fn = meta.get("original_filename", "dataset.csv")
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(226, 232, 240)
    pdf.cell(160, 10, _clean_text(f"Dataset: {orig_fn}"), align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(25)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(148, 163, 184)
    info_line = (f"{meta['cleaned_rows']:,} rows x {meta['cleaned_cols']} columns  |  "
                 f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    pdf.cell(160, 8, _clean_text(info_line), align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(28)

    pdf.set_x(25)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(167, 139, 250)
    pdf.cell(0, 7, "Column Types Detected:", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    type_counts: Dict[str, int] = {}
    for v in col_types.values():
        type_counts[v] = type_counts.get(v, 0) + 1
    for t, cnt in type_counts.items():
        pdf.set_x(35)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(226, 232, 240)
        pdf.cell(0, 6, _clean_text(f"{t.upper()}: {cnt} column(s)"), new_x="LMARGIN", new_y="NEXT")

    # ── Overview ─────────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.page_bg()
    pdf.section_title("1. Dataset Overview")
    mem_val = meta.get("memory_mb", quality.get("memory_mb", "0.1"))
    dup_val = quality.get("duplicates", {}).get("duplicate_rows", 0) if isinstance(quality.get("duplicates"), dict) else 0
    pdf.kpi_row([
        ("Rows", f"{meta.get('cleaned_rows', 0):,}"),
        ("Columns", meta.get("cleaned_cols", 0)),
        ("Memory", f"{mem_val} MB"),
        ("Duplicates", dup_val),
    ])

    pdf.body_text(f"Encoding: {meta['encoding']}   Delimiter: '{meta['delimiter']}'")
    if iris_says.get("overview"):
        pdf.sub_title("Iris Analysis:")
        pdf.body_text(iris_says["overview"])
    pdf.sub_title("Column Types")
    for ctype in set(col_types.values()):
        cols = [c for c, t in col_types.items() if t == ctype]
        pdf.body_text(f"  {ctype.upper()} ({len(cols)}):  {', '.join(cols[:12])}{'...' if len(cols) > 12 else ''}")

    # ── Data Quality ─────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.page_bg()
    dup = quality.get("duplicates", {}) if isinstance(quality.get("duplicates"), dict) else {}
    dup_rows = dup.get("duplicate_rows", 0)
    dup_pct = dup.get("duplicate_pct", 0.0)
    pdf.body_text(f"Duplicate rows: {dup_rows:,} ({dup_pct}%)")

    miss_df = quality.get("missing", pd.DataFrame())
    if isinstance(miss_df, pd.DataFrame) and len(miss_df) > 0:
        pdf.sub_title("Missing Values by Column")
        pdf.small_table(miss_df)
    else:
        pdf.body_text("[OK] No missing values - dataset is complete!")
    out_df = quality.get("outliers", pd.DataFrame())
    if isinstance(out_df, pd.DataFrame) and len(out_df) > 0:
        pdf.sub_title("Outlier Summary (IQR Method)")
        pdf.small_table(out_df.head(15))
    skew_df = quality.get("skewness", pd.DataFrame())
    if isinstance(skew_df, pd.DataFrame) and len(skew_df) > 0:
        pdf.sub_title("Skewness and Kurtosis")
        pdf.small_table(skew_df)
    if iris_says.get("quality"):
        pdf.sub_title("Iris Analysis:")
        pdf.body_text(iris_says["quality"])

    # ── Exploratory Data Analysis ──────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.page_bg()
    pdf.section_title("3. Exploratory Data Analysis")
    
    ENGINEERED_SUFFIXES = ("_scaled", "_minmax", "_encoded", "_binary", "_log", "_log1p", "_zscore")
    
    numeric_stats = {c: s for c, s in col_stats.items() if s.get("col_type") == "numeric" and not any(c.endswith(sfx) for sfx in ENGINEERED_SUFFIXES)}
    if numeric_stats:
        pdf.sub_title("Numeric Feature Distributions")
        for col, s in list(numeric_stats.items())[:15]:
            pdf.set_text_color(226, 232, 240)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(50, 6, _clean_text(col), new_x="RIGHT", new_y="TOP")
            
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(148, 163, 184)
            stat_text = f"Mean: {s.get('mean','?')} | Median: {s.get('median','?')} | Std: {s.get('std','?')} | Skew: {s.get('skewness','?')}"
            pdf.cell(0, 6, _clean_text(stat_text), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)
    cat_stats = {c: s for c, s in col_stats.items() if s.get("col_type") in ("categorical","boolean") and not any(c.endswith(sfx) for sfx in ENGINEERED_SUFFIXES)}
    if cat_stats:
        pdf.sub_title("Categorical Distributions")
        for col, s in list(cat_stats.items())[:10]:
            pdf.set_text_color(226, 232, 240)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(50, 6, _clean_text(col), new_x="RIGHT", new_y="TOP")
            
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(148, 163, 184)
            stat_text = f"{s.get('unique','?')} uniques | Top: '{s.get('top_value','?')}' ({s.get('top_freq_pct','?')}%)"
            pdf.cell(0, 6, _clean_text(stat_text), new_x="LMARGIN", new_y="NEXT")

    # ── Correlations ──────────────────────────────────────────────────────────────
    if corr and corr.get("top_pairs"):
        pdf.add_page()
        pdf.page_bg()
        pdf.section_title("4. Top Correlations")
        pdf.muted_text("Pearson r: +1 = perfect positive, -1 = perfect negative, 0 = no linear relationship.")
        for p in corr["top_pairs"][:10]:
            strength = "Strong" if abs(p["pearson_r"]) > 0.7 else "Moderate" if abs(p["pearson_r"]) > 0.4 else "Weak"
            pdf.body_text(f"  {p['col1']} <-> {p['col2']}: r = {p['pearson_r']}  ({strength})")
        if iris_says.get("correlations"):
            pdf.sub_title("Iris Analysis:")
            pdf.body_text(iris_says["correlations"])

    # ── ML Results ────────────────────────────────────────────────────────────────
    if ml_results and ml_results.get("results"):
        pdf.add_page()
        pdf.page_bg()
        pdf.section_title("5. Machine Learning Results")
        pdf.kpi_row([
            ("Task", ml_results.get("task", "-")),
            ("Best Model", ml_results.get("best_model", "-")),
            ("Target", ml_results.get("target", "-")),
        ])
        for model, metrics in ml_results.get("results", {}).items():
            if isinstance(metrics, dict):
                m_str = "  ".join(f"{k}={v}" for k, v in metrics.items())
            else:
                m_str = str(metrics)
            pdf.body_text(f"  {model}: {m_str}")
        if iris_says.get("ml"):
            pdf.sub_title("Iris Analysis:")
            pdf.body_text(iris_says["ml"])

    return bytes(pdf.output())


def build_cleaned_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
