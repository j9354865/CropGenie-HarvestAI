"""
Combine Pakistani mandi price CSVs — CropGenie HarvestAI
=========================================================

Reads the per-crop CSV files in ``data/pakistan_market_raw/`` (Pakistani
mandi/market-committee price history, ~2007–2022) and combines them into
one unified dataset:

    data/pakistan_market_price_data.csv

It also builds a compact summary artifact used by the Best Market API:

    models/pakistan_market_best.joblib
    (per-crop, per-market recent average prices — loads instantly
     at runtime instead of re-parsing the big CSV)

Size budget
-----------
The raw folder (318 MB) exceeds the ~300 MB project budget, so files are
included by priority until the combined CSV reaches a **280 MB budget**:

    Priority 1 — staple / most-traded crops (wheat, paddy, maize, ...)
    Priority 2 — secondary crops (lemon, grapes, jaggery, ...)
    Priority 3 — everything else (alphabetical)

The budget is applied to the EXACT size of each file's cleaned rows in
the final CSV (not the raw file size), so the written output is
guaranteed to stay under the limit.  Duplicate files (identical content,
e.g. Banana(Dozens).csv vs Banana.csv) are skipped.

Units (IMPORTANT — verified against real market rates)
-----------------------------------------------------
The raw ``Price`` column unit differs per crop:

    • Most crops        → PKR per maund  (1 maund = 40 kg)
    • Masoor / Mash     → PKR per quintal (100 kg)  [verified via price history]
    • Kinnow, Orange,   → PKR per 100 pieces
      Grapefruit, Musambi, Feutral
    • Banana            → PKR per dozen

The script stamps every row with a ``unit`` column so downstream code
can convert correctly.  Weight-based units are normalised to
Rs/maund at display time (quintal × 0.40).

Column mapping (all raw files share the same layout)
----------------------------------------------------
    City   → market     (mandi / market-committee town)
    Date   → date       (YYYY-MM-DD)
    Crop   → crop       (variety level, e.g. "Paddy Basmati")
    Price  → price      (whole PKR — unit varies, see above)

    commodity → derived from the crop name (e.g. "Paddy Basmati" → "paddy").
                NOTE: filenames are NOT reliable (e.g. "masoor pulse.csv"
                actually contains Mash data, "Peach.csv" also contains
                Papaya rows), so we map from the crop column instead.

Usage:
    python training/combine_pakistan_market_data.py
"""

import hashlib
import io
import logging
import os
import sys
from datetime import timedelta

# Windows consoles default to cp1252 and choke on Urdu crop names —
# force UTF-8 output.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import joblib
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "pakistan_market_raw")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "pakistan_market_price_data.csv")
OUTPUT_ARTIFACT = os.path.join(PROJECT_ROOT, "models", "pakistan_market_best.joblib")

# ── Budget ───────────────────────────────────────────────────────────
# 300 MB project budget → cap the COMBINED CSV at 280 MB (safety margin)
SIZE_BUDGET_BYTES = 280 * 1024 * 1024

# ── Crop priorities (from the task requirements) ─────────────────────
# Matched against the FILENAME (lowercase, prefix match, longest keyword first).

PRIORITY_1 = [
    "wheat", "paddy", "maize", "cotton", "sugarcane", "potato", "onion",
    "tomato", "gram", "masoor", "moong", "mango", "banana", "guava",
    "orange", "kinnow", "carrot", "garlic", "green chilli", "groundnut",
]

PRIORITY_2 = [
    "lemon", "grapes", "grapefruit", "jaman", "jujube", "jaggery",
    "green onion", "mustard", "ginger", "millet",
]

# ── Crop → commodity group mapping ───────────────────────────────────
# Lowercase substring rules, first match wins.  This handles the mixed
# files (e.g. "Peach.csv" also contains Papaya rows, "millet melon.csv"
# contains Millet + Melon + Mint).

CROP_TO_COMMODITY = [
    ("paddy", "paddy"),
    ("mango", "mango"),
    ("masoor", "masoor"),
    ("mash", "mash"),
    ("gram", "gram"),
    ("moong", "moong"),
    ("mongray", "moong"),          # mongray = split moong variety
    ("kinnow", "kinnow"),
    ("orange", "orange"),
    ("grapefruit", "grapefruit"),
    ("musambi", "musambi"),
    ("feutral", "feutral"),        # Feutrel's Early citrus
    ("mustard", "mustard"),
    ("melon", "melon"),
    ("millet", "millet"),
    ("mint", "mint"),
    ("carrot", "carrot"),
    ("garlic", "garlic"),
    ("ginger", "ginger"),
    ("lemon", "lemon"),
    ("grapes", "grapes"),
    ("banana", "banana"),
    ("guava", "guava"),
    ("groundnut", "groundnut"),
    ("jaggery", "jaggery"),
    ("apple", "apple"),
    ("apricot", "apricot"),
    ("peach", "peach"),
    ("papaya", "papaya"),
    ("pomegranate", "pomegranate"),
    ("cabbage", "cabbage"),
    ("canola", "canola"),
    ("capsicum", "capsicum"),
    ("cauliflower", "cauliflower"),
    ("cucumber", "cucumber"),
    ("coriander", "coriander"),
    ("cocoyam", "cocoyam"),
    ("cocunut", "coconut"),        # dataset has this typo
    ("coconut", "coconut"),
    ("brown sugar", "brown sugar"),
    ("batho", "batho"),
    ("green onion", "green onion"),
    ("green chilli", "green chilli"),
    ("green chickpeas", "green chickpeas"),
    ("green fodder", "green fodder"),
    ("jujube", "jujube"),
    ("jaman", "jaman"),
]

# ── Unit detection (per crop) ────────────────────────────────────────
# Verified against real Pakistani market rates:
#   • masoor/mash prices follow per-QUINTAL history (Rs/kg matches
#     known lentil price history 2015-2022)
#   • kinnow/orange/grapefruit/musambi/feutral carry "(100Pcs)" in name
#   • banana carries "(DOZENS)" in name
#   • everything else is per maund (40 kg) — standard Pakistani mandi unit

UNIT_LABELS = {
    "maund": "Rs. / maund (40 kg)",
    "quintal": "Rs. / quintal (100 kg)",
    "100pcs": "Rs. / 100 pieces",
    "dozen": "Rs. / dozen",
}

# Multiplier to convert a raw price into Rs/maund (None = not weight-based)
UNIT_TO_MAUND = {
    "maund": 1.0,
    "quintal": 0.40,   # 40 kg / 100 kg
    "100pcs": None,
    "dozen": None,
}


def detect_unit(crop_name: str) -> str:
    """Return the price unit for a crop based on its name."""
    lower = crop_name.lower()
    if "masoor" in lower or "mash" in lower:
        return "quintal"
    if "100" in lower and ("pcs" in lower or "pieces" in lower):
        return "100pcs"
    if "dozen" in lower:
        return "dozen"
    return "maund"


def map_commodity(crop_name: str, fallback: str) -> str:
    """Map a crop (variety) name to its commodity group."""
    lower = crop_name.lower()
    for keyword, commodity in CROP_TO_COMMODITY:
        if keyword in lower:
            return commodity
    return fallback  # filename stem (cleaned) if no rule matches


def md5_of_file(path: str) -> str:
    """MD5 hash of a file — used to detect byte-identical duplicates."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── File helpers ─────────────────────────────────────────────────────

def classify_file(filename: str) -> int:
    """
    Return the priority bucket (1, 2, or 3) for a raw CSV filename.

    Prefix matching against the filename stem (lowercase).  Longer
    keywords are checked first so e.g. "green onion" is not caught by
    the "onion" keyword.
    """
    stem = filename.lower().removesuffix(".csv").strip()
    for bucket, keywords in ((1, PRIORITY_1), (2, PRIORITY_2)):
        for kw in sorted(keywords, key=len, reverse=True):
            if stem.startswith(kw):
                return bucket
    return 3


def read_and_clean(filename: str) -> pd.DataFrame | None:
    """
    Read one raw CSV and return the cleaned, standardised DataFrame.

    Cleaning steps:
      • standardise column names (City/Date/Crop/Price → our schema)
      • parse dates (drop unparseable rows)
      • drop rows with Price <= 0  (0 = "no quote" marker in this dataset)
      • drop exact duplicate rows
      • add `commodity` (from crop name) and `unit` columns
    """
    path = os.path.join(RAW_DIR, filename)
    stem = filename.lower().removesuffix(".csv").strip().replace("_", " ")

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Normalise column names (all files use City,Date,Crop,Price —
    # but map defensively in case a file differs slightly)
    colmap = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("city", "market", "district"):
            colmap[c] = "market"
        elif cl == "date":
            colmap[c] = "date"
        elif cl in ("crop", "commodity"):
            colmap[c] = "crop"
        elif "price" in cl:
            colmap[c] = "price"
    df = df.rename(columns=colmap)
    missing = {"market", "date", "crop", "price"} - set(df.columns)
    if missing:
        logger.warning("  %s: missing columns %s — skipped", filename, missing)
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"] > 0]
    if df.empty:
        logger.warning("  %s: no valid rows after cleaning — skipped", filename)
        return None

    df = df.drop_duplicates()

    df["market"] = df["market"].str.strip()
    df["crop"] = df["crop"].str.strip()
    # Commodity group from the CROP name (filenames proved unreliable)
    df["commodity"] = [map_commodity(c, fallback=stem) for c in df["crop"]]
    # Price unit per crop
    df["unit"] = [detect_unit(c) for c in df["crop"]]
    # Whole rupees (mandi quotes are integers)
    df["price"] = df["price"].round().astype("int64")

    return df[["date", "market", "commodity", "crop", "price", "unit"]]


def output_bytes(df: pd.DataFrame) -> int:
    """
    Exact number of bytes these rows occupy in the final CSV
    (date,market,commodity,crop,price,unit + commas + newline).

    String columns are measured in UTF-8 BYTES — several crop names
    contain Urdu text (2 bytes per character), so plain len() would
    under-count and blow the size budget.
    """
    return int(
        (
            df["date"].dt.strftime("%Y-%m-%d").str.len()
            + df["market"].str.encode("utf-8").str.len()
            + df["commodity"].str.encode("utf-8").str.len()
            + df["crop"].str.encode("utf-8").str.len()
            + df["price"].astype(str).str.len()
            + df["unit"].str.len()
            + 6  # five commas + newline
        ).sum()
    )


# ── Best-market summary artifact ─────────────────────────────────────

def build_best_market_artifact(df: pd.DataFrame) -> dict:
    """
    Precompute per-crop, per-market recent price stats so the API can
    answer instantly (instead of re-parsing the big CSV on every start).

    For each crop we take its most recent 365 days of data and compute
    per-market average/min/max prices.  Markets with fewer than 5
    records in the window are excluded (too few quotes to trust).
    """
    crops_out: dict[str, dict] = {}

    for crop, cdf in df.groupby("crop"):
        last_date = cdf["date"].max()
        window_start = last_date - timedelta(days=365)
        window = cdf[cdf["date"] >= window_start]
        if window.empty:
            continue

        unit = window["unit"].iloc[0]
        markets: dict[str, dict] = {}
        for market, mdf in window.groupby("market"):
            if len(mdf) < 5:
                continue
            markets[market] = {
                "avg_price": round(float(mdf["price"].mean()), 1),
                "min_price": round(float(mdf["price"].min()), 1),
                "max_price": round(float(mdf["price"].max()), 1),
                "records": int(len(mdf)),
                "last_date": str(mdf["date"].max().date()),
            }

        if not markets:
            continue

        crops_out[crop] = {
            "commodity": window["commodity"].iloc[0],
            "unit": unit,
            "unit_label": UNIT_LABELS[unit],
            "per_maund_multiplier": UNIT_TO_MAUND[unit],
            "national_avg_price": round(float(window["price"].mean()), 1),
            "last_date": str(last_date.date()),
            "total_records": int(len(cdf)),
            "markets": markets,
        }

    return {
        "crops": crops_out,
        "window_days": 365,
        "min_market_records": 5,
        "source": "Pakistani mandi price data (2007-2022)",
        "unit_note": (
            "Raw prices are PKR per maund (40kg) except masoor/mash "
            "(per quintal = 100kg) and kinnow/orange/grapefruit/musambi/"
            "feutral (per 100 pcs) and banana (per dozen). Use "
            "per_maund_multiplier to convert to Rs/maund."
        ),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_ARTIFACT), exist_ok=True)

    all_files = sorted(
        f for f in os.listdir(RAW_DIR) if f.lower().endswith(".csv")
    )
    total_raw = sum(os.path.getsize(os.path.join(RAW_DIR, f)) for f in all_files)
    logger.info("=" * 70)
    logger.info(
        "STEP 1 — Reading & cleaning %d raw files (%.1f MB total)",
        len(all_files), total_raw / 1024 / 1024,
    )
    logger.info("=" * 70)

    # ── Read every (non-duplicate) file once ────────────────────────
    seen_hashes: dict[str, str] = {}          # md5 → filename
    cleaned: list[tuple[str, int, pd.DataFrame]] = []  # (file, priority, df)

    for filename in all_files:
        path = os.path.join(RAW_DIR, filename)
        digest = md5_of_file(path)
        if digest in seen_hashes:
            logger.info(
                "  SKIP  %-35s duplicate of '%s'", filename, seen_hashes[digest]
            )
            continue
        seen_hashes[digest] = filename

        df = read_and_clean(filename)
        if df is None:
            continue
        priority = classify_file(filename)
        cleaned.append((filename, priority, df))
        logger.info(
            "  READ  %-35s P%d  %7d rows  (%.1f MB raw)",
            filename, priority, len(df),
            os.path.getsize(path) / 1024 / 1024,
        )

    # ── Select files by priority until the OUTPUT budget is used up ──
    logger.info("=" * 70)
    logger.info(
        "STEP 2 — Selecting files by priority (output budget: %.0f MB)",
        SIZE_BUDGET_BYTES / 1024 / 1024,
    )
    logger.info("=" * 70)

    # Log missing Priority-1 staples (explains coverage gaps)
    for kw in PRIORITY_1:
        if not any(
            f.lower().removesuffix(".csv").strip().startswith(kw)
            for f in all_files
        ):
            logger.warning("  MISSING priority-1 crop: no file for '%s'", kw)

    # Sort: priority 1 → 2 → 3, alphabetical within each bucket
    cleaned.sort(key=lambda x: (x[1], x[0].lower()))

    selected: list[pd.DataFrame] = []
    used = 0
    for filename, priority, df in cleaned:
        size = output_bytes(df)
        if used + size > SIZE_BUDGET_BYTES:
            logger.info(
                "  SKIP  %-35s P%d  budget: combined file would reach %.1f MB",
                filename, priority, (used + size) / 1024 / 1024,
            )
            continue
        selected.append(df)
        used += size
        logger.info(
            "  KEEP  %-35s P%d  → %.1f MB in CSV (total: %.1f MB)",
            filename, priority, size / 1024 / 1024, used / 1024 / 1024,
        )

    logger.info("Selected %d / %d files — combined CSV will be ~%.1f MB",
                len(selected), len(all_files), used / 1024 / 1024)

    if not selected:
        logger.error("No files selected — aborting.")
        return

    # ── Combine, dedupe across files, and write ─────────────────────
    combined = pd.concat(selected, ignore_index=True)
    combined = combined.drop_duplicates()
    combined = combined.sort_values(["commodity", "crop", "market", "date"])
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    # lineterminator="\n" — without this, Windows pandas writes \r\n
    # (an extra byte per row × 5M+ rows would blow the size budget).
    combined.to_csv(OUTPUT_CSV, index=False, lineterminator="\n")

    csv_mb = os.path.getsize(OUTPUT_CSV) / 1024 / 1024
    logger.info("=" * 70)
    logger.info("STEP 3 — Combined dataset written")
    logger.info("=" * 70)
    logger.info("Combined CSV : %s", OUTPUT_CSV)
    logger.info("Total rows   : %s", f"{len(combined):,}")
    logger.info("File size    : %.1f MB (budget %.0f MB)", csv_mb,
                SIZE_BUDGET_BYTES / 1024 / 1024)
    logger.info("Date range   : %s → %s",
                combined["date"].min(), combined["date"].max())

    commodities = sorted(combined["commodity"].unique())
    logger.info("Commodities  : %d → %s", len(commodities), ", ".join(commodities))

    markets = sorted(combined["market"].unique())
    logger.info("Markets      : %d unique mandis/cities", len(markets))
    logger.info("  %s", ", ".join(markets))

    # ── Save the best-market summary artifact ───────────────────────
    logger.info("=" * 70)
    logger.info("STEP 4 — Building best-market summary artifact")
    logger.info("=" * 70)
    # Parse dates back for the rolling-window calculation
    combined["date"] = pd.to_datetime(combined["date"])
    artifact = build_best_market_artifact(combined)
    joblib.dump(artifact, OUTPUT_ARTIFACT)
    logger.info("Artifact     : %s", OUTPUT_ARTIFACT)
    logger.info("Crops in artifact: %d", len(artifact["crops"]))
    logger.info("Artifact size: %.1f KB",
                os.path.getsize(OUTPUT_ARTIFACT) / 1024)

    logger.info("DONE ✅  — run `python training/train_market.py` next to "
                "refresh the seasonal trends artifact.")


if __name__ == "__main__":
    main()
