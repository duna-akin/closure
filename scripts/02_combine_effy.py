"""
Combine 11 years of IPEDS 12-month Enrollment (EFFY) files into one long panel.

EFFY is keyed on UNITID x EFFYLEV (level of study). We keep three levels:
    EFFYLEV == 1  -> all students (total)
    EFFYLEV == 2  -> undergraduate
    EFFYLEV == 4  -> graduate

For each (UNITID, year) we emit one row with three enrollment columns
(total/undergrad/grad) plus a single imputation flag for the total headcount.

Output: processed_data/effy_combined.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_data"
OUT = ROOT / "processed_data"

YEAR_FILES = {y: f"effy{y}.csv" for y in range(2014, 2025)}

LEVEL_NAME = {1: "total_enrollment", 2: "undergrad_enrollment", 4: "grad_enrollment"}


def load_year(year: int) -> pd.DataFrame:
    path = RAW / str(year) / YEAR_FILES[year]
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()

    df = df[df["EFFYLEV"].isin(LEVEL_NAME.keys())].copy()
    df = df[["UNITID", "EFFYLEV", "EFYTOTLT", "XEYTOTLT"]]

    # Pivot so each (UNITID) becomes one row with columns per level.
    head = df.pivot_table(
        index="UNITID", columns="EFFYLEV", values="EFYTOTLT", aggfunc="first"
    ).rename(columns=LEVEL_NAME)

    # Imputation flag carried only for the level-1 (total) value, since that is
    # what we use to weight downstream model rows.
    flag = (
        df[df["EFFYLEV"] == 1]
        .set_index("UNITID")["XEYTOTLT"]
        .rename("XEYTOTLT_total")
    )

    out = head.join(flag, how="left").reset_index()
    out["year"] = year
    return out


def main() -> None:
    frames = []
    for year in sorted(YEAR_FILES):
        df = load_year(year)
        nz = {c: int(df[c].notna().sum()) for c in LEVEL_NAME.values() if c in df.columns}
        print(f"{year}: {len(df):>5} rows  |  non-null counts: {nz}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[
        ["UNITID", "year", "total_enrollment", "undergrad_enrollment",
         "grad_enrollment", "XEYTOTLT_total"]
    ]

    OUT.mkdir(exist_ok=True)
    out_path = OUT / "effy_combined.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  shape={combined.shape}")
    print(f"unique UNITIDs: {combined['UNITID'].nunique()}")
    print(f"year counts:\n{combined['year'].value_counts().sort_index().to_string()}")


if __name__ == "__main__":
    main()
