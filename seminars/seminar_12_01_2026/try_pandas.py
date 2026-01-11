"""
Pandas listing.
"""

from pathlib import Path

try:
    from pandas import read_csv
except ImportError:
    print('Library "pandas" not installed. Failed to import.')
    read_csv = None  # type: ignore


def main() -> None:
    """
    Entrypoint for the listing.
    """
    data_path = Path(__file__).parent / "assets" / "glue_mnli_example.csv"

    # 1. Load from file - in lab it is in memory from dataset.to_pandas()
    raw_data_df = read_csv(data_path, encoding="utf-8")

    # 2. Get number of samples
    num_samples = len(raw_data_df)
    print(f"Number of samples: {num_samples}")

    # 3. Get number of columns
    columns = raw_data_df.columns
    print(f"Number of columns: {len(columns)}")
    print(f"Columns: {columns}")

    # 4. Get duplicates
    duplicates = raw_data_df[raw_data_df.duplicated()]
    print(f"Number of duplicates: {len(duplicates)}")

    # 5. Get empty samples
    empty = raw_data_df[raw_data_df.isna().any(axis=1)]
    print(f"Number of empty rows: {len(empty)}")

    # 6. Get only needed columns
    subset = raw_data_df[["premise", "hypothesis", "label"]]
    print(subset.head(3))

    # 7. Drop unnecessary columns
    raw_data_df = raw_data_df.drop("idx", axis=1)
    print(f"After removal columns are: {raw_data_df.columns}")

    # 8. Rename columns
    raw_data_df = raw_data_df.rename(
        columns={"premise": "my_premise", "hypothesis": "my_hypothesis", "label": "my_label"}
    )
    print(raw_data_df.head(3))

    # 9. Drop invalid samples (N/A, duplicates)
    cleaned = raw_data_df.dropna().drop_duplicates()
    print(f"Cleaned dataset has {len(cleaned)} samples")

    # 10. Apply transformation to each value in column
    raw_data_df["my_label"] = raw_data_df["my_label"].apply(lambda x: "No" if x == 0 else "Yes")
    print(raw_data_df.head(3))

    # 11. Get N-first rows
    print(raw_data_df.head(3))


if __name__ == "__main__":
    main()
