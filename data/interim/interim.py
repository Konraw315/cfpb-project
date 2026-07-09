"""
Filter and preprocess complaint data stored in a Parquet file.

This module reads selected columns from a Parquet dataset using DuckDB,
filters out rows with missing values in key columns, converts the resulting
columns to the pandas 'category' dtype, and saves the processed data to a
new Parquet file.
"""

import duckdb 
import pandas as pd

DATA_URL = "https://cfpb-project-742435394310-us-east-1-an.s3.us-east-1.amazonaws.com/raw/raw.parquet"


def load_filtered_columns(parquet_path: str) -> pd.DataFrame:
    """
    Read selected columns from a Parquet file and filter incomplete rows.

    Args:
        parquet_path: Local path or URL to the Parquet file.

    Returns:
        DataFrame containing selected and filtered columns.
    """
    query = """
        SELECT
            "Product",
            "Sub-product",
            "Issue",
            "Sub-issue",
            "Company",
            "State",
            "Submitted via",
            "Company public response",
            "Consumer consent provided?",
            "Company response to consumer"
        FROM read_parquet(?)
        WHERE "Issue" IS NOT NULL
          AND "Company response to consumer" IS NOT NULL
          AND "Company response to consumer" NOT IN (
              'Closed with relief',
              'Closed without relief'
          )
    """

    with duckdb.connect() as con:
        data = con.execute(query, [parquet_path]).df()

    return data

def save_categorical_data(data: pd.DataFrame) -> None:
    """
    Convert all columns in the DataFrame to categorical type and save output. Moreover

    Args:
        data: The input DataFrame to transform.

    Returns:
        None. The processed data is written to 'processed_data.parquet'.
    """
    for col in data.columns:
        data[col] = data[col].astype("category")
    
    columns_na = ['Consumer consent provided?', "Sub-product", "Sub-issue", "Company public response"]
    for columns in columns_na:
        if 'N/A' not in data[columns].cat.categories:
            data[columns] = data[columns].cat.add_categories('N/A')
    
    data[columns_na] = data[columns_na].fillna('N/A')

    return data.to_parquet('processed_data.parquet')

if __name__ == "__main__":
    
    data = load_filtered_columns(DATA_URL)
    save_categorical_data(data)