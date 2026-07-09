"""
Filter and process complaint data stored in a Parquet file.

The processed data is later used for machine learning.
"""

import os
import duckdb 
import pandas as pd

DATA_URL = "https://cfpb-project-742435394310-us-east-1-an.s3.us-east-1.amazonaws.com/interim/interim_data.parquet"
PROCESSED_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROCESSED_PATH, "processed_data.parquet")

def load_filtered_columns(parquet_path: str, data_path: str) -> pd.DataFrame:
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
            "Issue",
            "Company",
            "Submitted via",
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

    return data.to_parquet(data_path)


if __name__ == "__main__":
    
    data = load_filtered_columns(DATA_URL, DATA_PATH)
