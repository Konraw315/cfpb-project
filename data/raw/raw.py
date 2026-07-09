"""
Process the data from CSV file to the Parquet file.
"""
import duckdb 

DATA_URL = "https://cfpb-project-742435394310-us-east-1-an.s3.us-east-1.amazonaws.com/raw/raw.csv"


def transforimng_to_parquet(URL: str) -> None:
    """
    Transforming raw.csv data to parquet data.

    Parameters:
    URL - the url link to data
    """

    duckdb.sql(f"COPY (SELECT * FROM read_csv('{URL}')) TO 'raw.parquet' (FORMAT parquet);")

    return


if __name__ == "__main__":
    
    transforimng_to_parquet(DATA_URL)
