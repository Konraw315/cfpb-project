""""
Saving the feature names and their categories into a JSON file.
"""
import json
import os
import pandas as pd

DATA_URL = "https://cfpb-project-742435394310-us-east-1-an.s3.us-east-1.amazonaws.com/processed/processed_data.parquet"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(SCRIPT_DIR, "categories.json")


df = pd.read_parquet(DATA_URL)

def create_dictionary(data: pd.DataFrame) -> dict:
  """
  Create a dictionary containing column names and unique values.

  Parameters
  data: dataset from which column names and values are extracted.

  Returns
  dict: dictionary containing column names and their unique values.
  """
  
  columns = data.columns.to_list()
  columns.remove("Company response to consumer")
  categories  = {}

  for column in columns:
    categories[column] = data[column].cat.categories.to_list()

  return categories 

def save_json(categories: dict):
  """
  Saving created dictionary as JSON.

  Parameters:
  categories: dictionary containing column names and their unique values
  """

  with open(JSON_PATH, "w", encoding= "utf-8") as file:
    json.dump(
      categories,
      file
    )
  
  print(f"Categories saved to: {JSON_PATH}")

def main() -> None:
  categories = create_dictionary(df)
  save_json(categories)
  
if __name__ == "__main__":
    main()

