from google.cloud import bigquery
import re

#Aim: know which files have already been processed or not 

def get_processed_file_timestamps(project_id: str, dataset_id: str) -> set:
    """
    Retrieves the timestamps of all files that have already been processed
    and uploaded to a BigQuery dataset.

    Args:
        project_id (str): Your Google Cloud project ID.
        dataset_id (str): The ID of the BigQuery dataset where angles are stored (e.g., 'angle_csvs').

    Returns:
        set: A set of strings, where each string is a timestamp (e.g., '20150302T114239')
              extracted from the BigQuery table names.
    """
    client = bigquery.Client(project=project_id)
    dataset_ref = client.dataset(dataset_id, project=project_id)

    processed_timestamps = set()
    timestamp_pattern = re.compile(r'angles_(\d{8}T\d{6})') # Matches e.g., 'angles_20150302T114239'

    try:
        tables = client.list_tables(dataset_ref)
        print(f"Checking tables in dataset: {dataset_id}")
        for table in tables:
            match = timestamp_pattern.match(table.table_id)
            if match:
                processed_timestamps.add(match.group(1))
                # print(f"Found processed timestamp: {match.group(1)}") # For debugging
    except Exception as e:
        print(f"Error listing tables in BigQuery dataset {dataset_id}: {e}")
        return set() # Return empty set on error

    return processed_timestamps
