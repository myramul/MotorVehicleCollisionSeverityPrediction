"""
fetch_data.py

Fetches a random sample of rows from the NYC Open Data Motor Vehicle Collisions dataset
using the API and saves it as a CSV file in the data/ directory.

Run:
    python code/fetch_data.py
"""

import os
import random
import time
import requests


NUMBER_ROWS = 150000

DATASET_ID = "h9gi-nx95"
SOCRATA_DOMAIN = "data.cityofnewyork.us"
SOCRATA_RESOURCE = f"https://{SOCRATA_DOMAIN}/resource/{DATASET_ID}.csv"


def fetch_to_csv(out_csv="data/raw/motor_vehicle_collisions_sample.csv", retries=5, timeout=240):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Get the token
    token = os.getenv("SOCRATA_APP_TOKEN")
    headers = {"X-App-Token": token} if token else {}

    total_rows_fetched = 0  
    start_time = time.time()
    wrote_header = False

    # where clause is that these can not all be null or we have no target
    where_clause = (
        "number_of_persons_injured IS NOT NULL OR "
        "number_of_persons_killed IS NOT NULL OR "
        "number_of_pedestrians_injured IS NOT NULL OR "
        "number_of_pedestrians_killed IS NOT NULL OR "
        "number_of_cyclist_injured IS NOT NULL OR "
        "number_of_cyclist_killed IS NOT NULL OR "
        "number_of_motorist_injured IS NOT NULL OR "
        "number_of_motorist_killed IS NOT NULL"
    )

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        while total_rows_fetched < NUMBER_ROWS:
            remaining = NUMBER_ROWS - total_rows_fetched
            batch_size = min(5000, remaining) # get 5000 rows at a time

            # the api rquires this to get more than 1000 rows, random offset to ensure random sample
            params = {
                "$limit": batch_size,
                "$offset": random.randint(0, 2200000),
                "$where": where_clause,
            }

            attempts_left = retries

            while attempts_left > 0:
                try:
                    response = requests.get(
                        SOCRATA_RESOURCE,
                        params=params,
                        headers=headers,
                        timeout=timeout,
                    )
                    response.raise_for_status()
                    text = response.text.strip()

                    if not text:
                        print("Received empty response. Stopping.")
                        attempts_left = 0
                        break

                    lines = text.splitlines()

                    # If response only has header or is empty, stop
                    if len(lines) <= 1:
                        print("No data rows returned. Stopping.")
                        attempts_left = 0
                        break

                    # Write header only once
                    if not wrote_header:
                        f.write("\n".join(lines) + "\n")
                        wrote_header = True
                        rows_fetched = len(lines) - 1
                    else:
                        f.write("\n".join(lines[1:]) + "\n")
                        rows_fetched = len(lines) - 1

                    total_rows_fetched += rows_fetched
                    print(f"Fetched {rows_fetched} rows, total so far: {total_rows_fetched}")
                    break

                except requests.exceptions.RequestException as e:
                    attempts_left -= 1
                    print(f"Error fetching data: {e}. Retrying... ({attempts_left} retries left)")
                    if attempts_left <= 0:
                        print("Max retries reached for this batch. Stopping fetch.")
                        break
                    time.sleep(5)

            # if batch failed completely, stop outer loop too
            if attempts_left <= 0:
                break

            # slight delay
            time.sleep(0.5)

    end_time = time.time()
    metadata = {
        "total_rows_fetched": total_rows_fetched,
        "time_taken_seconds": end_time - start_time,
        "output_file": out_csv,
    }
    return metadata


if __name__ == "__main__":
    metadata = fetch_to_csv()
    print("\nFetch Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")