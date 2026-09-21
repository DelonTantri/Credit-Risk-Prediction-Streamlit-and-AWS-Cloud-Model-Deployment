import os
import pandas as pd

def ingest_data():
    container_input = "/opt/ml/processing/input"
    container_output = "/opt/ml/processing/ingested"

    if os.path.exists(container_input):
        input_dir = container_input
        output_dir = container_output
    else:
        input_dir = "./"
        output_dir = "./ingested"

    os.makedirs(output_dir, exist_ok=True)
    input_file = os.path.join(input_dir, "data_A.csv")
    print(f"Looking for file: {input_file}")

    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
        output_file = os.path.join(output_dir, "data_A.csv")
        df.to_csv(output_file, index=False)
        print(f"✅ Dataset ingested: {output_file}")
    else:
        print(f"❌ File not found: {input_file}")

if __name__ == "__main__":
    ingest_data()