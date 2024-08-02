import pandas as pd
import json

def jsonl_to_csv(input_file, output_file):
    # Read the JSONL file
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_file, index=False)


input_file = 'output/raft_dataset.jsonl'
output_file = 'output/raft_dataset.csv'

jsonl_to_csv(input_file, output_file)
