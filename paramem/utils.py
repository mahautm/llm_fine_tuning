import os
import pandas as pd
import pickle
import math
from pathlib import Path

def convert_csv_to_pickle(base_path):
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                pickle_path = os.path.splitext(csv_path)[0] + "-data.pickle"
                # add data as a prefix
                df = pd.read_csv(csv_path)
                df["expected_answers"] = df["expected_answers"].apply(pd.eval)
                output = {k: v.tolist() for k, v in df.items()}
                with open(pickle_path, "wb") as f:
                    pickle.dump(output, f)
                print(f"Converted {csv_path} to {pickle_path}")

def split_pickle_file(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    num_records = len(data[list(data.keys())[0]])
    chunk_size = math.ceil(num_records / 3)
    
    for i in range(3):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, num_records)
        chunk_data = {k: v[start_index:end_index] for k, v in data.items()}
        
        split_pickle_path = f"{str(pickle_path)[:-7]}-split{i+1}.pickle"
        with open(split_pickle_path, "wb") as f:
            pickle.dump(chunk_data, f)
        
        print(f"Split {pickle_path} into {split_pickle_path}")



if __name__=="__main__":
    # base_path = "/home/mmahaut/projects/paramem/hlayer"
    # convert_csv_to_pickle(base_path)
    pickle_path = "/home/mmahaut/projects/paramem/hlayer2"
    for f in Path(pickle_path).rglob("*pile*.pickle"):
        if "split" not in str(f):
            split_pickle_file(f)