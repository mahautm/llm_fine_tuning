import pickle
import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from scipy.cluster.vq import kmeans
from scipy.stats import shapiro
from scipy.stats import bartlett
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from graph import plot_layer_distance
import pickle

# from Marco
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import pickle
import sys
import typer
from typing import List, Optional
import pandas as pd
from data import prepare_data
from pathlib import Path


def cosine_distance(tensor1, tensor2, save=None, label=None):
    """
    compute average cosine distance for each layer
    """
    cos_dist=[]
    for layer in list(tensor1.keys())[1:]:
        cos_dist.append(np.mean([1 - (np.dot(tensor1[layer][i], tensor2[layer][i]) / (np.linalg.norm(tensor1[layer][i]) * np.linalg.norm(tensor2[layer][i]))) for i in range(len(tensor1[layer]))]))
        # cosine similarity
        # cos = np.dot(tensor1[layer], tensor2[layer]) / (np.linalg.norm(tensor1[layer]) * np.linalg.norm(tensor2[layer]))
        # cosine distance
        # cos_dist = 1 - cos
    if save is not None:
        sns.lineplot(x=range(1,len(cos_dist)+1), y=cos_dist, label=label)
        plt.title(f"Average cosine distance")
        plt.savefig(save)
    return cos_dist

if __name__ == "__main__":
    input_key = "query"
    model_name="Met7"    
    data_file = "/home/mmahaut/projects/paramem/data/wikidata_Met7.csv"
    _data = pd.read_csv(data_file)
    if "exact_match.1" in _data.columns:
        success_key = "exact_match.1"
    else:
        success_key = "exact_match"
    _data = _data.groupby(
        [input_key, "template", "expected_answers"], as_index=False).agg(
        {
            success_key: "any",
            # "nli_factual": "any",
        }
    )
    # only keep positive examples
    _data = _data[_data[success_key]==False]
    # The occupation of Samuel Giacosa is
    # The occupation of Georg Grothe is
    # Otto Thott Fritzner Müller found employment in
    # Fredrik Schulte found employment in    
    idxA = 103
    print(_data.iloc[idxA]["template"])
    idxB = 104
    print(_data.iloc[idxB]["template"])
    idxC = 102
    print(_data.iloc[idxC]["template"])
    idxD = 502
    print(_data.iloc[idxD]["template"])

    codes = ["Met7"]
    # Load pickled tensors
    for c in codes:
        file1 = f"/home/mmahaut/projects/paramem/hlayer/{c}-ft-small.pickle"
        file2 = f"/home/mmahaut/projects/paramem/hlayer/{c}-icl.pickle"
        with open(file1, "rb") as f:
            tensor1 = pickle.load(f)
            tensor1 = {k: [np.array(vv).astype(np.float32) for vv in v] for k, v in tensor1.items()}

        with open(file2, "rb") as f2:
            tensor2 = pickle.load(f2)
            tensor2 = {k: [np.array(vv).astype(np.float32) for vv in v] for k, v in tensor2.items()}
        # print(len(tensor1), len(tensor2), len(inputs))
        # assert len(tensor1[list(tensor1.keys())[0]]) == len(inputs), "Different number of inputs"

        # sample 10 idxs
        idxs = np.random.choice(range(len(tensor1[list(tensor1.keys())[0]])), 1)
        idxs2 = np.random.choice(range(len(tensor1[list(tensor1.keys())[0]])), 1)

        tensorA = {k: [v[i] for i in [idxA, idxB, idxC, idxD]] for k, v in tensor1.items()}
        tensorB = {k: [v[i] for i in [idxB, idxA, idxD, idxC]] for k, v in tensor1.items()}
        tensorC = {k: [v[i] for i in [idxC, idxD, idxA, idxB]] for k, v in tensor1.items()}
        tensor2 = {k: [v[i] for i in [idxA, idxB, idxC, idxD]] for k, v in tensor2.items()}
        print("c")
        inputs = _data.iloc[idxs][input_key].tolist()  
        print(f"Comparing {file1} and {file2}")
        co = cosine_distance(tensorA, tensor2, save=None, label=c)
        print("PAIRS",[co[i] for i in [25]])
        # reorder idxs
        inputs = _data.iloc[idxs2][input_key].tolist()  

        print(f"Comparing {file1} and {file2}")
        co = cosine_distance(tensorB, tensor2, save=None, label=c)
        print("Intra",[co[i] for i in [25]])

        co = cosine_distance(tensorC, tensor2, save=None, label=c)
        print("Extra",[co[i] for i in [25]])
