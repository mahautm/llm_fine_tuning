import jax
import jax.numpy as jnp
# from dadapy._utils.metric_comparisons import _return_imbalance
import numpy as np
import pickle
from pathlib import Path
from typing import List
from tqdm import tqdm
import warnings
import torch
import dadapy
import matplotlib.pyplot as plt
import seaborn as sns
import typer
import pandas as pd

app = typer.Typer()
# def information_imbalance(indices1: np.ndarray, indices2: np.ndarray, rng, k: int) -> list:
#     """Compute information imbalance between given indices and indices from a path."""    
#     assert len(indices1) == len(indices2), \
#         "Mismatch in number of samples, impossible to perform information imbalance"
    
#     return [
#         _return_imbalance(indices1, indices2, rng, k=k),
#         _return_imbalance(indices2, indices1, rng, k=k)
#     ]

def load_pickle(file_path: str):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_quantiles(a,alphamin,alphamax):
  qmin = jnp.quantile(a,q=alphamin,axis=1)
  qmax = jnp.quantile(a,q=alphamax,axis=1)
  return qmin,qmax

def clip(act):
    alphamin = 0.05
    alphamax = 0.95
    
    if len(act.shape)==3:
      reshape = True
    else:
      reshape = False

    if reshape:
      B,T,E = act.shape
      act = jnp.reshape(act,shape=(B,T*E))
    
    qmin,qmax = get_quantiles(act,alphamin,alphamax)
    act = jnp.clip(act.T,min=qmin,max=qmax).T

    # if reshape:
    #   act = jnp.reshape(act,shape=(B,T,E))
    return act

def normalized_L2_distance(x, y):
    x /= jnp.linalg.norm(x)
    y /= jnp.linalg.norm(y)
    u = (x-y)
    return jnp.sqrt(jnp.sum(u*u))

def pairwise_distances(distance, xs, ys):
  return jax.vmap(lambda x: jax.vmap(lambda y: distance(x, y))(xs))(ys).T

def compare(all_activations_A,all_activations_B,diagonal_only:bool=False,rank:int=0,output_folder:str=None):
    limit=2500
    all_activations_A = {f"layer_{i}": np.array(all_activations_A[i][:limit]) for i in all_activations_A.keys()}
    all_activations_B = {f"layer_{i}": np.array(all_activations_B[i][:limit]) for i in all_activations_B.keys()}
    # all_activations_A = {k: v.mean(axis=1, keepdims=False) for k, v in all_activations_A.items()}
    # all_activations_B = {k: v.mean(axis=1, keepdims=False) for k, v in all_activations_B.items()}

    sample_size = all_activations_B["layer_0"].shape[0]
    rows = all_activations_A.keys()
    cols = all_activations_B.keys()
    inf_imb_matrix = []
    reciprocal_inf_imb_matrix = []


    for row in tqdm(rows, desc="Compute Distances"):
        act_A = all_activations_A[row]
        act_A = np.array(clip(act_A))
        dt_A = dadapy.Data(act_A)
        dt_A.compute_distances(maxk=sample_size-1, metric="euclidean")

        for col in tqdm(cols, desc="Computing layer_B", leave=False):
            if diagonal_only and row != col:
                continue
            act_B = all_activations_B[col]
            # Clip activations
            act_B = np.array(clip(act_B))
            # dt_B = dadapy.Data(act_B)      
            # dt_B.compute_distances(maxk=sample_size-1, metric="euclidean")

            # ii_ab, ii_ba = _information_imbalance(dt_A.dist_indices, dt_B.dist_indices, key, k=sample_size-1)
            ii=dt_A.return_information_imbalace(act_B, subset_size=sample_size-1)
            ii_ab = ii[0][0]
            ii_ba = ii[1][0]
            tqdm.write(f"(II(A-->B) {ii_ab:.4f} ; II(B-->A) {ii_ba:.4f})")
            inf_imb_matrix.append((row, col, ii_ab))
            reciprocal_inf_imb_matrix.append((row, col, ii_ba))

    # Save the results
    if output_folder is None:
        return inf_imb_matrix, reciprocal_inf_imb_matrix
    else:
        output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / f"clipped_II_AB_{rank}.pkl", 'wb') as f:
        pickle.dump(inf_imb_matrix, f)
    with open(output_folder / f"clipped_II_BA_{rank}.pkl", 'wb') as f:
        pickle.dump(reciprocal_inf_imb_matrix, f)
    return inf_imb_matrix, reciprocal_inf_imb_matrix



def plot_and_save_II(inf_imb_matrix, output_file, title=None):
    # Plot the information imbalance matrices
    df=pd.DataFrame(inf_imb_matrix, columns=["Layer_A", "Layer_B", "Information Imbalance"])
    inf_imb_matrix = df.pivot(index="Layer_A", columns="Layer_B", values="Information Imbalance")
    inf_imb_matrix=inf_imb_matrix.reindex(index=[f"layer_{i}" for i in range(len(inf_imb_matrix.keys()))], columns=[f"layer_{i}" for i in range(len(inf_imb_matrix.keys()))])

    plt.figure(figsize=(10, 8))
    sns.heatmap(inf_imb_matrix, annot=False, cmap="coolwarm")
    plt.title(title if title else "Information Imbalance Matrix")
    plt.savefig(output_file)
    plt.close()


@app.command()
def main( 
         output_folder:str=None,
         diagonal_only:bool=False,  # New option to compute only the diagonal
         rank:int=0,
         ):

    # get all files in the folder
    input_path = Path("/home/mmahaut/projects/paramem/hlayer/")
    files = sorted(input_path.glob("*.pickle"))
    # for file in files:
    #     pickle_data = load_pickle(file)
    #     print(f"Loaded {file.name} with keys: {list(pickle_data.keys())}")
    #     ii, rii = compare(pickle_data, pickle_data, diagonal_only=diagonal_only, rank=rank, output_folder=output_folder)
    #     plot_and_save_II(ii, Path(output_folder) / f"{file.stem}_vs_{file.stem}.png")
    #     plot_and_save_II(rii, Path(output_folder) / f"{file.stem}_vs_{file.stem}_reciprocal.png")

    # Make comparisons
    for file in files:
        if "lora" not in file.name:
            continue
        pickle_data1 = load_pickle(file)
        # remove lora from the name
        file_name2 = file.name.replace("-lora", "")
        try:
            pickle_data2 = load_pickle(input_path / file_name2)
        except FileNotFoundError:
            print(f"File {file_name2} not found, skipping.")
            continue
        print(f"Loaded {file.name} and {file_name2} with keys: {list(pickle_data1.keys())} and {list(pickle_data2.keys())}")
        ii, rii = compare(pickle_data1, pickle_data2, diagonal_only=diagonal_only, rank=rank, output_folder=output_folder)
        plot_and_save_II(ii, Path(output_folder) / f"{Path(file).stem}_vs_{Path(file_name2).stem}.png", title=f"II {Path(file).stem} vs {Path(file_name2).stem}")
        plot_and_save_II(rii, Path(output_folder) / f"{Path(file).stem}_vs_{Path(file_name2).stem}_reciprocal.png", title=f"Reciprocal II {Path(file_name2).stem} vs {Path(file).stem}")
        print(f"Comparison done for {file} and {file_name2}")
            
if __name__ == "__main__":
    app()