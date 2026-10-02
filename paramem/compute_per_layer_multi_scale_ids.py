import sys
import pickle
import numpy as np
from dadapy import Data
import glob
import re
import concurrent
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

range_max = int(sys.argv[1])

def cumul_mus(mus):
    N=len(mus)
    mu_sorted = np.sort(mus)
    cdf = np.arange(1, N+1)/N
    return cdf, mu_sorted


pickle_path = sys.argv[2]
model_name = sys.argv[3]

with open(pickle_path, 'rb') as pickle_file:
    data = pickle.load(pickle_file)



ids = {}
for i, (k,v) in enumerate(data.items()):
    print(f"Block {k} has shape {np.array(v).shape}")
    dada_data= Data(coordinates=np.array(v))
    ids_scaling, _, _ = dada_data.return_id_scaling_gride(range_max = range_max, set_attr=True)
    ids[k] = ids_scaling

    if i == 10:
        mus = dada_data.intrinsic_dim_mus_gride[:, 0]
        cdf, mu_sorted = cumul_mus(mus)

plt.scatter(np.log(mu_sorted), -(np.log(1-cdf)))
id_10=ids[10][0]
plt.plot(np.log(mu_sorted), np.log(mu_sorted)*id_10)
plt.savefig(f"{model_name}_id_10.png")



# Plotting the ids_scaling for each block
plot_data = []
for block_num, ids_scaling in ids.items():
    for scale in range(ids_scaling.shape[0]):
        plot_data.append([block_num, 2**scale, ids_scaling[scale]])

df = pd.DataFrame(plot_data, columns=['Block', 'Scale', 'ID'])
save_path = f"{model_name}_ids_gride_per_layer.png"
Path(save_path).parent.mkdir(parents=True, exist_ok=True)


plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Block', y='ID', hue='Scale', markers=True, dashes=False)
plt.title(f'{model_name} ID Scaling across Blocks')
plt.xlabel('Block Number')
plt.ylabel('ID Scaling')
plt.legend(title='Scale')
plt.savefig(save_path)
plt.close()