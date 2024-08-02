# plot layer per layer distances
# Layer 0  0.7741215008933889 0.2258784991066111
# Layer 1  0.2808219178082192 0.7191780821917808
# Layer 2  0.7459797498511018 0.2258784991066111
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_layer_distance(data_file, sep=" "):
    data = pd.read_csv(data_file, sep=sep, header=None)

    # only start with line that says Layer-0
    data = data.loc[1:].reset_index(drop=True)
    # drop last column
    data = data.drop(columns=[3])
    print(data)

    data.columns = ["epoch", "loss1", "loss2"]
    data["loss1"] = data["loss1"].astype(float)
    data["loss2"] = data["loss2"].astype(float)
    data["distance"] = data["loss1"] - data["loss2"]
    data["distance"] = data["distance"].abs()
    sns.lineplot(x="epoch", y="distance", data=data)
    # save
    plt.savefig(data_file.replace(data_file.split(".")[-1], "png"))

def plot_from_log(log_path="", key="", title=""):
    with open(log_path, "r") as log_file:
        vals=[]
        legends = []
        lines = log_file.readlines()
        legend = ""
        layers=[]
        i=0
        for line in lines:
            if key in line:
                val=line.split(key)[1].strip()
                val = float(val)
                vals.append(val)
                legends.append(legend)
                layers.append(i)
                i+=1
            if "Comparing" in line:
                legend = line.strip().split("/")[-1].split("-")[0]
                i=0
        sns.lineplot(x=layers, y=vals, hue=legends)
        plt.title(title)
        plt.savefig(f"{Path(log_path).parent}/{key[:-1]}_{title.split(' ')[0]}.png")
        plt.clf()

if __name__ == "__main__":
    plot_from_log(log_path="/home/mmahaut/projects/exps/la/555513_0_log.out", key="Accuracy:", title="Discrimination accuracy from hidden layers")
    plot_from_log(log_path="/home/mmahaut/projects/exps/la/555513_0_log.out", key="Coefficients:", title="Coefficients used for discrimination between hidden layers")

    



