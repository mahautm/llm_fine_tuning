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
from pathlib import Path
def prepare_data(tensor1, tensor2):
    data = pd.DataFrame()
    data["hidden"] = tensor1[layer]
    # explode list as columns
    data = data["hidden"].apply(pd.Series)
    data["category"] = [0] * len(tensor1[layer])
    data2 = pd.DataFrame()
    data2["hidden"] = tensor2[layer]
    # explode list as columns
    data2 = data2["hidden"].apply(pd.Series)
    data2["category"] = [1] * len(tensor2[layer])
    data = pd.concat([data, data2])

    y = data2["category"]
    x = data.drop(columns=["category"])
    return x, y    
def do_manova(tensor1, tensor2):
    for layer in list(tensor1.keys()):
        # Prepare data for repeated measures ANOVA
        exog, endog = prepare_data(tensor1[layer], tensor2[layer])

        # manova = MANOVA.from_formula(" + ".join(endog.columns.astype(str)) + " ~ " + " + ".join(exog.columns.astype(str)), data)
        manova = MANOVA(endog=endog, exog=exog)
        print(manova.mv_test())

def do_kmeans(tensor1, tensor2, log_file):
    for layer in list(tensor1.keys())[1:]:
        data = np.concatenate([tensor1[layer], tensor2[layer]])
        centroids, _ = kmeans(data, 2)
        # print(centroids)
        # check if elements from tensor1 are closer to centroid 1 than centroid 2
        c1 = np.mean(np.linalg.norm(data[:len(tensor1[layer])] - centroids[0], axis=1) < np.linalg.norm(data[:len(tensor1[layer])] - centroids[1], axis=1))
        # check if elements from tensor2 are closer to centroid 2 than centroid 1
        c2 = np.mean(np.linalg.norm(data[len(tensor1[layer]):] - centroids[1], axis=1) < np.linalg.norm(data[len(tensor1[layer]):] - centroids[0], axis=1))
        log_file.write(f"Layer-{layer} {c1} {c2}\n")
        print(f"Layer-{layer} {c1} {c2}")

def co_activations(tensor1, tensor2, plot:str=None):
    # following https://online.stat.psu.edu/stat462/node/137/
    # check if the same neurons are activated in both tensors
    r2s = []
    for layer in list(tensor1.keys()):
        exog, endog = prepare_data(tensor1[layer], tensor2[layer])
        # Multiple Linear Regression
        # y = Xb + e, with y = endog, X = exog, b = coefficients, e = residuals
        # b = (X^T X)^-1 X^T y
        # b = np.dot(np.dot(np.linalg.inv(np.dot(exog.T, exog)), exog.T), endog)
        b, e, rk, _ = np.linalg.lstsq(exog, endog, rcond=None)
        _e = endog - exog @ b
        # explained variance
        r2 = 1 - np.sum(e**2) / np.sum(endog**2)
        _r2 = 1 - np.sum(_e**2) / np.sum(endog**2)
        r2s.append(_r2)
        print(f"Layer-{layer} Explained variance: {r2}, ({_r2})")
        print("rank of X", rk)
        # check if the same neurons are activated in both tensors
        # H0: the same neurons are activated
        # H1: the same neurons are not activated
        f, p = f_oneway(*[endog[:, i] for i in range(endog.shape[1])])
        print(f"Layer-{layer} One-way ANOVA: {f} {p}")
    if plot is not None:
        sns.lineplot(x=range(len(r2s)+1), y=r2s)
        plt.title(f"Explained variance {plot.split('/')[-1].split('.')[0]}")
        plt.savefig(plot)

def paired_classifier(tensor1, tensor2, plot:str=None, seed:int=42):
    accs = []
    ranks = []
    for layer in list(tensor1.keys()):
        data = pd.DataFrame()
        data["t1"] = tensor1[layer]
        data["t2"] = tensor2[layer]
        # check size of tensors

        random_y = np.random.randint(0, 2, len(data))
        # swap t1 and t2 when random_y is 1
        data["category"] = random_y
        hidden = []
        for i, row in data.iterrows():
            if row["category"]:
                hidden.append(row["t1"].astype(np.float32) - row["t2"].astype(np.float32))
            else:
                hidden.append(row["t2"].astype(np.float32) - row["t1"].astype(np.float32))
        data["concat"] = hidden
        # logistic Regression
        # y = 1 / (1 + exp(-Xb)), with b = coefficients
        # split X and y into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(np.stack(data["concat"].to_numpy()), data["category"].to_numpy(), test_size=0.25, random_state=seed)
        # instantiate the model (using the default parameters)
        logreg = LogisticRegression(random_state=seed, penalty="l1", solver="saga", max_iter=1000)
        # fit the model with data
        logreg.fit(X_train, y_train)

        y_pred = logreg.predict(X_test)
        acc = np.mean(y_pred & y_test)
        print(f"Layer-{layer} Accuracy: {acc}")
        print(f"Layer-{layer} Coefficients: {np.sum(logreg.coef_ != 0)}")
        accs.append(acc)
        ranks.append(np.sum(logreg.coef_ != 0))
    if plot is not None:
        sns.lineplot(x=range(len(accs)+1), y=accs, label=f"{plot.split('/')[-1].split('.')[0]}")
        plt.title(f"Per layer logreg Accuracy")
        plt.savefig(plot)
        sns.lineplot(x=range(len(ranks)+1), y=ranks, label=f"{plot.split('/')[-1].split('.')[0]}")
        plt.title(f"Per layer logreg Coefficients")
        plt.savefig(plot.replace("acc", "coef") if "acc" in plot else plot[:-4] + "-coef.png")
        

def logreg(tensor1, tensor2, plot:str=None, seed:int=42):
    accs = []
    # skip layer 0, it is the embedding layer
    for layer in list(tensor1.keys()):
        x, y = prepare_data(tensor1[layer], tensor2[layer])
        # logistic Regression
        # y = 1 / (1 + exp(-Xb)), with y = endog, X = exog, b = coefficients
        # split X and y into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
        # instantiate the model (using the default parameters)
        logreg = LogisticRegression(random_state=seed)#, penalty="l2", solver="saga")
        # fit the model with data
        logreg.fit(X_train, y_train)

        y_pred = logreg.predict(X_test)
        acc = np.mean(y_pred & y_test)
        accs.append(acc)
        print(f"Layer-{layer} Accuracy: {acc}")
        print(f"Layer-{layer} Coefficients: {np.sum(logreg.coef_ != 0)}")


    if plot is not None:
        sns.lineplot(x=range(len(accs)+1), y=accs, label=f"{plot.split('/')[-1].split('.')[0]}")
        plt.title(f"Per layer logreg Accuracy")
        plt.savefig(plot)

def average_difference(tensor1, tensor2, save=None):
    vect = []
    for layer in list(tensor1.keys())[1:]:
        vect.append(np.mean(np.array(tensor1[layer]) - np.array(tensor2[layer])))
    if save is not None:
        # use pickle to save the vectors
        with open(save, "wb") as f:
            pickle.dump(vect, f)

def cosine_distance(tensor1, tensor2, save=None, label=None, title=None):
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
        sns.lineplot(x=range(1,len(cos_dist)+1), y=cos_dist, label=label, alpha=0.5)
        plt.title(title)
        plt.savefig(save)

def open_launch(f1,f2,func,shuffle_f1=False,kwargs={}):
    with open(f1, "rb") as f:
        tensor1 = pickle.load(f)
        # float32
        tensor1 = {k: [np.array(vv).astype(np.float32) for vv in v] for k, v in tensor1.items()}
        if shuffle_f1:
            tensor1 = {k: np.random.permutation(v) for k, v in tensor1.items()}

    with open(f2, "rb") as f2:
        tensor2 = pickle.load(f2)
        # float32
        tensor2 = {k: [np.array(vv).astype(np.float32) for vv in v] for k, v in tensor2.items()}
    if len(tensor1[list(tensor1.keys())[1]]) != len(tensor2[list(tensor2.keys())[1]]):
        print("WARNING: Tensors are not the same size, choosing smallest tensor")
        min_size = min(len(tensor1[list(tensor1.keys())[1]]), len(tensor2[list(tensor2.keys())[1]]))
        tensor1 = {k: v[:min_size] for k, v in tensor1.items()}
        tensor2 = {k: v[:min_size] for k, v in tensor2.items()}

    print(f"Comparing {f1} and {f2}")
    func(tensor1, tensor2, **kwargs)
    # plot_layer_distance(log_file)

if __name__ == "__main__":
    # codes = ["Mis7P","Mis7iP","Met7P","Met7iP","OLM7P"]
    # codes = ["Mis7","Mis7i","Met7","Met7i","OLM7"]
    codes = ["OLM7"]
    # Load pickled tensors
    base_path="/home/mmahaut/projects/paramem/hlayer2"
    for c in codes:
        # file1 = f"/home/mmahaut/projects/paramem/hlayer/{c}-ft-small.pickle"
        # # file2 = f"/home/mmahaut/projects/paramem/hlayer/{c}-icl.pickle"
        # file2 = f"/home/mmahaut/projects/paramem/hlayer/{c}-base.pickle"
        # log_file = f"/home/mmahaut/projects/paramem/logs/{c}-ft-icl.txt"
        # fig_file = f"/home/mmahaut/projects/paramem/logs/{c}-ft-icl-permacc-s.png"
        keys=[["sanity1", "sanity1"], ["sanity1", "sanity2"], ["sanity1", "sanity3"], ["sanity2", "sanity3"]]
        # keys=[["sanity1-ft", "sanity1-ft"], ["sanity1-ft", "sanity2-ft"], ["sanity1-ft", "sanity3-ft"], ["sanity2-ft", "sanity3-ft"]]
        # keys=[["sanity1-kn", "sanity1-kn"], ["sanity1-kn", "sanity2-kn"], ["sanity1-kn", "sanity3-kn"], ["sanity2-kn", "sanity3-kn"]]
        # keys=[["sanity1-ft-kn", "sanity1-ft-kn"], ["sanity1-ft-kn", "sanity2-ft-kn"], ["sanity1-ft-kn", "sanity3-ft-kn"], ["sanity2-ft-kn", "sanity3-ft-kn"]]
        # keys=[["nc", "nc-kn"], ["nc", "nc"], ["nc-kn", "nc-kn"], ["nc-ft", "nc-ft-kn"], ["nc-ft", "nc-ft"], ["nc-ft-kn", "nc-ft-kn"]]
        # keys=[["pile8-split1", "pile8-split2"], ["pile8-split1", "pile18-split1"], ["pile18-split1", "pile18-split1"], ["pile8-ft-split1", "pile8-ft-split2"], ["pile8-ft-split1", "pile18-ft-split1"], ["pile18-ft-split1", "pile18-ft-split1"]]
        # keys=[["pile8-ft", "pile8-ft"],["pile18-ft", "pile18-ft"],["pile8-ft", "pile18-ft"]]


        labels = ["same context and query", "different context, same query", "same context, different query", "different context, different query"]
        # labels = ["pile8 vs pile8", "pile8 vs pile18", "pile18 vs pile18", "pile8 vs pile8 (ft)", "pile8 vs pile18 (ft)", "pile18 vs pile18 (ft)"]
        for k in keys:
            # find file in base_path with k in the name
            f1 = [f for f in Path(base_path).rglob(f"{c}-{k[0]}.pickle")][0]
            f2 = [f for f in Path(base_path).rglob(f"{c}-{k[1]}.pickle")][0]
            save = f"/home/mmahaut/projects/paramem/logs/{c}-pile.png"
            open_launch(f1,f2,cosine_distance,shuffle_f1=k[0]==k[1],kwargs={"save":save, "label":labels.pop(0), "title":f"{c} average cosine distance"})
        plt.clf()
        # plot_layer_distance(log_file)
