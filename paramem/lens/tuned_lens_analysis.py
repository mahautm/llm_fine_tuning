import torch
import typer
from typing import Optional
from tuned_lens.nn.lenses import TunedLens
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.plotting import PredictionTrajectory
from paramem.data import prepare_data, load_csv_data, load_pile_data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import pickle


def plot_ranks(df, save_path="my_lenses/ranks.png", label=None):
    n_layers = len(df.iloc[0]["ranks"])
    # df["ranks"] = df["ranks"].apply(lambda x: [sum(x[i])/len(x[i]) for i in range(n_layers)])
    df["ranks"] = df["ranks"].apply(lambda x: [y[-1] for y in x])
    df[[f"layer_{i}" for i in range(n_layers)]] = pd.DataFrame(df["ranks"].tolist(), index= df.index)
    df = pd.melt(df, id_vars=["ranks"], value_vars=[f"layer_{i}" for i in range(n_layers)], var_name="Layer", value_name="Rank")
    # make columns into categorical variables for seaborn
    sns.lineplot(data=df, x="Layer", y="Rank", label=label)
    # angle x labels
    plt.xticks(rotation=45)
    plt.title("Ranks")
    # axis
    plt.xlabel("Layer")
    plt.ylabel("Rank")
    plt.savefig(save_path)

def main(
    model_name:str,
    lens_name:str,
    batch_size:int,
    input_key:str,
    dataset_path:str,
    outpath:str,
    provide_answer:bool,
    n_samples:int,
    ans_in_prompt:bool,
    device:Optional[str] = "cpu",
    dry_run:bool = False,
    plot_save_path:Optional[str] = None,
    use_kn_threshold:bool = True,
    kn_threshold:Optional[float] = None,
    ):
    device = torch.device(device)
    # To try a diffrent modle / lens check if the lens is avalible then modify this code
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id=lens_name, map_location=device)
    tuned_lens = tuned_lens.to(device)

    # dataset_path = "/home/mmahaut/projects/paramem/data/wikidata_Met7.csv"
    # input_key = "query"
    if dataset_path.endswith(".csv"):
        _d = load_csv_data(dataset_path, input_key=input_key, threshold=kn_threshold, threshold_knowledge=use_kn_threshold)
        df = pd.DataFrame(_d)
        print(df.head(1))
    elif dataset_path.endswith(".txt"):
        _d = load_pile_data(dataset_path, input_key=input_key)
        df = pd.DataFrame(_d)
    else:
        raise ValueError("data_file should be a csv or a txt file")
    # keep only 10
    if dry_run:
        df = df.sample(100)
    ranks = []
    for i in tqdm.tqdm(range(len(df))):
        t1 = df.iloc[i][input_key] + df.iloc[i]["expected_answers"]
        ans_size = len(tokenizer.encode(df.iloc[i]["expected_answers"]))
        toked_inputs = tokenizer.encode(t1)
        toked_targets = toked_inputs[1:] + [tokenizer.eos_token_id]
        pred = PredictionTrajectory.from_lens_and_model(
            tuned_lens,
            model,
            tokenizer=tokenizer,
            input_ids=toked_inputs,
            targets=toked_targets,
        ).slice_sequence([-ans_size, -ans_size])
        ranks.append(pred.rank().stats)
    df["ranks"] = ranks
    print(df.head(1)["ranks"])
    # save ranks
    with open(outpath, 'wb') as f:
        pickle.dump(df, f)
    if plot_save_path is not None:
        plot_ranks(df, plot_save_path, label=f"kn_threshold={kn_threshold}")

if __name__ == "__main__":
    # typer.run(main)
    main(
        model_name="meta-llama/Meta-Llama-3-8B",
        lens_name="./my_lenses/Met7",
        batch_size=1,
        input_key="query",
        dataset_path="./data/wikidata_Met7.csv",
        outpath="my_lenses/ranks.pkl",
        provide_answer=False,
        n_samples=1,
        ans_in_prompt=False,
        device="cpu",
        dry_run=True,
        plot_save_path="my_lenses/ranks.png",
        kn_threshold=0.9
    )
    main(
        model_name="meta-llama/Meta-Llama-3-8B",
        lens_name="./my_lenses/Met7",
        batch_size=1,
        input_key="query",
        dataset_path="./data/wikidata_Met7.csv",
        outpath="my_lenses/ranks.pkl",
        provide_answer=False,
        n_samples=1,
        ans_in_prompt=False,
        device="cpu",
        dry_run=True,
        plot_save_path="my_lenses/ranks.png",
        kn_threshold=None
    )
    # with open("my_lenses/ranks.pkl", 'rb') as f:
    #     df = pickle.load(f)
    # plot_ranks(df)