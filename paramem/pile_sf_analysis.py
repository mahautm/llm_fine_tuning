import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
if __name__ == "__main__":
# dataframe columns are query,expected_answers,generated,direct_follow,exact_match,nli_factual,entropy
    base_path="/home/mmahaut/projects/paramem/data2/"
    file_prefix="pile8"
    # models=["Mis7","Mis7i","Met7","Met7i","OLM7"]
    models=["Mis7","Met7","Met7i","OLM7"]

    all_df=pd.DataFrame()
    for model in models:
        df=pd.read_csv(f"{base_path}{file_prefix}_{model}.csv")
        df["ft"]=["base"]*len(df)
        df_ft=pd.read_csv(f"{base_path}{file_prefix}ft_{model}.csv")
        df_ft["ft"]=["fine-tuned"]*len(df_ft)
        df=pd.concat([df,df_ft]) 
        df["entropy"]=df["entropy"].apply(lambda x: pd.eval(x)[0])
        df["model"]=[model]*len(df)
        all_df=pd.concat([all_df,df])
    # barplot both entropies
    sns.boxplot(data=all_df, x="ft", y="entropy", hue="model")
    print(all_df.groupby(["ft","model"])["entropy"].mean())
    # compute percentage of exact matches
    plt.savefig(f"entropy_{file_prefix}.png")

