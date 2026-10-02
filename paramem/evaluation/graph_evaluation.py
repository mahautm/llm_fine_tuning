from pathlib import Path
from typing import List, Optional
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def find_completed_checkpoints(model_dir: str, step: Optional[int] = None) -> List[dict]:
    """
    Find all completed checkpoints in the given model directory and extract metadata.
    
    Args:
        model_dir (str): Path to the model directory.
        step (Optional[int]): Specific step to look for. If None, all steps are considered.
        
    Returns:
        List[dict]: List of dictionaries with checkpoint metadata.
    """
    completed_checkpoints = []
    model_path = Path(model_dir)
    
    if not model_path.exists():
        print(f"Model directory {model_dir} does not exist.")
        return completed_checkpoints
    
    for checkpoint in model_path.glob("checkpoint-*"):
        checkpoint_name = checkpoint.name
        if step is not None and f"checkpoint-{step}" != checkpoint_name:
            continue
        
        slurm_logs = checkpoint / "slurm_logs"
        if slurm_logs.exists():
            for log_file in slurm_logs.glob("*.out"):
                with open(log_file, 'r') as f:
                    content = f.read()
                    if "COMPLETED" in content:
                        # Extract metadata from model directory and checkpoint
                        # Example: /home/mmahaut/projects/paramem/models2/Mistral-7B-v0.3-fsdp-0-pile/checkpoint-1000
                        model_dir_name = model_path.name
                        parts = model_dir_name.split('-')
                        model_name = parts[0] if len(parts) > 0 else ""
                        use_lora = parts[4] if len(parts) > 4 else ""
                        dataset = parts[5] if len(parts) > 5 else ""
                        checkpoint_step = checkpoint_name.replace("checkpoint-", "")
                        completed_checkpoints.append({
                            "model_name": model_name,
                            "use_lora": use_lora,
                            "dataset": dataset,
                            "checkpoint_step": checkpoint_step,
                            "logfile_path": str(log_file),
                            "log_content": content
                        })
    return completed_checkpoints

# there are two types of log files:
# 1. slurm_logs/ID_matrixent.out
# 2. slurm_logs/performance_evaluation.out

# for the first type, we are interested in results which look like:
# dataset_name_intrinsic_dimension:
#   layer_X: [value1, value2, value3, value4, value5]
#   ...
#   layer_Y: [value1, value2, value3, value4, value5]
# dataset_name_entropy:
#   layer_X: value
#   ...
#   layer_Y: value
# for example:
# wikidata_Mis7_train_intrinsic_dimension:
#   layer_0: [ 0.37  0.32  0.33  0.91 35.3 ]
#     ...
#   layer_31: [8.82 8.28 7.28 6.87 6.64]
#   layer_32: [7.83 7.47 6.35 6.25 5.82]
# wikidata_Mis7_train_entropy:
#   layer_0: 7.152555099310121e-07
#     ...
#   layer_31: 1.0728830375228426e-06
#   layer_32: 1.4305104514278355e-06

# for the second type, we are interested in results which look like:
# Benchmark Results:
# mmlu_accuracy: 0.000
# arc_accuracy: 0.000
# hellaswag_accuracy: 0.000
# truthfulqa_accuracy: 0.000
# gsm8k_accuracy: 0.000
# winogrande_accuracy: 0.000
# openbookqa_accuracy: 0.000
# lambada_accuracy: 0.000
# COMPLETED EVALUATION

def parse_id_matrixent_log(log_content: str) -> dict:
    """
    Parse the intrinsic dimension and entropy from the log content.
    we use regular expressions to extract the values.
    re allows us to find dataset name and layer numbers dynamically, as well as differenciate between intrinsic dimension and entropy.

    Args:
        log_content (str): Content of the log file.
        
    Returns:
        dict: Parsed intrinsic dimension and entropy values.
    """
    results = {} # ID/entropy, Layer 1...
    id_pattern = re.compile(r"(?P<dataset_name>[\w_]+)_intrinsic_dimension:\s*(?P<layers>(?:\s*layer_\d+:\s*\[.*?\]\s*)+)", re.DOTALL)
    entropy_pattern = re.compile(r"(?P<dataset_name>[\w_]+)_entropy:\s*(?P<layers>(?:\s*layer_\d+:\s*[-+]?\d*\.\d+e[-+]?\d+\s*)+)", re.DOTALL)
    id_matches = id_pattern.finditer(log_content)
    entropy_matches = entropy_pattern.finditer(log_content)    

    for match in id_matches:
        dataset_name = match.group("dataset_name")
        layers = match.group("layers")
        layer_values = re.findall(r"layer_(\d+):\s*\[(.*?)\]", layers)
        for layer, values in layer_values:
            results[f"layer_{layer}"] = {
                "intrinsic_dimension": [float(v) for v in values.split()],
                "entropy": None,
                "dataset_name": dataset_name
            }

    for match in entropy_matches:
        dataset_name = match.group("dataset_name")
        layers = match.group("layers")
        layer_values = re.findall(r"layer_(\d+):\s*([-+]?\d*\.\d+e[-+]?\d+)", layers)
        for layer, value in layer_values:
            if f"layer_{layer}" in results:
                results[f"layer_{layer}"]["entropy"] = float(value)

    return results

def parse_performance_evaluation_log(log_content: str) -> dict:
    ### example:
    # Benchmark Results:
    # mmlu_accuracy: 0.000
    # arc_accuracy: 0.000
    # hellaswag_accuracy: 0.000
    # truthfulqa_accuracy: 0.000
    # gsm8k_accuracy: 0.000
    # winogrande_accuracy: 0.000
    # openbookqa_accuracy: 0.000
    # lambada_accuracy: 0.000
    # COMPLETED EVALUATION
    results = {}
    pattern = re.compile(r"(?P<metric>[\w_]+):\s*(?P<value>[-+]?\d*\.\d+|\d+)", re.MULTILINE)
    matches = pattern.finditer(log_content)
    for match in matches:
        metric = match.group("metric")
        value = match.group("value")
        results[metric] = float(value)
    return results

def plot_intrinsic_dimension(
    parsed_id_data: List[dict],
    output_path: str = "intrinsic_dimension_plot.png"
):
    """Plot the intrinsic dimension from the parsed data.
    Args:
        parsed_id_data (List[dict]): List of parsed intrinsic dimension data.
        output_path (str): Path to save the plot.
    """    
    # pandas shenanigans to create a DataFrame + plotting
    df = pd.DataFrame(parsed_id_data)
    df["model_name"] = [log_data['model_name'] for log_data in id_logdatas]
    df["use_lora"] = [log_data['use_lora'] for log_data in id_logdatas]
    df["dataset"] = [log_data['dataset'] for log_data in id_logdatas]
    df["checkpoint_step"] = [log_data['checkpoint_step'] for log_data in id_logdatas]
    # make the layer_0, layer_1, ... columns elements of LAYER column
    df = df.melt(id_vars=['model_name', 'use_lora', 'dataset', 'checkpoint_step'],
                 var_name='layer', 
                 value_name='values')
    df["layer"] = df["layer"].str.replace("layer_", "").astype(int)
    df["intrinsic_dimension"] = df["values"].apply(lambda x: x.get("intrinsic_dimension") if isinstance(x, dict) else None)
    df["entropy"] = df["values"].apply(lambda x: x.get("entropy") if isinstance(x, dict) else None)
    df.drop(columns=['values'], inplace=True)

    n_scales = len(df.iloc[0]["intrinsic_dimension"]) if isinstance(df.iloc[0]["intrinsic_dimension"], list) else 0
    for scale in range(n_scales):
        df[f"{2**(scale+1)}"] = df["intrinsic_dimension"].apply(lambda x: x[scale] if isinstance(x, list) else None)
    df.drop(columns=['intrinsic_dimension'], inplace=True)

    df=df.melt(id_vars=['model_name', 'use_lora', 'dataset', 'checkpoint_step', 'layer', 'entropy'],
                 value_vars=[f"{2**(scale+1)}" for scale in range(n_scales)],
                 var_name='scale',
                 value_name='Intrinsic Dimension')
    print(df.head())

    # Plotting
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df[df["checkpoint_step"]=="4000"], x='layer', y='Intrinsic Dimension', hue='scale', style='use_lora', markers=True, dashes=False)
    plt.title('Intrinsic Dimension Across Layers')
    plt.xlabel('Layer')
    plt.ylabel('Intrinsic Dimension')
    plt.legend(title='Model Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved as '{output_path}'")

def plot_performance_evaluation(
    parsed_perf_data: List[dict],
    output_path: str = "performance_evaluation_plot.png"
):
    """Plot the performance evaluation from the parsed data.
    Args:
        parsed_perf_data (List[dict]): List of parsed performance evaluation data.
        output_path (str): Path to save the plot.
    """
    df = pd.DataFrame(parsed_perf_data)
    df["model_name"] = [log_data['model_name'] for log_data in perf_log_datas]
    df["use_lora"] = [log_data['use_lora'] for log_data in perf_log_datas]
    df["dataset"] = [log_data['dataset'] for log_data in perf_log_datas]
    df["checkpoint_step"] = [log_data['checkpoint_step'] for log_data in perf_log_datas]

    # Melt the DataFrame to have a long format
    df = df.melt(id_vars=['model_name', 'use_lora', 'dataset', 'checkpoint_step'],
                 var_name='metric', 
                 value_name='value')
    df["checkpoint_step"] = df["checkpoint_step"].astype(int)

    # Plotting
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='checkpoint_step', y='value', hue='metric', style='use_lora', markers=True, dashes=False)
    plt.title('Performance Evaluation Metrics')
    plt.xlabel('Checkpoint Step')
    plt.ylabel('Acc.')
    plt.xticks(rotation=45)
    plt.legend(title='Model Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved as '{output_path}'")

if __name__ == "__main__":
    # Example usage
    model_dir = "/home/mmahaut/projects/paramem/models2/Mistral-7B-v0.3-fsdp-1-pile"
    log_datas = find_completed_checkpoints(model_dir, step=None)

    id_logdatas = [log_data for log_data in log_datas if "ID_matrixent" in log_data['logfile_path']]
    parsed_id_data = [parse_id_matrixent_log(log_data['log_content']) for log_data in id_logdatas]
    plot_intrinsic_dimension(parsed_id_data, output_path="intrinsic_dimension_plot.png")
    # perf_log_datas = [log_data for log_data in log_datas if "performance_evaluation" in log_data['logfile_path']]
    # parsed_perf_data = [parse_performance_evaluation_log(log_data['log_content']) for log_data in perf_log_datas]
    # plot_performance_evaluation(parsed_perf_data, output_path="performance_evaluation_plot.png")


