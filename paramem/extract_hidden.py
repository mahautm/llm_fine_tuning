# from Marco
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import pickle
import sys
import typer
from typing import List, Optional
import pandas as pd
from data import prepare_data, single_example_to_csv_format, no_context_to_csv_format, load_csv_data, load_pile_data
from pathlib import Path
from datasets import load_dataset
from paramem.load_distrib_model import load_distrib_model
import logging

def model_pass(raw_inputs, tokenizer, model, device, save_attention=False):
    # for now, I don't constraint to a max length
    inputs = tokenizer(raw_inputs, padding=True, return_tensors="pt").to(device)

    # making sure we collect hidden state after the last "true" token
    # and not a padding token
    last_true_token_indices = []
    for att_mask in inputs.attention_mask:
        if not(0 in att_mask):
            last_true_token_indices.append(len(att_mask)-1)
        else:
            last_true_token_indices.append(att_mask.tolist().index(0)-1)

    with torch.no_grad():
        g = model(**inputs,output_hidden_states=True,output_attentions=True)
        hidden_states = g.hidden_states[1:] # removing the embedding layer
        attentions = g.attentions

    
    per_layer_activations = []
    per_layer_attentions = []
    for layer_idx, raw_activation in enumerate(hidden_states): # shape of hidden_states: layers x batch_size x tokens x d
        # shape of attention: layers * batch_size * num_heads * sequence_length * sequence_length
        # traversing layers
        last_token_activations = []
        last_attentions = []
        for i in range(len(last_true_token_indices)):
            # traversing batch items
            last_token_activation = raw_activation[i][last_true_token_indices[i]].cpu().numpy()
            last_token_activations.append(last_token_activation)
            if save_attention:
                last_attention = attentions[layer_idx][i].cpu().numpy()
                last_attentions.append(last_attention)
        # appending a list of all the last-token-activations of the current layer to a list of lists
        per_layer_activations.append(last_token_activations)
        if save_attention:
            per_layer_attentions.append(last_attentions)

    if not save_attention:
        per_layer_attentions = None
    return(per_layer_activations, per_layer_attentions)

def main(
    model_name:str,
    batch_size:int,
    data_file:str, 
    override_data_file:bool=False,
    out_pickle_prefix:str="",
    checkpoint_path:str=None,
    input_key:str="query",
    sanity_check:bool=False,
    no_context:bool=False,
    kn_threshold:Optional[float]=None,
    save_attention:bool=False
    ):
    Path(out_pickle_prefix).parent.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info("device is " + device, file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",torch_dtype=torch.float16)
    if checkpoint_path is not None:
        try:
            state_dict = torch.load(checkpoint_path)
            model.load_state_dict(state_dict)
        except Exception as e:
            logging.error(f"Error loading model checkpoint: {e} Trying to load as distributed model.")
            model = load_distrib_model(model, checkpoint_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if override_data_file:
        _temp_base=out_pickle_prefix.replace("-ft","")
        _df_path=_temp_base + "inputs.txt"
        if Path(_df_path).exists():
            logging.warning(f"Overriding data path {data_file} with {_df_path} which already exists")
            data_file=_df_path
        with open(data_file, "r") as f:
            inputs = f.readlines()

    elif ".csv" in data_file:
        inputs = load_csv_data(data_file, threshold=kn_threshold, sanity_check=sanity_check, no_context=no_context, input_key=input_key)[input_key]
    elif ".txt" in data_file:
        inputs = load_pile_data(data_file, input_key)[input_key]

    elif data_file in ["tiny_shakespeare", "tiny_shakespeare_test"]:
        inputs = load_dataset(data_file)["train"]["text"]
        # # use karpathy/tiny_shakespeare as input
        # inputs = load_dataset("tiny_shakespeare")["train"]["text"]
        # n_chars = 1000
        # # use map to cut after n tokens, and add n+1 as expected output
        # _inp = pd.DataFrame()
        # _inp[input_key] = list(map(lambda x: x.split(" ")[:n_chars], inputs))
        # _inp["expected_answers"] = list(map(lambda x: x[n_chars:n_chars+1], inputs))

    else:
        raise ValueError("data_file should be a csv, a txt file, or a huggingface dataset name")
    # save
    with open(out_pickle_prefix + "inputs.txt", "w") as f:
        f.write("\n".join(inputs))
    cases_count = len(inputs)
    first_index = 0
    current_batch_size = batch_size
    states = dict()
    attentions = dict()
    if (current_batch_size>cases_count):
        current_batch_size = cases_count
    while ((first_index+current_batch_size)<cases_count):
        layer_output, attention_output = model_pass(inputs[first_index:first_index+current_batch_size], tokenizer, model, device, save_attention=save_attention)
        for i in range(len(layer_output)):
            if not i in states:
                states[i] = []
            states[i] = states[i] + layer_output[i]
        if save_attention:
            for i in range(len(attention_output)):
                if not i in attentions:
                    attentions[i] = []
                attentions[i] = attentions[i] + attention_output[i]
        first_index=first_index+current_batch_size
    # in case cases_count is not a multiple of batch_size
    if first_index<cases_count:
        layer_output, attention_output = model_pass(inputs[first_index:cases_count], tokenizer, model, device)
        for i in range(len(layer_output)):
            states[i] = states[i] + layer_output[i]
        if save_attention:
            for i in range(len(attention_output)):
                attentions[i] = attentions[i] + attention_output[i]

    out_pickle_name = out_pickle_prefix + ".pickle"
    with open(out_pickle_name, 'wb') as f:
        pickle.dump(states, f)
    if save_attention:
        att_pickle_name = out_pickle_prefix + "_att.pickle"
        with open(att_pickle_name, 'wb') as f:
            pickle.dump(attentions, f)

if __name__ == "__main__":
    typer.run(main)
