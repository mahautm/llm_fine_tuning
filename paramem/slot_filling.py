import logging
from glob import glob
from pathlib import Path
import random
import pandas as pd
import torch
import tqdm
import typer
from typing import List, Optional
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
from paramem.data import prepare_data, no_context_to_csv_format
from paramem.load_distrib_model import load_distrib_model
from data import load_csv_data, load_pile_data

import accelerate
app = typer.Typer()

class NLI:
    # NLI v1 --> DeBerta
    def __init__(self) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "cross-encoder/nli-deberta-v3-large"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-large")

    def check_equivalence(self, batch, targets):
        # two sentences are supposed equivalent iff they entail each other
        # as in https://arxiv.org/pdf/2302.09664.pdf (Kuhn et al.) we approach semantic equivalence
        # as an equivalence class which is reflexive, symmetric and transitive
        # we therefore only need to check a sentence is equivalent to one sentence in a group
        # to know if it belongs to that group
        # TODO: verify if in practice, testing a few members of the group increases accuracy
        toked = self.tokenizer(
            batch, targets, padding=True, truncation=True, return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            scores = self.model(**toked).logits
            entails = scores.argmax(dim=1) == 1  # 0 is contradiction, 1 is entailment, 2 is neutral
        # check entailment in other direction, skipping those already eliminated
        batch2 = [batch[i] for i, e in enumerate(entails) if e]
        targets2 = [targets[i] for i, e in enumerate(entails) if e]

        if len(batch2) != 0:
            toked2 = self.tokenizer(
                batch2,
                targets2,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                scores2 = self.model(**toked2).logits
                entails2 = scores2.argmax(dim=1) == 1
            # combine
            for i, e in enumerate(entails):
                _count = 0
                if e:
                    entails[i] = entails2[_count].cpu().item()
                    _count += 1
        return entails.cpu().tolist()

def slot_fill(model, tokenizer, input_sentence, expected_output, max_new_tokens, device, nli=None, collect_entropy=False, do_sample=False):
    """
    get a batch of sentences from sentence with missing completion
    generate an amount of words greedily that corresponds to the possible answer
    check and label if answer is correct

    Parameters
    ----------
    model : torch.nn.Module
    tokenizer : transformers.PreTrainedTokenizerFast
    input_sentence : list of str
    expected_output : list of str
    max_new_tokens : int
    device : torch.device  
    

    Returns
    -------
    direct_follow : list of bool
    exact_match : list of bool
    nli_factual : list of bool
    generated : list of str
    """
    assert len(input_sentence) == len(expected_output), "input and expected output must have the same length"

    # prepare model input and run model
    input_tok = tokenizer(
        input_sentence, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    gen_o = model.generate(
        **input_tok, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, do_sample=do_sample,
    )
    if collect_entropy:
        logits = model(**input_tok).logits
        entropy = torch.distributions.Categorical(logits=logits).entropy().cpu().tolist()
    else:
        entropy = None
    generated = tokenizer.batch_decode(gen_o, skip_special_tokens=True)

    # Testing the query is not in the generated text, some models do that
    if input_sentence[0] in generated[0]:
        for i in range(len(input_sentence)):
            if input_sentence[i] in generated[i]:
                generated[i] = generated[i].replace(input_sentence[i], "")

    # direct follow, if the expected output is the very next word
    direct_follow = []
    for i in range(len(input_sentence)):
        for j in range(len(expected_output[i])):
            _exp = " " + expected_output[i][j] # !! FIXME hardcoded common missing space
            _len = min(len(generated[i]), len(_exp))
            if _exp[:_len] == generated[i][:_len]:
                direct_follow.append(True)
                break
        if len(direct_follow) != i + 1:
            direct_follow.append(False)

    # exact matching, if the expected output is in the generated text
    exact_match = []
    for i, _gen in enumerate(generated):
        exact_match.append(any([e in _gen for e in expected_output[i]]))

    # nli matching (takes into account synonims, for that reason we only check one of the expected outputs)
    _truth = [
        f"{input_sentence[i]} {expected_output[i][0]}" for i in range(len(input_sentence))
    ]
    if nli is None:
        nli = NLI()
    nli_factual = nli.check_equivalence(_truth, generated)

    # logging
    logging.debug(f"input: {input_sentence}")
    logging.debug(f"generated: {generated}")
    logging.debug(f"expected: {expected_output}")
    logging.debug(f"direct_follow: {direct_follow}")
    logging.debug(f"exact_match: {exact_match}")
    logging.debug(f"nli_factual: {nli_factual}")
    if collect_entropy:
        logging.debug(f"entropy: {entropy}")


    return direct_follow, exact_match, nli_factual, generated, entropy

@app.command()
def test_generation(
    outpath="./data/wikidata_sf.csv",
    log_path="./logs/improved_sf",
    dataset_path="./benchmark/train.jsonl",
    ans_in_prompt: bool=False,
    threshold_knowledge: bool=True,
    model_name="tiiuae/falcon-7b-instruct",
    checkpoint_path: str=None,
    instruction: str="",
    num_return_sequences:int=10,
    max_new_tokens:int=25,
    intermediate_saves=False,
    save_inputs:str=None,
    batch_size:int = 64,
    input_key:str="query",
    no_context:bool=False,
    sanity_check:bool=False,
    kn_threshold:Optional[float]=None,
    collect_entropy:bool=False
):
    """
    This script is used to test the slot filling capabilities of a model. It generates completions for a prompt and checks if the completions are correct. It saves the results in a csv file.

    Parameters
    ----------
    outpath : str
        The path to save the results.
    log_path : str
        The path to save the logs.
    dataset_path : str
        The path to the dataset.
    ans_in_prompt : bool
        Whether the answer is included in the prompt. (useful to check if the model can use context)
    only_unknowns : bool
        Whether to exclude already known data from the dataset. Will exclude ligns which display True in the exact_match and nli_factual columns.
    model_name : str
        The name of the model to use.
    checkpoint_path : str
        The path to the model checkpoint to use.
    instruction : str
        The instruction to give to the model. This is the first part of the prompt
        ex: "Accurately fill in the following sentence with the correct word, creating factual sentences as in the examples:"
    num_return_sequences : int
        The number of completions to generate for each prompt.
    max_new_tokens : int
        The maximum number of tokens to generate for each completion.
    intermediate_saves : bool
        Whether to save the results in a partial file during the generation.
    save_inputs : str
        The path to save the inputs, providing a trace of input sequences with their prompts. If None, the inputs are not saved.
    batch_size : int
        The batch size to use for generation.
    input_key : str
        The key to use to access the input data in the dataset.
    no_context : bool
        Whether to remove the context from the dataset. If True, the dataset will be transformed to a format without context.
    """
    ## INIT
    # dirs and files
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)
    Path(log_path).mkdir(exist_ok=True, parents=True)
    if intermediate_saves:
        partial_save_path = f"{outpath}.partial"
    logging.basicConfig(
        level=logging.DEBUG,
        filename=f"{log_path}/slot_filling.log",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # prompt
    # model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="balanced")#, load_in_4bit=True)
    if checkpoint_path is not None:
        try:
            state_dict = torch.load(checkpoint_path)
            model.load_state_dict(state_dict)
        except Exception as e:
            logging.error(f"Error loading model checkpoint: {e} Trying to load as distributed model.")
            model = load_distrib_model(model, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # dataset preparation # refactor, not perfect yet
    if "jsonl" in dataset_path:
        df = pd.read_json(dataset_path, lines=True)
        df = prepare_data(df, provide_answer=False, instruction=instruction, ans_in_prompt=ans_in_prompt)
    elif "csv" in dataset_path:
        df = pd.DataFrame(load_csv_data(dataset_path, threshold=kn_threshold, sanity_check=sanity_check, no_context=no_context, input_key=input_key, threshold_knowledge=threshold_knowledge))
    elif ".txt" in dataset_path:
        inputs = load_pile_data(dataset_path, input_key)
        df = pd.DataFrame(inputs)
    else:
        raise ValueError("Dataset format not supported")
        
    if save_inputs is not None:
        df.to_csv(save_inputs, index=False)
    tr_dataset = Dataset.from_pandas(df)
    def collate_fn(batch):
        return {
            key: [item[key] for item in batch] for key in batch[0].keys()
        }
    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    nli = NLI()
    out = {key: [] for key in ["generated", "direct_follow", "exact_match", "nli_factual"]}
    if collect_entropy:
        out["entropy"] = []
    ## END INIT

    ## GENERATION
    for batch in tqdm.auto.tqdm(tr_dataloader):
        input_sentence = batch[input_key]
        expected_outputs = batch["expected_answers"]
        direct_follow, exact_match, nli_factual, generated, entropy = slot_fill(model, tokenizer, input_sentence, expected_outputs, max_new_tokens, device, nli, collect_entropy=collect_entropy, do_sample=num_return_sequences>1)
        out["generated"].extend(generated)
        out["direct_follow"].extend(direct_follow)
        out["exact_match"].extend(exact_match)
        out["nli_factual"].extend(nli_factual)
        if collect_entropy:
            out["entropy"].extend(entropy)

        if partial_save_path is not None:
            pd.DataFrame(out).to_csv(partial_save_path, index=False)
    ## END GENERATION

    ## SAVE
    complete = pd.concat([df, pd.DataFrame(out)], axis=1)
    complete.to_csv(outpath, index=False)

@app.command()
def check_acc(
    model_path: str,
    data_file: str,
    data_file2: str=None,
    instruction: str="",
    outpath: str="./data/sf_acc.csv",
    checkpoint_path: str=None,
    examples:Optional[List[str]] = typer.Option(None),
    ans_in_prompt: bool=False,
    use_nli: bool=False,
    n_samples:int=None,
    num_return_sequences:int=10,
    max_new_tokens:int=25,
    ):
    print(model_path, data_file, data_file2, instruction, outpath, examples, ans_in_prompt, use_nli, n_samples, num_return_sequences, max_new_tokens)
    prompt = {
        "examples": examples,
        "instruction": instruction,
    }
    _data = pd.read_csv(data_file)
    _data2 = pd.read_csv(data_file2) if data_file2 is not None else None
    _data = prepare_data(_data, _data2, provide_answer=False, prompt=prompt, ans_in_prompt=ans_in_prompt)
    # drop and regenerate index
    _data = _data.reset_index(drop=True)
    # sample if needed
    if n_samples is not None:
        _data = _data.sample(n_samples)
    # drop "Unnamed: 0" column
    if "Unnamed: 0" in _data.columns:
        _data = _data.drop(columns=["Unnamed: 0"])

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="balanced", torch_dtype=torch.float16)
    if checkpoint_path is not None:
        try:
            state_dict = torch.load(checkpoint_path)
            model.load_state_dict(state_dict)
        except Exception as e:
            logging.error(f"Error loading model checkpoint: {e} Trying to load as distributed model.")
            model = load_distrib_model(model, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    success = evaluate_all_slot_filling(
            model, tokenizer, _data, use_nli=use_nli, partial_save_path=outpath, num_return_sequences=num_return_sequences, max_new_tokens=max_new_tokens
        )
    success=pd.DataFrame(success)
    success.to_csv(outpath)
    _temp_reload = pd.read_csv(outpath) 

    print(f"Accuracy: {len(prepare_data(_temp_reload))/len(_temp_reload)}")
    

if __name__ == "__main__":
    app()
