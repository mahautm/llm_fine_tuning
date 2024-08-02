from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

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
            entails = [e and e2 for e, e2 in zip(entails, entails2)]
        return entails

class paraphraser(torch.nn.Module):
    # we use generative models to paraphrase
    def __init__(self, model, tokenizer) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, prompt, batch, max_new_tokens):
        batch = [prompt.replace("$sentence", t) for t in batch["template"]]
        encoded_source_sentence = self.tokenizer(
            batch, padding=True, return_tensors="pt"
        )
        # for k, v in encoded_source_sentence.items():
        #     encoded_source_sentence[k] = v.to(accelerator.device)
        generated_target_tokens = self.model.generate(
            **encoded_source_sentence,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        target_sentence = self.tokenizer.batch_decode(
            generated_target_tokens, skip_special_tokens=True
        )
        return target_sentence


def paraphrase(
    model_name, dataset_path, prompt, batch_size=100, save_frq=100, save_path=None
):
    Path(save_path).mkdir(exist_ok=True, parents=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="balanced", load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # check if deepspeed is setup on accelerator
    paraphraser_model = paraphraser(model, tokenizer)
    if ".csv" not in dataset_path:
        dataset = load_from_disk(dataset_path)
    else:
        dataset = load_dataset("csv", data_files=dataset_path, split="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    # model, tokenizer, dataloader = accelerator.prepare(model, tokenizer, dataloader)

    # index through df, add paraphrase to new column
    batches, paraphs = [], []
    for i, batch in enumerate(tqdm(dataloader)):
        paraph = paraphraser_model(prompt, batch, 400)
        # batch to cpu to df
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cpu()
        batch = pd.DataFrame(batch)
        batches.append(batch)
        paraphs.append(pd.DataFrame(paraph, columns=["paraphrase"]))
        if len(batches) % save_frq == 0 or len(batches) == len(dataloader):
            df_batches = pd.concat(batches)
            df_batches["paraphrase"] = pd.concat(paraphs)
            df_batches.to_csv(
                Path(save_path)
                / f"paraph_{i//save_frq:03d}.csv"
            )
            batches, paraphs = [], []
    # df_batches = pd.concat(batches)
    # df_batches["paraphrase"] = paraphs
    # return df_batches


def filter_raw_paraphrase(raw_paraph, text, nli=None):
    # remove last line if no full stop
    # if "." not in raw_paraph.split("\n")[-1] or "?" not in raw_paraph.split("\n")[-1]:
    #     raw_paraph = raw_paraph.rsplit("\n", 1)[0]
    raw_paraph = raw_paraph.rsplit("Paraphrases:", 1)[1]
    # if not "-" in raw_paraph:
        #find text between \n and .
    res = []
    for t in raw_paraph.split("\n")[1:]:
        if "." in t:
            res.append(t.split(".")[1])
                

    clean_paraphs = res #raw_paraph.replace("\n", "").split("- ")[1:]

    if nli is not None and len(clean_paraphs) > 1:
        nli_mask = nli.check_equivalence(clean_paraphs, [text] * len(clean_paraphs))
        clean_paraphs = list(set([p for p, m in zip(clean_paraphs, nli_mask) if m and p != text]))
    return clean_paraphs

def format_for_marco(path, save_path):
    lama_rels = ['P1001', 'P101', 'P103', 'P106', 'P108', 'P127', 'P1303', 'P131', 'P1376', 'P138', 'P140', 'P1412', 'P159', 'P17', 'P176', 'P178', 'P19', 'P190', 'P20', 'P264', 'P27', 'P36', 'P37', 'P39', 'P47', 'P276', 'P279', 'P361', 'P364', 'P413', 'P449', 'P463', 'P495', 'P530', 'P740', 'P937']
    rel_num=[]
    with open(save_path, "w") as save_file:
        for f in Path(path).glob("*.csv"):
            data = pd.read_csv(f)
            for i, row in data.iterrows():
                if row["relation"] in lama_rels:
                    continue
                if row["relation"] not in rel_num:
                    rel_num.append(row["relation"])
                    save_file.write(f"{row['relation']}\n")
                    save_file.write(f"L: {row['label']}\n")
                    save_file.write(f"D: {row['description']}\n")
                    save_file.write(f"{row['template']} [Y]\n")
                save_file.write(f"{row['paraphrase'][1:]}\n")
    print(rel_num)

def marco_to_csv(path, save_path):
    with open(path, "r") as f:
        lines = f.readlines()
        rels = []
        labels = []
        descriptions = []
        templates = []
        paraphs = []
        for i, l in enumerate(lines):
            if l[0] == "R" or l[0] == "*":
                base = i
                rels.append(l.strip().split("R: " if "R: " in l else "*")[1].split(" ")[0])
                if "L: " in lines[i+1]:
                    labels.append(lines[i+1].strip().split("L: ")[1])
                    descriptions.append(lines[i+2].strip().split("D: ")[1])
                    templates.append(lines[i+3].strip())
                    paraphs.append([])
                else :
                    labels.append("")
                    descriptions.append("")
                    templates.append(lines[i+1][2:].strip())
                    paraphs.append([])  
                    paraphs[-1].append(lines[i+2][2:].strip())
                    paraphs[-1].append(lines[i+3][2:].strip())       
            elif i - base > 3:
                paraphs[-1].append(l.strip() if l[:2] != "S:" and l[:2] != "Q:" else l[2:].strip())
        df = pd.DataFrame({
            "relation": rels,
            "label": labels,
            "description": descriptions,
            "template": templates,
            "paraphrase": paraphs
        })
        df.to_csv(save_path, index=False)

if __name__ == "__main__":
    # prompt = "Please provide 50 DIFFERENT paraphrases of the folowing sentence, using [X] and [Y] as placehoders. Paraphrases must be in a list, with bullet points. Paraphrase must maintain the same meaning. Sentence: $sentence [Y]\nParaphrases:"
    # prompt = "Provide 10 DIFFERENT questions about [X] that can be answered with [Y]. Questions must be in a list, with bullet points. Questions must maintain the same meaning. Example: [X] is the capital of [Y]. Question: What is [X] the capital of? Answer: [Y]. Example: $sentence. Question:"
    # for f in [Path("/home/mmahaut/projects/parametric_mem/extended_paraphrases.csv")]:#Path("~/projects/parametric_mem").glob("*.csv"):
    #     save_path=f"~/projects/parametric_mem/paraphs/{f.stem}_questions"
    #     paraphrases = paraphrase(
    #         "mistralai/Mistral-7B-Instruct-v0.2",
    #         str(f),
    #         prompt,
    #         save_path=save_path
    #     )
    #     nli = NLI()
    #     concatenated_data = []

    #     for sf in Path(save_path).glob("*.csv"):
    #         dataset = pd.DataFrame(load_dataset("csv", data_files=str(sf), delimiter=",")["train"])
    #         # unique
    #         # dataset = dataset.drop_duplicates(subset=["text"])
    #         dataset["paraphrase"] = dataset.apply(
    #             lambda x: filter_raw_paraphrase(x["paraphrase"], x["template"], nli=nli), axis=1
    #         )
    #         # remove empty paraphrase
    #         dataset = dataset[dataset["paraphrase"].map(len) > 0]
    #         dataset = dataset.explode("paraphrase")

    #         concatenated_data.append(dataset)
    #         print(dataset.head(5))
    #         # concatenate
    #         pd.concat(concatenated_data).to_csv(save_path + "_clean.csv")
    #         # concatenate_datasets(concatenated_data).save_to_-disk(save_path)
    # format_for_marco(path="/home/mmahaut/projects/parametric_mem/paraphs", save_path="/home/mmahaut/projects/parametric_mem/paraphs/marco_formated.txt")
    marco_to_csv(path="/home/mmahaut/projects/parametric_mem/wikidata_relations/orig-lama-relation-paraphrases.txt", save_path="/home/mmahaut/projects/parametric_mem/wikidata_relations/orig-lama-relation-paraphrases.csv")
