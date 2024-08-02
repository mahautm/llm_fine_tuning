import pandas as pd
import itertools
import random
import sys
import numpy as np

# TODO: generalise usage of these sentences throughout the code instead of reimplementing each time
def load_csv_data(data_file, threshold=None, sanity_check=False, no_context=False, input_key="query", threshold_knowledge=True, seed=42):
    _data = pd.read_csv(data_file)
    if "exact_match.1" in _data.columns:
        # this happens if multiple evaluations of slot filling ability were done. We use the last one
        # TODO: use last one instead of .1
        print("WARNING: exact_match.1 column found, using it instead of exact_match", file=sys.stderr)
        success_key = "exact_match.1"
    else:
        success_key = "exact_match"
    if threshold_knowledge:
        _data = _data.groupby(
            ["template", input_key, "expected_answers"], as_index=False).agg(
            {
                success_key: "any" if threshold is None else "mean",
            }
        )
        # only keep positive examples
        _data = _data[_data[success_key]==False] if threshold is None else _data[_data[success_key]>=threshold]
        if threshold is not None:
            print(f"Keeping only examples with {threshold} or more success", file=sys.stderr)
        else:
            print("Keeping only failed generations", file=sys.stderr)

    if sanity_check:
        random.seed(seed)
        np.random.seed(seed)
        random_generator = np.random.default_rng(seed)
        _data=single_example_to_csv_format(_data, random_generator=random_generator)
    elif no_context:
        _data=no_context_to_csv_format(_data)
    return _data.to_dict(orient="list")

def load_jsonl_data(data_file, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    random_generator = np.random.default_rng(seed)
    pass

def load_pile_data(data_file, input_key="query", seed=42):
    random.seed(seed)
    np.random.seed(seed)
    random_generator = np.random.default_rng(seed)
    # used to load line by line extracts from Pile dataset
    with open(data_file, "r") as f:
        inputs = f.readlines()
    # remove last space-delimited word
    query = list(map(lambda x: " ".join(x.split(" ")[:-1]).strip(), inputs))
    expected_answers = list(map(lambda x: x.split(" ")[-1].strip(), inputs))
    return {input_key: query, "expected_answers": expected_answers}


def no_context_to_csv_format(data, random_state=42, n_samples=1):
    output = pd.DataFrame()
    if "result_names" in data.columns:
        output["expected_answers"] = data["result_names"]
        data["expected_answers"] = data["result_names"]
    elif "expected_answers" in data.columns:
        output["expected_answers"] = data["expected_answers"].apply(pd.eval)
        data["expected_answers"] = data["expected_answers"].apply(pd.eval)
    else:
        raise ValueError("data should have 'result_names' or 'expected_answers' column")
    
    output["template"] = data["template"]
    output["query"] = data["template"].apply(lambda x: x.replace(" [Y]", ""))
    if n_samples > 1:
        output = pd.concat([output]*n_samples, ignore_index=True)
    return output
    
def single_example_to_csv_format(data, random_generator=None):
    output = pd.DataFrame()
    def format_line(row):
        # approximation of same relation, not exact
        same_temp = data[data["template"].apply(lambda x: len(set(x.split()) & set(row["template"].split()))) / len(row["template"].split()) > 0.2]
        same_temp["result_names"] = same_temp["expected_answers"].apply(lambda x: x[0])
        same_temp = same_temp[same_temp["expected_answers"] != row["expected_answers"][0]]
        same_temp = same_temp.sample(1, random_state=random_generator)
        same_temp = same_temp.apply(lambda x: x["template"].replace("[Y]", x["result_names"]), axis=1)
        same_temp = same_temp.tolist()[0]

        random_ex = data.sample(2, random_state=random_generator)
        random_ex = random_ex.apply(lambda x: x["template"].replace("[Y]", x["expected_answers"][0]), axis=1)
        random_ex = random_ex.tolist()
        
        query = row["template"].replace(" [Y]", "")
        ans = row["template"].replace(" [Y]", row["expected_answers"][0])
        query2 = data.sample(1, random_state=random_generator)["template"].tolist()[0].replace(" [Y]", "")
        while query in random_ex or query2[0] in random_ex:
            random_ex = data.sample(2, random_state=random_generator).apply(lambda x: x["template"].replace("[Y]", x["expected_answers"][0]), axis=1).tolist()
        return (
            random_ex[0] + ". " + query,
            random_ex[1] + ". " + query,
            random_ex[0] + ". " + query2,
            same_temp + ". " + query,
            ans + ". " + query,
        )
    if "result_names" in data.columns:
        output["expected_answers"] = data["result_names"]
        data["expected_answers"] = data["result_names"]
    elif "expected_answers" in data.columns:
        output["expected_answers"] = data["expected_answers"].apply(pd.eval)
        data["expected_answers"] = data["expected_answers"].apply(pd.eval)
    else:
        raise ValueError("data should have 'result_names' or 'expected_answers' column")
    output["template"] = data["template"]
    c1s = []
    c2s = []
    c3s = []
    c4s = []
    c5s = []
    for row in data.iterrows():
        c1q1, c2q1, c1q2, c4, c5 = format_line(row[1])
        c1s.append(c1q1)
        c2s.append(c2q1)
        c3s.append(c1q2)
        c4s.append(c4)
        c5s.append(c5)
    output["context1_query1"] = c1s
    output["context2_query1"] = c2s
    output["context1_query2"] = c3s
    output["similar_context"] = c4s
    output["answer"] = c5s
    return output

def benchmark_to_csv_format(data, instruction, provide_answer=False, convert_paraphrases=False, random_generator=None):
    # add in_context directly
    # have a shuffled prompt with
    def format_line(row):
        same_temp = data[data["property"] == row["property"]] 
        same_temp["result_names"] = same_temp["result_names"].apply(lambda x: x[0])
        same_temp = same_temp[same_temp["result_names"] != row["result_names"][0]]
        same_temp = same_temp.sample(1, random_state=random_generator)
        same_temp = same_temp.apply(lambda x: x["template"].replace("[Y]", x["result_names"]), axis=1)
        same_temp = same_temp.tolist()


        random_ex = data.sample(2, random_state=random_generator)
        random_ex = random_ex.apply(lambda x: x["template"].replace("[Y]", x["result_names"][0]), axis=1)
        random_ex = random_ex.tolist()

        no_context_ex = random_ex + same_temp
        rand_idx = random.randint(0, len(no_context_ex) - 1)
        no_context_ex[-1], no_context_ex[rand_idx] = no_context_ex[rand_idx], no_context_ex[-1]

        context_ex = random_ex + [row["template"].replace("[Y]", row["result_names"][0])]
        context_ex[-1], context_ex[rand_idx] = context_ex[rand_idx], context_ex[-1]

        no_context = instruction + " " + ". ".join(no_context_ex)
        context = instruction + " " + ". ".join(context_ex)
        query = row["template"].replace("[Y]", row["result_names"][0]) if provide_answer else row["template"].replace(" [Y]", "")
        return (
            context + (". " if context != "" else "") + query, 
            no_context + (". " if no_context != "" else "") + query,
        )

    output = pd.DataFrame() # with columns text, expected_answers        
    data["result_names"] = data["result_names"].apply(lambda x: [item for sublist in x for item in sublist])
    # delete all unique "property" 
    data = data[data["property"].duplicated(keep=False)]

    # first with originals
    output["template"] = data["template"] # we keep the template for reference
    # output["context_query"], output["query"] = data.apply(format_line, axis=1, result_type="expand")
    _cqs = []
    _qs = []
    for row in data.iterrows():
        _cq, _q = format_line(row[1])
        _cqs.append(_cq)
        _qs.append(_q)
    output["expected_answers"] = data["result_names"]

    if convert_paraphrases:
        # next with paraphrases
        data = data.explode("paraphrases")
        _t = data["template"].copy()
        data["template"] = data["paraphrases"]
        for row in data.iterrows():
            _cq, _q = format_line(row[1])
            _cqs.append(_cq)
            _qs.append(_q)
        output["expected_answers"] = pd.concat([output["expected_answers"], data["result_names"]], ignore_index=True)
        output["template"] = pd.concat([output["template"], _t], ignore_index=True)
    output["context_query"] = pd.Series(_cqs)
    output["query"] = pd.Series(_qs)
    return output

def add_ans_in_prompt(data, prompt):
    # add "template" + "result_names" to "query"
    def format_line_c(x):
        _ans = x["template"].replace("[Y]", x["expected_answers"][0])
        idx = random.randint(0, len(prompt["examples"]) - 1)
        if random.choice([True, False]):
            _ans = _ans + " " + prompt["examples"][idx]
        else:
            _ans = prompt["examples"][idx] + " " + _ans
        out = x["query"].split(prompt["examples"][idx])[0] + _ans + x["query"].split(prompt["examples"][idx])[1]
        return out
    data["expected_answers"] = data["expected_answers"].apply(pd.eval)
    data["query"] = data.apply(format_line_c, axis=1)
    return data

def prepare_data(data, provide_answer=True, instruction="", ans_in_prompt=False, n_samples=1):
    # input has the folowing columns:
    # template, query_qid, query_name, result_qids, result_names, property, domain, paraphrases
    # instead we want to provide input text, and expected_answers, all of this in csv format

    # the folowing prompt format is expected.
    # ex: 
    # "instruction": "Accurately fill in the following sentence with the correct word, creating factual sentences as in the examples:",
    
    # each paraphrase is prepared as a sentence with the result name
    # data["paraphrases"] = data["paraphrases"].apply(lambda x: pd.eval(x))
    if "result_names" in data.columns:
        output = benchmark_to_csv_format(data, instruction, provide_answer=provide_answer)
    elif ans_in_prompt:
        # data has already been prepared once, we're just here to add ans in prompt
        output = add_ans_in_prompt(data, instruction)
    else:
        # issue warning
        print("Warning: data has no 'result_names' column, and 'ans_in_prompt' is False. Returning data as is.")
        if "expected_answers" in data.columns:
            data["expected_answers"] = data["expected_answers"].apply(pd.eval)
        output = data
    

    if n_samples > 1:
        output = pd.concat([output]*n_samples, ignore_index=True)
    return output


    

def legacy_prepare_data(data1, data2=None, provide_answer=True, prompt=None, ans_in_prompt=False):
    """
    data1 and data2 should have the same sentences in the same order. This does a boolean OR of the generation_correct,
    meaning if any of the methods report True, True is the value for that sentence.
    """
    if prompt is not None:
        assert "examples" in prompt.keys() and "instruction" in prompt.keys(), "prompt should have 'examples' and 'instruction' keys"
    else:
        prompt = {"examples": [""], "instruction": ""}

    # conservative approach: if any of the paraphrases is correct, the sentence is considered memorized
    data1["generation_correct"] = data1["generation_correct"].apply(lambda x: any(pd.eval(x) if isinstance(x, str) else x))
    data1["result_names"] = data1["result_names"].apply(lambda x: list(itertools.chain(*pd.eval(x)) if isinstance(x, str) and "[" in x else x)) 
    # if two experiments were made for the same data, combine them
    if data2 is not None:
        assert data1["template"].equals(data2["template"]), "data1 and data2 should have the same sentences in the same order"
        data2["generation_correct"] = data2["generation_correct"].apply(lambda x: any(pd.eval(x) if isinstance(x, str) else x))
        data1["generation_correct"] = data1["generation_correct"] | data2["generation_correct"]
    # select only data the model got wrong
    data1 = data1[~data1["generation_correct"]]
    data1 = data1.drop(columns=["generation_correct", "generated"] if "generated" in data1.columns else ["generation_correct"])
    # each template is prepared as a sentence with the result name
    data1=data1.explode("result_names")
    _temp_data = data1.copy()
    _temp_data["text"] = _temp_data.apply(
        lambda x: prompt["instruction"] +
        " ".join(random.sample(prompt["examples"], len(prompt["examples"])) +
            [x["template"].replace("[Y]", x["result_names"]) if ans_in_prompt else ""]
        ) +
        x["template"].replace("[Y]", x["result_names"]) if provide_answer else x["template"].replace(" [Y]", ""),
        axis=1
        )
    # each paraphrase is prepared as a sentence with the result name
    data1["paraphrases"] = data1["paraphrases"].apply(lambda x: pd.eval(x))
    data1 = data1.explode("paraphrases")
    data1["text"] = data1.apply(
        lambda x: prompt["instruction"] +
        " ".join(random.sample(prompt["examples"], len(prompt["examples"])) +
            [x["template"].replace("[Y]", x["result_names"]) if ans_in_prompt else ""]
        ) +
        x["template"].replace("[Y]", x["result_names"]) if provide_answer else x["template"].replace(" [Y]", ""),
        axis=1
        )
    # concat everything
    data1 = pd.concat([data1, _temp_data], ignore_index=True)
    data1 = data1.drop(columns="paraphrases")
    return data1

def prepare_as_tsv(data, output_file):
    # reorder columns - text first, then first element of result names, then everything else
    data["ans"] = data["result_names"].apply(lambda x: x[0][0])
    data = data[["text", "ans", *[col for col in data.columns if col not in ["text", "result_names"]]]]
    data.to_csv(output_file, sep="\t", index=False)   

def find_common_data_for_training(dataset_paths, intersection_column, kept_column):
    # for every file keep only the index where the intersection metric is True
    intersection_idxs=[]
    for dp in dataset_paths:
        df=pd.load_csv(dp)
        idxs=df.index[df[intersection_column] == True].tolist()
        if len(intersection_idxs) == 0:
            intersection_idxs = set(idxs)
        else:
            intersection_idxs=set.intersection(intersection_idxs,set(idxs))
    inputs=df[df["index"]==list(intersection_idxs)][kept_column]