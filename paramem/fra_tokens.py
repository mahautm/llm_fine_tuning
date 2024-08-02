##      INNER - Interpretability of Neural Networks in Entity Representation   -   https://github.com/franfranz/INNER
##                                              
##      seesoft ultralight.py
##
##          v 2.2.1

# this code: 
#            1) sends sentence prompts, structured in a \t separated file, such as:
# '''
#house and straight to the bar . i grabbed the bottle of bourbon and a glass and went upstairs to	the	ADP	PRON
#and play with the mounds of homework awaiting his full undivided attention . bo chuckles then reminds . monitored .	when	PUNCT	SCONJ
#'''

#             2) collects ranks and probs for each vocabulary item occurring at least once in top k 


import os
import argparse
import json
#import scipy
import csv
import pandas as pd
import torch
import datetime
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import logging
from pathlib import Path
from tqdm import tqdm
#from transformers import GPTNeoXForCausalLM, AutoTokenizer
from data import prepare_data
parser = argparse.ArgumentParser()

parser.add_argument( '-rp',
                    '--rel_filepath', 
                    help = "path where the input file is located",
                    #type = str,
                    default = os.path.abspath(os.path.dirname(__file__)) + '/data'
                    )

parser.add_argument( '-rp2',
                    '--rel_filepath2', 
                    help = "path where the second input file is located",
                    #type = str,
                    default = os.path.abspath(os.path.dirname(__file__)) + '/data'
                    )

parser.add_argument( '-of',
                    '--output_filepath', 
                    help = "path of the output file",
                    #type = str,
                    default = os.path.abspath(os.path.dirname(__file__)) + '/out'
                    )

parser.add_argument( '-sf',
                    '--summary_filepath', 
                    help = "name of the file summarizing accuracy and elapsed time.",
                    #type = str,
                    default = os.path.abspath(os.path.dirname(__file__)) + '/summary'
                    )

parser.add_argument( '-md',
                    '--model_name', 
                    help = "name of the directory containing the relations file.",
                    #type = str,
                    default = "EleutherAI/pythia-70m" 
                    #"EleutherAI/pythia-1.4b-deduped" 
                    #"EleutherAI/pythia-12b-deduped"
                    #"EleutherAI/pythia-6.9b"

                    #"facebook/opt-125m"
                    #"facebook/opt-1.3b"
                    #"facebook/opt-6.7b"
                    #"facebook/opt-13b"
                    
                    # "mistralai/Mistral-7B-v0.1"
                    # "allenai/OLMo-7B"
                    # "allenai/OLMo-1B"
                    )

parser.add_argument( '-dv',
                    '--device', 
                    help = "default = 'cuda' ",
                    #type = str,
                    default = "cuda"
                    )

parser.add_argument( '-k',
                    '--knum', 
                    help = "k number of output tokens",
                    type = int,
                    default = 10
                    )
 
# get start time
# get and print start time
sts = datetime.datetime.now()
print( 'Prompt and seesoft ultralight v2.2.1 - starting job: ', sts)

## parse arguments
args = parser.parse_args()
# paths
rel_filepath = args.rel_filepath
rel_filepath2 = args.rel_filepath2
#rel_filename = args.rel_filename
output_filepath = args.output_filepath
summary_filepath = args.summary_filepath
#rel_filein = rel_filepath + '/' + rel_filename
# model arguments
model_name = args.model_name
device = args.device
# number of topk 
knum = args.knum
#
strmodel = model_name.replace("/", "_")

print(model_name)

# model_type 
if 'pythia' in model_name:
    model_class = 'pythia'
    print (model_class)
elif 'opt' in model_name:
    model_class = 'opt'
    print (model_class)
elif 'mistral' in model_name:
    model_class = 'mistral'
    print (model_class)
elif 'falcon' in model_name:
    model_class = 'falcon'
    print (model_class)
elif "lama" in model_name:
    model_class = 'lama'
# elif 'OLMo' in model_name:
#     model_class = 'olmo'
#     import hf_olmo
    print (model_class)
else:
    print (model_name)
    #break

#if model_class == "pythia": #this seems solved now, careful
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             torch_dtype=torch.float16,
                                             #padding_side = 'left',
                                             output_hidden_states=True).to(device)
#else:
    # model = GPTNeoXForCausalLM.from_pretrained(model_name, 
    #                                          torch_dtype=torch.float16,
    #                                          #padding_side = 'left',
    #                                          output_hidden_states=True).to(device)

# if model_class == 'opt':
#     tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = False) 
                    # with OPT it should be == False, turns out it works even when True
# elif model_class == 'pythia' or model_class == 'mistral': 
#     tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True) 
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True, padding_side = 'left') # MATEO test without padding side here 

# vocabulary size in tokenizer
#vocsize = 10 # run this on clipped vocabulary during test runs, then use actual number uncommenting below
vocsize = model.config.vocab_size

### collect ranks on the whole vocabulary. Careful! this argument will override knum assigned as kwarg
###knum = vocsize

# initialize dict to store accurate responses
acc_st = 0 
nonacc_st = 0
accline = {}

df_out = {}

# MATÉO Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file = output_filepath if Path(output_filepath).is_file() else Path(output_filepath) /f"{model_name.split('/')[-1]}-log.txt"
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

_data1 = pd.read_csv(rel_filepath)
_data2 = pd.read_csv(rel_filepath2)
data_list = prepare_data(_data1, _data2, provide_answer=False) #, prompt=prompt, ans_in_prompt=False) MATÉO
output_filename = (output_filepath)
summary_filename = Path(summary_filepath) / f"{model_name.split('/')[-1]}-summary.txt"
for row_idx, row in tqdm(data_list.iterrows(), total=len(data_list)):
    # unique row identifier 
    row_out = []
    to_send = row["text"]
    dict_to_send = {'prompt' : to_send}
    dict_acc = {'target_obj' : row["result_names"][0][0]}
    pos1 = row["template"]
    pos2 = row["domain"]
    row_out.append(dict_to_send)
    row_out.append(dict_acc)
    row_out.append(pos1)
    row_out.append(pos2)
# df_out.update(dict_acc)
    out_acc_obj = row[0]
#         # tokenize and input prompt into model
    tok_to_send = tokenizer(to_send, return_tensors = 'pt').to(device)
    mod_out = model(**tok_to_send)
#         # get logits from output
    pred_logits = mod_out.logits[0, -1]
# print(pred_logits)
#         ### output tokens of last layer: max and topk
#         # top selected token
    max_out = torch.argmax(pred_logits)
#         # topk tokens
    topk_out = torch.topk(pred_logits, k = knum)
#         # decode argmax token (to compare with ground truth for acc)
    mymax = tokenizer.decode(max_out)
    #print(mymax)
#       # initialize dicts to collect variables
    int_entropy = {}
    top_values_cmf = {}   
    alltopk_ranks = {} 
    alltopk_probs = {}
    out_intlay_MI = {}

    topk_outsoft = []
#             # softmax, last layer
    soft_pred_prob = torch.softmax(pred_logits, dim = -1)
    top_probs_ind_out = torch.topk(soft_pred_prob, k = vocsize).indices
#             # probabilities for topk 
    top_probs_out = soft_pred_prob[top_probs_ind_out]
#             # tensor indices to list
    top_probs_ind_out = top_probs_ind_out.tolist()
#             # sorted dict with indices and probs
    prob_dict_out = {index: prob.item() for index, prob in zip(top_probs_ind_out, top_probs_out)}
    #print(prob_dict_out)
    topk_outsoft.append(prob_dict_out)
#             # IDs into vocabulary items
    decoded_outsoft = {tokenizer.decode(out_index): v_outprob for out_index, v_outprob in prob_dict_out.items()}
    #print(decoded_outsoft)
        
    all_int_IDs = {}
    all_int_toks = {}

#             # store last layer states 
#             # softmax for intermediate layers
    for l_idx in range(len(mod_out.hidden_states)):
        layname = str(l_idx)
#                 # access hidden states
        intermediate_output = mod_out.hidden_states[l_idx]

            # logits: this has different names across models, careful
        if model_class == 'pythia':
            intermediate_logits = model.embed_out(intermediate_output)
        elif model_class == 'opt' or model_class == 'mistral' or model_class == 'lama' or model_class == 'falcon':
            intermediate_logits = model.lm_head(intermediate_output)

            
        int_logits = intermediate_logits[0,-1]
#                 # topk output tokens for each layer
        topk_intermediate_out = torch.topk( int_logits, knum)
        topk_intermediate_IDs = topk_intermediate_out.indices.tolist()
        
            # decode topk IDs into tokens, collect them in a list 
        k_int_toklist = []
        for my_ID in topk_intermediate_IDs: 
            k_int_token = tokenizer.decode(my_ID)
            k_int_toklist.append(k_int_token) 

#                 # now store layname and the list of IDs and of tokens
            layerIDs = {layname: topk_intermediate_IDs}
            layertoks = {layname: k_int_toklist}
            all_int_IDs.update(layerIDs)
            all_int_toks.update(layertoks)
#             # list of all tokens occurring at least one in topk
    alltoks_lists = list(all_int_toks.values())
    #print(all_int_toks)
    alltoks_one_list = []
    for token_list in alltoks_lists:
        for token in token_list:
            alltoks_one_list.append(token)
#             # frequencies of topk across layers
    topk_tokenfreqs = Counter(alltoks_one_list)
        # unique list of all tokens occurring on topk
    unique_toptokens = list(set(alltoks_one_list))
#             # this is to fix
#             #unique_toptokens = []
#             #for toptoken in topk_tokenfreqs:
#             #    unique_toptokens.append(toptoken)

#         ### softmax, intermediate states
            
    for l_idx in range(len(mod_out.hidden_states)):
        layname = str(l_idx)         
#             # access hidden states
        intermediate_output = mod_out.hidden_states[l_idx]
#             # logits: this has different names across models, careful
        if model_class == 'pythia':
            intermediate_logits = model.embed_out(intermediate_output)
        #elif model_class == 'opt' or model_class == 'mistral':
        else:
            intermediate_logits = model.lm_head(intermediate_output)
            
        int_logits = intermediate_logits[0,-1]
#             # topk output tokens for each layer
        topk_intermediate_out = torch.topk( int_logits, knum)
        topk_intermediate_IDs = topk_intermediate_out.indices.tolist()

        int_softmax_IDs = [] 
        int_softmax_toks = []
#             layminus_soft = {}

#             # dictionary to collect softmax: IDs 
        lay_softmax_IDs = {mymax: l_idx}
#             # dictionary to collect softmax: tokens 
        lay_softmax_v_units = {mymax: l_idx}
#             # softmax, intermediate layers    
        soft_int_prob = torch.softmax(int_logits, dim = -1)
#             # this in on a clipped voc set for testing (then with vocsize == model vocabulary size)
        top_probs_ind_int = torch.topk(soft_int_prob, k = vocsize).indices
#             # probabilities for topk 
        top_probs_int = soft_int_prob[top_probs_ind_int]
#             # tensor indices to list
        top_probs_ind_int = top_probs_ind_int.tolist()
#             # sorted dict with indices and probs
        int_prob_dict = {int_index: int_prob.item() for int_index, int_prob in zip(top_probs_ind_int, top_probs_int)}
#             # decode IDs into vocabulary items 
        lay_softmax_tokens = {tokenizer.decode(v_index): v_prob for v_index, v_prob in int_prob_dict.items()}
#             # collect IDs into dictionary
        lay_softmax_IDs.update(int_prob_dict)
#             # collect vocabulary items into dictionary
        lay_softmax_v_units.update(lay_softmax_tokens)


#             # sort probs in descending order
        sorted_lay_2 = {k: v for k, v in sorted(lay_softmax_tokens.items(), key = lambda item: item[1], reverse=True)}
        # dictionaries to store ranks and probs
        lay_topk_ranks = {}
        lay_topk_probs = {}
#             # find the rank in layer for each token occurring at least once in topk    
        for mytoken in unique_toptokens:
            rank = list(sorted_lay_2.keys()).index(mytoken) + 1
            lay_topk_ranks[mytoken] = rank
            if mytoken in sorted_lay_2:
                lay_topk_probs[mytoken] = sorted_lay_2[mytoken]
            
            lt_ranks = {layname : lay_topk_ranks}      
            lt_probs = {layname : lay_topk_probs} 
            alltopk_ranks.update(lt_ranks)
            alltopk_probs.update(lt_probs)

    # turn output into dictionaries
    maxdict = {'token_out' : mymax}
    ranksdict = {'ranks' : alltopk_ranks}
    probsdict = {'probs' : alltopk_probs}

    # append model name
    moddict = {'model_name' : strmodel}

    row_out.append(maxdict)

    #row_out.append(alltopk_ranks)
    row_out.append(ranksdict)
    #row_out.append(alltopk_probs)
    row_out.append(probsdict)
    row_out.append(moddict)

    # output results
    str_row = str(row_out)

    # Log the row_out
    logger.info(str_row)
    row_out = []    
# output summary file
ets = datetime.datetime.now()

with open(summary_filename, 'w') as summary_file:
    summary_file.writelines([str(model_name), ' start time: ', str(sts), ' end time: ', str(ets), '\n'])
    summary_file.writelines(['sent prompts: ', str(row_idx), '\n'])

# get and print ending time
ets = datetime.datetime.now()
print( 'end of job: ', ets)        