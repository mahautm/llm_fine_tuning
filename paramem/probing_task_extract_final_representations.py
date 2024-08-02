from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import pickle
import sys
import logging

model_name = sys.argv[1]
batch_size = int(sys.argv[2])
data_file = sys.argv[3]
out_pickle_prefix = sys.argv[4]
if len(sys.argv)>5:
    checkpoint_path = sys.argv[5]
else:
    checkpoint_path = None

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device is " + device, file=sys.stderr)

model = AutoModelForCausalLM.from_pretrained(model_name,load_in_8bit=True,device_map="auto")
if checkpoint_path is not None:
    try:
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
    except Exception as e:
        logging.error(f"Error loading model checkpoint: {e} Trying to load as distributed model.")
        model = load_distrib_model(model, checkpoint_path)
model.eval()

if ("facebook/opt" in model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

def model_pass(raw_inputs):
    # for now, I don't constrain to a max length
    if "OLMo" in model_name:
        inputs = tokenizer(raw_inputs, padding=True, return_tensors="pt")
    else:
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
        if "OLMo" in model_name:
            hidden_states = model(inputs.input_ids.to(device),output_hidden_states=True).hidden_states
        else:
            hidden_states = model(**inputs,output_hidden_states=True).hidden_states
    
    per_layer_activations = []
    for raw_activation in hidden_states: # shape of hidden_states: layers x batch_size x tokens x d
        # traversing layers
        last_token_activations = []
        for i in range(len(last_true_token_indices)):
            # traversing batch items
            last_token_activation = raw_activation[i][last_true_token_indices[i]].cpu().numpy()
            last_token_activations.append(last_token_activation)
        # appending a list of all the last-token-activations of the current layer to a list of lists
        per_layer_activations.append(last_token_activations)
    return(per_layer_activations)

inputs = []
f = open(data_file)
for line in f:
    input_fields = line.strip("\n").split("\t")
    inputs.append(input_fields[2])
f.close()

cases_count = len(inputs)

first_index = 0
current_batch_size = batch_size
states = dict()
if (current_batch_size>cases_count):
    current_batch_size = cases_count
while ((first_index+current_batch_size)<cases_count):
    curr_output = model_pass(inputs[first_index:first_index+current_batch_size])
    for i in range(len(curr_output)):
        if not i in states:
            states[i] = []
        states[i] = states[i] + curr_output[i]
    first_index=first_index+current_batch_size
# in case cases_count is not a multiple of batch_size
if first_index<cases_count:
    curr_output = model_pass(inputs[first_index:cases_count])
    for i in range(len(curr_output)):
        states[i] = states[i] + curr_output[i]

out_pickle_name = out_pickle_prefix + ".pickle"

with open(out_pickle_name, 'wb') as f:
    pickle.dump(states, f)
