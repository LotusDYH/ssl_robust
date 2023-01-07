
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import sys
import torch.nn as nn
import numpy as np
import argparse
# from my_model import CLModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

data_args = parser.add_argument_group('attack')
data_args.add_argument('--dataset', type=str, default='ag_news')
data_args.add_argument('--n_classes', type=int, default=4)
data_args.add_argument('--max_length', type=int, default=128)
data_args.add_argument('--ft_model_path', type=str)
data_args.add_argument('--clft_model_path', type=str)
data_args.add_argument('--adv_model_path', type=str)
data_args.add_argument('--cladv_model_path', type=str)
data_args.add_argument('--adv_data_path', type=str)
data_args.add_argument('--model_g_path', type=str)

args = parser.parse_args()
pre_trained_model = [args.ft_model_path, args.clft_model_path, args.adv_model_path, args.cladv_model_path]


def get_inputs(sentences, tokenizer):
    outputs = tokenizer(sentences, truncation=True, padding=True)
    input_ids = outputs["input_ids"]
    attention_mask = outputs["attention_mask"]
    return torch.tensor(input_ids).to(device), torch.tensor(attention_mask).to(device)


def split_forward(input_ids, attention_mask, model, model_g):
    splits = 32
    input_ids_splits = input_ids.split(splits, dim=0)
    attention_mask_splits = attention_mask.split(splits, dim=0)
    all_new_sent_vectors = []
    with torch.no_grad():
        for idx in range(len(input_ids_splits)):
            outputs = model(input_ids=input_ids_splits[idx], attention_mask=attention_mask_splits[idx],
                                 output_hidden_states=True, return_dict=False)
            outputs = model_g(model.bert.pooler(outputs[1][-1]))
            outputs = nn.Tanh()(outputs)
            all_new_sent_vectors.append(outputs)
            # all_new_sent_vectors.append(model.bert.pooler(outputs[1][-1]))

    all_new_sent_vectors = torch.cat(all_new_sent_vectors, dim=0)

    return all_new_sent_vectors

file_path = args.adv_data_path
checkpoint_PATH = args.model_g_path
num_labels = args.n_classes
model_max_length = args.max_length
f = open(file_path, 'r')
label = f.readline().strip('\n')
total = 0
skip = 0
success = 0
fail = 0
sentences = []
labels = []
while label:
    total += 1
    origin = f.readline().strip('\n')
    attack = f.readline().strip('\n')
    stat = f.readline().strip('\n')
 
    if stat == 'fail':
        fail += 1
    if stat == 'success':
        success += 1
    if stat == 'skip':
        skip += 1

    if stat != 'skip':
        labels.append(int(label))
        labels.append(int(label))
        sentences.append(origin)
        sentences.append(attack)

    if (len(labels) == 2000):
        break

    label = f.readline().strip('\n')
    
for name in pre_trained_model:
    model = BertForSequenceClassification.from_pretrained(name, num_labels=num_labels).to(device)
    tokenizer = BertTokenizer.from_pretrained(name, do_lower_case=True, model_max_length=model_max_length)
    input_ids, attention_mask = get_inputs(sentences=sentences, tokenizer=tokenizer)
 
    # model_g = CLModel(hidden_size=768).to(device)
    model_g = nn.Linear(768, 128).to(device)
    model_CKPT = torch.load(checkpoint_PATH)
    model_g.load_state_dict(model_CKPT['state_dict'])

    vectors = split_forward(input_ids, attention_mask, model, model_g).cpu()
    tmp = []
    tmp1 = []
    save_1 = []
    save_2 = []
    save_3 = []
    for i in range(0, len(vectors), 2):
        sys.stdout.flush()
        v = vectors[i] - vectors
        v = v.norm(dim=1).cpu()
        save_1.append(vectors[i].numpy())
        save_2.append(vectors[i+1].numpy())
        save_3.append(labels[i])
        tmp.append(v[i+1])
        tmp1.extend(v[:i])
        tmp1.extend(v[i+2:])
    print(torch.mean(torch.tensor(tmp)))
    print(torch.mean(torch.tensor(tmp1)))
    save_1 = np.array(save_1)
    save_2 = np.array(save_2)
    save_3 = np.array(save_3)
    arr = name.split('/')
    print(arr[1] + '_' + arr[0])
    np.savez_compressed(arr[1] + '_' + arr[0], a=save_1, b=save_2, c=save_3)

