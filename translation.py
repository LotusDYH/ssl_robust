from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
from transformers import BertTokenizer as BertTokenizer
import pickle as p
import torch
import sys
import argparse

print('PyTorch V ersion {}, GPU enabled {}'.format(torch.__version__, torch.cuda.is_available()))

parser = argparse.ArgumentParser()
data_args = parser.add_argument_group('adv train options')
data_args.add_argument('--dataset_path', type=str, default='dataset/imdb/test_dataset.pkl')
data_args.add_argument('--output_path', type=str, default='dataset/imdb/back_translation_test.pkl')
args = parser.parse_args()


with open(args.dataset_path, 'rb') as datasetFile:
    dataset = p.load(datasetFile)
print('dataset loaded from {}'.format(args.dataset_path))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name1 = 'Helsinki-NLP/opus-mt-en-roa'
model_name2 = 'Helsinki-NLP/opus-mt-roa-en'
tokenizer1 = MarianTokenizer.from_pretrained(model_name1)
tokenizer2 = MarianTokenizer.from_pretrained(model_name2)
model1 = MarianMTModel.from_pretrained(model_name1).to(device)
model2 = MarianMTModel.from_pretrained(model_name2).to(device)
back_translation = {}
sentences = []
ids = []
count = 0
for e in tqdm(dataset):
    sentence = '>>fra<< ' + e['text']
    sentences.append(sentence)
    if len(sentences) == 10:
        translated = model1.generate(**tokenizer1(sentences, return_tensors="pt", padding=True, truncation=True).to(device))
        tmp = [tokenizer1.decode(t, skip_special_tokens=True) for t in translated]
        translated = model2.generate(**tokenizer2(tmp, return_tensors="pt", padding=True, truncation=True).to(device))
        results = [tokenizer2.decode(t, skip_special_tokens=True) for t in translated]
        for i in range(len(results)):
            words = tokenizer.tokenize(results[i])
            result = tokenizer.convert_tokens_to_string(words)
            back_translation[count] = result
            count += 1
        sentences = []
        sys.stdout.flush()

assert len(back_translation) == len(dataset)
with open(args.output_path, 'wb') as datasetFile:
    p.dump(back_translation, datasetFile)

