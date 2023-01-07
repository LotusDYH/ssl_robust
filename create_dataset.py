from datasets import load_dataset
import pickle
import argparse

parser = argparse.ArgumentParser()
data_args = parser.add_argument_group('create_dataset')
data_args.add_argument('--dataset', type=str, default='ag_news')
args = parser.parse_args()

train_val_dataset, test_dataset = load_dataset(args.dataset, split=['train', 'test'])
train_val_dataset = train_val_dataset.train_test_split(train_size=0.8, shuffle=True)
train_dataset = train_val_dataset['train']
val_dataset = train_val_dataset['test']

with open('dataset/' + args.dataset + '/test_dataset', 'wb') as datasetFile:
	pickle.dump(test_dataset, datasetFile)
with open('dataset/' + args.dataset + '/val_dataset', 'wb') as datasetFile:
	pickle.dump(val_dataset, datasetFile)
with open('dataset/' + args.dataset + '/train_dataset', 'wb') as datasetFile:
	pickle.dump(train_dataset, datasetFile)
