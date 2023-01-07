from transformers import BertForSequenceClassification as BertForSequenceClassification
from transformers import BertTokenizer as BertTokenizer
from textattack.models.wrappers import HuggingFaceModelWrapper
import torch
from textattack.attack_recipes.bert_attack_li_2020 import BERTAttackLi2020
from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
from textattack.attack_recipes.genetic_algorithm_alzantot_2018 import GeneticAlgorithmAlzantot2018
from textattack.attack_recipes.faster_genetic_algorithm_jia_2019 import FasterGeneticAlgorithmJia2019
from textattack.attack_recipes.iga_wang_2019 import IGAWang2019
from textattack.attack_recipes.pwws_ren_2019 import PWWSRen2019
from textattack.attack_recipes.bae_garg_2019 import BAEGarg2019
from textattack.attack_recipes.morpheus_tan_2020 import MorpheusTan2020
from textattack.attack_recipes.hotflip_ebrahimi_2017 import HotFlipEbrahimi2017
from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.attack_recipes.kuleshov_2017 import Kuleshov2017
import numpy as np
from tqdm import tqdm
import argparse
import pickle
import sys
from textattack.attack_recipes.textbugger_li_2018 import TextBuggerLi2018

attacker = {
    'BERTAttackLi2020': BERTAttackLi2020,
    'TextFoolerJin2019': TextFoolerJin2019,
    'PWWSRen2019': PWWSRen2019,
    'BAEGarg2019': BAEGarg2019,
    'TextBuggerLi2018': TextBuggerLi2018,
    'HotFlipEbrahimi2017': HotFlipEbrahimi2017,
    'Kuleshov2017': Kuleshov2017
}

parser = argparse.ArgumentParser()

data_args = parser.add_argument_group('attack')
data_args.add_argument('--dataset', type=str, default='ag_news')
data_args.add_argument('--pre_trained_model', type=str, default='bert-base-uncased')
data_args.add_argument('--n_classes', type=int, default=4)
data_args.add_argument('--max_length', type=int, default=128)
data_args.add_argument('--attacker', type=str)
data_args.add_argument('--output', type=str)

args = parser.parse_args()
out = open(args.output, "w")

## load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained(args.pre_trained_model, num_labels=args.n_classes).to(device)
tokenizer = BertTokenizer.from_pretrained(args.pre_trained_model,
                                          model_max_length=args.max_length, do_lower_case=True)
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

## load dataset
dataset_file_name = 'dataset/' + args.dataset + '/test_dataset.pkl'
with open(dataset_file_name, 'rb') as datasetFile:
    test_dataset = pickle.load(datasetFile)

if args.dataset == 'imdb':
    test_dataset = test_dataset.select(range(1000))
if args.dataset == 'dbpedia':
    test_dataset = test_dataset.select(range(5000))
if args.dataset == 'yelp_polarity':
    test_dataset = test_dataset.select(range(2000))
custom_dataset = []
for sample in test_dataset:
    custom_dataset.append((sample['text'], sample['label']))

## create attack
attack = attacker[args.attacker].build(model_wrapper)

## attack samples
results_iterable = attack.attack_dataset(custom_dataset)
perturbed_word_percentages = np.zeros(len(custom_dataset))
words_changed = np.zeros(len(custom_dataset))
failed_attacks = 0
skipped_attacks = 0
successful_attacks = 0
max_words_changed = 0
total_attacks = len(custom_dataset)
with tqdm(total=len(custom_dataset)) as progress_bar:
    for i, result in enumerate(results_iterable):
        if successful_attacks + failed_attacks != 0 and i % 10 == 0:
            print(i, successful_attacks / (successful_attacks + failed_attacks))
        original_text, label = custom_dataset[i]
        out.write(str(label) + '\n')
        out.write(original_text + '\n')
        out.write(result.perturbed_result.attacked_text.text + '\n')
        sys.stdout.flush()
        if isinstance(result, FailedAttackResult):
            failed_attacks += 1
            progress_bar.update(1)
            out.write('fail\n')
            continue
        elif isinstance(result, SkippedAttackResult):
            skipped_attacks += 1
            progress_bar.update(1)
            out.write('skip\n')
            continue
        out.write('success\n')
        out.flush()
        successful_attacks += 1
        num_words_changed = len(
            result.original_result.attacked_text.all_words_diff(result.perturbed_result.attacked_text))
        words_changed[i] = num_words_changed
        max_words_changed = max(max_words_changed or num_words_changed, num_words_changed)
        if len(result.original_result.attacked_text.words) > 0:
            perturbed_word_percentage = (num_words_changed * 100.0 / len(result.original_result.attacked_text.words))
        else:
            perturbed_word_percentage = 0
        perturbed_word_percentages[i] = perturbed_word_percentage
        progress_bar.update(1)

# Original classifier success rate on these samples.
original_accuracy = (total_attacks - skipped_attacks) * 100.0 / (total_attacks)
original_accuracy = str(round(original_accuracy, 2)) + "%"

# New classifier success rate on these samples.
accuracy_under_attack = (failed_attacks) * 100.0 / (total_attacks)
accuracy_under_attack = str(round(accuracy_under_attack, 2)) + "%"

# Attack success rate.
if successful_attacks + failed_attacks == 0:
    attack_success_rate = 0
else:
    attack_success_rate = (successful_attacks * 100.0 / (successful_attacks + failed_attacks))
    attack_success_rate = str(round(attack_success_rate, 2)) + "%"

perturbed_word_percentages = perturbed_word_percentages[perturbed_word_percentages > 0]
average_perc_words_perturbed = perturbed_word_percentages.mean()
average_perc_words_perturbed = str(round(average_perc_words_perturbed, 2)) + "%"

words_changed = words_changed[words_changed > 0]
average_words_changed = words_changed.mean()
average_words_changed = str(round(average_words_changed, 2)) + "%"

summary_table_rows = [
    ["Number of successful attacks:", str(successful_attacks)],
    ["Number of failed attacks:", str(failed_attacks)],
    ["Number of skipped attacks:", str(skipped_attacks)],
    ["Original accuracy:", original_accuracy],
    ["Accuracy under attack:", accuracy_under_attack],
    ["Attack success rate:", attack_success_rate],
    ["Average perturbed word %:", average_perc_words_perturbed],
    ["Average num. words changed:", average_words_changed],
]
print(summary_table_rows)
out.close()
