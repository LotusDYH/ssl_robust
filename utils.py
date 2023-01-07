from transformers import BertTokenizer
from transformers import AdamW
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup
import pytorch_lightning as pl
import pickle
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
import torch
from nltk.corpus import stopwords
from torch.nn import CosineSimilarity
import string
import numpy as np
import os
from nltk.corpus import wordnet
from transformers import BertForMaskedLM


class BertBase(pl.LightningModule):
    def __init__(self, args):
        super(BertBase, self).__init__()
        self.pre_trained_model = args.pre_trained_model
        self.n_classes = args.n_classes
        self.max_length = args.max_length
        self.dataset = args.dataset
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.warmup_proportion = args.warmup_proportion
        self.ngpu = args.ngpu
        self.train_size = args.train_size
        self.val_size = args.val_size
        self.test_size = args.test_size

        self.tokenizer = BertTokenizer.from_pretrained(self.pre_trained_model, do_lower_case=True,
                                                       model_max_length=self.max_length)

        dataset_file_name = 'dataset/' + self.dataset + '/test_dataset.pkl'
        with open(dataset_file_name, 'rb') as datasetFile:
            self.test_dataset = pickle.load(datasetFile)
        dataset_file_name = 'dataset/' + self.dataset + '/val_dataset.pkl'
        with open(dataset_file_name, 'rb') as datasetFile:
            self.val_dataset = pickle.load(datasetFile)
        dataset_file_name = 'dataset/' + self.dataset + '/train_dataset.pkl'
        with open(dataset_file_name, 'rb') as datasetFile:
            self.train_dataset = pickle.load(datasetFile)
        self.train_dataset = self.train_dataset.select(range(int(len(self.train_dataset)*self.train_size)))
        self.val_dataset = self.val_dataset.select(range(int(len(self.val_dataset) * self.val_size)))
        self.test_dataset = self.test_dataset.select(range(int(len(self.test_dataset) * self.test_size)))
        print("Train Samples: {}\nVal Samples: {}\nTest Samples: {}".format(
            len(self.train_dataset), len(self.val_dataset), len(self.test_dataset)))

    def _collate_fn(self, batch):
        label = torch.tensor([item['label'] for item in batch])
        text = [item['text'] for item in batch]
        outputs = self.tokenizer(text, truncation=True, padding=True)
        input_ids = torch.tensor(outputs["input_ids"])
        attention_mask = torch.tensor(outputs["attention_mask"])
        return {'label': label, 'text': text, 'input_ids': input_ids, 'attention_mask': attention_mask}

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = FusedAdam(params=optimizer_grouped_parameters, lr=self.learning_rate)
        ## optimizer = DeepSpeedCPUAdam(model_params=optimizer_grouped_parameters, lr=self.learning_rate)

        num_training_steps = self.epochs * int(len(self.train_dataset) / self.batch_size / self.ngpu)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.warmup_proportion * num_training_steps),
            num_training_steps=num_training_steps,
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

    def train_dataloader(self):
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            dataset=self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn
        )
        return val_dataloader

    def test_dataloader(self):
        if self.dataset == 'imdb':
            test_dataloader = DataLoader(         
                dataset=self.test_dataset.select(range(1000)),
                shuffle=False,
                batch_size=self.batch_size,
                collate_fn=self._collate_fn
            )
            return test_dataloader        

        if self.dataset == 'dbpedia':
            test_dataloader = DataLoader(
                dataset=self.test_dataset.select(range(5000)),
                shuffle=False,
                batch_size=self.batch_size,
                collate_fn=self._collate_fn
            )
            return test_dataloader

        if self.dataset == 'yelp_polarity':
            test_dataloader = DataLoader(
                dataset=self.test_dataset.select(range(2000)),
                shuffle=False,
                batch_size=self.batch_size,
                collate_fn=self._collate_fn
            )
            return test_dataloader
        
        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn
        )
        return test_dataloader


filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves']
filter_words = set(filter_words)

class BaseAttack:
    def __init__(self, args, tokenizer):
        self.stopwords = set(stopwords.words('english'))
        self.model = None
        self.cosine_similarity = CosineSimilarity(dim=1, eps=1e-6)
        self.tokenizer = tokenizer
        self.num_classes = args.n_classes
        self.device = None
        self.msk_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.msk_model.eval()
        self.words_to_sub_words = None
        self.max_length = args.max_length
        self.n_candidates = args.n_candidates
        self.max_loops = args.max_loops
        self.sim_thred = args.sim_thred
        self.splits = args.splits

        self.word2id = self.tokenizer.get_vocab()
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.cos_sim = None
        self.sim_word2id = None
        self.sim_id2word = None
        self.synonym = args.synonym
        self.cos_sim_dict = None
        if args.cos_sim:
            self.init_matrix(args.embedding_path, args.sim_path)

    def get_important_scores(self, grads, words_to_sub_words):
        index_scores = [0.0] * len(words_to_sub_words)
        for i in range(len(words_to_sub_words)):
            matched_tokens = words_to_sub_words[i]
            agg_grad = np.sum(grads[matched_tokens], axis=0)
            index_scores[i] = np.linalg.norm(agg_grad, ord=1)
        return index_scores

    def init_matrix(self, embedding_path, sim_path):
        embeddings = []
        self.sim_id2word = {}
        self.sim_word2id = {}

        ## constuct cosine similarity matrix
        with open(embedding_path, 'r') as ifile:
            for line in ifile:
                embedding = [float(num) for num in line.strip().split()[1:]]
                embeddings.append(embedding)
                word = line.split()[0]
                if word not in self.sim_id2word:
                    self.sim_id2word[len(self.sim_id2word)] = word
                    self.sim_word2id[word] = len(self.sim_id2word) - 1
        if os.path.exists(sim_path):
            self.cos_sim = np.load(sim_path)
        else:
            embeddings = np.array(embeddings)
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = np.asarray(embeddings / norm, "float32")
            self.cos_sim = np.dot(embeddings, embeddings.T)

        ## construct top-k similar words for each word
        if self.synonym == 'cos_sim':
            self.cos_sim_dict = {}
            for idx, word in self.sim_id2word.items():
                candidates = set()
                indices = torch.topk(torch.tensor(self.cos_sim[idx]), k=self.n_candidates).indices
                for i in indices:
                    i = int(i)
                    if self.cos_sim[idx][i] < self.sim_thred:
                        break
                    if i == idx:
                        continue
                    candidates.add(self.sim_id2word[i])
                if len(candidates) == 0:
                    candidates = [word]
                self.cos_sim_dict[idx] = candidates


    def check_word(self, word):
        return word == '[PAD]' or word == '[UNK]' or word == '[CLS]' or \
               word == '[SEP]' or word in self.stopwords or word in string.punctuation or \
               word in filter_words or word in '...' or word == '[MASK]'

    def get_synonym_by_cos(self, word):
        if not (word in self.sim_word2id):
            return [word]
        idx = self.sim_word2id[word]
        return self.cos_sim_dict[idx]

    def get_synonym(self, word):
        candidates = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                w = l.name()
                if self.check_word(w):
                    continue
                if w in candidates:
                    continue
                candidates.add(w)
        if self.tokenizer.tokenize(word)[0] != word:
            for syn in wordnet.synsets(self.tokenizer.tokenize(word)[0]):
                for l in syn.lemmas():
                    w = l.name()
                    if self.check_word(w):
                        continue
                    if w in candidates:
                        continue
                    candidates.add(w)
        if len(candidates) == 0:
            candidates = [word]
        return candidates

    def calc_words_to_sub_words(self, words, batch_size):
        self.words_to_sub_words = []
        for i in range(batch_size):
            position = 0
            self.words_to_sub_words.append({})
            for idx in range(len(words[i])):
                length = len(self.tokenizer.tokenize(words[i][idx]))
                if position + length > self.max_length - 2:
                    break
                self.words_to_sub_words[i][idx] = np.arange(position, position + length)
                position += length

    def get_inputs(self, sentences):
        outputs = self.tokenizer(sentences, truncation=True, padding=True)
        input_ids = outputs["input_ids"]
        attention_mask = outputs["attention_mask"]
        return torch.tensor(input_ids).to(self.device), torch.tensor(attention_mask).to(self.device)


## TODO: it is better to check USE score for each attack sample
# import tensorflow.compat.v1 as tf
# import tensorflow_hub as hub
# tf.disable_v2_behavior()

# class USE(object):
#     def __init__(self):
#         super(USE, self).__init__()
#         os.environ['TFHUB_CACHE_DIR'] = 'tmp'
#         module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
#         self.embed = hub.Module(module_url)
#         config = tf.ConfigProto()
#         config.gpu_options.allow_growth = True
#         self.sess = tf.Session(config=config)
#         self.build_graph()
#         self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

#     def build_graph(self):
#         self.sts_input1 = tf.placeholder(tf.string, shape=(None))
#         self.sts_input2 = tf.placeholder(tf.string, shape=(None))

#         sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
#         sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
#         self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
#         clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
#         self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

#     def semantic_sim(self, sents1, sents2):
#         scores = self.sess.run(
#             [self.sim_scores],
#             feed_dict={
#                 self.sts_input1: sents1,
#                 self.sts_input2: sents2,
#             })
#         return scores[0]
