import torch
from copy import deepcopy
from my_loss import pairwise_similarity
from my_loss import get_loss
from utils import BaseAttack
import numpy as np


class GreedyAttack(BaseAttack):

    def __init__(self, args, tokenizer):
        super(GreedyAttack, self).__init__(args, tokenizer)
        self.ngpu = args.ngpu
        self.replace_history = None
        self.model_g = None

    def get_grad(self, input_ids, attention_mask):

        """
        solve gradients for word embeddings and latent space
        """

        embedding_layer = self.model.get_input_embeddings()
        linear_layer = self.model_g.linear
        original_stat_emb = embedding_layer.weight.requires_grad
        original_stat_linear = linear_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True
        linear_layer.weight.requires_grad = True

        emb_grads = []
        linear_grads = []

        def emb_grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        def linear_grad_hook(module, grad_in, grad_out):
            linear_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(emb_grad_hook)
        linear_hook = linear_layer.register_backward_hook(linear_grad_hook)

        self.model.zero_grad()
        self.model_g.zero_grad()
        with torch.enable_grad():
            bert_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
            outputs_z = self.model_g(bert_outputs[1])
            similarity_matrix = pairwise_similarity(outputs=outputs_z, ngpu=self.ngpu)
            loss = get_loss(similarity_matrix=similarity_matrix, device=self.device)
            loss.backward()

        grads = emb_grads[0].float().cpu().numpy()
        grads_z = linear_grads[0]
        embedding_layer.weight.requires_grad = original_stat_emb
        linear_layer.weight.requires_grad = original_stat_linear
        emb_hook.remove()
        linear_hook.remove()

        return grads_z, grads, outputs_z

    def compute_word_importance(self, words, batch_size):

        """
        select the most important word
        """

        sentences = [' '.join(e) for e in words]
        input_ids, attention_mask = self.get_inputs(sentences)
        grads_z, grads, outputs_z = self.get_grad(input_ids, attention_mask)
        sep_idx = (input_ids == self.tokenizer._convert_token_to_id('[SEP]')).nonzero()
        assert len(sep_idx) == batch_size * 2

        ## [batch, len, dim]
        grads = grads[:batch_size]
        replace_idx = []
        for i in range(batch_size):
            temp_idx = None
            norms = self.get_important_scores(grads[i][1:], self.words_to_sub_words[i])
            indices = torch.topk(torch.tensor(norms), k=len(norms)).indices
            max_len = int(sep_idx[i][1] * 0.2)
            for idx in indices:
                if self.check_word(words[i][idx]):
                    continue
                if (self.sim_word2id is not None) and not(words[i][idx] in self.sim_word2id):
                    continue
                if idx in self.replace_history[i]:
                    continue
                if len(self.replace_history[i]) >= max_len:
                    continue
                temp_idx = idx
                break
            if temp_idx is None:
                temp_idx = indices[0]
            replace_idx.append(temp_idx)
            self.replace_history[i].add(temp_idx)

        return replace_idx, grads_z[:batch_size], outputs_z[:batch_size]

    def construct_new_samples(self, word_idx, words, batch_size):

        ori_words = deepcopy(words)
        all_new_text = []
        all_nums = []

        ## attack use synonym (not recommend)
        if self.synonym == 'synonym':
            for i in range(batch_size):
                candidates = self.get_synonym(ori_words[i][word_idx[i]])
                for new_word in candidates:
                    ori_words[i][word_idx[i]] = new_word
                    all_new_text.append(' '.join(ori_words[i]))
                all_nums.append(len(candidates))
            return all_nums, all_new_text

        ## attack use cosin similarity matrix
        if self.synonym == 'cos_sim':
            for i in range(batch_size):
                candidates = self.get_synonym_by_cos(ori_words[i][word_idx[i]])
                for new_word in candidates:
                    ori_words[i][word_idx[i]] = new_word
                    all_new_text.append(' '.join(ori_words[i]))
                all_nums.append(len(candidates))
            return all_nums, all_new_text

        ## attack use Bert
        old_words = []
        for i in range(batch_size):
            old_words.append(ori_words[i][word_idx[i]])
            ori_words[i][word_idx[i]] = '[MASK]'
        sentences = [' '.join(e) for e in ori_words]
        input_ids, attention_mask = self.get_inputs(sentences)

        with torch.no_grad():
            outputs = self.msk_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        logits = outputs[0]
        mask_idx = (input_ids == self.tokenizer._convert_token_to_id('[MASK]')).nonzero()
        assert len(mask_idx) == batch_size

        for i in range(batch_size):
            x, y = mask_idx[i][0], mask_idx[i][1]
            indices = torch.topk(logits[x][y], k=self.n_candidates * 4).indices.squeeze(0)

            old_word = old_words[i]
            count = 0
            for idx in indices:
                new_word = self.id2word[int(idx)]
                if old_word == new_word or self.check_word(new_word) or '##' in new_word:
                    continue
                ## check cosin similarity between the original word and the candidate word
                if self.cos_sim is not None:
                    if not (new_word in self.sim_word2id) or not (old_word in self.sim_word2id):
                        continue
                    syn_word_id = self.sim_word2id[new_word]
                    old_word_id = self.sim_word2id[old_word]
                    score = self.cos_sim[syn_word_id][old_word_id]
                    if score < self.sim_thred:
                        continue

                ori_words[i][word_idx[i]] = new_word
                all_new_text.append(' '.join(ori_words[i]))
                count += 1

                if count > self.n_candidates:
                    break
            if count == 0:
                all_new_text.append(' '.join(words[i]))
                count += 1

            all_nums.append(count)

        return all_nums, all_new_text

    def split_forward(self, input_ids, attention_mask):

        input_ids_splits = input_ids.split(self.splits, dim=0)
        attention_mask_splits = attention_mask.split(self.splits, dim=0)

        outputs = []
        with torch.no_grad():
            for idx in range(len(input_ids_splits)):
                bert_outputs = self.model(input_ids=input_ids_splits[idx],
                                          attention_mask=attention_mask_splits[idx],
                                          return_dict=False)
                outputs_z = self.model_g(bert_outputs[1])
                outputs.append(outputs_z)

        outputs = torch.cat(outputs, dim=0)
        return outputs

    def adv_attack_samples(self, model, model_g, device, input_ids):

        self.device = device
        original_words = [self.tokenizer.decode(ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False).split(" ") for ids in input_ids]
        batch_size = len(input_ids)
        cur_words = deepcopy(original_words)
        ## construct the mapping from words index to sub-words index
        self.calc_words_to_sub_words(cur_words, batch_size)

        self.model = deepcopy(model)
        self.model_g = deepcopy(model_g)
        self.model.eval()
        self.model_g.eval()
        self.msk_model.to(self.device)

        ## avoid repeat modification
        self.replace_history = [set() for _ in range(batch_size)]

        for iter_idx in range(self.max_loops):

            replace_idx, vector_z, ori_z = self.compute_word_importance(words=cur_words + original_words,
                                                                        batch_size=batch_size)
            all_nums, all_new_text = self.construct_new_samples(word_idx=replace_idx, words=cur_words,
                                                                batch_size=batch_size)

            input_ids, attention_mask = self.get_inputs(all_new_text)
            outputs = self.split_forward(input_ids=input_ids, attention_mask=attention_mask)
            outputs = torch.split(outputs, all_nums)

            count = 0
            for i, cur_z in enumerate(outputs):
                cur_z = cur_z.float() - ori_z[i].float()
                z = torch.repeat_interleave(vector_z[i].float().unsqueeze(0), repeats=len(cur_z), dim=0)
                cur_z_norm = cur_z.norm(dim=1)
                cosin_z = self.cosine_similarity(cur_z, z)
                project_z = torch.mul(cur_z_norm, cosin_z)
                selected_idx = torch.argmax(project_z)
                if project_z[selected_idx] > 0:
                    cur_words[i] = all_new_text[int(selected_idx) + count].split(' ')
                    ## modify the mapping from words index to sub-words index
                    self.words_to_sub_words[i] = {}
                    position = 0
                    for idx in range(len(cur_words[i])):
                        length = len(self.tokenizer.tokenize(cur_words[i][idx]))
                        if position + length > self.max_length - 2:
                            break
                        self.words_to_sub_words[i][idx] = np.arange(position, position + length)
                        position += length

                count += len(cur_z)

        del self.model
        del self.model_g
        return cur_words
