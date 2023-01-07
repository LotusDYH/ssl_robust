import torch
from copy import deepcopy
from deepfool import DeepFool
import numpy as np
from utils import BaseAttack


class GreedyAttack(BaseAttack):
    def __init__(self, args, tokenizer):
        super(GreedyAttack, self).__init__(args, tokenizer)
        self.attack = DeepFool(num_classes=self.num_classes, max_iters=20)

    def get_grad(self, input_ids, attention_mask, labels):

        """
        solve gradients for word embeddings
        """

        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        with torch.enable_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                 output_hidden_states=True, return_dict=False)
            loss = outputs[0]
            loss.backward()

        grads = emb_grads[0].float().cpu().numpy()
        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        sent_vecs = self.model.bert.pooler(outputs[2][-1]).data

        return sent_vecs, grads

    def compute_replace_words_idx(self, words, input_ids, attention_mask, labels, batch_size):

        """
        calculate top-k most important words
        """

        sent_vecs, grads = self.get_grad(input_ids, attention_mask, labels)
        sep_idx = (input_ids == self.tokenizer._convert_token_to_id('[SEP]')).nonzero()
        assert len(sep_idx) == batch_size

        replace_orders = []
        for i in range(batch_size):
            sub_replace_orders = []
            norms = self.get_important_scores(grads[i][1:], self.words_to_sub_words[i])
            indices = torch.topk(torch.tensor(norms), k=len(norms)).indices
            max_len = int(sep_idx[i][1] * 0.4)
            for idx in indices:
                if self.check_word(words[i][idx]):
                    continue
                if (self.sim_word2id is not None) and not(words[i][idx] in self.sim_word2id):
                    continue
                sub_replace_orders.append(idx)
                if len(sub_replace_orders) >= min(max_len, self.max_loops):
                    break
            replace_orders.append(sub_replace_orders)
        return replace_orders, sent_vecs

    def construct_new_samples(self, labels, word_idx, words, finish_mask, iter, batch_size):

        ori_words = deepcopy(words)
        all_new_labels = []
        all_new_text = []
        all_num = []

        ## attack use synonym (not recommend)
        if self.synonym == 'synonym':
            for i in range(batch_size):
                if finish_mask[i]:
                    continue
                candidates = self.get_synonym(ori_words[i][word_idx[i][iter]])
                for new_word in candidates:
                    ori_words[i][word_idx[i][iter]] = new_word
                    all_new_text.append(' '.join(ori_words[i]))
                all_new_labels.extend([labels[i] * len(candidates)])
                all_num.append(len(candidates))
            return all_new_labels, all_new_text, all_num

        ## attack use cosin similarity matrix
        if self.synonym == 'cos_sim':
            for i in range(batch_size):
                if finish_mask[i]:
                    continue
                candidates = self.get_synonym_by_cos(ori_words[i][word_idx[i][iter]])
                for new_word in candidates:
                    ori_words[i][word_idx[i][iter]] = new_word
                    all_new_text.append(' '.join(ori_words[i]))
                all_new_labels.extend([labels[i] * len(candidates)])
                all_num.append(len(candidates))
            return all_new_labels, all_new_text, all_num

        ## attack use Bert
        old_words = []
        for i in range(batch_size):
            if finish_mask[i]:
                continue
            old_words.append(ori_words[i][word_idx[i][iter]])
            ori_words[i][word_idx[i][iter]] = '[MASK]'
        sentences = [' '.join(ori_words[i]) for i in range(batch_size) if finish_mask[i] == 0]
        input_ids, attention_mask = self.get_inputs(sentences)

        with torch.no_grad():
            outputs = self.msk_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        logits = outputs[0]
        mask_idx = (input_ids == self.tokenizer._convert_token_to_id('[MASK]')).nonzero()
        assert len(mask_idx) == batch_size - torch.sum(finish_mask)

        cur_step = 0
        for i in range(batch_size):
            if finish_mask[i]:
                continue
            x, y = mask_idx[cur_step][0], mask_idx[cur_step][1]
            indices = torch.topk(logits[x][y], k=self.n_candidates * 4).indices.squeeze(0)

            old_word = old_words[cur_step]
            count = 0
            for idx in indices:
                new_word = self.id2word[int(idx)]
                if old_word == new_word or self.check_word(new_word) or '##' in new_word:
                    continue

                ## check cosin similarity between the original word and the candidate word
                if self.cos_sim is not None:
                    if not(new_word in self.sim_word2id):
                        continue
                    syn_word_id = self.sim_word2id[new_word]
                    old_word_id = self.sim_word2id[old_word]
                    score = self.cos_sim[syn_word_id][old_word_id]
                    if score < self.sim_thred:
                        continue

                ori_words[i][word_idx[i][iter]] = new_word
                all_new_text.append(' '.join(ori_words[i]))
                count += 1

                if count > self.n_candidates:
                    break

            cur_step += 1
            if count == 0:
                all_new_text.append(' '.join(words[i]))
                count += 1

            all_new_labels.extend([labels[i] * count])
            all_num.append(count)

        return all_new_labels, all_new_text, all_num

    def split_forward(self, input_ids, attention_mask):

        input_ids_splits = input_ids.split(self.splits, dim=0)
        attention_mask_splits = attention_mask.split(self.splits, dim=0)

        all_new_logits = []
        all_new_sent_vectors = []
        with torch.no_grad():
            for idx in range(len(input_ids_splits)):
                outputs= self.model(input_ids=input_ids_splits[idx], attention_mask=attention_mask_splits[idx],
                                    output_hidden_states=True, return_dict=False)
                all_new_logits.append(outputs[0])
                all_new_sent_vectors.append(self.model.bert.pooler(outputs[1][-1]))

        all_new_logits = torch.cat(all_new_logits, dim=0)
        all_new_sent_vectors = torch.cat(all_new_sent_vectors, dim=0)

        return all_new_logits, all_new_sent_vectors

    def adv_attack_samples(self, input_ids, model, labels, attention_mask,
                           original_predictions, original_probs, device):

        self.device = device
        batch_size = len(labels)
        original_words = [self.tokenizer.decode(ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False).split(" ") for ids in input_ids]
        cur_words = deepcopy(original_words)
        finish_mask = torch.zeros(batch_size).bool()
        ## construct the mapping from words index to sub-words index
        self.calc_words_to_sub_words(cur_words, batch_size)

        self.model = deepcopy(model)
        self.msk_model = self.msk_model.to(self.device)
        self.model.eval()
        net = self.model.classifier

        ## select top-k most important words
        replace_orders, sent_vecs = self.compute_replace_words_idx(words=original_words, input_ids=input_ids,
                                                                   attention_mask=attention_mask,
                                                                   labels=labels, batch_size=batch_size)
        intermediate_pred_probs = [original_probs]
        one_hot_mask = torch.arange(self.num_classes).unsqueeze(0).repeat(batch_size, 1)
        one_hot_mask = (one_hot_mask == original_predictions.unsqueeze(1))

        cur_predictions = deepcopy(original_predictions)
        cur_sent_vecs = deepcopy(sent_vecs)

        for i in range(batch_size):
            if len(replace_orders[i]) == 0:
                finish_mask[i] = True

        for iter_idx in range(self.max_loops):

            if finish_mask.sum() == batch_size:
                break

            temp_sent_vecs = torch.masked_select(cur_sent_vecs,
                                                 (~finish_mask).view(-1, 1)).view((~finish_mask).sum(), -1)

            self.model.zero_grad()
            ## deepfool
            cur_normals, cur_pert_vecs, cur_original_predictions = self.attack(vecs=temp_sent_vecs, net_=net)
            ## generate attack samples
            all_new_labels, all_new_text, all_num = self.construct_new_samples(labels=labels, word_idx=replace_orders,
                                                                               words=cur_words,
                                                                               finish_mask=finish_mask, iter=iter_idx,
                                                                               batch_size=batch_size)

            input_ids, attention_mask = self.get_inputs(all_new_text)

            all_new_logits, all_new_sent_vectors = self.split_forward(input_ids=input_ids, attention_mask=attention_mask)

            all_new_predictions = torch.argmax(all_new_logits, dim=-1).data.cpu()
            all_new_probs = torch.softmax(all_new_logits, dim=-1).data.cpu()

            repeats = torch.tensor(all_num).to(self.device)
            all_cur_sent_vecs = torch.repeat_interleave(temp_sent_vecs, repeats=repeats, dim=0)
            all_cur_normals = torch.repeat_interleave(cur_normals, repeats=repeats, dim=0)
            all_new_r_tot = all_new_sent_vectors - all_cur_sent_vecs

            all_new_r_tot_length = all_new_r_tot.norm(dim=1)
            all_cosines = self.cosine_similarity(all_new_r_tot, all_cur_normals)
            all_projections = torch.mul(all_new_r_tot_length, all_cosines)

            all_projections = torch.split(all_projections, split_size_or_sections=all_num)
            all_new_predictions = torch.split(all_new_predictions, split_size_or_sections=all_num)
            all_new_sent_vectors = torch.split(all_new_sent_vectors, split_size_or_sections=all_num, dim=0)
            all_new_probs = torch.split(all_new_probs, split_size_or_sections=all_num, dim=0)

            cur_step = 0
            cur_pred_probs = deepcopy(intermediate_pred_probs[-1])
            count = 0
            for i in range(batch_size):
                if finish_mask[i]:
                    continue
                selected_index = torch.argmax(all_projections[cur_step])
                selected_projection = torch.max(all_projections[cur_step])
                selected_sent_vec = all_new_sent_vectors[cur_step][selected_index]
                selected_new_probs = all_new_probs[cur_step][selected_index]
                selected_prediction = all_new_predictions[cur_step][selected_index]

                if selected_projection > 0 or selected_prediction != original_predictions[i]:
                    cur_predictions[i] = selected_prediction
                    cur_sent_vecs[i] = selected_sent_vec
                    cur_words[i] = all_new_text[count + int(selected_index)].split(" ")
                    cur_pred_probs[i] = torch.masked_select(selected_new_probs, one_hot_mask[i])

                    ## modify the mapping from words index to sub-words index
                    self.words_to_sub_words[i] = {}
                    position = 0
                    for idx in range(len(cur_words[i])):
                        length = len(self.tokenizer.tokenize(cur_words[i][idx]))
                        if position + length > self.max_length - 2:
                            break
                        self.words_to_sub_words[i][idx] = np.arange(position, position + length)
                        position += length

                ## finish attacking if (1) no word to be replaced (2) the model has already been fooled
                if iter_idx + 1 >= len(replace_orders[i]) or selected_prediction != original_predictions[i]:
                    finish_mask[i] = True
                count += all_num[cur_step]
                cur_step += 1

            intermediate_pred_probs.append(cur_pred_probs)

        return cur_predictions, cur_words, original_words, intermediate_pred_probs