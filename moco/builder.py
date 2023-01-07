# copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.distributed as dist
import diffdist.functional as distops
import sys


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, pre_trained_path, hidden_size, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.k_tmp = None

        # create the encoders
        self.encoder_q = BertModel.from_pretrained(pre_trained_path)
        self.encoder_k = BertModel.from_pretrained(pre_trained_path)
        self.encoder_q_linear = nn.Linear(hidden_size, dim)
        self.encoder_k_linear = nn.Linear(hidden_size, dim)

        if mlp:
            dim_mlp = self.encoder_q_linear.weight.shape[1]
            self.encoder_q_linear = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q_linear)
            self.encoder_k_linear = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k_linear)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.encoder_q_linear.parameters(), self.encoder_k_linear.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.Tanh()(self.queue)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def get_latent_space(self, input_ids, attention_mask):
        output = self.encoder_q(input_ids, attention_mask, return_dict=False)
        outputs_z = self.encoder_q_linear(output[1])
        outputs_z = nn.Tanh()(outputs_z)
        return outputs_z


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.encoder_q_linear.parameters(), self.encoder_k_linear.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, input_ids, attention_mask, update=True):

        # compute query features
        batch_size = int(len(input_ids) / 2)
        sen_q, mask_q, sen_k, mask_k = input_ids[:batch_size], attention_mask[:batch_size], \
                                       input_ids[batch_size:], attention_mask[batch_size:]
        q = self.encoder_q(sen_q, mask_q, return_dict=False)  # queries: NxC
        q = self.encoder_q_linear(q[1])
        q = nn.Tanh()(q)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if update:
                self._momentum_update_key_encoder()  # update the key encoder

            if update or self.k_tmp is None:

                k = self.encoder_k(sen_k, mask_k, return_dict=False)  # keys: NxC
                k = self.encoder_k_linear(k[1])
                k = nn.Tanh()(k)
                self.k_tmp = k

        if not update:
            l_pos = torch.einsum('nc,nc->n', [q, self.k_tmp]).unsqueeze(-1)
        else:
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        if update:
            self._dequeue_and_enqueue(k)
            return logits, labels

        return logits, labels, q


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output