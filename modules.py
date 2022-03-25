import copy
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class Packed(nn.Module):

    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn

    @property
    def batch_first(self):
        return self.rnn.batch_first

    def forward(self, inputs, lengths, hidden=None, max_length=None):
        lens, indices = torch.sort(lengths, 0, True)
        inputs = inputs[
            indices] if self.batch_first else inputs[:,
                                                     indices]  # rank and then package
        outputs, (h, c) = self.rnn(
            nn.utils.rnn.pack_padded_sequence(inputs,
                                              lens.tolist(),
                                              batch_first=self.batch_first),
            hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=self.batch_first, total_length=max_length)
        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices] if self.batch_first else outputs[:,
                                                                     _indices]
        # h, c = h[:, _indices, :], c[:, _indices, :]
        # return outputs, (h, c)
        return outputs


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def l_relu(x, n_slope=0.01):
    return F.leaky_relu(x, n_slope)


class ConditionGate(nn.Module):
    """docstring for ConditionGate"""

    def __init__(self, h_dim):
        super(ConditionGate, self).__init__()
        self.gate = nn.Linear(2 * h_dim, h_dim, bias=False)
        # self.q_to_x = nn.Linear(h_dim, h_dim)
        # self.q_to_y = nn.Linear(h_dim, h_dim)

    def forward(self, q, x, y, gate_mask):
        """
        :param q: (bsz, 1, hsz) question representation
        :param x: (bsz, max_n_doc*max_doc_len, hsz) words' entities info
        :param y: (bsz, max_n_doc*max_doc_len, hsz) document representation
        :param gate_mask: (bsz, max_n_doc*max_doc_len, 1) word_ent_info_mask
        :return: document representation (bsz, max_n_doc*max_doc_len, hsz)
        """
        q_x_sim = x * q  # (bsize, max_n_doc*max_doc_len, emb_dim)
        q_y_sim = y * q  # (bsize, max_n_doc*max_doc_len, emb_dim)
        gate_val = self.gate(
            torch.cat([q_x_sim, q_y_sim],
                      dim=-1)).sigmoid()  # (bsz, max_n_doc*max_doc_len, hsz)
        gate_val = gate_val * gate_mask
        return gate_val * x + (1 - gate_val) * y


class ModifyConditionGate(nn.Module):
    """docstring for ConditionGate"""

    def __init__(self, h_dim):
        super(ModifyConditionGate, self).__init__()
        self.gate = nn.Linear(10 * h_dim, h_dim, bias=False)
        self.q_to_y = nn.Bilinear(h_dim, h_dim, h_dim)
        self.x_to_y = nn.Bilinear(h_dim, h_dim, h_dim)

    def forward(self, q, x, y, gate_mask):
        """
        :param q: (bsz, 1, hsz) question representation
        :param x: (bsz, max_n_doc*max_doc_len, hsz) words' entities info
        :param y: (bsz, max_n_doc*max_doc_len, hsz) document representation
        :param gate_mask: (bsz, max_n_doc*max_doc_len, 1) word_ent_info_mask
        :return: document representation (bsz, max_n_doc*max_doc_len, hsz)
        """
        q_x_sim = x * q  # (bsize, max_n_doc*max_doc_len, hsz)
        q_y_sim = y * q
        x_y_sim = x * y
        y_q_sub = y - q
        y_x_sub = y - x
        y_q_bil = self.q_to_y(y, q.expand_as(y).contiguous())
        y_x_bil = self.x_to_y(y, x)

        gate_val = self.gate(
            torch.cat([
                q.expand_as(y).contiguous(), x, y, q_x_sim, q_y_sim, x_y_sim,
                y_q_sub, y_x_sub, y_q_bil, y_x_bil
            ],
                      dim=-1)).sigmoid()  # (bsz, max_n_doc*max_doc_len, hsz)
        gate_val = gate_val * gate_mask
        return gate_val * x + (1 - gate_val) * y


class Fusion(nn.Module):
    """docstring for Fusion"""

    def __init__(self, d_hid):
        super(Fusion, self).__init__()
        self.r = nn.Linear(d_hid * 4, d_hid, bias=False)
        self.g = nn.Linear(d_hid * 4, d_hid, bias=False)

    def forward(self, x, y):
        # x question self-attentive encoding
        # y topic entity knowledge of the question, both (bsize, hidden size)
        r_ = self.r(torch.cat([x, y, x - y, x * y], dim=-1)).tanh()
        g_ = torch.sigmoid(self.g(torch.cat([x, y, x - y, x * y], dim=-1)))
        return g_ * r_ + (1 - g_) * x


class AttnEncoder(nn.Module):
    """docstring for ClassName"""

    def __init__(self, d_hid):
        super(AttnEncoder, self).__init__()
        self.attn_linear = nn.Linear(d_hid, 1, bias=False)

    def forward(self, x, x_mask):
        """
        x: (B, len, d_hid)
        x_mask: (B, len)
        return: (B, d_hid)
        """
        x_attn = self.attn_linear(x)  # weights (bsize, len, 1)
        x_attn = x_attn - (1 - x_mask.unsqueeze(2)) * 1e8
        x_attn = F.softmax(x_attn, dim=1)
        return (x * x_attn).sum(1)


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.
    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttn, self).__init__()
        self.normalize = normalize

        # If identity is true, we just use a dot product without transformation.
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.normalize:
            if self.training:
                # In training we output log-softmax for NLL
                alpha = F.log_softmax(xWy, dim=-1)
            else:
                # ...Otherwise 0-1 probabilities
                alpha = F.softmax(xWy, dim=-1)
        else:
            alpha = xWy.exp()
        return alpha


class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)  # a single layer NN
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)  # a single layer NN
        else:
            x_proj = x
            y_proj = y

        ##################### dot product attention #####################
        scores = x_proj.bmm(y_proj.transpose(2, 1))  # (bsize, x_len, y_len)
        ##################### dot product attention #####################
        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data,
                                 -float('inf'))  # (bsize, x_len, y_len)

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=-1)
        alpha = alpha_flat.view(-1, x.size(1),
                                y.size(1))  # (bsz, x_len, y_len)

        # Take weighted average
        matched_seq = alpha.bmm(y)

        return matched_seq


class QueryReform(nn.Module):
    """docstring for QueryReform"""

    def __init__(self, h_dim):
        super(QueryReform, self).__init__()
        # self.q_encoder = AttnEncoder(h_dim)
        self.fusion = Fusion(h_dim)
        self.q_ent_attn = nn.Linear(h_dim, h_dim)

    def forward(self, q_node, ent_emb, seed_info, ent_mask):
        '''
        q_node: (B, q_len, h_dim)
        q_mask: (B,q_len)
        q_ent_span: (B,q_len)
        ent_emb: (bsz, max_local_candidates, hsz)
        seed_info: (B, max_local_candidates)
        ent_mask: (bsz, max_local_candidates) padding with the number of entities

        :return: (bsz, hsz)
        '''
        # q_node = self.q_encoder(q, q_mask)
        q_ent_attn = (self.q_ent_attn(q_node).unsqueeze(1) * ent_emb).sum(
            2, keepdim=True)
        q_ent_attn = F.softmax(q_ent_attn - (1 - ent_mask.unsqueeze(2)) * 1e8,
                               dim=1)
        # attn_retrieve = (q_ent_attn * ent_emb).sum(1)

        # (bsz, 1, h_dim) integrate entities info
        # sum the representation of entities related to the query
        seed_retrieve = torch.bmm(seed_info.unsqueeze(1), ent_emb).squeeze(1)
        # how to calculate the gate

        # return  self.fusion(q_node, attn_retrieve)
        return self.fusion(q_node, seed_retrieve)

        # retrieved = self.transform(torch.cat([seed_retrieve, attn_retrieve], dim=-1)).relu()
        # gate_val = self.gate(torch.cat([q.squeeze(1), seed_retrieve, attn_retrieve], dim=-1)).sigmoid()
        # return self.fusion(q.squeeze(1), retrieved).unsqueeze(1)
        # return (gate_val * q.squeeze(1) + (1 - gate_val) * torch.tanh(self.transform(torch.cat([q.squeeze(1), seed_retrieve, attn_retrieve], dim=-1)))).unsqueeze(1)


class ModifyQueryReform(nn.Module):

    def __init__(self, h_dim):
        super(ModifyQueryReform, self).__init__()

    def forward(self, q_emb, q_mask, ent_emb, seed_info):
        """
        :param q_emb: (bsz, q_len, hsz)
        :param q_mask: (bsz, q_len) 1 for word, 0 for padding
        :param ent_emb: (bsz, max_local_candidates, hsz)
        :param seed_info: (bsz, max_local_candidates) 1 for entity relative to question
        :return q_update: (bsz, hsz)
        """
        # (bsz, hsz) aggregate the entities which are relative to query
        seed_info_avg = seed_info / seed_info.sum(-1).unsqueeze(1)
        seed_info_avg[torch.isnan(
            seed_info_avg
        )] = 0.0  # nan for query without relative to any entities
        ent_relative_q = torch.bmm(seed_info_avg.unsqueeze(1),
                                   ent_emb).squeeze(1)
        # (bsz, q_len)
        q_ent_score = torch.bmm(ent_relative_q.unsqueeze(1),
                                q_emb.transpose(1, 2)).squeeze(1)
        q_ent_score.masked_fill_(q_mask == 0, -1e9)
        q_ent_weight = F.softmax(q_ent_score, dim=-1)
        q_update = torch.bmm(q_ent_weight.unsqueeze(1), q_emb).squeeze(1)
        return q_update


class ModifyQueryReformWithWeight(nn.Module):

    def __init__(self, h_dim, s_type='general'):
        super(ModifyQueryReformWithWeight, self).__init__()
        self.stype = h_dim
        self.stype = s_type
        if self.stype == 'general':
            self.ent_q_match = nn.Bilinear(h_dim, h_dim, 1, bias=False)
        elif self.stype == 'concat':
            self.ent_q_match = nn.Linear(2 * h_dim, 1, bias=False)
        elif self.stype == 'perceptron':
            self.ent_q_match = nn.Linear(2 * h_dim, h_dim, bias=False)
            self.ent_q_w = nn.Linear(h_dim, 1, bias=False)
        else:
            raise RuntimeError("No such type of calculating score")

    def forward(self, q_emb, q_mask, ent_emb, seed_info):
        """
        :param q_emb: (bsz, q_len, hsz)
        :param q_mask: (bsz, q_len) 1 for word, 0 for padding
        :param ent_emb: (bsz, max_local_candidates, hsz)
        :param seed_info: (bsz, max_local_candidates) 1 for entity relative to question
        :return q_update: (bsz, hsz)
        """
        # (bsz, hsz) aggregate the entities which are relative to query
        seed_info_avg = seed_info / seed_info.sum(-1).unsqueeze(1)
        seed_info_avg[torch.isnan(
            seed_info_avg
        )] = 0.0  # nan for query without relative to any entities
        ent_relative_q = torch.bmm(seed_info_avg.unsqueeze(1),
                                   ent_emb).squeeze(1)

        ent_expand = ent_relative_q.unsqueeze(1).expand_as(q_emb)
        if self.stype == 'general':
            q_ent_score = self.ent_q_match(ent_expand.contiguous(),
                                           q_emb).transpose(1, 2).squeeze(
                                               1)  # (bsz, q_len)
        elif self.stype == 'concat':
            q_ent_score = self.ent_q_match(
                torch.cat([ent_expand, q_emb], dim=-1)).squeeze(-1)
        else:
            q_ent_score = self.ent_q_w(
                self.ent_q_match(torch.cat([ent_expand, q_emb],
                                           dim=-1)).tanh()).squeeze(-1)

        q_ent_score.masked_fill_(q_mask == 0, -1e9)
        q_ent_weight = F.softmax(q_ent_score, dim=-1)
        q_update = torch.bmm(q_ent_weight.unsqueeze(1), q_emb).squeeze(1)
        return q_update


class SelfAttnEncoder(nn.Module):
    """Model size and number of heads"""

    def __init__(self, num_heads, input_dim, output_dim, dropout_rate=0.2):
        super(SelfAttnEncoder, self).__init__()
        assert input_dim % num_heads == 0
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_k = input_dim // num_heads
        self.num_heads = num_heads

        # self.query_project = nn.Linear(input_dim, input_dim)
        # self.key_project = nn.Linear(input_dim, input_dim)
        # self.value_project = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.concat_projection = nn.Linear(input_dim, output_dim)

    def attention(self, query, key, value, mask=None, dropout=None):
        """
        query, key value size: (bsz, num_heads, max_seq_len, emb_size)
        mask size: (bsz, num_heads, max_seq_len, max_seq_len), mask position 0, non-mask position 1
        dropout
        return1 size: (bsz, num_heads, max_seq_len, emb_size)
        return2 size: (bsz, num_heads, max_seq_len)
        """
        d_k = query.size(-1)
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            d_k)  # (bsz, num_heads, max_seq_len)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(score, dim=-1)
        p_attn = p_attn * mask

        if dropout is not None:
            p_attn = dropout(p_attn)

            return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        """
        query, key value size: (bsz, max_seq_len, emb_size)
        mask size: (bsz,  max_seq_len), mask position 0, non-mask position 1
        return1 size: (bsz, max_seq_len, emb_size)
        return2 size: (bsz, num_heads, max_seq_len, max_seq_len)
        """
        batch_size = query.size(0)
        max_seq_len = query.size(1)

        # query (bsz, max_seq_len, dk), dk = d_model //h
        if mask is not None:
            # Same mask applied to all heads
            mask = mask.unsqueeze(2)
            score_mask = torch.matmul(mask, mask.transpose(2, 1))
            score_mask = score_mask.unsqueeze(1).expand(
                batch_size, self.num_heads, max_seq_len, max_seq_len)

        # step1 linear projections in batch the from d_model to (h x d_k)
        # query = self.query_project(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # key = self.query_project(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # value = self.query_project(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        query = query.view(batch_size, -1, self.num_heads,
                           self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads,
                       self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads,
                           self.d_k).transpose(1, 2)

        # step2 apply attention on all the projected vectors in batch
        # x size: (bsz, num_heads, max_seq_len, emb_size)
        x, _ = self.attention(query,
                              key,
                              value,
                              mask=score_mask,
                              dropout=self.dropout)

        # step3 concat
        # (hsz, seq_len, input_dim)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1,
                                                self.num_heads * self.d_k)
        out_x = self.concat_projection(x) * mask
        return out_x


class EntAwareSelfAttnEncoder(nn.Module):

    def __init__(self, hidden_size):
        super(EntAwareSelfAttnEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.weight1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.weight2 = nn.Linear(self.hidden_size, 1)

    def forward(self, word_ent_info, word_ent_info_mask, doc, doc_mask):
        """
        :param word_ent_info:  (bsz*num_doc, doc_len, hsz)
        :param word_ent_info_mask: (bsz*num_doc, doc_len) zero for word without linking any entities
        :param doc: (bsz*num_doc, doc_len, hsz)
        :param doc_mask: (bsz*num_doc, doc_len)
        :return: (bsz*num_doc, doc_len, hsz)
        """

        num_doc = doc.size(0)
        max_doc_len = doc.size(1)
        # (bsz*num_doc, 1, hsz)
        word_ent_infoagg = torch.bmm(word_ent_info_mask.float().unsqueeze(1),
                                     word_ent_info)
        word_ent_infoagg_expand = word_ent_infoagg.expand(
            num_doc, max_doc_len, -1)

        # (bsz*num_doc, doc_len, hsz)  wi * ent_agg
        word_ent_agg = doc * word_ent_infoagg_expand
        word_ent_agg_expand = word_ent_agg.unsqueeze(2).expand(
            num_doc, max_doc_len, max_doc_len, -1)

        # (bsz*num_doc, doc_len, doc_len, hsz)  wi * wj
        doc_expand = doc.unsqueeze(2).expand(num_doc, max_doc_len, max_doc_len,
                                             -1)
        doc_self_info = doc_expand * doc_expand.transpose(1, 2)

        # (bsz*num_doc, doc_len, doc_len)
        score = self.weight2(
            self.weight1(
                torch.cat([doc_self_info, word_ent_agg_expand],
                          dim=-1)).tanh()).squeeze(-1) / math.sqrt(
                              self.hidden_size)

        # get doc mask
        doc_mask = doc_mask.unsqueeze(2)
        score_mask = torch.matmul(doc_mask, doc_mask.transpose(
            1, 2))  # (bsz*num_doc, doc_len, doc_len)
        score[score_mask == 0.0] = -1e9  # mask the padding word
        word_ent_w = F.softmax(score, dim=-1)
        word_ent_w = word_ent_w * score_mask

        # (bsz*num_doc, doc_len, hsz)
        doc_ent_enc = torch.matmul(word_ent_w, doc)
        return doc_ent_enc


class ConditionalSelfAttnEncoder(nn.Module):

    def __init__(self, hidden_size):
        super(ConditionalSelfAttnEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.weight1 = nn.Linear(3 * self.hidden_size, self.hidden_size)
        self.weight2 = nn.Linear(self.hidden_size, 1)

    def forward(self, word_ent_info, word_ent_info_mask, doc, doc_mask,
                q_for_text):
        """
        :param word_ent_info:  (bsz*num_doc, doc_len, hsz)
        :param word_ent_info_mask: (bsz*num_doc, doc_len) zero for word without linking any entities
        :param doc: (bsz*num_doc, doc_len, hsz)
        :param doc_mask: (bsz*num_doc, doc_len)
        :param q_for_text: (bsz, hsz)
        :return: (bsz*num_doc, doc_len, hsz)
        """

        bsz = q_for_text.size(0)
        num_doc = doc.size(0)
        max_doc_len = doc.size(1)
        # (bsz*num_doc, 1, hsz)
        word_ent_infoagg = torch.bmm(word_ent_info_mask.float().unsqueeze(1),
                                     word_ent_info)
        word_ent_infoagg_expand = word_ent_infoagg.expand(
            num_doc, max_doc_len, -1)

        # wi * q
        q_for_text_expand_ = q_for_text.unsqueeze(1).expand(
            bsz, int(num_doc / bsz), -1)
        q_for_text_expand = q_for_text_expand_.contiguous().view(
            num_doc, 1, -1)  # (bsz*num_doc, 1, hsz)
        word_q_agg_expand_ = doc * q_for_text_expand  # (bsz*num_doc, max_doc_len, hsz)
        word_q_agg_expand = word_q_agg_expand_.unsqueeze(1).expand(
            num_doc, max_doc_len, max_doc_len, -1)

        # (bsz*num_doc, doc_len, hsz)  wi * ent_agg
        word_ent_agg = doc * word_ent_infoagg_expand
        word_ent_agg_expand = word_ent_agg.unsqueeze(2).expand(
            num_doc, max_doc_len, max_doc_len, -1)

        # (bsz*num_doc, doc_len, doc_len, hsz)  wi * wj
        doc_expand = doc.unsqueeze(2).expand(num_doc, max_doc_len, max_doc_len,
                                             -1)
        doc_self_info = doc_expand * doc_expand.transpose(1, 2)

        # (bsz*num_doc, doc_len, doc_len)
        score = self.weight2(
            self.weight1(
                torch.cat(
                    [doc_self_info, word_ent_agg_expand, word_q_agg_expand],
                    dim=-1)).tanh()).squeeze(-1) / math.sqrt(self.hidden_size)

        # get doc mask
        doc_mask = doc_mask.unsqueeze(2)
        score_mask = torch.matmul(doc_mask, doc_mask.transpose(
            1, 2))  # (bsz*num_doc, doc_len, doc_len)
        score[score_mask == 0.0] = -1e9  # mask the padding word
        word_ent_w = F.softmax(score, dim=-1)
        word_ent_w = word_ent_w * score_mask

        # (bsz*num_doc, doc_len, hsz)
        doc_ent_enc = torch.matmul(word_ent_w, doc)
        return doc_ent_enc


# Transformer relative class


def clones(module, num_layers):
    """Produce N identical layers"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])


class LayerNorm(nn.Module):
    """ layerNorm module """

    def __init__(self, feature_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(feature_size))
        self.b = nn.Parameter(torch.zeros(feature_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class MultiheadAttention(nn.Module):
    """Model size and number of heads"""

    def __init__(self, num_heads, d_model, dropout_rate=0.1):
        super(MultiheadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.query_project = nn.Linear(d_model, d_model)
        self.key_project = nn.Linear(d_model, d_model)
        self.value_project = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.concat_projecttion = nn.Linear(d_model, d_model)

    def attention(self, query, key, value, mask=None, dropout=None):
        """
        query, key value size: (bsz, num_heads, max_seq_len, emb_size)
        mask size: (bsz, num_heads, max_seq_len, max_seq_len), mask position 0, non-mask position 1
        dropout
        return1 size: (bsz, num_heads, max_seq_len, emb_size)
        return2 size: (bsz, num_heads, max_seq_len)
        """
        d_k = query.size(-1)
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            d_k)  # (bsz, num_heads, max_seq_len)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(score, dim=-1)
        p_attn = p_attn * mask

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        """
        query, key value size: (bsz, max_seq_len, emb_size)
        mask size: (bsz,  max_seq_len), mask position 0, non-mask position 1
        return1 size: (bsz, max_seq_len, emb_size)
        return2 size: (bsz, num_heads, max_seq_len, max_seq_len)
        """
        batch_size = query.size(0)
        max_seq_len = query.size(1)

        # query (bsz, max_seq_len, dk), dk = d_model //h
        if mask is not None:
            # Same mask applied to all heads
            mask = mask.unsqueeze(2)
            scord_mask = torch.matmul(mask, mask.transpose(2, 1))
            scord_mask = scord_mask.unsqueeze(1).expand(
                batch_size, self.num_heads, max_seq_len, max_seq_len)

        # step1 linear projections in batch the from d_model to (h x d_k)
        query = self.query_project(query).view(batch_size, -1, self.num_heads,
                                               self.d_k).transpose(1, 2)
        key = self.query_project(key).view(batch_size, -1, self.num_heads,
                                           self.d_k).transpose(1, 2)
        value = self.query_project(value).view(batch_size, -1, self.num_heads,
                                               self.d_k).transpose(1, 2)

        # step2 apply attention on all the projected vectors in batch
        # x size: (bsz, num_heads, max_seq_len, emb_size)
        x, self.score = self.attention(query,
                                       key,
                                       value,
                                       mask=scord_mask,
                                       dropout=self.dropout)

        # step3 concat and apply a final Linear
        x = x.transpose(1, 2).contiguous().view(batch_size, -1,
                                                self.num_heads * self.d_k)

        return self.concat_projecttion(x), self.score


class EncoderLayer(nn.Module):
    """a layer encoder containing self attn and feed forward"""

    def __init__(self,
                 d_model,
                 num_heads,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu"):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = MultiheadAttention(num_heads,
                                            d_model,
                                            dropout=dropout)
        # parameter of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = self._get_activation_fn(activation)

        # Add and Norm Layer
        # self.norm1 = LayerNorm(d_model)
        # self.norm2 = LayerNorm(d_model)
        # or
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "leaky_relu":
            return F.leaky_relu
        else:
            return RuntimeError(
                "activation should be relu/leaky_relu, not {}".format(
                    activation))

    def forward(self, src, src_mask):
        """
        self-attn --> add & norm --> feedforward --> add & norm
        :param src: (bsz, max_seq_len, emb_size)
        :param src_mask:(bsz, num_heads, max_seq_len, max_seq_len), mask position 0, non-mask position 1
        :return: layer_enc:
        """
        # self attention
        # attn_enc (bsz, max_seq_len, emb_size)
        attn_enc, attn_score = self.self_attn(src, src, src, src_mask)
        # add & norm
        attn_enc = attn_enc + self.dropout1(attn_enc)
        attn_enc = self.norm1(attn_enc)

        # feedforward
        feed_enc = self.linear2(
            self.dropout(self.activation(self.linear1(attn_enc))))
        feed_enc = feed_enc + self.dropout2(feed_enc)
        layer_enc = self.norm2(feed_enc)

        return layer_enc


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # precompute the positional encoding once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: (bsz, max_seq_len, emb_size)
        :return: the same size of input
        """
        x = x + nn.Parameter(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Encoder is a stack of N layers"""

    def __init__(self, layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.layers = clones(layer, num_layers)
        self.norm = nn.LayerNorm(layer.d_model)
        self.positions = PositionalEncoding(layer.d_model, dropout=0.1)

    def forward(self, src, mask=None):
        """
        :param src: (bsz, max_seq_len, emb_size)
        :param mask:(bsz,  max_seq_len), mask position 0, non-mask position 1
        :return1: (bsz, max_seq_len, emb_size)
        :return2: (bsz, emb_size)
        """
        output = src + self.positions(src)
        for layer in self.layers:
            output = layer(output, mask)
        agg_output = torch.matmul(mask.unsqueeze(1), output)

        return self.norm(output), agg_output.squeeze(1)


if __name__ == "__main__":

    bsz = 4
    max_seq_len = 5
    wsz = 15
    hsz = 10

    q_emb = torch.rand(bsz, max_seq_len, wsz)
    q_mask = torch.ones(bsz, max_seq_len)
    q_mask[0, -2:] = 0
    q_mask[1, -3:] = 0
    q_mask[2, -1] = 0

    model = SelfAttnEncoder(num_heads=1, input_dim=wsz, output_dim=hsz)
    q_selfenc = model(q_emb, q_emb, q_emb, q_mask)

    print(q_selfenc)
    '''
    # Test the ConditionalSelfAttnEncoder
    bsz = 3
    max_seq_len = 4
    hsz = 10

    d_emb = torch.rand(bsz, max_seq_len, hsz)
    # d_emb[0, -1], d_emb[0, -2], d_emb[1, -1] = 0, 0, 0
    d_mask = torch.ones(bsz, max_seq_len)
    d_mask[0, -1], d_mask[0, -2], d_mask[1, -1] = 0, 0, 0

    word_ent_info = torch.rand(bsz, max_seq_len, hsz)
    word_ent_info_mask = torch.zeros(bsz, max_seq_len)
    word_ent_info_mask[0, 1], word_ent_info_mask[1, 1:3], word_ent_info_mask[2, 0] = 1, 1, 0

    model = ConditionalSelfAttnEncoder(hsz)
    d_enc = model(word_ent_info, word_ent_info_mask, d_emb, d_mask)

    print(d_enc)

    '''
    """
    # Test the ModifyQueryReform
    bsz = 3
    max_seq_len = 5
    hsz = 21
    max_local_candidates = 4
    q_emb = torch.rand(bsz, max_seq_len, hsz)
    q_mask = torch.ones(bsz, max_seq_len)
    q_mask[0, -1], q_mask[0, -2], q_mask[1, -1] = 0, 0, 0
    ent_emb = torch.rand(bsz, max_local_candidates, hsz)
    seed_info = torch.zeros(bsz, max_local_candidates)
    seed_info[0, 1], seed_info[1, 0], seed_info[1, 2] = 1, 1, 1
    model = ModifyQueryReformWithWeight(hsz, s_type='perceptron')
    q_update = model(q_emb, q_mask, ent_emb, seed_info)

    print(q_update)
    """
    """
    # Test the ModifyQueryReform
    bsz = 3
    max_seq_len = 5
    hsz = 21
    max_local_candidates = 4
    q_emb = torch.rand(bsz, max_seq_len, hsz)
    q_mask = torch.ones(bsz, max_seq_len)
    q_mask[0, -1], q_mask[0, -2], q_mask[1, -1] = 0, 0, 0
    ent_emb = torch.rand(bsz, max_local_candidates, hsz)
    seed_info = torch.zeros(bsz, max_local_candidates)
    seed_info[0, 1], seed_info[1, 0], seed_info[1, 2] = 1, 1, 1
    model = ModifyQueryReform(hsz)
    q_update = model(q_emb, q_mask, ent_emb, seed_info)

    print(q_update)
    """

    # Test the ModifyConditionGate
    # bsz = 2
    # max_seq_len = 4
    # d_model = 20
    # dim_feedforward = 128
    # q = torch.rand(bsz, 1, d_model)
    # x = torch.rand(bsz, max_seq_len, d_model)
    # y = torch.rand(bsz, max_seq_len, d_model)
    # gate_mask = torch.rand(bsz, max_seq_len, 1)
    # gate_function = ModifyConditionGate(d_model)
    # new_y = gate_function(q, x, y, gate_mask)
    #
    # print(new_y)

    # # Test the postional encoding
    # max_seq_len = 4
    # d_model = 20
    # x = torch.zeros(1, max_seq_len, 20)
    # pe = PositionalEncoding(d_model=d_model, dropout=0.2)
    # y = pe(x)
    #
    # print(x)
    # print(y)

    # Test the Transformer module
    # bsz = 2
    # max_seq_len = 4
    # d_model = 20
    # num_heads = 5
    # num_layers = 3
    # dim_feedforward = 128
    # x = torch.rand(bsz, max_seq_len, d_model)
    # x[0, -1, :], x[0, -2, :], x[1, -1, :] = 0, 0, 0
    # mask = torch.ones(bsz, max_seq_len)
    # mask[0, -1], mask[0, -2], mask[1, -1] = 0, 0, 0
    # enc_layer = EncoderLayer(d_model=d_model, num_heads=num_heads, dim_feedforward=dim_feedforward)
    # enc_model = TransformerEncoder(layer=enc_layer, num_layers=num_layers)
    # layer_enc = enc_model(x, mask)
    #
    # print(x)
    # print(layer_enc)
    """
    # Test the EncoderLayer module
    bsz = 2
    max_seq_len = 4
    d_model = 12
    x = torch.rand(bsz, max_seq_len, d_model)
    x[0, -1, :], x[0, -2, :], x[1, -1, :] = 0, 0, 0
    mask = torch.ones(2, 4)
    mask[0, -1], mask[0, -2], mask[1, -1] = 0, 0, 0
    enc_layer = EncoderLayer(d_model=d_model, num_heads=3,  dim_feedforward=24)
    layer_enc = enc_layer(x, mask)

    print(x)
    print(layer_enc)
    """

    # # Test the MultiheadAttention module
    # x = torch.rand(2, 4, 12)
    # x[0, -1, :], x[0, -2, :], x[1, -1, :] = 0, 0, 0
    # mask = torch.ones(2, 4)
    # mask[0, -1], mask[0, -2], mask[1, -1] = 0, 0, 0
    # mha = MultiheadAttention(3, 12)
    # x_enc, x_attn = mha(x, x, x, mask=mask)
    #
    # print(x)
    # print(x_enc)
    # print(x_attn)
    """
    # test the attention of MultiheadAttention module
    x = torch.rand(2, 3, 4, 12)
    mask = torch.ones(2, 4)
    mask[0, -1], mask[0,-2], mask[1,-1] = 0, 0, 0
    mask = mask.unsqueeze(2)
    scord_mask = torch.matmul(mask, mask.transpose(2, 1)) # (bsz, max_seq_len, max_seq_len)
    scord_mask = scord_mask.unsqueeze(1).expand(2, 3, 4, 4)
    mha = MultiheadAttention(3, 12)
    x_enc, x_attn = mha.attention(x, x, x, mask=scord_mask)

    print(x)
    print(x_enc)
    print(x_attn)
    """