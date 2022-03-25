import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import AttnEncoder
from modules import Packed
from modules import SeqAttnMatch
from modules import l_relu
from modules import QueryReform
from modules import ConditionGate
from modules import EncoderLayer
from modules import TransformerEncoder
from modules import ModifyConditionGate
from modules import ModifyQueryReform
from modules import ModifyQueryReformWithWeight
from modules import SelfAttnEncoder
from modules import EntAwareSelfAttnEncoder
from modules import ConditionalSelfAttnEncoder
from util import load_dict


class KAReader(nn.Module):
    """docstring for ClassName"""

    def __init__(self, args):
        super(KAReader, self).__init__()

        self.entity2id = load_dict(args['data_folder'] + args['entity2id'])
        self.word2id = load_dict(args['data_folder'] + args['word2id'])
        self.relation2id = load_dict(args['data_folder'] + args['relation2id'])
        self.num_entity = len(self.entity2id)
        self.num_relation = len(self.relation2id)
        self.num_word = len(self.word2id)
        self.num_layer = args['num_layer']
        self.use_doc = args['use_doc']
        self.word_drop = args['word_drop']
        self.hidden_drop = args['hidden_drop']
        self.label_smooth = args['label_smooth']
        self.use_cuda = args['use_cuda']
        self.hidden_dim = 64

        for k, v in args.items():
            if k.endswith('dim'):
                setattr(self, k, v)
            if k.endswith('emb_file'):
                setattr(self, k, args['data_folder'] + v)

        # pretrained entity embeddings
        self.entity_emb = nn.Embedding(self.num_entity + 1,
                                       self.entity_dim,
                                       padding_idx=self.num_entity)
        self.entity_emb.weight.data.copy_(
            torch.from_numpy(
                np.pad(np.load(self.entity_emb_file), ((0, 1), (0, 0)),
                       'constant')))  # 在最后一行加上零向量
        self.entity_emb.weight.requires_grad = False
        self.entity_emb = nn.DataParallel(self.entity_emb)
        self.entity_linear = nn.Linear(self.entity_dim, self.hidden_dim)

        # word embeddings
        self.word_emb = nn.Embedding(self.num_word,
                                     self.word_dim,
                                     padding_idx=1)
        self.word_emb.weight.data.copy_(
            torch.from_numpy(np.load(self.word_emb_file)))
        self.word_emb.weight.requires_grad = False
        self.word_emb = nn.DataParallel(self.word_emb)

        self.word_emb_match = SeqAttnMatch(self.word_dim)

        # question and doc encoder dim change from wsz to hsz
        ############################ from LSTM to SelfAttnEncoder ############################
        self.question_encoder = Packed(
            nn.LSTM(self.word_dim,
                    self.hidden_dim // 2,
                    batch_first=True,
                    bidirectional=True))
        # self.question_encoder = Packed(nn.GRU(self.word_dim, self.hidden_dim // 2, batch_first=True, bidirectional=True))
        # self.question_encoder = SelfAttnEncoder(num_heads=1, input_dim=self.word_dim, output_dim=self.hidden_dim)
        ############################ from LSTM to SelfAttnEncoder ############################

        ######################## Transformer encoding (not effective) ########################
        self.self_att_r = AttnEncoder(self.hidden_dim)
        self.self_att_q = AttnEncoder(self.hidden_dim)
        # layer = EncoderLayer(d_model=self.hidden_dim, num_heads=1, dim_feedforward=256)
        # self.self_att_r = TransformerEncoder(layer=layer, num_layers=1)
        # self.self_att_q = TransformerEncoder(layer=layer, num_layers=1)
        ######################## Transformer encoding (not effective) ########################

        self.combine_q_rel = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        # doc encoder

        self.ent_info_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.input_proj = nn.Linear(2 * self.word_dim, self.hidden_dim)
        ######################## SelfAttnEncoder ########################
        # self.doc_encoder = Packed(nn.LSTM(self.hidden_dim, self.hidden_dim // 2,\
        #                           batch_first=True, bidirectional=True))
        # self.doc_encoder = SelfAttnEncoder(num_heads=1, d_model=self.hidden_dim)
        # self.doc_encoder = ConditionalSelfAttnEncoder(hidden_size=self.hidden_dim)
        self.doc_encoder = EntAwareSelfAttnEncoder(hidden_size=self.hidden_dim)
        ######################## SelfAttnEncoder ########################

        ######################## conditional gate function ########################
        self.ent_info_gate = ConditionGate(self.entity_dim)
        self.ent_info_gate_out = ConditionGate(self.entity_dim)
        # self.ent_info_gate = ModifyConditionGate(self.hidden_dim)
        # self.ent_info_gate_out = ModifyConditionGate(self.hidden_dim)
        ######################## conditional gate function ########################

        self.kg_prop = nn.Linear(self.hidden_dim + self.hidden_dim,
                                 self.hidden_dim)
        self.kg_gate = nn.Linear(self.hidden_dim + self.hidden_dim,
                                 self.hidden_dim)
        self.self_prop = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.combine_q = nn.Linear(2 * self.hidden_dim, self.hidden_dim)

        self.reader_gate = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        ################ try to modify queryReform ################
        # self.query_update = QueryReform(self.hidden_dim)
        self.query_update = ModifyQueryReform(self.hidden_dim)
        # self.query_update = ModifyQueryReformWithWeight(self.hidden_dim)
        ################ try to modify queryReform ################

        self.attn_match = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.attn_match_q = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.loss = nn.BCEWithLogitsLoss()

        self.word_drop = nn.Dropout(self.word_drop)
        self.hidden_drop = nn.Dropout(self.hidden_drop)
        self.ans_gate = nn.Linear(2 * self.hidden_dim, self.hidden_dim)

    def forward(self, feed):
        # encode questions
        question = feed['questions']  # (bsize, max_q_len)
        q_mask = (question != 1).float()
        q_len = q_mask.sum(-1)  # (B, q_len)
        q_word_emb = self.word_drop(self.word_emb(question))
        # (bsize, max_len, wsz) ---> (bsize, max_len, hsz) reserve zero padding
        ############################## from LSTM to SelfAttnEncoder ##############################
        q_emb = self.question_encoder(q_word_emb,
                                      q_len,
                                      max_length=question.size(1))
        # q_emb = self.question_encoder(q_word_emb, q_word_emb, q_word_emb, q_mask)
        ############################## from LSTM to SelfAttnEncoder ##############################
        q_emb = self.hidden_drop(q_emb)

        B, max_q_len = question.size(0), question.size(1)

        # candidate ent embeddings and encode
        ent_emb_ = self.entity_emb(
            feed['candidate_entities'])  # (bsz, max_local_entities, hsz)
        ent_emb = l_relu(
            self.entity_linear(ent_emb_))  # (bsz, max_local_entities, hsz)

        # # keep a copy of the initial ent_emb
        # init_ent_emb = ent_emb
        ent_mask = (feed['candidate_entities'] !=
                    self.num_entity).float()  # (bsz, max_local_entities)

        # linked relations
        max_num_neighbors = feed['entity_link_ents'].size(2)
        max_num_candidates = feed['candidate_entities'].size(1)
        neighbor_mask = (feed['entity_link_ents'] !=
                         self.num_entity).float()  # (B, |C|, |N|)

        # encode all relations with question encoder
        rel_word_ids = feed[
            'rel_word_ids']  # (num_rel+1, word_lens) load directly
        rel_word_mask = (rel_word_ids != 1).float()
        rel_word_lens = rel_word_mask.sum(-1)
        rel_word_lens[rel_word_lens == 0] = 1
        ############################## from LSTM to SelfAttnEncoder ##############################
        rel_encoded = self.question_encoder(
            self.word_drop(self.word_emb(rel_word_ids)),
            rel_word_lens,
            max_length=rel_word_ids.size(1))  # (|R|, r_len, h_dim)

        # rel_word_emb = self.word_emb(rel_word_ids)
        # rel_encoded = self.question_encoder(rel_word_emb, rel_word_emb, rel_word_emb, rel_word_mask)
        ############################## from LSTM to SelfAttnEncoder ##############################
        rel_encoded = self.hidden_drop(rel_encoded)
        rel_encoded = self.self_att_r(
            rel_encoded, rel_word_mask
        )  # (num_rel+1, max_rel_len, emb_size) --> (num_rel+1, hid_dim)

        neighbor_rel_ids = feed['entity_link_rels'].long().view(
            -1)  # Embedding index
        neighbor_rel_emb = torch.index_select(rel_encoded,
                                              dim=0,
                                              index=neighbor_rel_ids).view(
                                                  B * max_num_candidates,
                                                  max_num_neighbors,
                                                  self.hidden_dim)
        # encode entities , relarions
        # for look up Extra Attention over Topic Entity Neighbors
        neighbor_ent_local_index = feed['entity_link_ents'].long(
        )  # (B, |C|, max_num_neighbors)
        neighbor_ent_local_index = neighbor_ent_local_index.view(B, -1)
        neighbor_ent_local_mask = (neighbor_ent_local_index != -1).long()
        fix_index = torch.arange(B).long() * max_num_candidates
        if self.use_cuda:
            fix_index = fix_index.to(torch.device('cuda'))
        neighbor_ent_local_index = neighbor_ent_local_index + fix_index.view(
            -1, 1)
        neighbor_ent_local_index = (neighbor_ent_local_index +
                                    1) * neighbor_ent_local_mask
        neighbor_ent_local_index = neighbor_ent_local_index.view(-1)

        # seed entity (query entity) will have 1.0 score
        ent_seed_info = feed['query_entities'].float()
        if self.use_cuda:
            ent_is_seed = torch.cat([
                torch.zeros(1).to(torch.device('cuda')),
                ent_seed_info.view(-1)
            ],
                                    dim=0)
        else:
            ent_is_seed = torch.cat(
                [torch.zeros(1), ent_seed_info.view(-1)], dim=0)
        ent_seed_indicator = torch.index_select(
            ent_is_seed, dim=0, index=neighbor_ent_local_index).view(
                B * max_num_candidates,
                max_num_neighbors)  # (B*max_num_candidates, max_num_neighbors)

        # v0.0 more find-grained attention
        # question-relation matching:
        # question (bsz, max_q_len, hid_size) relation (bsz*max_local_entities, max_num_neighbors, hid_size) -->
        q_emb_expand = q_emb.unsqueeze(1).expand(B, max_num_candidates,
                                                 max_q_len, -1).contiguous()
        q_emb_expand = q_emb_expand.view(B * max_num_candidates, max_q_len,
                                         -1)  # (bsz*m_n_c, m_q_l, hsz)
        q_mask_expand = q_mask.unsqueeze(1).expand(
            B, max_num_candidates,
            -1).contiguous()  # (bsz, m_q_l) --> (bsz, m_num_c, m_q_l)
        q_mask_expand = q_mask_expand.view(
            B * max_num_candidates,
            -1)  # (bsz*max_num_candidates, max_q_len, hsz)
        ##################### dot product attention #####################
        # (bsz*max_num_candidates, max_q_len, max_num_neighbors)
        # dot product to get question-relation match scores
        q_n_affinity = torch.bmm(q_emb_expand,
                                 neighbor_rel_emb.transpose(1, 2))
        ##################### dot product attention #####################
        # (bsz*m_n_c, m_q_l, m_n_neigh) mask the attention scores of padding of the question
        q_n_affinity_mask_q = q_n_affinity - (
            1 - q_mask_expand.unsqueeze(2)) * 1e20
        q_n_affinity_mask_n = q_n_affinity - (
            1 -
            neighbor_mask.view(B * max_num_candidates, 1, max_num_neighbors)
        )  # (bsz*m_n_c, m_q_l, m_n_neigh)
        normalize_over_q = F.softmax(q_n_affinity_mask_q, dim=1)
        normalize_over_n = F.softmax(q_n_affinity_mask_n, dim=2)
        retrieve_q = torch.bmm(
            normalize_over_q.transpose(1, 2), q_emb_expand
        )  # (bsz*m_n_c, m_n_neigh, hsz) 实际上是问题表征，只有bsz个，bsz中不同实体对应的问题表征是一样的
        # (bsz*max_local_entities, max_num_neigh)  s_r
        q_rel_simi = torch.sum(neighbor_rel_emb * retrieve_q, dim=2)

        retrieve_r = torch.bmm(normalize_over_n,
                               neighbor_rel_emb)  # (bsz*m_n_c, m_q_l, hsz)
        q_and_rel = torch.cat([q_emb_expand, retrieve_r],
                              dim=2)  # a single layer neural network
        rel_aware_q = self.combine_q_rel(q_and_rel).tanh().view(
            B, max_num_candidates, -1, self.hidden_dim
        )  # (bsize, max_num_candidates, max_q_len, hidden_size) representation of question over relation info

        # pooling over the q_len dim retrieve the most representable vector
        q_node_emb = rel_aware_q.max(2)[0]  # (bsz, max_num_candidates, hsz)

        # Information Propagation from Neighbors
        ent_emb = l_relu(
            self.combine_q(
                torch.cat([ent_emb, q_node_emb],
                          dim=2)))  # (bsize, max_num_candidates, hidden_size)
        ent_emb_for_lookup = ent_emb.view(-1, self.hidden_dim)
        if self.use_cuda:
            ent_emb_for_lookup = torch.cat([
                torch.zeros(1, self.hidden_dim).to(torch.device('cuda')),
                ent_emb_for_lookup
            ],
                                           dim=0)
        else:
            ent_emb_for_lookup = torch.cat(
                [torch.zeros(1, self.hidden_dim), ent_emb_for_lookup], dim=0)
        neighbor_ent_emb = torch.index_select(ent_emb_for_lookup,
                                              dim=0,
                                              index=neighbor_ent_local_index)
        neighbor_ent_emb = neighbor_ent_emb.view(B * max_num_candidates,
                                                 max_num_neighbors, -1)
        neighbor_vec = torch.cat([neighbor_rel_emb, neighbor_ent_emb],
                                 dim=-1).view(B * max_num_candidates,
                                              max_num_neighbors,
                                              -1)  # for propagation [ri; ei]
        neighbor_scores = q_rel_simi * ent_seed_indicator  # s^~(ri,ei) (bsz*m_n_c, m_n_neigh)
        neighbor_scores = neighbor_scores - (1 - neighbor_mask.view(
            B * max_num_candidates, max_num_neighbors)) * 1e8
        attn_score = F.softmax(neighbor_scores, dim=1)
        aggregate = self.kg_prop(neighbor_vec) * attn_score.unsqueeze(
            2
        )  # \sum(s x W_e[ri;ei]) (bsize*max_num_candidates, max_num_neighbors, hidden_size)
        aggregate = l_relu(aggregate.sum(1)).view(
            B, max_num_candidates, -1
        )  # (bsize, max_num_candidates, hidden size) /sum the neighbors and then activate
        self_prop_ = l_relu(
            self.self_prop(ent_emb))  # linear transformer of entity encoding
        gate_value = self.kg_gate(
            torch.cat([aggregate, ent_emb],
                      dim=-1)).sigmoid()  # gamma^e = g(e, aggregate)
        # Information Propagation from Neighbors (bsz, max_local_ent, hsz)
        ent_emb = gate_value * self_prop_ + (1 - gate_value) * aggregate

        # read documents
        if self.use_doc:
            '''
            Query Reformulation in Latent Space
            '''
            # (bsz, max_q_len, word_emb_size)-->(bsz, hsz)  self-attentive encoder of query reformulation
            # init_q_emb = self.self_att_q(q_emb, q_mask)
            # (bsz, hsz) -->(bsz, hsz)
            # q_for_text = self.query_update(init_q_emb, ent_emb, ent_seed_info, ent_mask)
            ##################### w/o Query Reformulation #####################
            # q_for_text = init_q_emb
            ##################### w/o Query Reformulation #####################
            ##################### ModifyQueryReform #####################
            q_for_text = self.query_update(q_emb, q_mask, ent_emb,
                                           ent_seed_info)
            ##################### ModifyQueryReform #####################

            # q_for_text = q_node_emb.mean(1)
            # q_for_text = init_q_emb
            # (bsz, max_num_candidates, 2*hsz)
            q_node_emb = torch.cat([
                q_node_emb,
                q_for_text.unsqueeze(1).expand_as(q_node_emb).contiguous()
            ],
                                   dim=-1)
            '''
            document embedding
            '''
            # (bsz, max_local_entities, max_num_doc, max_doc_len)
            ent_linked_doc_spans = feed['ent_link_doc_spans']
            doc = feed['documents']  # (B, |D|, d_len)
            max_num_doc = doc.size(1)
            max_d_len = doc.size(2)
            doc_mask = (doc != 1).float()
            doc_len = doc_mask.sum(-1)  # (bsize, max_num_doc)
            doc_len += (doc_len == 0).float()  # padded documents have 0 words
            doc_len = doc_len.view(-1)  # (bsize * max_num_doc)
            # (bsz*num_doc, doc_len, emb_dim) esz = 3hsz
            d_word_emb = self.word_drop(
                self.word_emb(doc.view(-1, doc.size(-1))))
            '''
            encode document over original question representation 这一段在论文中是没有的
            '''
            ################## w/o word_emb_match (better do not) ###############
            # input features for documents over question info (seq attn and FFN)
            q_word_emb = q_word_emb.unsqueeze(1).expand(
                B, max_num_doc, max_q_len, self.word_dim).contiguous()
            q_word_emb = q_word_emb.view(B * max_num_doc, max_q_len, -1)
            q_mask_ = (question == 1).unsqueeze(1).expand(
                B, max_num_doc, max_q_len).contiguous()
            q_mask_ = q_mask_.view(B * max_num_doc, -1)
            # d^{->} document representation (bsz*num_doc, doc_len, esz)
            q_weighted_emb = self.word_emb_match(d_word_emb, q_word_emb,
                                                 q_mask_)
            ################## effective w/o feed['documents_em'] ##################
            # doc_em = feed['documents_em'].float().view(B*max_num_doc, max_d_len, 1) # ???
            # doc_input = torch.cat([d_word_emb, q_weighted_emb, doc_em], dim=-1) # 2*word_dim + 1
            doc_input = torch.cat([d_word_emb, q_weighted_emb],
                                  dim=-1)  # 2*word_dim
            ################## effective w/o feed['documents_em'] ###################
            # doc representation over question (bsz*num_doc, doc_len, hsz)
            doc_input = self.input_proj(doc_input).tanh()
            # doc_input = self.input_proj(d_word_emb).tanh()
            ################## w/o word_emb_match (better do not) ###############
            '''
            # Knowledge-aware Passage Enhancement with conditional gating function
            '''
            # get the doc words linking entities (bsz, num_doc*doc_len, max_local_entity)
            word_entity_id = ent_linked_doc_spans.view(B, max_num_candidates,
                                                       -1).transpose(1, 2)
            word_ent_info_mask = (
                word_entity_id.sum(-1, keepdim=True) != 0
            ).float(
            )  # (bsz, num_doc*doc_len, 1) the word w/o linking entities is 0, otherwise 1
            # word_ent_info (bsz, max_n_doc*max_doc_len, hsz)
            # 把doc中的word关联的所有entities的表征加起来得到结果的每一行向量，
            # 没有关联任何entities的word的行向量为0
            word_ent_info = torch.bmm(word_entity_id.float(), ent_emb)
            word_ent_info = self.ent_info_proj(word_ent_info).tanh()
            #################################### ConditionalSelfAttnEncoder ####################################
            # # gate function
            # doc_input = self.ent_info_gate(q_for_text.unsqueeze(1), word_ent_info, doc_input.view(B, max_num_doc*max_d_len, -1), word_ent_info_mask) # Knowledge-aware Passage Enhancement 的公式 (bsz, max_num_doc*max_doc_len, emb_size)
            # ######################## SelfAttn encoding ########################
            # # (bsz*m_n_doc, m_doc_len, hsz)
            # # d_emb, _ = self.doc_encoder(doc_input.view(B*max_num_doc, max_d_len, -1), doc_len, max_length=doc.size(2)) # 3.2中的Entity Info Aggregation from Text Reading 的bi-LSTM, which takes several token-level h^{d}_{wi}
            # doc_input_ = doc_input.view(B*max_num_doc, max_d_len, -1)
            # d_emb = self.doc_encoder(doc_input_, doc_input_, doc_input_, doc_mask.view(B*max_num_doc, -1))
            # ######################## SelfAttn encoding ########################
            # d_emb = self.hidden_drop(d_emb)
            # # gate function again (bsz*num_doc, max_doc_len, hsz)
            # d_emb = self.ent_info_gate_out(q_for_text.unsqueeze(1), word_ent_info, d_emb.view(B, max_num_doc*max_d_len, -1), word_ent_info_mask).view(B*max_num_doc, max_d_len, -1)

            word_ent_info_ = word_ent_info.contiguous().view(B, max_num_doc, max_d_len, -1) \
                .view(B * max_num_doc, max_d_len, -1)
            word_ent_info_mask_ = word_ent_info_mask.squeeze(-1).view(B, max_num_doc, max_d_len)\
                .view(B * max_num_doc, max_d_len)
            doc_mask_ = doc_mask.contiguous().view(B * max_num_doc,
                                                   max_d_len).float()
            d_emb = self.doc_encoder(word_ent_info_, word_ent_info_mask_,
                                     doc_input, doc_mask_)
            d_emb = self.hidden_drop(d_emb)
            # d_emb = self.doc_encoder(word_ent_info_, word_ent_info_mask_, d_emb, doc_mask_)
            #################################### ConditionalSelfAttnEncoder ####################################
            '''
            Entity Info Aggregation from Text Reading
            '''
            q_for_text = q_for_text.unsqueeze(1).expand(
                B, max_num_doc, self.hidden_dim).contiguous()
            q_for_text = q_for_text.view(B * max_num_doc, -1)  # (B*|D|, h_dim)
            d_emb = d_emb.view(B * max_num_doc, max_d_len,
                               -1)  # (B*|D|, d_len, h_dim)
            q_over_d = torch.bmm(q_for_text.unsqueeze(1), d_emb.transpose(
                1, 2)).squeeze(1)  # (B*|D|, d_len)
            q_over_d = F.softmax(
                q_over_d -
                (1 - doc_mask.view(B * max_num_doc, max_d_len)) * 1e8,
                dim=-1)
            # (bsz, max_num_doc, hsz)  get each document’s representation over question
            q_retrieve_d = torch.bmm(q_over_d.unsqueeze(1),
                                     d_emb).view(B, max_num_doc, -1)
            #  ent_linked_doc_spans.sum(-1): (B, num_candidate, num_doc)
            # 每一行表示所有entities对应的50个document，
            # 每一个元素是该document中出现该entities的次数
            # 这一结果表示每一个entity与哪一个document有关，有关的为1
            ent_linked_doc = (ent_linked_doc_spans.sum(-1) != 0).float()
            # (bsz, num_candidate, hsz) 对所有包含实体e的document的表征相加取平均得到该实体的表征
            ent_emb_from_doc = torch.bmm(ent_linked_doc, q_retrieve_d)
            ent_emb_from_doc = F.dropout(ent_emb_from_doc, 0.5, self.training)
            '''
            retrieve_span entities representation by sum relative document words
            '''
            # (bsz, num_candidate, m_n_doc, m_doc_len)
            ent_link_doc_norm_spans = feed['ent_link_doc_norm_spans'].float()
            ent_emb_from_span = torch.bmm(
                ent_link_doc_norm_spans.view(B, max_num_candidates, -1),
                d_emb.view(B, max_num_doc * max_d_len, -1))  # 值向量是每个单词的词向量
            ent_emb_from_span = F.dropout(ent_emb_from_span, 0.2,
                                          self.training)
        '''
        Answer Prediction
        '''
        # refine KB ent_emb
        # refined_ent_emb = self.refine_ent(ent_emb, ent_emb_from_doc)
        if self.use_doc:
            ent_emb = l_relu(
                self.attn_match(
                    torch.cat([ent_emb, ent_emb_from_doc, ent_emb_from_span],
                              dim=-1)))  # (bsz, num_can, hsz)
            ######################### less info (effective) #########################
            # ent_emb = l_relu(self.attn_match(torch.cat([ent_emb, ent_emb_from_doc], dim=-1)))
            # ent_emb = l_relu(self.attn_match(torch.cat([ent_emb, ent_emb_from_span], dim=-1)))
            # ent_emb = l_relu(self.attn_match(ent_emb_from_doc)) # not good
            ######################### less info (effective) #########################
            ######################### gate undate #########################
            ans_gamma = self.ans_gate(
                torch.cat([ent_emb, ent_emb_from_doc], dim=-1)).sigmoid()
            ent_emb = ans_gamma * ent_emb + (
                1 - ans_gamma) * ent_emb_from_doc  # (bsz, max_local_ent, hsz)
            ######################### gate undate #########################
            q_node_emb = self.attn_match_q(
                q_node_emb
            )  # 如果不使用ans_gate，则本行注释掉 q_node_emb 为(bsz, num_can, 2hsz)， self.attn_match(3hsz, 2hsz)

        ent_scores = (q_node_emb * ent_emb).sum(2)  # (bsz, max_num_candidates)
        ############################## try to select high pagerank score only ##############################
        # highpr_ent = feed['highpr_ent']  # (bsz, max_num_candidates)
        # ent_scores[highpr_ent != 1] = -1e9
        ############################## try to select high pagerank score only ##############################
        answers = feed['answers'].float()

        if self.label_smooth:  # not good
            answers = ((1.0 - self.label_smooth) *
                       answers) + (self.label_smooth / answers.size(1))

        ########################### non negative constraint ###########################
        # re_non_neg = F.relu(-self.entity_emb.weight).sum() / self.entity_emb.weight.size(0)
        ########################### non negative constraint ###########################
        # both (bsz, max_local_candidate)  1 for answer 0 for non-answer
        loss = self.loss(ent_scores, feed['answers'].float())  # + re_non_neg

        pred_dist = (ent_scores - (1 - ent_mask) * 1e8).sigmoid() * ent_mask
        pred = torch.max(ent_scores, dim=1)[1]

        return loss, pred, pred_dist


class KVMemNN(nn.Module):

    def __init__(self, args):
        super(KVMemNN, self).__init__()

        self.entity2id = load_dict(args['data_folder'] + args['entity2id'])
        self.word2id = load_dict(args['data_folder'] + args['word2id'])
        self.relation2id = load_dict(args['data_folder'] + args['relation2id'])
        self.num_entity = len(self.entity2id)
        self.num_relation = len(self.relation2id)
        self.num_word = len(self.word2id)
        self.num_layer = args['num_layer']
        self.use_doc = args['use_doc']
        self.word_drop = args['word_drop']
        self.hidden_drop = args['hidden_drop']
        self.label_smooth = args['label_smooth']
        self.use_cuda = args['use_cuda']

        for k, v in args.items():
            if k.endswith('dim'):
                setattr(self, k, v)
            if k.endswith('emb_file'):
                setattr(self, k, args['data_folder'] + v)

        self.num_hop = 3
        self.hidden_size = 64

        self.entity_linear = nn.Linear(self.entity_dim, self.hidden_size)
        self.self_att_r = AttnEncoder(self.word_dim)

        self.entity_emb = nn.Embedding(self.num_entity + 1,
                                       self.entity_dim,
                                       padding_idx=self.num_entity)
        self.entity_emb.weight.data.copy_(
            torch.from_numpy(
                np.pad(np.load(self.entity_emb_file), ((0, 1), (0, 0)),
                       'constant')))
        self.entity_emb.weight.requires_grad = False
        self.entity_emb = nn.DataParallel(self.entity_emb)

        self.word_emb = nn.Embedding(self.num_word,
                                     self.word_dim,
                                     padding_idx=1)  # rel, query, doc
        self.word_emb.weight.data.copy_(
            torch.from_numpy(np.load(self.word_emb_file)))
        self.word_emb.weight.requires_grad = False
        self.word_emb = nn.DataParallel(self.word_emb)

        self.rel_encoder = nn.Linear(self.word_dim, self.hidden_size)
        self.rel_dropout = nn.Dropout(p=self.hidden_drop)

        self.query_encoder = nn.Linear(self.word_dim, self.hidden_size)
        self.query_dropout = nn.Dropout(p=self.hidden_drop)

        self.key_kb_encoder = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_doc_encoder = nn.Linear(self.word_dim, self.hidden_size)
        self.key_dropout = nn.Dropout(p=self.hidden_drop)

        self.value_encoder = nn.Linear(self.entity_dim, self.hidden_size)
        self.value_dropout = nn.Dropout(p=self.hidden_drop)

        self.out_project = nn.Linear(self.hidden_size, self.hidden_size)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, feed):
        """
        :param query: (bsz, q_len) query word index, padding with 1
        :param key_kb: (bsz, num_mem, 2) padding memory with 1
        :param value_kb: (bsz, num_mem)
        :param key_doc: (bsz, num_mem, doc_len)
        :param value_doc: (bsz, num_mem)
        :param answers: (bsz, num_candidates)
        :return: loss
        :return: pred
        """
        '''
        encode query
        '''
        query = feed['questions']
        # (bsz, q_len, wsz)
        emb_q = self.word_emb(query)
        mask_q = (query != 1).float()
        # (bsz, q_len, hsz)
        init_enc_q = self.query_encoder(emb_q).tanh()
        init_enc_q = self.query_dropout(init_enc_q)
        init_enc_q = init_enc_q * mask_q.unsqueeze(-1)
        # (bsz, 1, hsz)
        encoded_q = torch.sum(init_enc_q, dim=1, keepdim=True)
        '''
        encode key and concat
        '''
        key_kb = feed['key_kb']  # (bsz, num_mem, 2) padding memory with 0
        bsz = key_kb.size(0)
        num_mem = key_kb.size(1)
        key_kb_subject = key_kb[:, :, 0]  # (bsz, num_mem)
        key_kb_relation = key_kb[:, :, 1]

        # subject entities encoding
        key_kb_subject_emb_ = self.entity_emb(
            key_kb_subject)  # (bsz, num_mem, hsz)
        key_kb_subject_emb = l_relu(
            self.entity_linear(key_kb_subject_emb_))  # (bsz, num_mem, hsz)

        # relations encoding
        rel_word_ids = feed[
            'rel_word_ids']  # (num_rel+1, word_lens) load directly from files
        rel_word_mask = (rel_word_ids != 1).float()
        rel_word_emb = self.word_emb(rel_word_ids)
        rel_encoded_agg = self.self_att_r(rel_word_emb,
                                          rel_word_mask)  # (num_rel+1, wsz)
        rel_encoded = self.rel_encoder(
            rel_encoded_agg).tanh()  # (num_rel+1, hsz)
        rel_encoded = self.rel_dropout(rel_encoded)
        # find the representation of relation from index
        key_kb_relation = key_kb_relation.long().view(-1)
        key_kb_rel_emb = torch.index_select(rel_encoded, dim=0, index=key_kb_relation).\
                                            view(bsz, num_mem, -1).unsqueeze(2)  # (bsz, num_mem, 1, hsz)

        # (bsz, num_mem, 2, wsz)
        emb_key_kb = torch.cat(
            [key_kb_subject_emb.unsqueeze(2), key_kb_rel_emb], dim=2)
        mask_key_kb = (key_kb != 0).float()
        # (bsz, num_mem, 2, hsz)
        init_enc_key_kb = self.key_kb_encoder(emb_key_kb).tanh()
        init_enc_key_kb = self.key_dropout(init_enc_key_kb)
        init_enc_key_kb = init_enc_key_kb * mask_key_kb.unsqueeze(-1)
        # (bsz, num_mem, hsz)
        encoded_key_kb = torch.sum(init_enc_key_kb, dim=2, keepdim=False)

        key_doc = feed['key_doc']
        # (bsz, num_mem, doc_len, wsz)
        emb_key_doc = self.word_emb(key_doc)
        mask_key_doc = (key_doc != 1).float()
        # (bsz, num_mem, doc_len, hsz)
        init_enc_key_doc = self.key_doc_encoder(emb_key_doc).tanh()
        init_enc_key_doc = self.key_dropout(init_enc_key_doc)
        init_enc_key_doc = init_enc_key_doc * mask_key_doc.unsqueeze(-1)
        # (bsz, num_mem, hsz)
        encoded_key_doc = torch.sum(init_enc_key_doc, dim=2, keepdim=False)

        # concat kb's key and doc's key (bsz, 2*num_mem, hsz)
        encoded_key = torch.cat([encoded_key_kb, encoded_key_doc], dim=1)
        '''
        encode value and concat
        '''
        value_kb = feed['val_kb']
        # (bsz, num_mem, esz)
        emb_value_kb = self.entity_emb(value_kb)
        mask_value_kb = (value_kb != 0).float()
        # (bsz, num_mem, hsz)
        init_enc_value_kb = self.value_encoder(emb_value_kb).tanh()
        init_enc_value_kb = self.value_dropout(init_enc_value_kb)
        init_enc_value_kb = init_enc_value_kb * mask_value_kb.unsqueeze(-1)

        # (bsz, num_mem, esz)
        value_doc = feed['val_doc']
        emb_value_doc = self.entity_emb(value_doc)
        mask_value_doc = (value_doc != 0).float()
        # (bsz, num_mem, hsz)
        init_enc_value_doc = self.value_encoder(emb_value_doc).tanh()
        init_enc_value_doc = self.value_dropout(init_enc_value_doc)
        init_enc_value_doc = init_enc_value_doc * mask_value_doc.unsqueeze(-1)

        # (bsz, 2*num_mem, hsz)
        encoded_value = torch.cat([init_enc_value_kb, init_enc_value_doc],
                                  dim=1)
        '''
        encode candidate entities
        '''
        candidates = feed['candidate_entities']
        # (bsz, num_candidates, hsz)
        emb_candidates = self.entity_emb(candidates)
        encoded_candidates = l_relu(self.entity_linear(
            emb_candidates))  # (bsz, max_local_entities, hsz)
        # it should mask padding here
        # expect padding memory is zero
        for hop in range(self.num_hop):
            # (bsz, 1, 2*num_mem)
            ph = torch.bmm(encoded_q, encoded_key.transpose(1, 2))
            ph[ph == 0.0] = -1e9
            score = F.softmax(ph, dim=-1)
            # (bsz, 1, hsz)
            out = torch.bmm(score, encoded_value)
            encoded_q = self.out_project(encoded_q + out)
        '''
        final prediction
        '''
        # (bsz, num_candidates) with 0 padding
        score_pred = (encoded_q * encoded_candidates).sum(2)
        answers = feed['answers'].float()

        loss = self.loss(score_pred, answers.float())
        pred = torch.max(score_pred, dim=1)[1]
        pred_dist = score_pred.sigmoid()  # probability need to be nomalization

        return loss, pred, pred_dist
