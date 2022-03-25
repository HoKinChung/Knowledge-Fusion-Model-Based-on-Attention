import json
import nltk
import numpy as np
import random
import torch

from collections import defaultdict
from tqdm import tqdm
from util import get_config
from util import load_dict
from util import load_documents


class DataLoader():

    def __init__(self, config, documents, mode='train'):
        self.mode = mode
        self.use_doc = config['use_doc']
        self.use_inverse_relation = config['use_inverse_relation']
        self.max_query_word = config['max_query_word']
        self.max_document_word = config['max_document_word']
        self.max_char = config['max_char']
        self.documents = documents
        self.data_file = config['data_folder'] + config['{}_data'.format(mode)]
        self.batch_size = config['batch_size'] if mode == 'train' else config[
            'batch_size']
        self.max_rel_words = config['max_rel_words']
        self.type_rels = config['type_rels']
        self.fact_drop = config['fact_drop']
        self.use_cuda = config['use_cuda']

        # read all data (passages, subgraph, query, answers, entities)
        self.data = []
        with open(self.data_file) as f:
            for line in tqdm(list(f)):
                self.data.append(json.loads(line))

        # word and kb vocab
        self.word2id = load_dict(config['data_folder'] + config['word2id'])
        self.relation2id = load_dict(config['data_folder'] +
                                     config['relation2id'])
        self.entity2id = load_dict(config['data_folder'] + config['entity2id'])
        self.id2entity = {i: entity for entity, i in self.entity2id.items()}

        self.rel_word_idx = np.load(config['data_folder'] + 'rel_word_idx.npy')

        # for batching
        self.max_local_entity = 0  # max num of candidates
        self.max_relevant_docs = 0  # max num of retired documents
        # self.max_relevant_docs = 50  # max num of retired documents
        self.max_kb_neighbors = config[
            'max_num_neighbors']  # max num of neighbors for entity
        self.max_kb_neighbors_ = config[
            'max_num_neighbors']  # kb relations are directed
        self.max_linked_entities = 0  # max num of linked entities for each doc
        self.max_linked_documents = 50  # max num of linked documents for each entity

        self.num_kb_relation = 2 * len(
            self.relation2id) if self.use_inverse_relation else len(
                self.relation2id)

        # get the batching parameters
        self.get_stats()

    def get_stats(self):
        if self.use_doc:
            # max_linked_entities
            self.useful_docs = {
            }  # collect all useful document filter out documents with out linked entities
            for docid, doc in self.documents.items():
                linked_entities = 0
                if 'title' in doc:
                    linked_entities += len(doc['title']['entities'])
                    offset = len(nltk.word_tokenize(doc['title']['text']))
                else:
                    offset = 0
                for ent in doc['document']['entities']:
                    if ent['start'] + offset >= self.max_document_word:
                        continue
                    else:
                        linked_entities += 1
                if linked_entities > 1:
                    self.useful_docs[
                        docid] = doc  # num(self.documents) >= num(self.useful_docs)
                self.max_linked_entities = max(self.max_linked_entities,
                                               linked_entities)
            print('max num of linked entities: ', self.max_linked_entities)

        # decide how many neighbors should we consider
        # num_neighbors = []

        num_tuples = []

        # max_linked_documents, max_relevant_docs, max_local_entity
        for line in tqdm(self.data):
            candidate_ents = set()
            rel_docs = 0
            # collect candidate entities
            # question entity
            for ent in line['entities']:
                candidate_ents.add(ent['text'])
            # kb entities
            for ent in line['subgraph']['entities']:
                candidate_ents.add(ent['text'])

            num_tuples.append(line['subgraph']['tuples'])

            if self.use_doc:
                # add entities in doc to candidate_ents
                for passage in line['passages']:
                    if passage[
                            'document_id'] not in self.useful_docs:  # useful_docs, docs containing entities
                        continue
                    rel_docs += 1
                    document = self.useful_docs[int(passage['document_id'])]
                    for ent in document['document']['entities']:
                        candidate_ents.add(ent['text'])
                    if 'title' in document:
                        for ent in document['title']['entities']:
                            candidate_ents.add(ent['text'])

            neighbors = defaultdict(list)
            neighbors_ = defaultdict(list)
            # read subgraph
            for triple in line['subgraph']['tuples']:
                s, r, o = triple
                neighbors[s['text']].append((r['text'], o['text']))
                neighbors_[o['text']].append((r['text'], s['text']))

            self.max_relevant_docs = max(self.max_relevant_docs, rel_docs)
            self.max_local_entity = max(self.max_local_entity,
                                        len(candidate_ents))

        # np.save('num_neighbors_', num_neighbors)

        print('mean num of triples: ', len(num_tuples))

        print('max num of relevant docs: ', self.max_relevant_docs)
        print('max num of candidate entities: ', self.max_local_entity)
        print('max_num of neighbors: ', self.max_kb_neighbors)
        print('max_num of neighbors inverse: ', self.max_kb_neighbors_)

    def batcher(self, shuffle=False):
        if shuffle:
            random.shuffle(self.data)

        device = torch.device('cuda')

        for batch_id in tqdm(range(0, len(self.data), self.batch_size)):
            batch = self.data[batch_id:batch_id + self.batch_size]

            batch_size = len(batch)
            questions = np.full((batch_size, self.max_query_word),
                                1,
                                dtype=int)
            documents = np.full(
                (batch_size, self.max_relevant_docs, self.max_document_word),
                1,
                dtype=int)
            entity_link_documents = np.zeros(
                (batch_size, self.max_local_entity, self.max_linked_documents,
                 self.max_document_word),
                dtype=int)
            entity_link_doc_norm = np.zeros(
                (batch_size, self.max_local_entity, self.max_linked_documents,
                 self.max_document_word),
                dtype=int)
            documents_ans_span = np.zeros(
                (batch_size, self.max_relevant_docs, 2), dtype=int)
            entity_link_ents = np.full(
                (batch_size, self.max_local_entity, self.max_kb_neighbors_),
                -1,
                dtype=int)  # incoming edges
            entity_link_rels = np.zeros(
                (batch_size, self.max_local_entity, self.max_kb_neighbors_),
                dtype=int)
            candidate_entities = np.full((batch_size, self.max_local_entity),
                                         len(self.entity2id),
                                         dtype=int)
            ent_degrees = np.zeros((batch_size, self.max_local_entity),
                                   dtype=int)
            true_answers = np.zeros((batch_size, self.max_local_entity),
                                    dtype=float)
            query_entities = np.zeros((batch_size, self.max_local_entity),
                                      dtype=float)
            highpr_ent = np.zeros((batch_size, self.max_local_entity),
                                  dtype=int)
            answers_ = []
            questions_ = []

            for i, sample in enumerate(batch):
                doc_global2local = {}
                '''
                collect answer set
                '''
                answers = set()
                for answer in sample['answers']:
                    keyword = 'text' if type(
                        answer['kb_id']) == int else 'kb_id'
                    answers.add(self.entity2id[answer[keyword]])

                if self.mode != 'train':
                    answers_.append(list(answers))
                    questions_.append(sample['question'])
                '''
                collect candidate entities
                '''
                candidates = set()
                highpr_candidates = set()
                query_entity = set()
                ent2linked_docId = defaultdict(list)
                for ent in sample['entities']:
                    candidates.add(self.entity2id[ent['text']])
                    query_entity.add(self.entity2id[ent['text']])
                # ranking the entities by pagerank score
                subgraph_entities = sample['subgraph']['entities']
                subgraph_entities = sorted(subgraph_entities,
                                           key=lambda x: x['pagerank_score'],
                                           reverse=True)
                select_rate = 0.2
                select_num = int(len(subgraph_entities) * select_rate)
                for c, ent in enumerate(subgraph_entities):
                    # for ent in sample['subgraph']['entities']:
                    if c < select_num:
                        highpr_candidates.add(self.entity2id[ent['text']])
                    candidates.add(self.entity2id[ent['text']])
                '''
                collect linking document(high pagerank score) and add entities in document to candidates
                '''
                if self.use_doc:
                    # fliter some passages
                    # select_rate = 0.3
                    # select_num = int(len(sample['passages']) * select_rate)
                    sample_passages = sample['passages'][0:]
                    for local_id, passage in enumerate(sample_passages):
                        # for local_id, passage in enumerate(sample['passages']):
                        if passage['document_id'] not in self.useful_docs:
                            continue
                        doc_id = int(passage['document_id'])
                        doc_global2local[
                            doc_id] = local_id  # rank the id about document
                        document = self.useful_docs[doc_id]
                        for word_pos, word in enumerate(
                            ['<bos>'] +
                                document['tokens']):  # convert word to id
                            if word_pos < self.max_document_word:
                                documents[i, local_id,
                                          word_pos] = self.word2id.get(
                                              word, self.word2id['<unk>']
                                          )  # (bsz, num_doc, doc_len)
                        for ent in document['document']['entities']:
                            if self.entity2id[ent['text']] in answers:
                                documents_ans_span[i, local_id, 0] = min(
                                    ent['start'] + 1, self.max_document_word -
                                    1)  # answer span start position
                                documents_ans_span[i, local_id, 1] = min(
                                    ent['end'] + 1, self.max_document_word -
                                    1)  # answer span end position
                            s, e = ent['start'] + 1, ent['end'] + 1
                            ent2linked_docId[self.entity2id[
                                ent['text']]].append((doc_id, s, e))
                            candidates.add(self.entity2id[ent['text']])
                        if 'title' in document:
                            for ent in document['title']['entities']:
                                candidates.add(self.entity2id(ent['text']))
                '''
                collect kb information
                '''
                connections = defaultdict(list)  # one subgraph one question

                if self.fact_drop and self.mode == 'train':
                    all_triples = sample['subgraph']['tuples']
                    random.shuffle(all_triples)
                    num_triples = len(all_triples)
                    keep_ratio = 1 - self.fact_drop
                    all_triples = all_triples[:int(num_triples * keep_ratio)]
                else:
                    all_triples = sample['subgraph']['tuples']
                '''
                collect subgraph tuples and add into connections
                '''
                for tpl in all_triples:
                    s, r, o = tpl
                    # only consider one direction of information propagation
                    connections[self.entity2id[o['text']]].append(
                        (self.relation2id[r['text']], self.entity2id[s['text']]
                         ))  # element of dict: o_id: [(r_id, s_id)]

                    if r['text'] in self.type_rels:  # reversed relation
                        connections[self.entity2id[s['text']]].append(
                            (self.relation2id[r['text']],
                             self.entity2id[o['text']]))

                ent_global2local = {}
                candidates = list(candidates)
                '''
                entities_query link and entities_document link
                '''
                for j, entid in enumerate(candidates):
                    if entid in query_entity:  # entities_query link
                        query_entities[i, j] = 1.0
                    candidate_entities[i, j] = entid  # (bsz, num_candidate)
                    if entid in highpr_candidates:
                        highpr_ent[i, j] = 1
                    ent_global2local[entid] = j
                    if entid in answers: true_answers[i, j] = 1.0
                    for linked_doc in ent2linked_docId[
                            entid]:  # entities_document link
                        start, end = linked_doc[1], linked_doc[2]
                        if end - start > 0:  # so entity_link_documents is the same as entity_link_doc_norm???
                            entity_link_documents[
                                i, j, doc_global2local[linked_doc[0]], start:
                                end] = 1.0  # (batch_size, max_local_entity, num_doc, doc_len)
                            entity_link_doc_norm[
                                i, j, doc_global2local[linked_doc[0]],
                                start:end] = 1.0
                '''
                collect candidates subgraph
                '''
                for j, entid in enumerate(candidates):
                    for count, neighbor in enumerate(connections[entid]):
                        if count < self.max_kb_neighbors_:
                            r_id, s_id = neighbor
                            # convert the global ent id to subgraph id, for graph convolution
                            s_id_local = ent_global2local[s_id]
                            entity_link_rels[i, j, count] = r_id
                            entity_link_ents[i, j, count] = s_id_local
                            ent_degrees[
                                i,
                                s_id_local] += 1  # degree is the number of edge of a vertex
                '''
                collect questions (word2index)
                '''
                for j, word in enumerate(sample['question'].split()):
                    if j < self.max_query_word:
                        if word in self.word2id:
                            questions[i, j] = self.word2id[word]
                        else:
                            questions[i, j] = self.word2id['<unk>']

            # if self.use_doc:
            #     # actually do not use
            #     # exact match features for docs
            #     # exact match features (if the word which appears in the question also appears in the document, it will be set to 1)
            #     d_cat = documents.reshape((batch_size, -1))
            #     em_d = np.array([np.isin(d_, q_) for d_, q_ in zip(d_cat, questions)],
            #                     dtype=int)
            #     em_d = em_d.reshape((batch_size, self.max_relevant_docs, -1))

            batch_dict = {
                'questions': questions,  # (B, q_len)
                'candidate_entities': candidate_entities,
                'entity_link_ents': entity_link_ents,
                'answers': true_answers,
                'query_entities': query_entities,
                'answers_': answers_,
                'questions_': questions_,
                'rel_word_ids': self.rel_word_idx,  # (num_rel+1, word_lens)
                'entity_link_rels':
                entity_link_rels,  # (bsize, max_num_candidates, max_num_neighbors)
                'ent_degrees': ent_degrees,
                'highpr_ent': highpr_ent  # (bsz, max_num_candidate)
            }  # summary batch data

            if self.use_doc:
                batch_dict['documents'] = documents
                # batch_dict['documents_em'] = em_d
                batch_dict['ent_link_doc_spans'] = entity_link_documents
                # batch_dict['documents_ans_span'] = documents_ans_span  # no use
                batch_dict['ent_link_doc_norm_spans'] = entity_link_doc_norm

            for k, v in batch_dict.items():
                if k.endswith('_'):  # answers_ and questions_
                    batch_dict[k] = v
                    continue
                if not self.use_doc and 'doc' in k:
                    continue
                if self.use_cuda:
                    batch_dict[k] = torch.from_numpy(v).to(
                        device)  # numpy --> torch
                else:
                    batch_dict[k] = torch.from_numpy(v)
            yield batch_dict


class DataLoaderForKV():

    def __init__(self, config, documents, mode='train'):
        self.mode = mode
        self.use_doc = config['use_doc']
        self.use_inverse_relation = config['use_inverse_relation']
        self.max_query_word = config['max_query_word']
        self.max_document_word = config['max_document_word']
        self.max_char = config['max_char']
        self.documents = documents
        self.data_file = config['data_folder'] + config['{}_data'.format(mode)]
        self.batch_size = config['batch_size'] if mode == 'train' else config[
            'batch_size']
        self.max_rel_words = config['max_rel_words']
        self.type_rels = config['type_rels']
        self.fact_drop = config['fact_drop']
        self.use_cuda = config['use_cuda']

        # read all data (passages, subgraph, query, answers, entities)
        self.data = []
        with open(self.data_file) as f:
            for line in tqdm(list(f)):
                self.data.append(json.loads(line))

        # word and kb vocab
        self.word2id = load_dict(config['data_folder'] + config['word2id'])
        self.relation2id = load_dict(config['data_folder'] +
                                     config['relation2id'])
        self.entity2id = load_dict(config['data_folder'] + config['entity2id'])
        self.id2entity = {i: entity for entity, i in self.entity2id.items()}

        self.rel_word_idx = np.load(config['data_folder'] + 'rel_word_idx.npy')

        # for batching
        self.max_local_entity = 0  # max num of candidates
        self.max_relevant_docs = 0  # max num of retired documents
        self.max_kb_neighbors = config[
            'max_num_neighbors']  # max num of neighbors for entity
        self.max_kb_neighbors_ = config[
            'max_num_neighbors']  # kb relations are directed
        self.max_linked_entities = 0  # max num of linked entities for each doc
        self.max_linked_documents = 50  # max num of linked documents for each entity

        self.num_kb_relation = 2 * len(
            self.relation2id) if self.use_inverse_relation else len(
                self.relation2id)

        # for model
        self.max_num_memory = 1800
        self.win_size = 3

        # get the batching parameters
        self.get_stats()

    def get_stats(self):
        if self.use_doc:
            # max_linked_entities
            self.useful_docs = {
            }  # collect all useful document filter out documents with out linked entities 只有链接了实体的document才算作useful docs
            for docid, doc in self.documents.items():
                linked_entities = 0
                if 'title' in doc:
                    linked_entities += len(doc['title']['entities'])
                    offset = len(nltk.word_tokenize(doc['title']['text']))
                else:
                    offset = 0
                for ent in doc['document']['entities']:
                    if ent['start'] + offset >= self.max_document_word:
                        continue
                    else:
                        linked_entities += 1
                if linked_entities > 1:
                    self.useful_docs[docid] = doc  # !!!
                self.max_linked_entities = max(self.max_linked_entities,
                                               linked_entities)
            print('max num of linked entities: ', self.max_linked_entities)

        # decide how many neighbors should we consider
        # num_neighbors = []

        num_tuples = []

        # max_linked_documents, max_relevant_docs, max_local_entity
        for line in tqdm(self.data):
            candidate_ents = set()
            rel_docs = 0
            # collect candidate entities
            # question entity
            for ent in line['entities']:
                candidate_ents.add(ent['text'])
            # kb entities
            for ent in line['subgraph']['entities']:
                candidate_ents.add(ent['text'])

            num_tuples.append(line['subgraph']['tuples'])

            if self.use_doc:
                # add entities in doc to candidate_ents
                for passage in line['passages']:
                    if passage['document_id'] not in self.useful_docs:
                        continue
                    rel_docs += 1
                    document = self.useful_docs[int(passage['document_id'])]
                    for ent in document['document']['entities']:
                        candidate_ents.add(ent['text'])
                    if 'title' in document:
                        for ent in document['title']['entities']:
                            candidate_ents.add(ent['text'])

            neighbors = defaultdict(list)
            neighbors_ = defaultdict(list)
            # read subgraph
            for triple in line['subgraph']['tuples']:
                s, r, o = triple
                neighbors[s['text']].append((r['text'], o['text']))
                neighbors_[o['text']].append((r['text'], s['text']))

            self.max_relevant_docs = max(self.max_relevant_docs, rel_docs)
            self.max_local_entity = max(self.max_local_entity,
                                        len(candidate_ents))

        print('mean num of triples: ', len(num_tuples))
        print('max num of relevant docs: ', self.max_relevant_docs)
        print('max num of candidate entities: ', self.max_local_entity)
        print('max_num of neighbors: ', self.max_kb_neighbors)
        print('max_num of neighbors inverse: ', self.max_kb_neighbors_)

    def batcher(self, shuffle=False):
        if shuffle:
            random.shuffle(self.data)
        device = torch.device('cuda')
        for batch_id in tqdm(range(0, len(self.data), self.batch_size)):
            batch = self.data[batch_id:batch_id + self.batch_size]
            batch_size = len(batch)

            true_answers = np.zeros((batch_size, self.max_local_entity),
                                    dtype=float)
            questions = np.full((batch_size, self.max_query_word),
                                1,
                                dtype=int)
            candidate_entities = np.full((batch_size, self.max_local_entity),
                                         len(self.entity2id),
                                         dtype=int)
            key_kb = np.full((batch_size, self.max_num_memory, 2),
                             0,
                             dtype=int)
            val_kb = np.full((batch_size, self.max_num_memory), 0, dtype=int)
            key_doc = np.full(
                (batch_size, self.max_num_memory, self.max_document_word),
                0,
                dtype=int)
            val_doc = np.full((batch_size, self.max_num_memory), 0, dtype=int)
            answers_ = []
            questions_ = []
            for i, sample in enumerate(batch):
                '''
                collect answer set
                '''
                answers = set()
                for answer in sample['answers']:
                    keyword = 'text' if type(
                        answer['kb_id']) == int else 'kb_id'
                    answers.add(self.entity2id[answer[keyword]])

                if self.mode != 'train':
                    answers_.append(list(answers))
                    questions_.append(sample['question'])
                '''
                collect questions (word2index)
                '''
                for j, word in enumerate(sample['question'].split()):
                    if j < self.max_query_word:
                        if word in self.word2id:
                            questions[i, j] = self.word2id[word]
                        else:
                            questions[i, j] = self.word2id['<unk>']
                '''
                collect candidate entities
                '''
                candidates = set()
                for ent in sample['entities']:
                    candidates.add(self.entity2id[ent['text']])
                # ranking the entities by pagerank score
                subgraph_entities = sample['subgraph']['entities']
                subgraph_entities = sorted(subgraph_entities,
                                           key=lambda x: x['pagerank_score'],
                                           reverse=True)
                # select_rate = 0.3
                # select_num = int(len(subgraph_entities) * select_rate)
                select_num = min(10, len(subgraph_entities))
                subgraph_entities = subgraph_entities[0:select_num]
                for ent in subgraph_entities:
                    candidates.add(self.entity2id[ent['text']])
                '''
                collect doc
                '''
                if self.use_doc:
                    mem_count = 0
                    # fliter some passages
                    # select_rate = 0.3
                    # select_num = int(len(sample['passages']) * select_rate)
                    sample_passages = sample['passages']
                    select_num = min(10, len(sample_passages))
                    sample_passages = sample_passages[0:select_num]
                    for local_id, passage in enumerate(sample_passages):
                        if passage['document_id'] not in self.useful_docs:
                            continue
                        doc_id = int(passage['document_id'])
                        document = self.useful_docs[
                            doc_id]  # read a useful document
                        for ent in document['document']['entities']:
                            if mem_count > self.max_num_memory - 1:
                                break
                            val_doc[i, mem_count] = self.entity2id[ent['text']]
                            s = min(ent['start'], self.max_document_word - 1)
                            e = min(ent['end'], self.max_document_word - 1)
                            mem_w_pos = 0
                            for word_pos in range(
                                    max(0, s - self.win_size),
                                    min(
                                        len(document['tokens']) - 1,
                                        e + self.win_size + 1)):
                                if word_pos < self.max_document_word:
                                    token = document['tokens'][word_pos]
                                    key_doc[i, mem_count,
                                            mem_w_pos] = self.word2id.get(
                                                token, self.word2id['<unk>'])
                                    mem_w_pos += 1
                            key_doc[i, mem_count, mem_w_pos:] = 1
                            mem_count += 1
                            candidates.add(self.entity2id[ent['text']])
                        if 'title' in document:
                            for ent in document['title']['entities']:
                                candidates.add(self.entity2id(ent['text']))

                candidates = list(
                    candidates)  # all candidate entities in a sample
                '''
                collect kb information
                '''
                connections = defaultdict(list)  # all triples in a sample

                if self.fact_drop and self.mode == 'train':
                    all_triples = sample['subgraph']['tuples']
                    random.shuffle(all_triples)
                    num_triples = len(all_triples)
                    keep_ratio = 1 - self.fact_drop
                    all_triples = all_triples[:int(num_triples * keep_ratio)]
                else:
                    all_triples = sample['subgraph']['tuples']

                for tpl in all_triples:
                    s, r, o = tpl
                    # element of dict: o_id: [(r_id, s_id)]
                    connections[self.entity2id[o['text']]].append(
                        (self.relation2id[r['text']],
                         self.entity2id[s['text']]))
                '''
                collect the candidate entities kb info
                '''
                for j, entid in enumerate(candidates):
                    if j > self.max_num_memory - 1:
                        break
                    if entid in answers:
                        true_answers[i, j] = 1.0
                    candidate_entities[i, j] = entid
                    for count, neighbor in enumerate(
                            connections[entid]
                    ):  # some candidates has no kb info
                        if count < self.max_kb_neighbors:
                            r_id, s_id = neighbor
                            key_kb[i, j, 0] = s_id
                            key_kb[i, j, 1] = r_id
                            val_kb[i, j] = entid

            batch_dict = {
                'questions':
                questions,  # (bsz, q_len) query word index, padding word with 1
                'candidate_entities':
                candidate_entities,  # (bsz, max_num_candidates), padding with len(self.entity2id)
                'key_kb': key_kb,  # (bsz, num_mem, 2) padding memory with 0
                'val_kb': val_kb,  # (bsz, num_mem) padding memory with 0
                'key_doc':
                key_doc,  # (bsz, num_mem, doc_len) padding memory with 0, padding word with 1
                'val_doc': val_doc,  # (bsz, num_mem), padding memory with 0
                'answers': true_answers,  # (bsz, max_num_candidates)
                'answers_': answers_,  # (bsz, num_ans) answer idx
                'questions_': questions_,  # (bsz, ) query words
                'rel_word_ids': self.rel_word_idx,  # (num_rel+1, word_lens)
                # 'ent_link_doc_norm_spans': entity_link_doc_norm
            }  # summary batch data

            for k, v in batch_dict.items():
                if k.endswith('_'):  # answers_ and questions_ ????
                    batch_dict[k] = v
                    continue
                if not self.use_doc and 'doc' in k:
                    continue
                if self.use_cuda:
                    batch_dict[k] = torch.from_numpy(v).to(
                        device)  # numpy --> torch
                else:
                    batch_dict[k] = torch.from_numpy(v)
            yield batch_dict


if __name__ == '__main__':
    cfg = get_config()
    documents = load_documents(cfg['data_folder'] +
                               cfg['{}_documents'.format(cfg['mode'])])
    # cfg['batch_size'] = 2
    train_data = DataLoader(cfg, documents)
    # build_squad_like_data(cfg['data_folder'] + cfg['{}_data'.format(cfg['mode'])], cfg['data_folder'] + cfg['{}_documents'.format(cfg['mode'])])
    for batch in train_data.batcher():
        print(batch['documents_ans_span'])
        assert False
