import random

import numpy as np
import torch

from model import KVMemNN
from data_generator import DataLoaderForKV
from util import get_config, cal_accuracy, load_documents


class Trainer():

    def __init__(self, cfg):
        self.num_epoch = cfg['num_epoch']
        self.model = KVMemNN(cfg)
        if cfg['use_cuda']:
            self.model = self.model.to(torch.device('cuda'))
        self.train_params = filter(lambda p: p.requires_grad,
                                   self.model.parameters())
        self.optimizer = torch.optim.Adam(self.train_params,
                                          lr=cfg['learning_rate'])
        if cfg['lr_schedule']:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, [30, 50], gamma=0.5)

        self.best_val_f1 = 0.0
        self.best_val_hits = 0.0

    def train(self, train_data, cfg):
        self.model.train()
        for epoch in range(self.num_epoch):
            print("Epoch {}".format(epoch))
            batcher = train_data.batcher(shuffle=True)
            train_loss, train_acc, train_max_acc = [], [], []
            for feed in batcher:
                loss, pred, score_pred = self.model(feed)
                train_loss.append(loss.item())
                acc, max_acc = cal_accuracy(pred,
                                            feed['answers'].cpu().numpy())
                train_acc.append(acc)
                train_max_acc.append(max_acc)

                self.optimizer.zero_grad()
                loss.backward()
                if cfg['gradient_clip'] != 0:
                    torch.nn.utils.clip_grad_norm_(self.train_params,
                                                   cfg['gradient_clip'])
                self.optimizer.step()
            print("Epoch {}: batch average training loss {}, "
                  "batch average training acc {}".format(
                      epoch, np.mean(train_loss), np.mean(train_acc)))

            # valid
            val_f1, val_hits = self.test(valid_data, cfg)
            if cfg['lr_schedule']:
                self.scheduler.step()
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
            if val_hits > self.best_val_hits:
                self.best_val_hits = val_hits
                torch.save(
                    self.model.state_dict(),
                    'model/{}/{}_best.pt'.format(cfg['name'], cfg['model_id']))
            print('evaluation best f1:{} current:{}'.format(
                self.best_val_f1, val_f1))
            print('evaluation best hits:{} current:{}'.format(
                self.best_val_hits, val_hits))

    def test(self, test_data, cfg):
        # self.model.load_state_dict(torch.load('model/{}/{}_best.pt'.format(cfg['name'], cfg['model_id'])))
        self.model.eval()
        batcher = test_data.batcher()
        id2entity = test_data.id2entity
        f1s, hits = [], []
        # accs, max_accs = [], []
        questions = []
        pred_answers = []
        for feed in batcher:
            _, pred, score_pred = self.model(feed)
            # acc, max_acc = cal_accuracy(pred, feed['answers'].cpu().numpy())
            # accs.append(acc)
            # max_accs.append(max_acc)
            batch_size = score_pred.size(0)
            batch_answers = feed['answers_']
            questions += feed['questions_']
            batch_candidates = feed['candidate_entities']
            pad_ent_id = len(id2entity)
            for batch_id in range(batch_size):  # a sample
                answers = batch_answers[batch_id]  # answer global id
                candidates = batch_candidates[
                    batch_id, :].tolist()  # candidates global id
                probs = score_pred[batch_id, :].tolist()
                candidate2prob = {}  # save global id entity 的probability
                for c, p in zip(candidates, probs):  # get rid of the padding
                    if c == pad_ent_id:
                        continue
                    else:
                        candidate2prob[c] = p
                f1, hit = self.f1_and_hits(answers, candidate2prob, cfg['eps'])
                best_ans = self.get_best_ans(candidate2prob)
                best_ans = id2entity.get(best_ans, '')

                pred_answers.append(best_ans)
                f1s.append(f1)
                hits.append(hit)
        print('evaluation...')
        print('how many eval samples...', len(f1s))
        # print('avg_acc', np.mean(acc))
        # print('max_acc', np.mean(max_acc))
        print('avg_f1', np.mean(f1s))
        print('avg_hits', np.mean(hits))
        return np.mean(f1s), np.mean(hits)

    @staticmethod
    def f1_and_hits(answers, candidate2prob, eps):
        retrieved = []
        correct = 0
        best_ans, max_prob = -1, 0  # save sample's prediction global id andprob
        for c, prob in candidate2prob.items():
            if prob > max_prob:
                max_prob = prob
                best_ans = c
            if prob > eps:
                retrieved.append(
                    c
                )  # as long as probability of the candidate surpass the pre-definded number
                if c in answers:
                    correct += 1  # as long as retrieves answer in candidaties
        if len(answers) == 0:
            if len(retrieved) == 0:
                return 1.0, 1.0
            else:
                return 0.0, 1.0
        else:
            hits = float(
                best_ans
                in answers)  # 1 if best answer is in answers otherwise 0
            if len(retrieved) == 0:
                return 0.0, hits
            else:
                p, r = correct / len(retrieved), correct / len(answers)
                f1 = 2.0 / (1.0 / p + 1.0 / r) if p != 0 and r != 0 else 0.0
                return f1, hits

    @staticmethod
    def get_best_ans(candidate2prob):
        best_ans, max_prob = -1, 0
        for c, prob in candidate2prob.items():
            if prob > max_prob:
                max_prob = prob
                best_ans = c
        return best_ans


if __name__ == "__main__":
    cfg = get_config()
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])

    documents = load_documents(cfg['data_folder'] +
                               cfg['{}_documents'.format(cfg['mode'])])
    train_data = DataLoaderForKV(cfg, documents)
    valid_data = DataLoaderForKV(cfg, documents, mode='dev')
    test_data = DataLoaderForKV(cfg, documents, mode='test')
    trainer = Trainer(cfg)
    trainer.train(train_data, cfg)
    trainer.test(test_data, cfg)
