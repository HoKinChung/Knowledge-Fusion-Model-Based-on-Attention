import torch
import numpy as np
import random

from data_generator import DataLoader
from model import KAReader
from util import get_config, cal_accuracy, load_documents

from tensorboardX import SummaryWriter


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
        hits = float(best_ans
                     in answers)  # 1 if best answer is in answers otherwise 0
        if len(retrieved) == 0:
            return 0.0, hits
        else:
            p, r = correct / len(retrieved), correct / len(answers)
            f1 = 2.0 / (1.0 / p + 1.0 / r) if p != 0 and r != 0 else 0.0
            return f1, hits


def get_best_ans(candidate2prob):
    best_ans, max_prob = -1, 0
    for c, prob in candidate2prob.items():
        if prob > max_prob:
            max_prob = prob
            best_ans = c
    return best_ans


def L1_regularization(model):
    """
    :param model: nn.Modules
    :return:
    """
    re_l1 = 0
    for para in model.parameters():
        re_l1 = re_l1 + torch.abs(para).sum()
    return re_l1


def L2_regularization(model):
    """
    :param model: nn.Modules
    :return:
    """
    re_l2 = 0
    for para in model.parameters():
        re_l2 = re_l2 + torch.sqrt(torch.pow(para, 2).sum())
    return re_l2


def train(cfg, alpha_l1=1e-4, alpha_l2=1e-4):
    tf_logger = SummaryWriter('tf_logs/' + cfg['model_id'])

    # train and test share the same set of documents
    # tokenize the document
    documents = load_documents(cfg['data_folder'] +
                               cfg['{}_documents'.format(cfg['mode'])])
    # train data
    train_data = DataLoader(cfg, documents)
    valid_data = DataLoader(cfg, documents, mode='dev')

    model = KAReader(cfg)
    if cfg['use_cuda']:
        model = model.to(torch.device('cuda'))

    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch.optim.Adam(trainable, lr=cfg['learning_rate'])

    if cfg['lr_schedule']:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [30, 50],
                                                         gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.5, last_epoch=-1)

    model.train()
    best_val_f1 = 0
    best_val_hits = 0
    for epoch in range(cfg['num_epoch']):
        print("Epoch {}".format(epoch))
        batcher = train_data.batcher(shuffle=True)
        train_loss, train_acc, train_max_acc = [], [], []
        for feed in batcher:
            loss, pred, pred_dist = model(feed)
            ################ L1 L2 regularization ################
            # l1_regularization = L1_regularization(model)
            # l2_regularization = L2_regularization(model)
            # loss = loss + alpha_l2 * l2_regularization
            ################ L1 L2 regularization ################
            train_loss.append(loss.item())
            acc, max_acc = cal_accuracy(pred, feed['answers'].cpu().numpy())
            train_acc.append(acc)
            train_max_acc.append(max_acc)
            optim.zero_grad()
            loss.backward()
            if cfg['gradient_clip'] != 0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg['gradient_clip'])
            optim.step()
        print("Epoch {}: batch average training loss {}, batch average training acc {}".format(epoch, np.mean(train_loss),\
                                                                                               np.mean(train_acc)))
        tf_logger.add_scalar('avg_batch_loss', np.mean(train_loss), epoch)

        val_f1, val_hits = test(model, valid_data, cfg['eps'])
        if cfg['lr_schedule']:
            scheduler.step()
        tf_logger.add_scalar('eval_f1', val_f1, epoch)
        tf_logger.add_scalar('eval_hits', val_hits, epoch)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
        if val_hits > best_val_hits:
            best_val_hits = val_hits
            torch.save(
                model.state_dict(),
                'model/{}/{}_best.pt'.format(cfg['name'], cfg['model_id']))
        print('best model saved in model/{}/{}_best.pt'.format(
            cfg['name'], cfg['model_id']))
        print('evaluation best f1:{} current:{}'.format(best_val_f1, val_f1))
        print('evaluation best hits:{} current:{}'.format(
            best_val_hits, val_hits))

    model_save_path = 'model/{}/{}_final.pt'.format(cfg['name'],
                                                    cfg['model_id'])
    print('save final model in {}'.format(model_save_path))
    torch.save(model.state_dict(), model_save_path)

    # model_save_path = 'model/{}/{}_best.pt'.format(cfg['name'], cfg['model_id'])
    # model.load_state_dict(torch.load(model_save_path))

    print('\n..........Finished training, start testing.......')

    test_data = DataLoader(cfg, documents, mode='test')
    model.eval()
    print('finished training, testing final model...')
    test(model, test_data, cfg['eps'])


#     print('testing best model...')
#     model_save_path = 'model/{}/{}_best.pt'.format(cfg['name'], cfg['model_id'])
#     model.load_state_dict(torch.load(model_save_path))
#     model.eval()
#     test(model, test_data, cfg['eps'])


def test(model, test_data, eps):
    model.eval()
    batcher = test_data.batcher()
    id2entity = test_data.id2entity
    f1s, hits = [], []
    accs, max_accs = [], []
    questions = []
    pred_answers = []
    for feed in batcher:
        _, pred, pred_dist = model(
            feed
        )  # pred (bsz): index of predicted entity ||| pred_dist (bsz, max_local_entities)
        acc, max_acc = cal_accuracy(pred,
                                    feed['answers'].cpu().numpy())  # max_acc
        accs.append(acc)
        max_accs.append(max_acc)
        batch_size = pred_dist.size(0)
        batch_answers = feed['answers_']
        questions += feed['questions_']
        batch_candidates = feed['candidate_entities']
        pad_ent_id = len(id2entity)
        for batch_id in range(batch_size):  # a sample
            answers = batch_answers[batch_id]  # answer global id
            candidates = batch_candidates[
                batch_id, :].tolist()  # candidates global id
            probs = pred_dist[batch_id, :].tolist()
            candidate2prob = {}  # save global id entity probability
            for c, p in zip(candidates, probs):  # remove the padding
                if c == pad_ent_id:
                    continue
                else:
                    candidate2prob[c] = p
            f1, hit = f1_and_hits(answers, candidate2prob, eps)
            best_ans = get_best_ans(candidate2prob)
            best_ans = id2entity.get(best_ans, '')

            pred_answers.append(best_ans)
            f1s.append(f1)
            hits.append(hit)
    print('evaluation.......')
    print('how many eval samples......', len(f1s))
    print('avg_acc', np.mean(acc))
    print('max_acc', np.mean(max_acc))
    print('avg_f1', np.mean(f1s))
    print('avg_hits', np.mean(hits))

    model.train()
    return np.mean(f1s), np.mean(hits)


if __name__ == "__main__":
    # config_file = sys.argv[2]
    cfg = get_config()
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    if cfg['mode'] == 'train':
        train(cfg)
    elif cfg['mode'] == 'test':
        documents = load_documents(cfg['data_folder'] +
                                   cfg['{}_documents'.format(cfg['mode'])])
        test_data = DataLoader(cfg, documents, mode='test')
        model = KAReader(cfg)
        if cfg['use_cuda']:
            model = model.to(torch.device('cuda'))
        model_save_path = 'model/{}/{}_final.pt'.format(
            cfg['name'], cfg['model_id'])
        print("Load model from {}".format(model_save_path))
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        test(model, test_data, cfg['eps'])
    else:
        assert False, "--train or --test?"
