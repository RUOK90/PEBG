import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from tqdm import tqdm
import numpy as np

from args import *
from PEBG import *
from dataset import *

# load dataset
with open(ARGS.train_dataset_path, 'rb') as f_r:
    question_tag_data_dic = pickle.load(f_r)
print('Done loading dataset')

# get dataset
questions = question_tag_data_dic['questions']
tags = question_tag_data_dic['tags']
question_attr_dic = question_tag_data_dic['question_attr_dic']
QT = question_tag_data_dic['QT']
QQ = question_tag_data_dic['QQ']
TT = question_tag_data_dic['TT']

question_to_emb_idx = {question: idx for idx, question in enumerate(questions, start=1)}
tag_to_emb_idx = {tag: idx for idx, tag in enumerate(tags, start=1)}

QT_qs, QT_ts, QT_labels = zip(*QT)
QT_q_idxs = [question_to_emb_idx[q] for q in QT_qs]
QT_t_idxs = [tag_to_emb_idx[t] for t in QT_ts]
QT_labels = list(QT_labels)
print('Done processing QT')

QQ_q1s, QQ_q2s, QQ_labels = zip(*QQ)
QQ_q1_idxs = [question_to_emb_idx[q] for q in QQ_q1s]
QQ_q2_idxs = [question_to_emb_idx[q] for q in QQ_q2s]
QQ_labels = list(QQ_labels)
print('Done processing QQ')

TT_t1s, TT_t2s, TT_labels = zip(*TT)
TT_t1_idxs = [tag_to_emb_idx[t] for t in TT_t1s]
TT_t2_idxs = [tag_to_emb_idx[t] for t in TT_t2s]
TT_labels = list(TT_labels)
print('Done processing TT')

diff_q_idxs = []
diff_t_idxs = []
diff_as = []
diff_labels = []
for question, attr in question_attr_dic.items():
    diff_q_idxs.append(question_to_emb_idx[question])
    diff_ts = [tag_to_emb_idx[tag] for tag in attr['tags']]
    diff_t_idxs.append(diff_ts + [Constants.PAD_IDX] * (Constants.MAX_NUM_TAGS_PER_ITEM - len(diff_ts)))
    diff_as.append(attr['avg_elapsed_time'])
    diff_labels.append(attr['difficulty'])
print('Done processing diff')

QT_loader = data.DataLoader(dataset=QTDataset(QT_q_idxs, QT_t_idxs, QT_labels), shuffle=True, batch_size=int(len(QT_labels)/ARGS.n_batches), num_workers=ARGS.num_workers)
QQ_loader = data.DataLoader(dataset=QQDataset(QQ_q1_idxs, QQ_q2_idxs, QQ_labels), shuffle=True, batch_size=int(len(QQ_labels)/ARGS.n_batches), num_workers=ARGS.num_workers)
TT_loader = data.DataLoader(dataset=TTDataset(TT_t1_idxs, TT_t2_idxs, TT_labels), shuffle=True, batch_size=int(len(TT_labels)/ARGS.n_batches), num_workers=ARGS.num_workers)
diff_loader = data.DataLoader(dataset=DiffDataset(diff_q_idxs, diff_t_idxs, diff_as, diff_labels), shuffle=True, batch_size=int(len(diff_labels)/ARGS.n_batches), num_workers=ARGS.num_workers)
print('Done setting dataloader')


# define model
model = PEBG(questions, tags, question_attr_dic, question_to_emb_idx, tag_to_emb_idx).to(ARGS.device)
bcelogit_loss = nn.BCEWithLogitsLoss(reduction='mean')
mse_loss = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=ARGS.lr)


# train
for epoch in range(1, ARGS.n_epochs+1):
    total_losses = []
    QT_losses = []
    QQ_losses = []
    TT_losses = []
    diff_losses = []
    for QT_batch, QQ_batch, TT_batch, diff_batch in zip(QT_loader, QQ_loader, TT_loader, diff_loader):
        QT_batch = {k: t.to(ARGS.device) for k, t in QT_batch.items()}
        QQ_batch = {k: t.to(ARGS.device) for k, t in QQ_batch.items()}
        TT_batch = {k: t.to(ARGS.device) for k, t in TT_batch.items()}
        diff_batch = {k: t.to(ARGS.device) for k, t in diff_batch.items()}

        QT_logit, QQ_logit, TT_logit, diff_hat, question_emb = model(QT_batch['QT_q'], QT_batch['QT_t'], QQ_batch['QQ_q1'], QQ_batch['QQ_q2'], TT_batch['TT_t1'], TT_batch['TT_t2'], diff_batch['diff_q'], diff_batch['diff_t'], diff_batch['diff_a'])
        QT_loss = bcelogit_loss(QT_logit, QT_batch['QT_label'])
        QQ_loss = bcelogit_loss(QQ_logit, QQ_batch['QQ_label'])
        TT_loss = bcelogit_loss(TT_logit, TT_batch['TT_label'])
        diff_loss = mse_loss(diff_hat, diff_batch['diff_label'])

        total_loss = QT_loss + QQ_loss + TT_loss + diff_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_losses.append(total_loss.item())
        QT_losses.append(QT_loss.item())
        QQ_losses.append(QQ_loss.item())
        TT_losses.append(TT_loss.item())
        diff_losses.append(diff_loss.item())

    total_epoch_loss = np.mean(total_losses)
    QT_epoch_loss = np.mean(QT_losses)
    QQ_epoch_loss = np.mean(QQ_losses)
    TT_epoch_loss = np.mean(TT_losses)
    diff_epoch_loss = np.mean(diff_losses)

    print(f'epoch: {epoch}, total_loss: {total_epoch_loss:.4f},'
          f'QT_loss: {QT_epoch_loss:.4f}, QQ_loss: {QQ_epoch_loss.item():.4f}, TT_loss: {TT_epoch_loss.item():.4f}, diff_loss: {diff_epoch_loss.item():.4f}')

    if ARGS.use_wandb:
        wandb.log({
            'QT loss': QT_epoch_loss,
            'QQ loss': QQ_epoch_loss,
            'TT loss': TT_epoch_loss,
            'Diff_loss': diff_epoch_loss,
            'Total loss': total_epoch_loss
        }, step=epoch)

    if epoch in [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]:
        model_weights = model.state_dict()
        torch.save(model_weights, f'{ARGS.weight_path}/{ARGS.name}_{epoch}.pt')