import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from args import *
from PEBG import PEBG

# load dataset
with open(ARGS.train_dataset_path, 'rb') as f_r:
    question_tag_data_dic = pickle.load(f_r)
print('Done loading dataset')

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

QT_q_idxs_tensor = torch.Tensor(QT_q_idxs).long().to(ARGS.device)
QT_t_idxs_tensor = torch.Tensor(QT_t_idxs).long().to(ARGS.device)
QT_labels_tensor = torch.Tensor(QT_labels).float().to(ARGS.device)

QQ_q1_idxs_tensor = torch.Tensor(QQ_q1_idxs).long().to(ARGS.device)
QQ_q2_idxs_tensor = torch.Tensor(QQ_q2_idxs).long().to(ARGS.device)
QQ_labels_tensor = torch.Tensor(QQ_labels).float().to(ARGS.device)

TT_t1_idxs_tensor = torch.Tensor(TT_t1_idxs).long().to(ARGS.device)
TT_t2_idxs_tensor = torch.Tensor(TT_t2_idxs).long().to(ARGS.device)
TT_labels_tensor = torch.Tensor(TT_labels).float().to(ARGS.device)

diff_q_idxs_tensor = torch.Tensor(diff_q_idxs).long().to(ARGS.device)
diff_t_idxs_tensor = torch.Tensor(diff_t_idxs).long().to(ARGS.device)
diff_as_tensor = torch.Tensor(diff_as).float().to(ARGS.device)
diff_labels_tensor = torch.Tensor(diff_labels).float().to(ARGS.device)
print('Done tensorizing')

# define model
model = PEBG(questions, tags, question_attr_dic, question_to_emb_idx, tag_to_emb_idx).to(ARGS.device)
bcelogit_loss = nn.BCEWithLogitsLoss(reduction='mean')
mse_loss = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=ARGS.lr)

for epoch in tqdm(range(ARGS.n_epochs)):
    QT_logit, QQ_logit, TT_logit, diff_hat, question_emb = model(QT_q_idxs_tensor, QT_t_idxs_tensor, QQ_q1_idxs_tensor, QQ_q2_idxs_tensor, TT_t1_idxs_tensor, TT_t2_idxs_tensor, diff_q_idxs_tensor, diff_t_idxs_tensor, diff_as_tensor)
    QT_loss = bcelogit_loss(QT_logit, QT_labels_tensor)
    QQ_loss = bcelogit_loss(QQ_logit, QQ_labels_tensor)
    TT_loss = bcelogit_loss(TT_logit, TT_labels_tensor)
    diff_loss = mse_loss(diff_hat, diff_labels_tensor)

    total_loss = QT_loss + QQ_loss + TT_loss + diff_loss

    # print(f'epoch: {epoch}, loss: {total_loss.item()}')
    if ARGS.use_wandb:
        wandb.log({
            'QT loss': QT_loss.item(),
            'QQ loss': QQ_loss.item(),
            'TT loss': TT_loss.item(),
            'Diff_loss': diff_loss.item(),
            'Total loss': total_loss.item()
        }, step=epoch)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # weight 저장
    # 우선 weight 저장 없이 쭉 돌려보기
