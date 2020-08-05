import torch
from torch.utils import data


class QTDataset(data.Dataset):
    def __init__(self, QT_q, QT_t, QT_label):
        self.QT_q = torch.Tensor(QT_q).long()
        self.QT_t = torch.Tensor(QT_t).long()
        self.QT_label = torch.Tensor(QT_label).float()

    def __len__(self):
        return len(self.QT_label)

    def __getitem__(self, item):
        return {'QT_q': self.QT_q[item], 'QT_t': self.QT_t[item], 'QT_label': self.QT_label[item]}


class QQDataset(data.Dataset):
    def __init__(self, QQ_q1, QQ_q2, QQ_label):
        self.QQ_q1 = torch.Tensor(QQ_q1).long()
        self.QQ_q2 = torch.Tensor(QQ_q2).long()
        self.QQ_label = torch.Tensor(QQ_label).float()

    def __len__(self):
        return len(self.QQ_label)

    def __getitem__(self, item):
        return {'QQ_q1': self.QQ_q1[item], 'QQ_q2': self.QQ_q2[item], 'QQ_label': self.QQ_label[item]}


class TTDataset(data.Dataset):
    def __init__(self, TT_t1, TT_t2, TT_label):
        self.TT_t1 = torch.Tensor(TT_t1).long()
        self.TT_t2 = torch.Tensor(TT_t2).long()
        self.TT_label = torch.Tensor(TT_label).float()

    def __len__(self):
        return len(self.TT_label)

    def __getitem__(self, item):
        return {'TT_t1': self.TT_t1[item], 'TT_t2': self.TT_t2[item], 'TT_label': self.TT_label[item]}


class DiffDataset(data.Dataset):
    def __init__(self, diff_q, diff_t, diff_a, diff_label):
        self.diff_q = torch.Tensor(diff_q).long()
        self.diff_t = torch.Tensor(diff_t).long()
        self.diff_a = torch.Tensor(diff_a).float()
        self.diff_label = torch.Tensor(diff_label).float()

    def __len__(self):
        return len(self.diff_label)

    def __getitem__(self, item):
        return {'diff_q': self.diff_q[item], 'diff_t': self.diff_t[item], 'diff_a': self.diff_a[item], 'diff_label': self.diff_label[item]}

