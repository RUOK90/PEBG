import numpy as np
import torch
from torch.utils import data


class QTDataset(data.Dataset):
    def __init__(self, QT_q, QT_t, QT_label):
        self.QT_q = np.asarray(QT_q, dtype=np.long)
        self.QT_t = np.asarray(QT_t, dtype=np.long)
        self.QT_label = np.asarray(QT_label, dtype=np.float32)

    def __len__(self):
        return len(self.QT_label)

    def __getitem__(self, item):
        return {'QT_q': self.QT_q[item], 'QT_t': self.QT_t[item], 'QT_label': self.QT_label[item]}


class QQDataset(data.Dataset):
    def __init__(self, QQ_q1, QQ_q2, QQ_label):
        self.QQ_q1 = np.asarray(QQ_q1, dtype=np.long)
        self.QQ_q2 = np.asarray(QQ_q2, dtype=np.long)
        self.QQ_label = np.asarray(QQ_label, dtype=np.float32)

    def __len__(self):
        return len(self.QQ_label)

    def __getitem__(self, item):
        return {'QQ_q1': self.QQ_q1[item], 'QQ_q2': self.QQ_q2[item], 'QQ_label': self.QQ_label[item]}


class TTDataset(data.Dataset):
    def __init__(self, TT_t1, TT_t2, TT_label):
        self.TT_t1 = np.asarray(TT_t1, dtype=np.long)
        self.TT_t2 = np.asarray(TT_t2, dtype=np.long)
        self.TT_label = np.asarray(TT_label, dtype=np.float32)

    def __len__(self):
        return len(self.TT_label)

    def __getitem__(self, item):
        return {'TT_t1': self.TT_t1[item], 'TT_t2': self.TT_t2[item], 'TT_label': self.TT_label[item]}


class DiffDataset(data.Dataset):
    def __init__(self, diff_q, diff_t, diff_a, diff_label):
        self.diff_q = np.asarray(diff_q, dtype=np.long)
        self.diff_t = np.asarray(diff_t, dtype=np.long)
        self.diff_a = np.asarray(diff_a, dtype=np.float32)
        self.diff_label = np.asarray(diff_label, dtype=np.float32)

    def __len__(self):
        return len(self.diff_label)

    def __getitem__(self, item):
        return {'diff_q': self.diff_q[item], 'diff_t': self.diff_t[item], 'diff_a': self.diff_a[item], 'diff_label': self.diff_label[item]}

