import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from sklearn import metrics
import random
import numpy as np
import pickle

from tqdm import tqdm_notebook

PATH = Path("/data2/yinterian/multi-task-romain")

def get_data_gap(PATH, gap="60min"):
    filename = "data_train_{gap}.pickle".format(gap=gap)
    with open(PATH/filename, 'rb') as f:
        train = pickle.load(f)
    filename = "data_valid_{gap}.pickle".format(gap=gap)
    with open(PATH/filename, 'rb') as f:
        valid = pickle.load(f)
    return train, valid

def get_test_data_gap(PATH, gap="60min"):
    filename1 = "data_validation_{gap}.pickle".format(gap=gap)
    with open(PATH/filename1, 'rb') as f:
        test_ext = pickle.load(f)

    filename2 = "data_test_{gap}.pickle".format(gap=gap)
    with open(PATH/filename2, 'rb') as f:
        test = pickle.load(f)
    print(filename1, filename2)
    return test_ext, test

def get_mean_std_series(train):
    ss = np.concatenate(train.series.values)
    ss = ss.reshape(-1,5)
    return ss.mean(axis=0), ss.std(axis=0)

def get_mean_std_static(train):
    res = {}
    for name in ["age", "sapsii", "sofa"]:
        values = train[name].values
        res[name] = (values.mean(), values.std())
    res["series"] = get_mean_std_series(train)
    return res

def stats_dict(train):
    subject_id_list = np.sort(np.unique(train.subject_id.values))
    id2index = {v: k+1 for k,v in enumerate(subject_id_list)}
    num_subjects = len(subject_id_list)
    norm_dict = get_mean_std_static(train)
    care2id = {v:k for k,v in enumerate(np.unique(train.care_unit.values))}
    return norm_dict, care2id, id2index, num_subjects

# Dataset
class MultiTask(Dataset):
    def __init__(self, df, norm_dict, id2index, care2id,  k=20, train=True):
        """
        Args:
            df: dataframe with data
            norm_dict: mean and std of all variables to normalize

        """
        self.norm_dict = norm_dict
        self.df = df
        self.df["care_unit"] = self.df["care_unit"].apply(lambda x: care2id[x])
        self.names = ["age", "sapsii", "sofa"] ## needs normalization
        self.names_binary = ["gender", "amine", "sedation", "ventilation"]
        self.id2index = id2index
        self.train = train
        self.pick_a_sample(k)

    def pick_a_sample(self, k=20):
        """ Picks sample with the same number of observations per patient"""
        if not self.train: # fix seed for validation and test
            np.random.seed(3)
        sample = self.df.groupby("subject_id", group_keys=False).apply(lambda x: x.sample(k, replace=True))
        sample = sample.copy()
        if self.train:
            self.subject_index = [self.id2index[subject_id] for subject_id in sample.subject_id.values]
            self.random = np.random.choice(2, sample.shape[0], p=[0.1, 0.9])
            self.subject_index = self.subject_index*self.random
        self.df_sample = sample

    def __getitem__(self, index):
        row = self.df_sample.iloc[index,:]
        x_series = (row.series - self.norm_dict["series"][0])/self.norm_dict["series"][1]
        x_cont = [(row[name]-self.norm_dict[name][0])/self.norm_dict[name][1] for name in self.names]
        x_binary = [row[name] for name in self.names_binary]
        subject_index = 0
        if self.train:
            subject_index = self.subject_index[index]
        x_cat = np.array([row["care_unit"], subject_index])
        x_cont = np.array(x_cont + x_binary)
        return x_series, x_cont, x_cat, row["prediction_mean_HR"], row["prediction_mean_MAP"]

    def __len__(self):
        return self.df_sample.shape[0]

def save_model(m, p): torch.save(m.state_dict(), p)
    
def load_model(m, p): m.load_state_dict(torch.load(p))

def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr

def val_metrics(model, valid_dl, C):
    model.eval()
    total = 0
    sum_loss = 0
    y_hat1 = []
    ys1 = []
    y_hat2 = []
    ys2 = []
    for x_series, x_cont, x_cat, y1, y2 in valid_dl:
        batch = y1.shape[0]
        x_series = x_series.float().cuda()
        x_cont = x_cont.float().cuda()
        x_cat = x_cat.long().cuda()
        y1 = y1.float().cuda()
        y2 = y2.float().cuda()
        out1, out2 = model(x_series, x_cont, x_cat)
        mse_loss1 = F.mse_loss(out1, y1.unsqueeze(-1))
        mse_loss2 = F.mse_loss(out2, y2.unsqueeze(-1))
        sum_loss += batch*(mse_loss2.item())
        total += batch
        y_hat1.append(out1.view(-1).detach().cpu().numpy())
        ys1.append(y1.view(-1).cpu().numpy())
        y_hat2.append(out2.view(-1).detach().cpu().numpy())
        ys2.append(y2.view(-1).cpu().numpy())
    
    y_hat1 = np.concatenate(y_hat1)
    y_hat2 = np.concatenate(y_hat2)
    ys1 = np.concatenate(ys1)
    ys2 = np.concatenate(ys2)
    r2_1 = 0
    if C > 0:
        r2_1 = metrics.r2_score(ys1, y_hat1)
    r2_2 = metrics.r2_score(ys2, y_hat2)
    return sum_loss/total, r2_1, r2_2

def cosine_segment(start_lr, end_lr, iterations):
    i = np.arange(iterations)
    c_i = 1 + np.cos(i*np.pi/iterations)
    return end_lr + (start_lr - end_lr)/2 *c_i

def get_cosine_triangular_lr(max_lr, iterations, div_start=2, div_end=5):
    min_start, min_end = max_lr/div_start, max_lr/div_end
    iter1 = int(0.3*iterations)
    iter2 = iterations - iter1
    segs = [cosine_segment(min_start, max_lr, iter1), cosine_segment(max_lr, min_end, iter2)]
    return np.concatenate(segs)



def train_mini_batch(model, optimizer, x_series, x_cont, x_cat, y1, y2, C):
    out1, out2 = model(x_series, x_cont, x_cat)
    mse_loss1 = F.mse_loss(out1, y1.unsqueeze(-1))
    mse_loss2 = F.mse_loss(out2, y2.unsqueeze(-1))
    loss = C*mse_loss1 + mse_loss2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return out1, out2, mse_loss1, mse_loss2

def valid_saving(model, modelS, valid_dl, loss1, loss2, lossS, prev_val_r2, prev_val_r2S, filename):
    val_loss, val_r2_1, val_r2_2 = val_metrics(model, valid_dl, 1)
    val_lossS, _, val_r2 = val_metrics(modelS, valid_dl, 0)
    print("\t Multi Train loss: {:.3f} {:.3f} map valid loss: {:.3f} valid r2 hr {:.3f} valid r2 map {:.3f}".format(
        loss1, loss2, val_loss, val_r2_1, val_r2_2))
    print("\t Single Train loss: {:.3f} map valid loss: {:.3f} valid r2 map {:.3f}".format(
        lossS, val_lossS, val_r2))

    if val_r2_2 > prev_val_r2:
        prev_val_r2 = val_r2_2
        if val_r2_2 > 0.7:
            path = "{0}/models/{1}_r2_{2:.0f}.pth".format(PATH, filename, 100*val_r2_2)
            save_model(model, path)
            print(path)
    if val_r2 > prev_val_r2S:
        prev_val_r2S = val_r2
        if val_r2 > 0.7:
            path = "{0}/models/{1}_single_r2_{2:.0f}.pth".format(PATH, filename, 100*val_r2)
            save_model(modelS, path)
            print(path)
    return prev_val_r2, prev_val_r2S


def train_epochs(model, modelS, optimizer, optimizerS, train_ds, valid_dl, filename, prev_val_r2, prev_val_r2S,\
        max_lr=0.04, epochs = 30, C=1/5):

    idx = 0
    train_dl = DataLoader(train_ds, batch_size=5000, shuffle=True)
    iterations = epochs*len(train_dl)
    pbar = tqdm_notebook(total=iterations)
    lrs = get_cosine_triangular_lr(max_lr, iterations)
    for i in range(epochs):
        model.train()
        modelS.train()
        sum_loss1 = 0
        sum_loss2 = 0
        sum_lossS = 0
        total = 0
        train_ds.pick_a_sample()
        train_dl = DataLoader(train_ds, batch_size=5000, shuffle=True)
        for x_series, x_cont, x_cat, y1, y2 in train_dl:
            update_optimizer(optimizer, lrs[idx])
            update_optimizer(optimizerS, lrs[idx])
            x_series = x_series.float().cuda()
            x_cont = x_cont.float().cuda()
            x_cat = x_cat.long().cuda()
            y1 = y1.float().cuda()
            y2 = y2.float().cuda()
            out1, out2, mse_loss1, mse_loss2 = train_mini_batch(model, optimizer, x_series, x_cont, x_cat, y1, y2, C)
            _, out, _, mse_loss = train_mini_batch(modelS, optimizerS, x_series, x_cont, x_cat, y1, y2, 0)
            sum_loss1 += len(y1) * mse_loss1.item()
            sum_loss2 += len(y1) * mse_loss2.item()
            sum_lossS += len(y1) * mse_loss.item()
            total += len(y1)
            idx +=1
            pbar.update()
        loss1 = sum_loss1/total
        loss2 = sum_loss2/total
        lossS = sum_lossS/total
        prev_val_r2, prev_val_r2S = valid_saving(model, modelS, valid_dl, loss1, loss2, lossS, prev_val_r2, prev_val_r2S, filename) 
    return prev_val_r2, prev_val_r2S 

class EventModel(nn.Module):
    def __init__(self, num_subjects, hidden_size=100, num2=50):
        super(EventModel, self).__init__()
        self.embedding1 = nn.Embedding(5, 1)
        self.embedding2 = nn.Embedding(num_subjects+1, 5)
        
        self.gru = nn.GRU(5, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.num1 = hidden_size + 1 + 5 + 7
        self.num2 = num2
        self.linear1 = nn.Linear(self.num1, self.num2)
        self.linear2 = nn.Linear(self.num2, self.num2)
        self.out1 = nn.Linear(self.num2, 1)
        self.out2 = nn.Linear(self.num2, 1)
        self.bn1 = nn.BatchNorm1d(self.num2)
        self.bn2 = nn.BatchNorm1d(self.num2)
        
    def forward(self, x_series, x_cont, x_cat):
        _, ht = self.gru(x_series)
        x_cat_1 = self.embedding1(x_cat[:,0])
        x_cat_2 = self.embedding2(x_cat[:,1])
        x = torch.cat((ht[-1], x_cat_1, x_cat_2, x_cont), 1)
        x = self.bn1(F.relu(self.linear1(x)))
        x = self.bn2(F.relu(self.linear2(x)))
        return self.out1(x), self.out2(x)


class EventModel0(nn.Module):
    def __init__(self, num_subjects, hidden_size=100, num2=50):
        super(EventModel0, self).__init__()
        self.embedding1 = nn.Embedding(5, 1)
        self.embedding2 = nn.Embedding(num_subjects+1, 5)

        self.gru = nn.GRU(5, hidden_size, num_layers=1, batch_first=True)
        self.num1 = hidden_size + 1 + 5 + 7
        self.num2 = num2
        self.linear1 = nn.Linear(self.num1, self.num2)
        self.linear2 = nn.Linear(self.num2, self.num2)
        self.out1 = nn.Linear(self.num2, 1)
        self.out2 = nn.Linear(self.num2, 1)
        self.bn1 = nn.BatchNorm1d(self.num2)
        self.bn2 = nn.BatchNorm1d(self.num2)

    def forward(self, x_series, x_cont, x_cat):
        _, ht = self.gru(x_series)
        x_cat_1 = self.embedding1(x_cat[:,0])
        x_cat_2 = self.embedding2(x_cat[:,1])
        x = torch.cat((ht[-1], x_cat_1, x_cat_2, x_cont), 1)
        x = self.bn1(F.relu(self.linear1(x)))
        x = self.bn2(F.relu(self.linear2(x)))
        return self.out1(x), self.out2(x)

