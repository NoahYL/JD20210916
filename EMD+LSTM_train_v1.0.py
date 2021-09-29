# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 09:28:58 2021

@author: Zhao
"""

import numpy as np
from numpy.lib.function_base import iterable
import pandas as pd
import torch 
import torch.nn as nn
from PyEMD import CEEMDAN
from sampen import sampen2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import time

#%%

class EMD_SE(CEEMDAN,StandardScaler):
    def __init__(self,X,trials=10,n_clusters=3,max_imf=-1,mm=2):
        super(EMD_SE, self).__init__()
        self.with_mean = True
        self.with_std = True
        self.copy = True
        self.X = np.array(X)
        self.trials = trials
        self.max_imf = max_imf
        self.mm = mm
        self.n_clusters = n_clusters
        self.X_scaler = self.std()        
    def std(self):
        self.fit(self.X.reshape(-1,1))
        return self.transform(self.X.reshape(-1,1))
    def _c_emd(self,x=None):
        if x is None:
            self.C_IMFs = self.ceemdan(self.X_scaler.reshape(1,-1)[0],max_imf=self.max_imf)
        else:
            return self.ceemdan(x)
    def _se(self,x=None):
        sample_entropy = {}
        se_series = []
        if x is None:
            data = self.C_IMFs[:-1]
        else:
            data = x
        for num,value in enumerate(data):
            sampen_of_series = sampen2(value.tolist(),mm=self.mm, r=0.2*self.scale_)
            sample_entropy[num] = sampen_of_series.copy()
            se_series.append(sampen_of_series[self.mm][1])
        if x is None:
            self.sample_entropy = sample_entropy
            self.se_series = np.array(se_series).reshape(-1,1)
        else:
            return sample_entropy,np.array(se_series).reshape(-1,1)
 
    def k_mean_train(self):
        start = time.time()
        if not hasattr(self,'se_series'):
            self._c_emd()
            self._se()
        assert self.C_IMFs.shape[0] >= self.n_clusters,'聚类类别数目大于信号分解数目!!!'
        self.KMeans_model = KMeans(n_clusters=self.n_clusters-1)
        self.KMeans_model.fit(self.se_series)
        se_label = self.KMeans_model.fit_predict(self.se_series)
        self.Imf_clusters = []
        for num in range(self.n_clusters-1):
            position = np.where(se_label==num)[0]
            self.Imf_clusters.append(np.sum(self.C_IMFs[position,:],axis=0))
        self.Imf_clusters.append(self.C_IMFs[-1].tolist())
        end = time.time()
        print('EMD分解 样本熵 KMeans共耗时 %.2f秒!'%(end-start))
        
class DatasetIterater(object):
    def __init__(self, dataset, batch_size, step,device='cpu'):
        self.batch_size = batch_size
        self.dataset = torch.tensor(dataset,dtype=torch.float32, device=device)
        self.n_batches =  self.dataset.shape[0]// batch_size
        self.step = step
        self.epoch_size = self.n_batches - step - 1
        self.dim = list(self.dataset.shape).__len__()
        self.indices = self._to_tensor()
        self.index = 0
        self.device = device
    def _to_tensor(self):
        if self.dim == 2:
            corpus_indices= self.dataset[0:self.batch_size*self.n_batches,:]
            return corpus_indices.view(self.batch_size,self.n_batches,-1)
        elif self.dim == 1:
            corpus_indices= self.dataset[0:self.batch_size*self.n_batches]
            return corpus_indices.view(self.batch_size,self.n_batches)
    def __next__(self):
        if self.index >= self.epoch_size:
            self.index = 0
            raise StopIteration
        else:
            if self.dim == 2:   
                batches_x = self.indices[:,self.index:self.index+self.step,:]
                batches_y = self.indices[:,self.index+self.step:self.index+self.step+1,:]
            elif self.dim == 1:
                batches_x = self.indices[:,self.index:self.index+self.step]
                batches_y = self.indices[:,self.index+self.step:self.index+self.step+1]
                print(batches_x)
                batches_x,batches_y = batches_x.view(self.batch_size,-1,1),batches_y.view(self.batch_size,-1,1)
            self.index += 1

            # 将维度做一个变换
            # batches_x = batches_x.transpose(1,0,2)
            # batches_y = batches_y.transpose(1,0,2)
            return batches_x,batches_y

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches 
    
def grad_clipping(params, threshold,device):
    norm = torch.tensor([0.0],device=device)
    for param in params:
        norm += (param.grad.data**2).sum()
    norm = norm.sqrt().item()
    if norm > threshold:
        for param in params:
            param.grad.data *= (threshold/norm)

class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,bidirectional=True,batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size *2
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size,bidirectional=bidirectional) #bidirectional选择单双向      
        self.fc = nn.Linear(self.hidden_size,input_size)
    def forward(self,x):
        ula, (h_out, _) = self.lstm(x)
        output = self.fc(h_out)
        return output
def training(model,data,epochs,loss,optimizer,device,threshold):
    model.to(device)
    # state = None
    time_cost = 0
    for epoch in range(epochs):
        loss_sum, n = 0,0
        start = time.time()
        for x,y in data:
            # if state is not None:
            #     if isinstance(state,tuple):
            #         state = (state[0].detach(),state[1].detach())
            #     else:
            #         state = state.detach()
            output = model(x)
            optimizer.zero_grad()
            l = loss(output,y)
            l.backward()
            grad_clipping(model.parameters(),threshold=threshold,device=device)
            optimizer.step()
            temp_tensor = torch.tensor(y.shape[:-1],dtype=torch.int32)
            num_ = torch.dot(temp_tensor,torch.ones(temp_tensor.__len__(),dtype=torch.int32))
            loss_sum += l.item()* num_
            n += num_
        end_time = time.time()
        time_cost += end_time - start
        if (epoch+1)%20 == 0 or epoch==0:
            print('Epoch %d: avg_loss %.4f, time %.2f'%(epoch+1,loss_sum/n,time_cost))
            time_cost = 0               

def predicting():
    pass

def get_rf_features(model,X_0,state=None):
    with torch.no_grad():
        rf_features = []
        y_std = []
        for x,y in X_0:
            out = model(x)
            for i in range(X_0.batch_size):
                for value in out[i].tolist():
                    rf_features.append(value)
                for yi in y[i].sum(dim=-1):
                    y_std.append(yi.item())
        rf_features = torch.tensor(rf_features).numpy()
        y_std = np.array(y_std).reshape(-1,1)
    return rf_features,y_std
#数据读取    
# f_path = r'E:\Python Script\随便写写\data\全国碳交易数据.xlsx'
f_path=r'全国碳交易数据.xlsx'
tables = pd.ExcelFile(f_path)
df = tables.parse('天津')
df.sort_values(by='交易日期', inplace=True,ascending=True)
price_list = df['成交价/元'].values   

#信号分解&信号聚类
ES =  EMD_SE(price_list,trials=10,n_clusters=3,max_imf=-1,mm=2)  
ES.k_mean_train()
#lstm 模块训练
X = torch.tensor(ES.Imf_clusters).T

# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(20,5))
# # plt.plot(price_list,c='black')
# plt.plot(ES.Imf_clusters[0],c='r')
# plt.plot(ES.Imf_clusters[1],c='b')
# plt.plot(ES.Imf_clusters[2],c='y')

X_0 =  DatasetIterater(X,batch_size=2,step=15,device='cpu')
x,y = X_0.__next__()
model = LSTM(ES.n_clusters,256,bidirectional=True)

model.forward(x)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
training(model,X_0,40,loss_func,optimizer,'cpu',0.005)
torch.save(model,'save.pt')
# torch.save(model.state_dict(),r'E:\Python Script\随便写写\data\EMD_LSTM_dict.pt')
model = torch.load('save.pt')
out = model(X_0.__next__()[0])

#rf模块特征获取&模型训练
rf_features,y_std = get_rf_features(model.lstm, X_0)
estimator = RandomForestRegressor(random_state=7, n_estimators=100,max_depth=4, min_samples_leaf=0.05)
estimator.fit(rf_features,y_std)
y_predict = estimator.predict(rf_features)
error = np.power(y_std.reshape(1,-1) - y_predict,2).mean()
print("rf整体方差:%.2f"%(error))

# %%
