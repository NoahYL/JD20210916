# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 09:28:58 2021

@author: Zhao，Noah
"""

'''
需要安装如下的包，其中PyEDM安装的方法为：
pip install EMD-signal
剩余都可以直接使用pip install 包名称 来安装
'''

import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from PyEMD import CEEMDAN
from sampen import sampen2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import time,os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

#%%
'''
1. 定义EMD_SE类，用来计算IMFs
2. 接着使用kmeans将算出来的IMFs进行分组，生成C_IMFs
3. C_IMFs将分别被放入LSTM进行训练
'''
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
    
'''
生成数据的Iterater，可以用for循环去取下一对数据
注：这里让多条C_IMFs组合（这里是3条），放到LSTM训练，这和分成多条LSTM效果一致
可以把这个想象成一个单词，单词长度是3，这样我们每次虽然输入1个单词，但是同时训练的是他的三个维度
'''
class DatasetIterater(object):
    def __init__(self, dataset, batch_size, step,device='cpu'):
        self.batch_size = batch_size
        self.dataset = torch.tensor(dataset,dtype=torch.float32, device=device)
        self.n_batches =  self.dataset.shape[0]// batch_size
        self.step = step
        self.epoch_size = (self.n_batches-1)// step
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
                batches_x = self.indices[:,self.index * self.step:(self.index + 1) * self.step,:]
                batches_y = self.indices[:,(self.index * self.step+1):((self.index + 1) * self.step + 1),:]
            elif self.dim == 1:
                batches_x = self.indices[:,self.index * self.step: (self.index + 1) * self.step]
                batches_y = self.indices[:,(self.index * self.step+1):((self.index + 1) * self.step + 1)]
                batches_x,batches_y = batches_x.view(self.batch_size,-1,1),batches_y.view(self.batch_size,-1,1)
            self.index += 1
            # 让每个y变成一维
            batches_y = batches_y.sum(axis=2).view(batches_y.shape[0],batches_y.shape[1],-1)
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

'''
LSTM 模型架构
'''
class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,bidirectional=bidirectional) #bidirectional选择单双向      
        self.hidden_size = self.lstm.hidden_size * (2 if self.lstm.bidirectional else 1)
        self.fc = nn.Linear(self.hidden_size,1)
    def forward(self,x,state):
        y, self.state = self.lstm(x,state)
        output = self.fc(y)
        return output,self.state

'''
train 训练过程
这里获取了最后一个state，用于后续的预测过程
'''
def training(model,data,epochs,loss,optimizer,device,threshold):
    model.to(device)
    # summary.add_graph(model,(input_summary,state_summary))
    state = None
    time_cost = 0
    for epoch in range(epochs):
        loss_sum, n = 0,0
        start = time.time()
        for x,y in data:
            if state is not None:
                if isinstance(state,tuple):
                    state = (state[0].detach(),state[1].detach())
                else:
                    state = state.detach()
            output,state = model(x,state)
            optimizer.zero_grad()
            l = loss(output,y)
            l.backward()
            grad_clipping(model.parameters(),threshold=threshold,device=device)
            optimizer.step()
            temp_tensor = torch.tensor(y.shape[:-1],dtype=torch.int32)
            num_ = torch.dot(temp_tensor,torch.ones(temp_tensor.__len__(),dtype=torch.int32))
            loss_sum += l.item()* num_
            n += num_
        # summary.add_scalar('train_loss',loss_sum/n,global_step=epoch)
        # for v1,v2 in model.lstm.named_parameters():
        #     summary.add_histogram(v1,v2,global_step=epoch)
        # for v1,v2 in model.fc.named_parameters():
        #     summary.add_histogram(v1,v2,global_step=epoch) 
        end_time = time.time()
        time_cost += end_time - start
        if (epoch+1)%20 == 0 or epoch==0:
            print('Epoch %d: avg_loss %.4f, time %.2f'%(epoch+1,loss_sum/n,time_cost))
            time_cost = 0  
    return state

'''
原本希望使用的特征选择器，用于随机森林预测
'''         
def get_rf_features(model,X_0,state=None):
    with torch.no_grad():
        rf_features = []
        y_std = []
        for x,y in X_0:
            out,state = model(x,state)
            for i in range(X_0.batch_size):
                for value in out[i].tolist():
                    rf_features.append(value)
                for yi in y[i].sum(dim=-1):
                    y_std.append(yi.item())
        rf_features = torch.tensor(rf_features).numpy()
        y_std = np.array(y_std).reshape(-1,1)
    return rf_features,y_std

'''
切分训练集和测试集
'''
def train_test_spliting(X,r):
    test_X = X[int(X.shape[0]*r):,:]
    train_X = X[:int(X.shape[0]*r),:]
    return train_X,test_X

'''
预测方法，需要使用到训练过程中最后一步的状态
'''

def predicting(model,datasets,state_f):
    real = []
    pred = []
    first_dim = 2 if model.lstm.bidirectional else 1
    with torch.no_grad(): 
        h_n = state_f[0][:,-1,:].clone().view(first_dim,1,model.hidden_size//first_dim).contiguous()
        c_n = state_f[1][:,-1,:].clone().view(first_dim,1,model.hidden_size//first_dim).contiguous()
        
    # h_n = torch.from_numpy(np.array([state_f[0][0][-1].cpu().detach().numpy(),state_f[0][1][-1].cpu().detach().numpy()]))
    # h_n = h_n.view(2,1,256).cuda()
    # c_n = torch.from_numpy(np.array([state_f[1][0][-1].cpu().detach().numpy(),state_f[1][1][-1].cpu().detach().numpy()]))
    # c_n = c_n.view(2,1,256).cuda()

        for x,y in datasets:
            output,state = model(x[0][0].view(1,1,-1),(h_n,c_n))
            h_n,c_n = state
            real.append(y.item())
            pred.append(output.item())
    return real,pred

'''
上面的方法全部运行即可
'''

'''
主程序，可以选择计算相应城市的结果，默认为天津。也可以选择训练集，测试集的比率，默认为0.8.
因此，在这里，剩余0.2的数据用来作图，展示测试集和训练集的数据结果

1. 模型参数是固定的，是我们训练出来效果还不错的，如果需要修改可以在main中修改，也可以写成接口放在参数里
2. 还用到了一些tensorboard的东西，用来随时查看参数，如果不需要或者没有安装对应包的可以注释掉（SummaryWriter）
3. 训练完的模型文件保存为model.pt 可以反复调用

'''
def main(city='天津',ratio=0.8,dev='cpu'):
    #数据读取   
    print('开始读取.....')
    f_path=r'全国碳交易数据.xlsx'  
    tables = pd.ExcelFile(f_path)
    df = tables.parse(city)
    df.sort_values(by='交易日期', inplace=True,ascending=True)
    price_list = df['成交价/元'].values
    print('数据读取完毕!')
    #信号分解&信号聚类
    print('开始分解序列.....')
    ES =  EMD_SE(price_list,trials=10,n_clusters=3,max_imf=-1,mm=2)  
    ES.k_mean_train()
    #lstm 模块训练
    print('开始训练数据.....')
    X = torch.tensor(ES.Imf_clusters).T

    train_X,test_X = train_test_spliting(X,ratio)

    X_0 =  DatasetIterater(train_X,batch_size=2,step=15,device=dev)
    X_1 = DatasetIterater(test_X,batch_size=1,step=1,device=dev)

    model = LSTM(ES.n_clusters,256,bidirectional=True)
    loss_func = nn.MSELoss()
    # writer_path = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/') + '/tensorboardX/summary1'
    # ES_writer = SummaryWriter(log_dir=writer_path)
    # input_summary = torch.rand([2,15,3]).to('cuda')
    # state_summary = (torch.rand([2, 15, 256]).to('cuda'),torch.rand([2, 15, 256]).to('cuda'))
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
    state_f = training(model,X_0,200,loss_func,optimizer,dev,0.005)  
    # torch.save(model.state_dict(),r'E:\Python Script\随便写写\data\EMD_LSTM_dict.pt')  
    torch.save(model,r'model.pt')

    '''
    预测，数据还原
    '''
    print('开始预测数据.....')
    real,pred = predicting(model,X_1,state_f)
    real_f = real*ES.scale_+ES.mean_
    pred_f = pred*ES.scale_+ES.mean_
    fig = plt.figure(figsize=(20,8))
    plt.plot(real_f,c='r',label='real')
    plt.plot(pred_f,c='b',label='pred')
    plt.legend(loc='best')

main()

'''
随机森林效果并不好，因此我们这里仅需要使用上面的结果
'''
#rf模块特征获取&模型训练
# rf_features,y_std = get_rf_features(model.lstm, X_0)
# estimator = RandomForestRegressor(random_state=7, n_estimators=100,max_depth=4, min_samples_leaf=0.05)
# estimator.fit(rf_features,y_std)
# y_predict = estimator.predict(rf_features)
# error = np.power(y_std.reshape(1,-1) - y_predict,2).mean()
# print("rf整体方差:%.2f"%(error))


