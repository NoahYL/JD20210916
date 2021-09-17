# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 22:21:22 2021

@author: Zhao
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from PyEMD import CEEMDAN
from sampen import sampen2

#%%

'''数据读入,以天津为例'''
# f_path = r'E:\Python Script\随便写写\data\全国碳交易数据.xlsx'
f_path=r'全国碳交易数据.xlsx'  
tables = pd.ExcelFile(f_path)
df = tables.parse('天津')
df.sort_values(by='交易日期', inplace=True,ascending=True)
price_list = df['成交价/元'].values
CEE = CEEMDAN()
CEE.trials = 10 #噪声序列的轮数
interval = np.linspace(0,10,price_list.size) #为画图提供x轴
C_IMFs = CEE.ceemdan(price_list,interval,max_imf=-1)
# C_IMFs.shape
fig = plt.figure(figsize=(14,8))
plt.subplot(C_IMFs.shape[0]+2,1, 1)
plt.plot(interval, price_list, 'r')
plt.xlim((0, 10))
plt.title("Original signal")
plt.subplot(C_IMFs.shape[0]+2,1, 2)
plt.plot(interval, CEE.residue, 'r')
plt.xlim((0, 10))
plt.title("Residue")
for num,value in enumerate(C_IMFs):
    plt.subplot(C_IMFs.shape[0]+2,1, num+3)
    plt.plot(interval, value,'g')
    plt.xlim((0, 10))
    plt.title("Imf "+str(num+1))

plt.show()


'''
计算每个IMF的样本熵,最后一个是确定性趋势,需要剔除
mm控制窗口大小,r是阈值,可随意调节,mm一般取2-3，r一般取0.2*std
'''
price_std = np.std(price_list)

sample_entropy = {}
for num,value in enumerate(C_IMFs[:-1]):
    sampen_of_series = sampen2(value.tolist(),mm=2, r=0.2*price_std)
    sample_entropy[num] = sampen_of_series.copy()

'''
使用mm=2时每个序列的SE均值为参考，寻找聚类方法
'''
se_series = []
for key in sample_entropy.keys():
    se_series.append(sample_entropy[key][2][1])

se_xlabels = []
for i in range(1,10):
    se_xlabels.append('IMF%s' % i)

fig = plt.figure(figsize=(10,8))
plt.plot(se_xlabels,se_series,'r',label='IMF',marker='o')
plt.plot(se_xlabels,[np.mean(se_series),]*9,linestyle='--',label='IMF_avg')
plt.title("IMFs")
plt.legend(loc='best')

'''
分为3组聚类，加总
IMF1-5，IMF6-7，IMF8-9
'''

imf_s1 = C_IMFs[0]+C_IMFs[1]+C_IMFs[2]+C_IMFs[3]+C_IMFs[4]
imf_s2 = C_IMFs[5]+C_IMFs[6]
imf_s3 = C_IMFs[7]+C_IMFs[8]
CC_IMFs = np.array([imf_s1,imf_s2,imf_s3])

fig = plt.figure(figsize=(14,8))
for num,value in enumerate(CC_IMFs):
    plt.subplot(CC_IMFs.shape[0],1, num+1)
    plt.plot(interval, value,'g')
    plt.title("Imf "+str(num+1))

#%%



