#import main
import pandas as pd
import os
import cvxpy as cp
import numpy as np
import math
from tqdm import tqdm,trange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from cvxpylayers.torch import CvxpyLayer
import torch.nn as nn
import argparse
"""
Solve the dual problem to optimality using cvxpy and mosek, and print the optimal function value.
We provide GPU-version and CPU-version
"""


def sigmoid(x):
    return 1/(1+np.exp(-x))





def Min_Regularizer(lambd,args):
    T = args.Time
    trained_preference_scores = np.load(os.path.join("tmp",args.base_model + "_" + args.Dataset + "_simulation.npy"))

    datas = pd.read_csv(os.path.join("dataset",args.Dataset,args.Dataset+".simulation.inter"),delimiter='\t')
    uid_field,iid_field,label_field,time_field,provider_field = datas.columns

    num_providers = len(datas[provider_field].unique())
    user_num, item_num = np.shape(trained_preference_scores)
    providerLen = np.array(datas.groupby(provider_field).size().values)
    rho = (1+1/num_providers)*providerLen/np.sum(providerLen)

    datas.sort_values(by=[time_field], ascending=True,inplace=True)
    batch_size = int(len(datas)* 0.1//T)

    data_val = np.array(datas[uid_field].values[-batch_size*T:]).astype(np.int)
    UI_matrix = trained_preference_scores[data_val]

    #normalize user-item perference score to [0,1]
    UI_matrix = sigmoid(UI_matrix)
    tmp = datas[[iid_field,provider_field]].drop_duplicates()
    item2provider = {x:y for x,y in zip(tmp[iid_field],tmp[provider_field])}

    #A is item-provider matrix
    A = np.zeros((item_num,num_providers))
    iid2pid = []
    for i in range(item_num):
        iid2pid.append(item2provider[i])
        A[i,item2provider[i]] = 1
    W_batch = []
    RRQ_batch, MMF_batch = [], []

    K = args.topk

    for b in trange(batch_size):
        min_index = b * T
        max_index = (b+1) * T
        batch_UI = UI_matrix[min_index:max_index,:]
        nor_dcg = []
        UI_matrix_sort = np.sort(batch_UI,axis=-1)
        for i in range(T):
            nor_dcg.append(0)
            for k in range(K):
                nor_dcg[i] = nor_dcg[i] + UI_matrix_sort[i,item_num-k-1]/np.log2(k+2)

        mu_t = np.zeros(num_providers)
        B_t = T*K*rho
        #print(np.float(B_t>0))
        sum_dual = 0
        result_x = []
        eta = args.eta / np.sqrt(T)
        gradient_cusum = np.zeros(num_providers)
        gradient_list = []
        for t in range(T):
            
            x_title = batch_UI[t,:] - lambd * np.matmul(A,(-B_t + np.min(B_t))/(T * rho))
            mask = np.matmul(A,(B_t>0).astype(np.float))

            mask = (1.0-mask) * -10000.0
            x = np.argsort(x_title+mask,axis=-1)[::-1]
            x_allocation = x[:K]
            re_allocation = np.argsort(batch_UI[t,x_allocation])[::-1]
            x_allocation = x_allocation[re_allocation]
            result_x.append(x_allocation)
            B_t = B_t - np.sum(A[x_allocation],axis=0,keepdims=False)

        ndcg = 0

        base_model_provider_exposure = np.zeros(num_providers)
        result = 0
        for t in range(T):
            dcg = 0
            x_recommended = result_x[t]
            #x_recommended = np.random.choice(list(range(0,item_num)),size=K,replace=False,p=x_value[t,:]/K)
            for k in range(K):
                base_model_provider_exposure[iid2pid[x_recommended[k]]] += 1
                dcg = dcg + batch_UI[t,x_recommended[k]]/np.log2(k+2)
                result = result + batch_UI[t,x_recommended[k]]

            ndcg = ndcg + dcg/nor_dcg[t]
        ndcg = ndcg/T
        rho_reverse = 1/(rho*T*K)
        MMF = np.min(base_model_provider_exposure*rho_reverse)
        W = result/T + lambd * MMF

        W_batch.append(W)
        RRQ_batch.append(ndcg)
        MMF_batch.append(MMF)

    W, RRQ, MMF = np.mean(W_batch), np.mean(RRQ_batch), np.mean(MMF_batch)
    print("W:%.4f RRQ: %.4f MMF: %.4f "%(W,RRQ ,MMF))
    return W, RRQ, MMF




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simulator")
    parser.add_argument('--base_model', default='bpr')
    parser.add_argument('--Dataset', nargs='?', default='yelp',
                        help='your data.')
    parser.add_argument('--Time', type=int, default=256,
                        help='fair time sep.')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--eta', type=float, default=1e-3)


    args = parser.parse_args()

    Min_Regularizer(lambd=1e-1,args=args)
