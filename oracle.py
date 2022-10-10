#import main
import pandas as pd
import os
import cvxpy as cp
import numpy as np
import math
from tqdm import tqdm,trange
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
"""
Solve the oracel primal problem to optimality using cvxpy and mosek, and print the optimal function value.
"""
def sigmoid(x):
    return 1/(1+np.exp(-x))

def oracel(lambd,args):

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
    K = args.topk
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

        x = cp.Variable((T, item_num))
        rho_reverse = 1/(rho*T*K)

        ##in order to acclerate the algorithm, we relaxed the intergal and make an expectation with the sample strategy
        objective = cp.Maximize(cp.sum(cp.multiply(batch_UI,x))/T + \
                                lambd * cp.min(cp.multiply(cp.sum(cp.matmul(x,A), axis=0),rho_reverse)) )
        constraints = [x>=0, x<=1, cp.sum(x, axis=1) == K, cp.sum(cp.matmul(x,A), axis=0) <= T*rho*K]

        prob = cp.Problem(objective, constraints)
        result_ori = prob.solve()

        ndcg = 0
        x_value = x.value
        result = 0
        base_model_provider_exposure = np.zeros(num_providers)
        for t in range(T):
            dcg = 0
            #make a sample strategy
            x_recommended = np.random.choice(list(range(0,item_num)),size=K,replace=False,p=x_value[t,:]/K)
            vec = batch_UI[t]
            sorted_index = np.argsort(vec[x_recommended])[::-1]

            for k in range(K):
                k_index = sorted_index[k]
                base_model_provider_exposure[iid2pid[x_recommended[k_index]]] += 1
                dcg = dcg + batch_UI[t,x_recommended[k_index]]/np.log2(k+2)
                result = result + batch_UI[t,x_recommended[k_index]]

            ndcg = ndcg + dcg/nor_dcg[t]

        ndcg = ndcg/T

        MMF = np.min(base_model_provider_exposure*rho_reverse)
        W = result_ori

        W_batch.append(W)
        RRQ_batch.append(ndcg)
        MMF_batch.append(MMF)

    W, RRQ, MMF = np.mean(W_batch), np.mean(RRQ_batch), np.mean(MMF_batch)
    print("W:%.4f RRQ: %.4f MMF: %.4f"%(W, RRQ, MMF))
    return W, RRQ, MMF


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="simulator")
    parser.add_argument('--base_model', default='bpr')
    parser.add_argument('--Dataset', nargs='?', default='yelp',
                        help='your data.')
    parser.add_argument('--Time', type=int, default=256,
                        help='fair time sep.')
    parser.add_argument('--topk', type=int, default=10)


    args = parser.parse_args()
    oracel(lambd=1e-1,args=args)