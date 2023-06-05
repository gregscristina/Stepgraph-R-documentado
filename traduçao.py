import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def cv_part(n, k):
    
    ntest = n // k  
    ntrain = n - ntest 
    ind = np.random.permutation(n) + 1  
    trainMat = np.empty((ntrain, k))  
    testMat = np.empty((ntest, k))  
    nn = np.arange(1, n + 1)  
    for j in range(k):
        sel = np.arange((j * ntest), ((j + 1) * ntest))  
        testMat[:, j] = ind[sel]  
        sel2 = nn[~np.isin(nn, sel)]  
        trainMat[:, j] = ind[sel2] 
    return {"trainMat": trainMat, "testMat": testMat}

def cv_stepGraph(x,fold,alpha_f_min,alpha_f_max,n_alpha,nei_max):
    n = x.shape[0]
    p = x.shape[1]
    part_list = cv_part(n, fold)
    alpha_seq = np.linspace(alpha_f_min, alpha_f_max, num=n_alpha)  
    alpha_f = np.repeat(alpha_seq, 2)
    alpha_b = np.concatenate((0.5 * alpha_seq, alpha_seq))
    alpha_grid = pd.DataFrame({'f': alpha_f, 'b': alpha_b})
    loss_re = np.zeros((alpha_grid.shape[0], fold))
    for k in range(1, fold+1):
        x_train = x[part_list['trainMat'][:, k-1], :]
        varepsilon_list = [stepGraph(x_train, alpha_grid.loc[i, 'f'], alpha_grid.loc[i, 'b'], nei_max) for i in range(alpha_grid.shape[0])]
        x_test = scale(x[part_list['testMat'][:, k-1], :])
        for i in range(alpha_grid.shape[0]):
            if len(varepsilon_list[i]) == 1:
                loss_re[i, k-1] = np.nan
            else:
                beta = varepsilon_list[i][1]
                loss_re[i, k-1] = np.sum(np.sum((x_test - x_test @ beta) ** 2))