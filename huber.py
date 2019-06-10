# Chester Holtz - chholtz@ucsd.edu
# HUBER Cov Estimator

import numpy as np
from numpy import linalg as la
from sklearn.linear_model import Lasso

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

MAXIT = 1000
TOL = 10**(-20)

def gen_5folds(X):
    X_folds = np.split(X, 5)
    S_folds = []
    for x_fold in X_folds:
        S = np.zeros((d,d))
        for iter,i in enumerate(range(d)):
            for j in range(iter,d):
                for k in range(x_fold.shape[0]):
                    S[i,j] += x_fold[k][i]*x_fold[k][j]/2
        S_folds.append(S)

    return X_folds, S_folds

def cv_tau(X, tau):
    X_folds, S_folds = gen_5folds(X)
    errs = []
    for i, fold in enumerate(tqdm(X_folds, desc='cv')):
        X = np.concatenate(X_folds[:i] + X_folds[(i + 1):])
        S_hat = huber_estimator(X, tau)
        errs.append(np.sum(np.abs(S_folds[i] - S_folds[i].shape[0]*S_hat)))
    return np.sum(errs)/X.shape[0]

def huber(X, th_0, w_0, tau):
    w = w_0
    th = th_0
    for j in range(MAXIT):
        th_prev = th
        w = np.multiply(1/tau * np.abs(X - th) - np.ones(N),np.where(np.abs(X - th)>=tau,1,0).T)
        th = np.sum(np.divide(X,w+1))/np.sum(1/(w+1))
        if np.abs(th_prev - th) < TOL:
            break
    return th

def huber_estimator(X, tau):
    w_0 = np.zeros(N).flatten()
    S_hat = np.zeros((d,d))

    for iter,i in enumerate(tqdm(range(d), desc="estimating covariance...")):
        for j in range(iter,d):
            y = []
            for k in range(N):
                y.append(Y[k][i]*Y[k][j]/2)
            th_0 = np.mean(y)
            tau = tau *np.std(y)
            s_hat = huber(y,th_0,w_0,tau) # predict cov between feature i and feature j given N samples of feature i, feature j
            S_hat[i,j] = s_hat
    return S_hat
n = 500
d = 10
NN = [500]
D = [8,16,32]
for n in NN:
    for d in D:
        N = int((n)*(n-1)/2)

        tau = 1

        X = []
        Y = []

        cov = np.eye(d)
        #cov =  np.fromfunction(lambda i, j: 0.5**(np.abs(i-j)), (d,d))
        mean = np.zeros(d)
        X = np.random.multivariate_normal(mean,cov,n)
        S = np.zeros((d,d))
        for i in tqdm(range(n), desc='const. Y'):
            for j in range(i+1,n):
                Y.append((X[i] - X[j]).flatten())

        for i in tqdm(range(n), desc='const. Y'):
            for j in range(i+1,n):
                Y.append((X[i] - X[j]).flatten())

        #C = [0.01,0.1,0.2,0.5,1,2,5,10]
        c = 200
        omega = np.sqrt(n/(2*np.log(n*d)))
        #for c in tqdm(C,desc="CV on c..."):
        tau = c * omega
        cverr = cv_tau(np.array(Y), tau)
        #tqdm.write(str((c, cverr)))
        tqdm.write(str((n,d, cverr)))
