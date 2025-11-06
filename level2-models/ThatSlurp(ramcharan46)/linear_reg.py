import numpy as np
import time
import pandas as pd

def sigmoid(x):
    return 1/(1+np.exp(-x))

def pred(X,w,b):
    z=np.dot(X,w)+b
    return sigmoid(z)

def mseloss(X,y,w,b):
    m =X.shape[0]
    h=pred(X,w,b)
    loss=(1/(2*m))*np.sum((h-y)**2)
    return np.squeeze(loss)

def calcgradient(X,y,w,b):
    m=X.shape[0]
    p=pred(X,w,b)
    error=p-y
    sd=p*(1-p)
    cgt=error*sd
    dw=(1/m)*np.dot(X.T,cgt)
    db=(1/m)*np.sum(cgt)
    return dw,db

def trainlinreg(X,y,lr,epoches):
    n_features=X.shape[1]
    w=np.zeros((n_features,1))
    b=0.0
    loss_history=[]
    start=time.time()
    for i in range(epoches):
        dw,db=calcgradient(X,y,w,b)
        w=w-lr*dw
        b=b-lr*db
        if i%100==0:
            loss_history.append(mseloss(X,y,w,b))
            print(f"Epoch {i}: Loss={loss_history[-1]}")
    end=time.time()
    print(f"Time for convergence: {end-start:.4f}s")
    return w,b,loss_history,(end-start)

