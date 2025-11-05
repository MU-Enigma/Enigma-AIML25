import numpy as np
import time
import pandas as pd

def predict(X,w):
    return np.where(np.dot(X,w)>=0,1,0)

def trainpercep(X,y,lr,epochs):
    w=np.zeros(X.shape[1])
    start=time.time()
    for epoch in range(epochs):
        errors=0
        for i in range(len(X)):
            y_p=1 if np.dot(X[i],w)>=0 else 0
            update=lr*(y[i]-y_p)
            if update!=0:
                w+=update*X[i]
                errors+=1
        if errors==0:
            break
    end=time.time()
    print(f"Time of convergence in {epoch+1} epochs, time: {end-start}s")
    return w,(end-start)

