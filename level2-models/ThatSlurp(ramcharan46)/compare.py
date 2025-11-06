from linear_reg import *
from perceptron import *
import pandas as pd

def resultperceptron(data):
    X=data[['x1','x2']].values
    y=data['label'].values
    w,tt=trainpercep(X,y,0.1,1000)
    start_p=time.time()
    y_p=predict(X,w)
    end_p=time.time()
    acc=np.mean(y_p==y)
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Time per prediction: {(end_p-start_p)/len(X)}s")

def resultlinearreg(data):
    X=data[['x1','x2']].values
    y=data[['label']].values
    w,b,lh,tt = trainlinreg(X,y,0.01,1000)
    start_p=time.time()
    y_pred=pred(X,w,b)
    end_p=time.time()
    y_class=(y_pred>=0.5).astype(int)
    acc=np.mean(y_class==y)
    print(f"Accuracy: {acc*100}%")
    print(f"Time per prediction: {(end_p-start_p)/len(X)}s")


#For linear dataset
#I have used an absolute path to access this file cause it wasnt working when i put the relative path ;-;
#change the path before using
data=pd.read_csv("C:/Users/ramch/OneDrive/Documents/Python/ThatSlurp(ramcharan46)/datasets/binary_classification.csv")
print("\n\nFor Linear Dataset:\n")
print("\nmetrics of linear regression model:")
resultlinearreg(data)
print("\nmetrics of perceptron model:")
resultperceptron(data)

#For non-linear dataset
data=pd.read_csv("C:/Users/ramch/OneDrive/Documents/Python/ThatSlurp(ramcharan46)/datasets/binary_classification_non_lin.csv")
print("\n\nFor Non-Linear Dataset:\n\n")
print("\nmetrics of linear regression model:")
resultlinearreg(data)
print("\nmetrics of perceptron model:")
resultperceptron(data)