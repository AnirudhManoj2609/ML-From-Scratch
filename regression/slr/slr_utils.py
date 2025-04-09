import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_csv(path):
    df = pd.read_csv(path)
    X = df.iloc[:,0].values #All rows from 1st column as input
    Y = df.iloc[:,1].values #All rows from 2nd column as output
    return X,Y

def plot_regression(X,Y,Y_pred,save_path=None):
    
    plt.figure(figsize=(8,5))
    plt.scatter(X,Y,color="blue",label="Actual")
    plt.plot(Y_pred,color="red",label="Predicted Line")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.title("Simple Linear Regression")

    if(save_path):
        plt.savefig(save_path)
    else:
        plt.show()
