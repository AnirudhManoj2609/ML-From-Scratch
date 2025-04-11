#!/usr/bin/env python
# coding: utf-8

# Importing libraries
# 

# In[ ]:


import numpy as np
from regression.mlr.mlr_model import MultiLinearRegression


# Define class

# In[ ]:


class PolynomialRegression:
    def __init__(self,degree):
        self.degree = degree
        self.mlr = MultiLinearRegression()


# Transforming the input

# In[ ]:


def transform_input(self,X):
    #X => 2D array with shape(n_samples,1)
    n_samples = X.shape[0]
    x_poly = np.ones((n_samples,self.degree))#x^0
    for i in range(1,self.degree + 1):
        x_poly[:,i - 1] = X[:,0] ** i
    return x_poly 
PolynomialRegression.transform_input = transform_input


# Training the model

# In[ ]:


def train(self,train_input,train_output,learning_rate,iters):
    x_poly = self.transform_input(train_input)
    self.mlr.train(x_poly,train_output,learning_rate,iters)
PolynomialRegression.train = train


# Predicting
# 

# In[ ]:


def predicting(self,train_input):
    x_poly = self.transform_input(train_input)
    return self.mlr.predicting(x_poly)
PolynomialRegression.predicting = predicting

