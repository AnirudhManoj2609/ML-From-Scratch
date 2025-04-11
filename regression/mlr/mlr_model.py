#!/usr/bin/env python
# coding: utf-8

# Importing libraries

# In[8]:


import numpy as np


# Define class
# 

# In[9]:


class MultiLinearRegression:
    def __init__(self):
        self.parameters = {}


# Prediction

# In[10]:


def predicting(self,train_input):
    m = self.parameters['m']
    c = self.parameters['c']
    return np.dot(train_input,m)
#using dot allows for vector multiplication which is needed here in case of MLR
MultiLinearRegression.predicting = predicting


# Training the model

# In[11]:


def error_calc(self,prediction,train_output):
    return np.mean((prediction-train_output)**2)
MultiLinearRegression.error_calc = error_calc


# Calculating the Gradient
# 

# In[12]:


def gradient(self,prediction,train_input,train_output):
    derivatives = {}
    diff = prediction - train_output
    derivatives['dm'] = 2 * np.dot(train_input.T,diff)/len(train_input)
    derivatives['dc'] = 2 * np.mean(diff)
    return derivatives 
MultiLinearRegression.gradient = gradient


# Update parameters based on gradient

# In[13]:


def update(self,learning_rate,derivatives):
    self.parameters['m'] = self.parameters['m'] - learning_rate * derivatives['dm']
    self.parameters['c'] = self.parameters['c'] - learning_rate * derivatives['dc']
MultiLinearRegression.update = update


# In[ ]:


def train(self,train_input,train_output,learning_rate,iters):
    n_features = train_input.shape[1]
    self.parameters['m'] = np.random.randn(n_features)
    self.parameters['c'] = np.random.randn()

    for i in range(iters):
        prediction = self.predicting(train_input)
        cost = self.error_calc(prediction,train_output)
        derivatives = self.gradient(prediction,train_input,train_output)
        self.update(learning_rate,derivatives)
MultiLinearRegression.train = train

