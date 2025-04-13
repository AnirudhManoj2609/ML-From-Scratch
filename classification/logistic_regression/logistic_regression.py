#!/usr/bin/env python
# coding: utf-8

# Import libraries

# In[1]:


import numpy as np


# Define Class

# In[ ]:


class LogisticRegression:
    def __init__(self):
        self.parameters = {}


# Sigmoid Function

# In[ ]:


def sigmoid(z):
    return 1/(1 + np.exp(-z))


# Predicting

# In[ ]:


def predicting(self,train_input):
    m = self.parameters['m']
    c = self.parameters['c']
    z = np.dot(train_input,m) + c 
    return sigmoid(z)
LogisticRegression.predicting = predicting


# Log Loss(Error calc)

# In[ ]:


def log_loss(self,prediction,train_output):
    epsilon = 1e-15
    p = np.clip(prediction,epsilon,1 - epsilon)
    loss = -np.mean(train_output * np.log(p) + (1 - train_output) * np.log(1 - p))
    return loss
LogisticRegression.log_loss = log_loss


# Calculate the gradient

# In[ ]:


def gradient(self,X,y_pred,y_true):
    derivatives = {}
    diff = y_pred - y_true
    derivatives['dm'] = np.dot(X.T,diff)/len(y_true)
    derivatives['dc'] = np.sum(diff)/len(y_true)
    return derivatives
LogisticRegression.gradient = gradient


# Update the model

# In[ ]:


def update(self,learning_rate,derivatives):
    self.parameters['m'] -= learning_rate * derivatives['dm']
    self.parameters['c'] -= learning_rate * derivatives['dc']
LogisticRegression.update = update


# Train the model

# In[ ]:


def train(self,train_input,train_output,learning_rate,iters):
    n_features = train_input.shape[1]
    self.parameters['m'] = np.random.randn(n_features)
    self.parameters['c'] = np.random.randn()

    for i in range(iters):
        prediction = self.predicting(train_input)
        loss = self.log_loss(prediction,train_output)
        derivatives = self.gradient(train_input,prediction,train_output)
        self.update(learning_rate,derivatives)
LogisticRegression.train = train

