#!/usr/bin/env python
# coding: utf-8

# Importing libraries

# In[1]:


import numpy as np


# Define class

# In[2]:


class LinearRegression:

    def __init__(self):
        self.parameters = {}


# Prediction 

# In[3]:


def predicting(self,train_input):
    m = self.parameters['m']
    c = self.parameters['c']
    return np.multiply(m,train_input) + c
LinearRegression.predicting = predicting


# Calculating MSE

# In[4]:


def error_calc(self,prediction,train_output):
    return np.mean((prediction - train_output)**2)
LinearRegression.error_calc = error_calc


# Calculating the gradient and storing

# In[5]:


def gradient(self,prediction,train_output,train_input):
    derivatives = {}
    diff = prediction - train_output
    derivatives['dm'] = 2 * np.mean(np.multiply(diff,train_input))
    derivatives['dc'] = 2 * np.mean(diff)
    return derivatives
LinearRegression.gradient = gradient


# Updating parameters based on gradient

# In[6]:


def update(self,learning_rate,derivatives):
    self.parameters['m'] = self.parameters['m'] - learning_rate * derivatives['dm']
    self.parameters['c'] = self.parameters['c'] - learning_rate * derivatives['dc']
LinearRegression.update = update


# Function to call to train the model

# In[7]:


def train(self,train_input,train_output,learning_rate,iters):
    self.parameters['m'] = np.random.uniform(0,1) * -1
    self.parameters['c'] = np.random.uniform(0,1) * -1

    for i in range(iters):
        prediction = self.predicting(train_input)
        cost = self.error_calc(prediction,train_output)
        derivatives = self.gradient(prediction,train_output,train_input)
        self.update(learning_rate,derivatives)

