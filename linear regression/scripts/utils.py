# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 17:38:05 2020

@author: Justi
"""

# %%
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def traintest_split(x,y, train_per):
    
    """This function splits the x and y data into training and testings sets
    
    :param list x, y
    :returns x_train, x_test, y_train, y_test
    
    """
    
    x_shuffle = x.sample(frac = 1)
    msk = np.random.rand(len(x)) < train_per
    
    x_train = x_shuffle[msk]
    y_train = y[msk]
    x_test = x_shuffle[~msk]
    y_test = y[~msk]
    
    return np.asarray(x_train), np.asarray(x_test), np.asarray(y_train), np.asarray(y_test)

def weightUpdate(w,b,x,t):
    
    """
    This function computes the gradients
    
    Argument:
    w -- weight vector
    b -- bias term
    x -- feature matrix
    t -- label vector
    
    
    Returns:
    gradients for weights and bias
    """
    
    pred = np.dot(x,w) + b
    
    gradWeights = (1/len(x)) * np.sum(np.multiply((pred - t),x))
    gradBias = (1/len(x))*np.sum(pred - t)
    
    return gradWeights, gradBias

def cost(w,b,x,t):
    """
    This function computes the cost function
    
    Argument:
    w -- weight vector
    b -- bias term
    x -- feature matrix
    t -- label vector
    
    
    Returns:
    cost
    """
    
    cost = (1/2*len(x))*np.sum(np.square(np.dot(x,w) + b - t))
    
    return cost


def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
#    w = np.zeros(shape=(dim, 1))
    w = np.random.rand(dim,1)
    b = 0
    ### END CODE HERE ###

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b


