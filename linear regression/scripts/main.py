# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 23:35:22 2020

@author: Justi
"""

from utils import *
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Q1

# Write a simple implementation of a least-squares solution to linear regression 
# that applies an iterative update to adjust the weights. Demonstrate the success 
# of your approach on the sample data loaded below, and visualize the best fit 
# plotted as a line (consider using linspace) against a scatter plot of the x 
# and y test values.

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

diabetes_X = pd.DataFrame(data = diabetes.data, columns = diabetes.feature_names)
diabetes_Y = diabetes.target.reshape(-1,1)

# Use only one feature
diabetes_X = diabetes_X.filter({'bmi'})

# Split data into training and testing sets

x_train, x_test, y_train, y_test = traintest_split(diabetes_X, diabetes_Y, 0.8)

# %% Set Training Parameters
dim = np.shape(x_train)[1]
w,b = initialize_with_zeros(dim)
epoch = 0
lr = 0.85
epoch = 0
set_epoch = 2000

# %%
for i in range(set_epoch):
    
    check_cost = cost(w,b,x_train,y_train)
    print('cost:', check_cost, 'epoch:', epoch)
    
    w_update,b_update = weightUpdate(w,b,x_train, y_train)
    
    w = w - lr*w_update
    b = b - lr*b_update
    epoch = epoch + 1

# %%
trained_model = np.dot(x_train,w) + b
test_model = np.dot(x_test,w) + b

training_error = mean_squared_error(trained_model, y_train)
testing_error = mean_squared_error(test_model, y_test)

print('train error:', training_error)
print('test error:', testing_error)
print('Weights:', w)
print('Bias:', b)


