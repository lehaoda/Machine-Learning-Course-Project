#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:31:47 2020

@author: haodale
"""

# The true function
def f_true(x):
    y = 6.0 * (np.sin(x + 2) + np.sin(2 * x + 4))
    return y

# Generate a synthetic data set, with Gaussian noise
import numpy as np

n = 750
X = np.random.uniform(-7.5, 7.5, n)
e = np.random.normal(0.0, 5.0, n)
y = f_true(X) + e

# plot raw data with true function
import matplotlib.pyplot as plt
plt.figure()

# plot the data
plt.scatter(X, y, 12, marker = 'o')

# plot the true function, which is really unkonwn
x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)
plt.plot(x_true, y_true, marker = 'None', color = 'r')

# scikit-learn has many tools and utilities for model selection
from sklearn.model_selection import train_test_split
tst_frac = 0.3 # Fraction of examples to sample for the test set 
val_frac = 0.1 # Fraction of examples to sample for the validation set

# First, we use train_test_split to partition (X, y) into training and test sets
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)

# Next, we use train_test_split to further partition (X_trn, y_trn) into train ing and validation sets
X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)

# Plot the three subsets
plt.figure()
plt.scatter(X_trn, y_trn, 12, marker='o', color='orange') 
plt.scatter(X_val, y_val, 12, marker='o', color='green') 
plt.scatter(X_tst, y_tst, 12, marker='o', color='blue')


#-----------------------------------------------------------------------
# 1. **Regression with Polynomial Basis Functions**, 30 points.
#-----------------------------------------------------------------------

# 1.a.
# X float(n, ): univariate data 
# d int: degree of polynomial 
def polynomial_transform(X, d):
    Phi = np.asfarray([[None for y in range(d + 1)] for x in range(len(X))])
    for i in range(0, len(X)):
        for j in range(0, d + 1):
            Phi[i][j] = X[i] ** j
    return Phi



# 1.b.   
# Phi float(n, d): transformed data 
# y float(n, ): labels
def train_model(Phi, y):
    return np.linalg.inv(Phi.transpose().dot(Phi)).dot(Phi.transpose()).dot(y)



# 1.c.    
# Phi float(n, d): transformed data
# y float(n, ): labels
# w float(d, ): linear regression model
def evaluate_model(Phi, y, w):
    n = len(y)
    sum = 0
    for i in range(0, n):
        sum += (y[i] - w.transpose().dot(Phi[i])) ** 2
    res = sum / n
    return res
    
    
    
# 1.d.
w = {}               # Dictionary to store all the trained models
validationErr = {}   # Validation error of the models
testErr = {}         # Test error of all the models

for d in range(3, 25, 3):  # Iterate over polynomial degree
    Phi_trn = polynomial_transform(X_trn, d)                 # Transform training data into d dimensions
    w[d] = train_model(Phi_trn, y_trn)                       # Learn model on training data
    
    Phi_val = polynomial_transform(X_val, d)                 # Transform validation data into d dimensions
    validationErr[d] = evaluate_model(Phi_val, y_val, w[d])  # Evaluate model on validation data
    
    Phi_tst = polynomial_transform(X_tst, d)           # Transform test data into d dimensions
    testErr[d] = evaluate_model(Phi_tst, y_tst, w[d])  # Evaluate model on test data

# Plot all the models
plt.figure()
plt.plot(list(validationErr.keys()), list(validationErr.values()), marker='o', linewidth=3, markersize=12)
plt.plot(list(testErr.keys()), list(testErr.values()), marker='s', linewidth=3, markersize=12)
plt.xlabel('Polynomial degree', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([2, 25, 15, 70])


plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for d in range(9, 25, 3):
  X_d = polynomial_transform(x_true, d)
  y_d = X_d @ w[d]
  plt.plot(x_true, y_d, marker='None', linewidth=2)

plt.legend(['true'] + list(range(9, 25, 3)))
plt.axis([-8, 8, -15, 15])    
    
    
    
    
#-----------------------------------------------------------------------
# 2.**Regression with Radial Basis Functions**, 70 points
#-----------------------------------------------------------------------   

# 2.a.
# X float(n, ): univariate data
# B float(n, ): basis functions
# gamma float : standard deviation / scaling of radial basis kernel
def radial_basis_transform(X, B, gamma=0.1):
    Phi = np.asfarray([[None for y in range(len(B))] for x in range(len(X))])
    for i in range(0, len(X)):
        for j in range(0, len(B)):
            Phi[i][j] = np.e ** (-gamma * (X[i] - B[j]) ** 2)
    return Phi



# 2.b.
# Phi float(n, d): transformed data
# y float(n, ): labels
# lam float : regularization parameter
def train_ridge_model(Phi, y, lam):
    Phi = np.asfarray(Phi)
    m = len(Phi[0])
    I = np.identity(m)
    return np.linalg.inv(Phi.transpose().dot(Phi) + lam * I).dot(Phi.transpose()).dot(y)



# 2.c.
lams = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

w = {}               # Dictionary to store all the trained models
validationErr = {}   # Validation error of the models
testErr = {}         # Test error of all the models

for lam in lams:  # Iterate over polynomial degreel
    Phi_trn = radial_basis_transform(X_trn, X_trn)                # Transform training data into d dimensions
    w[lam] = train_ridge_model(Phi_trn, y_trn, lam)                   # Learn model on training data
    
    Phi_val = radial_basis_transform(X_val, X_trn)               # Transform validation data into d dimensions
    validationErr[lam] = evaluate_model(Phi_val, y_val, w[lam])  # Evaluate model on validation data
    
    Phi_tst = radial_basis_transform(X_tst, X_trn)          # Transform test data into d dimensions
    testErr[lam] = evaluate_model(Phi_tst, y_tst, w[lam])  # Evaluate model on test data

# Plot all the models
plt.figure()
plt.plot(list(validationErr.keys()), list(validationErr.values()), marker='o', linewidth=3, markersize=12)
plt.plot(list(testErr.keys()), list(testErr.values()), marker='s', linewidth=3, markersize=12)
plt.xlabel('lambda', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([-60, 1000, 25, 70])
#plt.axis([0, 0.1, 25, 70])

   

# 2.d.
plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for lam in lams:
    X_lam = radial_basis_transform(x_true, X_trn, 0.1)
    y_lam = X_lam @ w[lam]
    plt.plot(x_true, y_lam, marker='None', linewidth=2)

plt.legend(['true'] + lams)
plt.axis([-8, 8, -15, 15])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    