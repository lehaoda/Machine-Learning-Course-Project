#!/usr/bin/env python
# coding: utf-8

# ------------------------------------------------------------------------------
# # 1. **Support Vector Machines with Synthetic Data** 50 points. 
# ------------------------------------------------------------------------------

# DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH DATA GENERATION, 
# MAKE A COPY OF THIS FUNCTION AND THEN EDIT
#
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN


def generate_data(n_samples, tst_frac=0.2, val_frac=0.2):
  # Generate a non-linear data set
  X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=42)
   
  # Take a small subset of the data and make it VERY noisy; that is, generate outliers
  m = 30
  np.random.seed(30)  # Deliberately use a different seed
  ind = np.random.permutation(n_samples)[:m]
  X[ind, :] += np.random.multivariate_normal([0, 0], np.eye(2), (m, ))
  y[ind] = 1 - y[ind]

  # Plot this data
  cmap = ListedColormap(['#b30065', '#178000'])  
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')       
  
  # First, we use train_test_split to partition (X, y) into training and test sets
  X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, 
                                                random_state=42)

  # Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
  X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, 
                                                random_state=42)
  
  return (X_trn, y_trn), (X_val, y_val), (X_tst, y_tst)


#
#  DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH VISUALIZATION, 
#  MAKE A COPY OF THIS FUNCTION AND THEN EDIT 
#

def visualize(models, param, X, y):
  # Initialize plotting
  if len(models) % 3 == 0:
    nrows = len(models) // 3
  else:
    nrows = len(models) // 3 + 1
    
  fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5.0 * nrows))
  cmap = ListedColormap(['#b30065', '#178000'])

  # Create a mesh
  xMin, xMax = X[:, 0].min() - 1, X[:, 0].max() + 1
  yMin, yMax = X[:, 1].min() - 1, X[:, 1].max() + 1
  xMesh, yMesh = np.meshgrid(np.arange(xMin, xMax, 0.01), 
                             np.arange(yMin, yMax, 0.01))

  for i, (p, clf) in enumerate(models.items()):
    # if i > 0:
    #   break
    r, c = np.divmod(i, 3)
    ax = axes[r, c]

    # Plot contours
    zMesh = clf.decision_function(np.c_[xMesh.ravel(), yMesh.ravel()])
    zMesh = zMesh.reshape(xMesh.shape)
    ax.contourf(xMesh, yMesh, zMesh, cmap=plt.cm.PiYG, alpha=0.6)

    if (param == 'C' and p > 0.0) or (param == 'gamma'):
      ax.contour(xMesh, yMesh, zMesh, colors='k', levels=[-1, 0, 1], 
                 alpha=0.5, linestyles=['--', '-', '--'])

    # Plot data
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')       
    ax.set_title('{0} = {1}'.format(param, p))


# Generate the data
n_samples = 300    # Total size of data set 
(X_trn, y_trn), (X_val, y_val), (X_tst, y_tst) = generate_data(n_samples)




# ------------------------------------------------------------------------------
# 1.a
# (25 points) The effect of the regularization parameter, C
# ------------------------------------------------------------------------------

# Learn support vector classifiers with a radial-basis function kernel with 
# fixed gamma = 1 / (n_features * X.std()) and different values of C
C_range = np.arange(-3.0, 6.0, 1.0)
C_values = np.power(10.0, C_range)

models = dict()
trnErr = dict()
valErr = dict()

for C in C_values:
    # Insert your code here to learn SVM models
    clf = SVC(C = C, gamma = 'scale')
    clf.fit(X_trn, y_trn)
    trnErr.update({C: 1 - clf.score(X_trn, y_trn)})
    valErr.update({C: 1 - clf.score(X_val, y_val)})
    models.update({C: clf})

visualize(models, 'C', X_trn, y_trn)

# Insert your code here to perform model selection
lists = sorted(trnErr.items())
x, y = zip(*lists)
plt.figure(3)
plt.plot(x, y, label = "training error")

lists = sorted(valErr.items())
x, y = zip(*lists)
plt.figure(3)
plt.plot(x, y, label = "validation error")

plt.xlabel('C') 
plt.ylabel('Error') 

plt.legend() 
plt.xscale("log", basex = 10)   
plt.show() 

# print C best test accuracy
print("validation error for C:")
print(valErr)
print("C_best: 1.0")
clf = SVC(C = 1.0,  gamma = 'scale')
clf.fit(X_trn, y_trn)
print("accuracy:", clf.score(X_tst, y_tst))




# ------------------------------------------------------------------------------
# 1.b
# (25 points) The effect of the RBF kernel parameter
# ------------------------------------------------------------------------------

# Learn support vector classifiers with a radial-basis function kernel with 
# fixed C = 10.0 and different values of gamma
gamma_range = np.arange(-2.0, 4.0, 1.0)
gamma_values = np.power(10.0, gamma_range)

models = dict()
trnErr = dict()
valErr = dict()

for G in gamma_values:
    # Insert your code here to learn SVM models
    clf = SVC(C = 10.0, gamma = G)
    clf.fit(X_trn, y_trn)
    trnErr.update({G: 1 - clf.score(X_trn, y_trn)})
    valErr.update({G: 1 - clf.score(X_val, y_val)})
    models.update({G: clf})
  
visualize(models, 'gamma', X_trn, y_trn)

# Insert your code here to perform model selection
lists = sorted(trnErr.items())
x, y = zip(*lists)
plt.figure(4)
plt.plot(x, y, label = "training error")

lists = sorted(valErr.items())
x, y = zip(*lists)
plt.figure(4)
plt.plot(x, y, label = "validation error")

plt.xlabel('G') 
plt.ylabel('Error') 

plt.legend() 
plt.xscale("log", basex = 10)   
plt.show() 

# print C best test accuracy
print("validation error for gamma:")
print(valErr)
print("Gamma_best: 1.0")
clf = SVC(C = 10.0,  gamma = 1.0)
clf.fit(X_trn, y_trn)
print("accuracy:", clf.score(X_tst, y_tst))




# ------------------------------------------------------------------------------
# # 2. **Breast Cancer Diagnosis with Support Vector Machines**, 25 points. 
# ------------------------------------------------------------------------------

# Load the Breast Cancer Diagnosis data set; download the files from eLearning
# CSV files can be read easily using np.loadtxt()
#
# Insert your code here.
X_trn = np.loadtxt('wdbc_trn.csv', delimiter = ',', usecols = range(1,31, 1))
y_trn = np.loadtxt('wdbc_trn.csv', delimiter = ',', usecols = 0)
X_val = np.loadtxt('wdbc_val.csv', delimiter = ',', usecols = range(1,31, 1))
y_val = np.loadtxt('wdbc_val.csv', delimiter = ',', usecols = 0)
X_tst = np.loadtxt('wdbc_tst.csv', delimiter = ',', usecols = range(1,31, 1))
y_tst = np.loadtxt('wdbc_tst.csv', delimiter = ',', usecols = 0)


# Insert your code here to perform model selection
C_range = np.arange(-2.0, 5.0, 1.0)
C_values = np.power(10.0, C_range)
Gamma_range = np.arange(-3.0, 3.0, 1.0)
Gamma_values = np.power(10.0, Gamma_range)

trnErr = []
valErr = []

for C in C_values:
    trnErrC = []
    valErrC = []
    for G in Gamma_values:
        clf = SVC(C = C, gamma = G)
        clf.fit(X_trn, y_trn)
        trnErrC.append(round(1 - clf.score(X_trn, y_trn), 3))
        valErrC.append(round(1 - clf.score(X_val, y_val), 3))
    trnErr.append(trnErrC)
    valErr.append(valErrC)
    
    
# print table
fig, axes = plt.subplots(2, 1)

axes[0].axis('tight')
axes[0].axis('off')
axes[1].axis('tight')
axes[1].axis('off')

axes[0].table(cellText = trnErr, rowLabels = C_values, colLabels = Gamma_values, loc='center')
axes[1].table(cellText = valErr, rowLabels = C_values, colLabels = Gamma_values, loc='center')

axes[0].set_title(label = 'trnErr')
axes[1].set_title(label = 'valErr')
plt.show()
    
# best C and gamma, print accuracy
clf = SVC(C = 10000.0,  gamma = 0.001)
clf.fit(X_trn, y_trn)
print("C_best: 1000.0")
print("Gamma_best: 0.001")
print("accuracy:", clf.score(X_tst, y_tst))




# ------------------------------------------------------------------------------
# # 3. **Breast Cancer Diagnosis with $k$-Nearest Neighbors**, 25 points. 
# ------------------------------------------------------------------------------

# Insert your code here to perform model selection
K_values = [1, 5, 11, 15, 21]
models = dict()
trnErr = dict()
valErr = dict()

for K in K_values:
    clf = KNN(n_neighbors = K)
    clf.fit(X_trn, y_trn)
    trnErr.update({K: 1 - clf.score(X_trn, y_trn)})
    valErr.update({K: 1 - clf.score(X_val, y_val)})
    models.update({K: clf})

lists = sorted(trnErr.items())
x, y = zip(*lists)
plt.figure(5)
plt.plot(x, y, label = "training error")

lists = sorted(valErr.items())
x, y = zip(*lists)
plt.figure(5)
plt.plot(x, y, label = "validation error")

plt.xlabel('K') 
plt.ylabel('Error') 

plt.legend()  
plt.show() 
    
# best K, print accuracy
print(valErr)    
clf = KNN(n_neighbors = 5)
clf.fit(X_trn, y_trn)
print("K_best: 5")
print("accuracy:", clf.score(X_tst, y_tst))
    
    
    
    

