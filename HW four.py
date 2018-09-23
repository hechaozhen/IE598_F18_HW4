# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 08:59:27 2018

@author: hecha
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
import scipy as sp
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import urllib.request
import sys
import pylab
import scipy.stats as stats
from pandas import DataFrame
import matplotlib.pyplot as plot
from random import uniform
from math import sqrt


#  read the data
df = pd.read_csv('concrete.csv', header= 0 , sep=',')

help( pd.read_csv)


# use seaborn to plot 
cols = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg',
        'fineagg', 'age', 'strength']

sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
# plt.savefig('images/10_03.png', dpi=300)
plt.show()


# corelation
cm = np.corrcoef(df[cols].values.T)
#sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 8},
                 yticklabels=cols,
                 xticklabels=cols)

plt.tight_layout()

plt.show()






# # Evaluating the performance of linear regression models


X = df.iloc[:, :-1].values
y = df['strength'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)



slr = LinearRegression()

slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)



print('coefficient:')
print(slr.coef_)
print('Intercept: %.3f' % slr.intercept_)



# #    Residual plot

plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual error of linear Model')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()

plt.show()

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

print('')
print('')
print('')

# # Ridge Regression 

list1=[0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,
       100,1000]
list2=[]
for alpha1 in list1:
    ridge = Ridge(alpha= alpha1)
    ridge.fit(X_train,y_train)
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)


    list2.append(r2_score(y_test, y_test_pred))

print('different alpha in ridge regression:')
print(list2)
    

ridge = Ridge(alpha= 0.2)
ridge.fit(X_train,y_train)
y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)


print('')

print('coefficient:')
print(ridge.coef_)
print('Intercept: %.3f' % ridge.intercept_)
       
plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual error of Ridge Model')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()

plt.show()
    
print('MSE train: %.3f, test: %.3f' % (
    mean_squared_error(y_train, y_train_pred),
     mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
    r2_score(y_train, y_train_pred),
    r2_score(y_test, y_test_pred)))


print('')
print('')
print('')


##LASSO Regression


list1=[0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,
       100,1000]
list2=[]
for alpha1 in list1:
    lasso = Lasso(alpha= alpha1)
    lasso.fit(X_train,y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)


    list2.append(r2_score(y_test, y_test_pred))

print('different alpha in lasso regression:')
print(list2)
    

lasso = Lasso(alpha= 0.2)
lasso.fit(X_train,y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)


print('')

print('coefficient:')
print(lasso.coef_)
print('Intercept: %.3f' % lasso.intercept_)
       
plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual error of linear Model')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()

plt.show()
    
print('MSE train: %.3f, test: %.3f' % (
    mean_squared_error(y_train, y_train_pred),
     mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
    r2_score(y_train, y_train_pred),
    r2_score(y_test, y_test_pred)))



print('')
print('')
print('')



# Elastic Net regression:


list1=[0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,
       100,1000]
list2=[]
for l1 in list1:
    elanet = ElasticNet(alpha=1.0, l1_ratio=l1)
    elanet.fit(X_train, y_train)
    y_train_pred = elanet.predict(X_train)
    y_test_pred = elanet.predict(X_test)
    list2.append(r2_score(y_test, y_test_pred))

print('different l1_ratio in Elastic Net regression:')
print(list2)


elanet = ElasticNet(alpha=1.0, l1_ratio=l1)
elanet.fit(X_train, y_train)
y_train_pred = elanet.predict(X_train)
y_test_pred = elanet.predict(X_test)
list2.append(r2_score(y_test, y_test_pred))


print('')

print('coefficient:')
print(elanet.coef_)
print('Intercept: %.3f' % elanet.intercept_)


plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual error of Elastic Net Model')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()

plt.show()
    
print('MSE train: %.3f, test: %.3f' % (
    mean_squared_error(y_train, y_train_pred),
     mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
    r2_score(y_train, y_train_pred),
    r2_score(y_test, y_test_pred)))

print('')
print('******************************************************')
print("My name is {Chaozhen He}")
print("My NetID is: {che19}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")