# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 13:25:43 2019

@author: akash
"""
from datetime import date
from nsepy import get_history
import numpy as np
from sklearn import preprocessing, model_selection, linear_model

infydata = get_history(symbol='INFY', start=date(2018,1,1), end=date(2019,2,10))
infydata['label']= infydata['Close'].shift(-7)

infydata = infydata.dropna()

X = np.array(infydata[['Open', 'High', 'Low', 'Close']])
y = np.array(infydata['label'])
#
X = preprocessing.scale(X)
#
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
#
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
#
score = reg.score(X_test, y_test)
