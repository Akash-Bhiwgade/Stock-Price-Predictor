import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import datetime, math
import seaborn as sns

data = pd.read_csv('infydata.csv')
data['HL_PCT'] = ((data['High'] - data['Low'])/(data['Low']))*100
data['PCT_change'] = ((data['Close'] - data['Open'])/(data['Open']))*100

data = data[['Close', 'HL_PCT', 'PCT_change', 'Volume']]

forecast_col = 'Close'
data.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.06*len(data)))
data['label'] = data[forecast_col].shift(-forecast_out)

X = np.array(data.drop(['label'], 1))
X = X[:-forecast_out]
X_before = X[-forecast_out:]
X = preprocessing.scale(X)

data.dropna(inplace=True)
y = np.array(data['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

reg = LinearRegression()
reg.fit(X_train, y_train)
accuracy = reg.score(X_test, y_test)
