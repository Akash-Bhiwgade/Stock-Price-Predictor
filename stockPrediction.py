import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from nsepy import get_history
from datetime import date
from sklearn import svm, model_selection, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

style.use('ggplot')

infydata = get_history(symbol="INFY", 
                   start=date(2015,1,1), 
                   end=date(2016,12,31)
                 )
tcsdata = get_history(symbol="TCS", 
                   start=date(2015,1,1), 
                   end=date(2016,12,31)
                 )

infydata['10ma'] = infydata['Close'].rolling(window=10).mean()
infydata.dropna(inplace=True)

tcsdata['10ma'] = tcsdata['Close'].rolling(window=10).mean()
tcsdata.dropna(inplace=True)

'''df_ohlc = infydata['Close'].resample('10D').ohlc()
df_ohlc = infydata['Volume'].resample('10D').sum()
df_ohlc.reset_index(inplace=True)'''

def predict_price(X, y):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)
    clf = VotingClassifier([
                ('lsvc', svm.LinearSVC()),
                ('knn', neighbors.KNeighborsClassifier()),
                ('rfor', RandomForestClassifier())
            ])
    
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    prediction = clf.predict(X_test)
    
    plt.plot(y_test, prediction, color='red', label='Linear Model')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title('Stock Market Price Prediction')
    plt.legend()
    plt.show()
    
    return accuracy, prediction
