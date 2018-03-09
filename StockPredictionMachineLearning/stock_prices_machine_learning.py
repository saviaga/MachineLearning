import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import datetime

def create_dataset (stock_symbol, start_date, end_date, lags):

    #Fetch the stock data from Google Finance
    data = web.DataReader(stock_symbol, 'google', start_date, end_date)
    #print(data)

    #Create a new dataframe
    #We are going to use additional features: lagged returns...today's returns, yesterday's returns

    tslag = pd.DataFrame(index=data.index)
    tslag["Today"] = data["Close"]
    tslag["Volume"] = data["Volume"]

    #Create the shifted lag series of prior trading period close values
    for i in range(0,lags):
        tslag["Lag%s" % str(i+1)] = data["Close"].shift(i+1)

    #Create the returns DataFrame

    dfret = pd.DataFrame(index=tslag.index)
    dfret["Volume"] = tslag["Volume"]
    dfret["Today"] = tslag["Today"].pct_change()*100.0

    #Create the lagged percentage returns columns
    for i in range(0, lags):
        dfret["Lag%s" %str(i+1)] = tslag["Lag%s" % str(i+1)].pct_change()*100.0

    #"Direction" column (+1 or -1) indicates up/down day
    dfret["Direction"] = np.sign(dfret["Today"])

    #because of the shifts there are NaN values.. we need to get rid of those
    dfret.drop(dfret.index[:5], inplace=True)

    return dfret



if __name__=="__main__":
    # Setting start and end dates of reading
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2018, 2, 28)
    lags=5
    data = create_dataset('TSLA', start, end, lags)

    #Use the 5 prior days of returns as predictor

    X = data[["Lag1","Lag2","Lag3","Lag4"]]
    Y = data["Direction"]

    start_test = datetime.datetime(2017, 3, 7)

    #Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    Y_train = Y[Y.index < start_test]
    Y_test = Y[Y.index >= start_test]

    #********************************* USING KNN **************************************

    #Use KNN  as the machine learning model
    modelKNN = KNeighborsClassifier(300) #K=300, considers the 300 nearest neighbors

    #train the model
    modelKNN.fit(X_train,Y_train)

    #Make an array of predictions on the test set
    predKNN = modelKNN.predict(X_test)

    print("******************* USING KNN *****************************")
    print("Accuracy of KNN model: %0.3f" % modelKNN.score(X_test,Y_test))
    #The best is 69%
    print("Confusion Matrix: \n%s" %confusion_matrix(predKNN, Y_test))

    # ********************************* USING SVM **************************************

    # Use SVM  as the machine learning model
    modelSVM = SVC(gamma=0.001)

    # train the model
    modelSVM.fit(X_train, Y_train)

    # Make an array of predictions on the test set
    predSVM = modelSVM.predict(X_test)

    print("******************* USING SVM *****************************")
    print("Accuracy of KNN model: %0.3f" % modelSVM.score(X_test, Y_test))
    # The best is 69%
    print("Confusion Matrix: \n%s" % confusion_matrix(predSVM, Y_test))
