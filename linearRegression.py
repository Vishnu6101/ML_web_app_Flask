from math import sqrt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')
#Ridge Regression
def Ridge(Alpha=0):
    data = pd.read_csv('CO2 Emissions_Canada.csv')

    data.drop(['Make', 'Model'], axis=1, inplace=True)

    Y = data.pop('CO2 Emissions(g/km)')
    X = data

    cat_cols = X.dtypes == object
    cat_cols = X.columns[cat_cols].tolist()

    num_cols = X.dtypes != object
    num_cols = X.columns[num_cols].tolist()

    dDict = defaultdict(LabelEncoder)

    X[cat_cols] = X[cat_cols].apply(lambda col: dDict[col.name].fit_transform(col))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=10)

    scale_train = MinMaxScaler()
    xTrainScaled = scale_train.fit_transform(X_train[num_cols])
    X_train.loc[:, num_cols] = xTrainScaled

    scale_test = MinMaxScaler()
    xTestScaled = scale_test.fit_transform(X_test[num_cols])
    X_test.loc[:, num_cols] = xTestScaled

    ridgeReg = linear_model.Ridge(alpha=Alpha)

    ridgeModel = ridgeReg.fit(X_train, Y_train)

    ridgePredict = ridgeModel.predict(X_test)

    ridgeMSE = mean_squared_error(Y_test, ridgePredict)
    ridgeRMSE = sqrt(ridgeMSE)
    ridgeMAE = mean_absolute_error(Y_test, ridgePredict)
    ridgeR2 = r2_score(Y_test, ridgePredict)

    return ridgeModel.coef_, ridgeModel.intercept_, ridgeRMSE, ridgeMSE, ridgeMAE, ridgeR2