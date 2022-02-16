import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from collections import defaultdict

def LogRegwithC(cVal=1):
    data = pd.read_csv('Titanic_train.csv')
    data.drop(['Name', 'PassengerId', 'Cabin', 'Age'], axis=1, inplace=True)
    data.dropna(inplace=True)
    
    y = data.pop('Survived')
    X = data
    
    dDict = defaultdict(LabelEncoder)

    cat_cols = X.dtypes == object
    cat_labels = X.columns[cat_cols].tolist()

    X[cat_labels] = X[cat_labels].apply(lambda col: dDict[col.name].fit_transform(col))

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

    scale = MinMaxScaler()
    x_train_scaled = scale.fit_transform(x_train)
    x_train = pd.DataFrame(x_train_scaled, columns= x_train.columns)

    if cVal == 1:
        logReg = LogisticRegression(C = cVal, max_iter=1000)
    else:
        logReg = LogisticRegression(C=eval(cVal), max_iter=1000)

    logModel = logReg.fit(x_train, y_train)

    scale_test = MinMaxScaler()
    x_test_scaled = scale_test.fit_transform(x_test)
    x_test = pd.DataFrame(x_test_scaled, columns=x_test.columns)

    logPredict = logModel.predict(x_test)

    accuracy = accuracy_score(y_test, logPredict)
    precision = precision_score(y_test, logPredict)
    recall = recall_score(y_test, logPredict)
    F_Score = f1_score(y_test, logPredict)
    tn, fp, fn, tp = confusion_matrix(y_test, logPredict).ravel()
    total = tp + fp + tn + fn

    return logModel.coef_, logModel.intercept_, accuracy, precision, recall, F_Score, tp, fp, fn, tn, total