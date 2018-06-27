import pandas as pd
import numpy as np
#from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing 
from sklearn.externals import joblib
import datetime
import warnings
# Load dataset
path = "newTrainngdata.csv"
names = ['Suggestion','Category','Result']
rawdata = pd.read_csv(path, names=names)

for row in rawdata:   
    catRawData = rawdata['Category']
    resRawData = rawdata['Result']
    for idx, col in enumerate(catRawData):
        if ((catRawData[idx] == "Template issue") and (resRawData[idx] == "Failure")):
            resRawData[idx] = "Failure"
        elif ((catRawData[idx] == "Template issue") and (resRawData[idx] == "Warning")):
            resRawData[idx] = "Failure"
        else:
            resRawData[idx] = "Success"

array  = pd.get_dummies(rawdata, prefix=None, prefix_sep='_', dummy_na=False, columns=['Suggestion','Category'])
rowcount  = array.shape
#array = array.drop(['result'], axis=1)
array.to_csv("filtered_Array.csv",index=False)
testformatcsv = array
testformatcsv = testformatcsv.drop(testformatcsv.index[0:rowcount[0]-1])
testformatcsv = testformatcsv.replace(1, 0)
testformatcsv.to_csv("filtered_TestData.csv",index=False)
# Split-out validation dataset
arrayData = array.values

X = arrayData[:,1:rowcount[1]-1]
#X=X.astype('int')
Y = arrayData[:,0]
#Y=Y.astype('int')
validation_size = 0.70
seed = 6
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# make prediction (KNN)
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, Y_train)
knn_predictions = knn_model.predict(X_validation)
print(accuracy_score(Y_validation,knn_predictions))
joblib.dump(knn_model, "knnTempSuggestion", compress=3)


