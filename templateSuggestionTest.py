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

knn_model = joblib.load("knnTempSuggestion")

#Required format for test data
pathTestDataFormat = "filtered_TestData.csv"
testDataFormat = pd.read_csv(pathTestDataFormat)
#input from backend
pathInputDataFormat = "InputTestData.csv"
names = ['Result','Suggestion','Category']
rawInputData = pd.read_csv(pathInputDataFormat, names=names)
#binarizing the data
arrayInputData  = pd.get_dummies(rawInputData, prefix=None, prefix_sep='_', dummy_na=False, columns=['Suggestion','Category'])

#merging the input and format data
key0 = arrayInputData.columns.values[0]
key1 = arrayInputData.columns.values[1]
key2 = arrayInputData.columns.values[2]
resultTestData = pd.merge(testDataFormat, arrayInputData, how='right', on=[key0, key1, key2])
resultTestData.fillna(0, inplace=True)
resultTestData.to_csv("resultTestData.csv",index=False)

testdataset = resultTestData
finaltestdata = testdataset

#decoding the binarized data
finaltestdata = finaltestdata.replace(0, pd.np.nan)
final_columns  = finaltestdata.columns[finaltestdata.isnull().any()].tolist()
finaltestdata.drop(final_columns, axis=1,inplace=True)

#Test data
testarray = testdataset.values
testRow = testdataset.shape
testX = testarray[:,1:testRow[1]-1]

# make prediction (KNN)
testknn_predictions = knn_model.predict(testX)
df_prediction = pd.DataFrame(testknn_predictions,columns=['Prediction'])

testFinalOutPut=pd.concat([finaltestdata,df_prediction], axis=1, ignore_index=False)

testFinalOutPut.to_csv("finalResultdata.csv",index=False, header='')


