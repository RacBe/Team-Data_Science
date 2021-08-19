# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
import seaborn as sn
import matplotlib.pyplot as plt

#DataFlair - Read the data
df=pd.read_csv('D:\\DataFlair\\parkinsons.data')
df.head()

#DataFlair - Get the features and labels
features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values

#DataFlair - Get the count of each label (0 and 1) in labels
print(labels[labels==1].shape[0], labels[labels==0].shape[0])

#DataFlair - Scale the features to between -1 and 1
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels

#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)

#DataFlair - Train the model
model=XGBClassifier()
model.fit(x_train,y_train)

# DataFlair - Calculate the accuracy
y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)

import seaborn as sb 
corr_map=df.corr()
sb.heatmap(corr_map,square=True)

features = df.loc[:,df.columns!='status'].values[:,1:]
labels = df.loc[:,'status'].values

scaler = MinMaxScaler((-1,1))
x = scaler.fit_transform(features)
y = labels
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 7)

model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Accuracy: ",accuracy_score(y_test, y_pred)*100)

probabilities = model.predict_proba(x_test)
first,second,thresholds = roc_curve(y_test, probabilities[:,1])
cm = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None, normalize=None)
fig, axis = plt.subplots(1,2, figsize=(20, 7))
#plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='d',ax=axis[0])
axis[0].set(xlabel='Predicted', ylabel='Actual')
#plt.ylabel('Actual')
#plt.xlabel('Predicted')

#first,second,thresholds = roc_curve(y_test, probabilities[:,1])
# Lineplot
plt.title('Receiver Operating Characteristic')
sn.lineplot(first, second, ax=axis[1])
axis[1].set(xlabel = 'False Positive Rate', ylabel = 'True Positive Rate')
plt.show()