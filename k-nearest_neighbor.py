# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 15:43:01 2023

@author: ceren
"""

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from numpy import unique
import pandas as pd
import numpy as np
import seaborn as sns


#Veriseti İnceleme
df = pd.read_csv("airline_passenger_satisfaction.csv", sep = ",")
df.head()

df.shape #129880 samples, 24 feature
df.info()

df.dropna(inplace=True)



# Label encoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype=='object':
        df[col]=label_encoder.fit_transform(df[col])
df.head()

#Korelasyon
plt.figure(figsize =(24,24))
correlation_matrix =df[df.columns].corr()
sns.heatmap(correlation_matrix,annot=True)

df.drop('customer_class',axis=1,inplace=True)
df.drop('ease_of_online_booking',axis=1,inplace=True)
df.drop('inflight_entertainment',axis=1,inplace=True)


#%% Train Test Split 
#train-test split
x= df.drop('satisfaction',axis=1)
y= df['satisfaction']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30 ,random_state=42)

print(x_train.shape)  # (103589,23)
print(x_test.shape)   # (25898,23)

#Min-max Scaler
from sklearn.preprocessing import MinMaxScaler
mms= MinMaxScaler(feature_range=(0,1))
x_train= mms.fit_transform(x_train)
x_test= mms.fit_transform(x_test)
x_train= pd.DataFrame(x_train)


#%% KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors= 9, weights= "uniform", algorithm= "kd_tree",
                           leaf_size=20, p=2, metric= "minkowski",
                           n_jobs=None) 
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)
print('Test Accuracy of knn: {:.2f}'.format(knn.score(x_test, y_test)))


confusion_matrix = confusion_matrix(y_test, y_predict)
print(confusion_matrix)
sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, fmt='.2%', cmap='Blues')

print(classification_report(y_test, y_predict))

#Roc curve
from sklearn import metrics
y_pred_proba = knn.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)
print("AUC is ", auc)


#plot
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("K-Nearest Neighbour ROC Curve")
plt.show()