# -*- coding: utf-8 -*-
"""

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

#%% XG-Boost
from xgboost import XGBClassifier

xgb = XGBClassifier(booster="gbtree", max_depth= 3, eta=0.6, gamma=0,
                    min_child_weight= 1, process_type = "default",
                    scale_pos_weight=2 )

xgb.fit(x_train, y_train)
print(xgb)

y_predict = xgb.predict(x_test)
predictions = [round(value) for value in y_predict]

#k- fold cross validation
from sklearn.model_selection import cross_val_score
score = cross_val_score(xgb, x_train, y_train, cv= 10)
print("10- Fold Cross Validation Score: ", score.mean())
confusion_matrix = confusion_matrix(y_test, y_predict)
print(confusion_matrix)
sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, fmt='.2%', cmap='Blues')
print(classification_report(y_test, y_predict)) 

#Roc curve
from sklearn import metrics
y_pred_proba = xgb.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)
print("AUC is ", auc)

#plot
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("XG-Boost Roc Curve")
plt.show()








