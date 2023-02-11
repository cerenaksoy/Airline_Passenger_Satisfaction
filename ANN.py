# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 16:14:12 2023

@author: ceren
"""

#Kütüphaneler
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras

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

#%%ANN
from keras.wrappers.scikit_learn import KerasClassifier
def build_classifier():
    model = keras.Sequential([
    keras.layers.Dense(200, input_shape=(20,),activation ='relu'),
    keras.layers.Dense(200, activation ='relu'),
    keras.layers.Dense(100, activation ='relu'),
    keras.layers.Dense(100, activation ='relu'),
    
    keras.layers.Dense(1,activation='sigmoid')
    ])

    model.compile(optimizer='Adamax', loss='binary_crossentropy',  metrics=['accuracy'])
   #model.compile(optimizer='Adam', loss='binary_crossentropy',  metrics=['accuracy'])
   #model.compile(optimizer='Adadelta', loss='binary_crossentropy',  metrics=['accuracy'])
   #model.compile(optimizer='SGD', loss='binary_crossentropy',  metrics=['accuracy'])
   #model.compile(optimizer='RMSprop', loss='binary_crossentropy',  metrics=['accuracy'])

    return model

model = KerasClassifier(build_fn = build_classifier, epochs = 10, batch_size=100)

"""
#10 Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = x_train, y = y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: "+ str(mean))
"""

history = model.fit(x_train,y_train)
y_pred_proba= model.predict(x_test)
y_pred_ann= (y_pred_proba>0.5).astype('int')

confusion_matrix = confusion_matrix(y_test, y_pred_ann)
print(confusion_matrix)
sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, fmt='.2%', cmap='Blues')
print(classification_report(y_test,y_pred_ann))

#%%results
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test,y_pred_ann)
print("Accuracy Score:", acc_score)


#Roc Curve & AUC
from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_ann)

auc = metrics.roc_auc_score(y_test, y_pred_ann)
print("AUC is ", auc)

#plot
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("ANN ROC Curve")
plt.show()

