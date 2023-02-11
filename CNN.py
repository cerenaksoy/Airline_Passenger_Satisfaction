# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 16:26:35 2023

@author: ceren
"""

import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout,MaxPool1D, BatchNormalization, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
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

xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.20, random_state=8)

print(xtrain.shape)
print(xtest.shape)   
#Min-max Scaler
from sklearn.preprocessing import MinMaxScaler
mms= MinMaxScaler(feature_range=(0,1))
xtrain= mms.fit_transform(xtrain)
xtest= mms.fit_transform(xtest)
xtrain= pd.DataFrame(xtrain)
xtest= pd.DataFrame(xtest)


xtrain= xtrain.values.reshape(xtrain.shape[0], xtrain.shape[1], 1)
print(xtrain.shape)

xtest= xtest.values.reshape(xtest.shape[0], xtest.shape[1], 1)
print(xtest.shape)


#%% 1D CNN
batch = 500
epochs = 20

def build_model():
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape = (20,1)))
    model.add(Conv1D(50, 4, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu', input_shape = (20,1)))
    model.add(MaxPooling1D())
    model.add(Conv1D(50, 4, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid')) 
    model.summary()
    
    
    optimizer= tf.keras.optimizers.Adamax(
    learning_rate=0.0015)

    model.compile(loss='binary_crossentropy', optimizer= optimizer, metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn = build_model, epochs = epochs, batch_size= batch)

"""
accuracies = cross_val_score(estimator = model, X = xtrain, y = ytrain, cv = 10)
mean = accuracies.mean()
print("Accuracy mean: "+ str(mean))
"""

history =model.fit(xtrain,ytrain)
y_pred_proba= model.predict(xtest)
y_pred= (y_pred_proba>0.5).astype('int')

confusion_matrix = confusion_matrix(ytest, y_pred_proba)
print(confusion_matrix)
sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, fmt='.2%', cmap='Blues')
print(classification_report(ytest,y_pred_proba))


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



#%% Results
print("1D CNN")
print(classification_report(ytest,y_pred))


from sklearn.metrics import accuracy_score
acc_score = accuracy_score(ytest,y_pred)
print("Accuracy Score:", acc_score)


#Roc Curve & AUC
from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(ytest,  y_pred)

auc = metrics.roc_auc_score(ytest, y_pred)
print("AUC is ", auc)

#plot
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("CNN ROC Curve")
plt.show()






