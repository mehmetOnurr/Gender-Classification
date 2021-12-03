# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 16:11:40 2021

@author: Asus
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv("gender_classification_v7.csv")

print(data.shape)


data_include = data.iloc[:,0:-1].values
data_classes = data.iloc[:,-1].values
data_classes = data_classes.reshape(-1,1)


#%% Data Encoder for classes
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown='ignore',drop='first')

data_classes = ohe.fit_transform(data_classes).toarray()

print(data_classes.shape)


#%% Data scale 

data_sc_values = data_include[:,1:3]

from sklearn.preprocessing import MinMaxScaler

minmaxSc = MinMaxScaler().fit(data_sc_values)

data_sc_values = minmaxSc.transform(data_sc_values)

data_include = np.delete(data_include, [1,2],1)

data_include = np.concatenate((data_include,data_sc_values), axis=1)



#%%  train test split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test =train_test_split(data_include,data_classes, test_size = 0.3, random_state= 42 )
 


#%% -------------------------------logistic regression-------------------------
from sklearn.linear_model import LogisticRegression

# scale edilmemiş
lr = LogisticRegression()
lr.fit(X_train,y_train)

predict_class = lr.predict(X_test)

#%% ------------------------------ Knn Classification

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors= 6)

knn.fit(X_train,y_train)

knn_predict = knn.predict(X_test)

#%%  Support Vector Classification

from sklearn.svm import SVC

svm_class = SVC(kernel="poly")

svm_class.fit(X_train,y_train)

svc_predict = svm_class.predict(X_test)

#%% Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb_gau = GaussianNB()

nb_gau.fit(X_train,y_train)

nb_gau_predict = nb_gau.predict(X_test)

#%% Decision Tree

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion="gini",min_samples_split=4)

dtc.fit(X_train,y_train)

dtc_predict = dtc.predict(X_test)

#%% Random Forest Classfication


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)

rfc_predict = rfc.predict(X_test)





#%% confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,predict_class)
print("Logistic Classification")
print(cm)

cm_knn = confusion_matrix(y_test,knn_predict)
print("Knn Classification")
print(cm_knn)

cm_svc = confusion_matrix(y_test, svc_predict)
print("SVC Classification")
print(cm_svc)

cm_nb_gau = confusion_matrix(y_test,nb_gau_predict)
print("Naive Bayes Gausian Classification")
print(cm_nb_gau)


cm_dtc = confusion_matrix(y_test, dtc_predict)
print("Decision Tree Classification")
print(cm_dtc)

cm_rfc = confusion_matrix(y_test, rfc_predict)
print("Random Forest Classification")
print(cm_rfc) 