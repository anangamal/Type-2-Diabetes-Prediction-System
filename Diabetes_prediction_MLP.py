import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings('ignore')
diabetes = pd.read_csv('diabetes.csv')
diabetes_mod = diabetes[(diabetes.BloodPressure != 0) & (diabetes.BMI != 0) & (diabetes.Glucose != 0)]
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = diabetes_mod[feature_names]
y = diabetes_mod.Outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = diabetes_mod.Outcome, test_size=0.1)
sc = StandardScaler()
sc=sc.fit(X_train)
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
best_result=0.000
i=1
while True:
        mlp=MLPClassifier(hidden_layer_sizes=(i, i, i, i, i), verbose=False, learning_rate='adaptive', shuffle=True, early_stopping=False)
        mlp.fit(X_train, y_train)
        y_pred=mlp.predict(X_test)
        y_score=accuracy_score(y_test, y_pred)
        if y_score<best_result:
                break
        else:
                best_result=y_score
                i=i+1
best_result=0.000
b=1
while True:
        mlp=MLPClassifier(hidden_layer_sizes=(i, i, i, i, i), batch_size=b, verbose=False, learning_rate='adaptive', shuffle=True, early_stopping=False)
        mlp.fit(X_train, y_train)
        y_pred=mlp.predict(X_test)
        y_score=accuracy_score(y_test, y_pred)
        if y_score<best_result:
                break
        else:
                best_result=y_score
                b=b+1
best_result=0.000
m=1
while True:
        mlp=MLPClassifier(hidden_layer_sizes=(i, i, i, i, i), batch_size=b, verbose=False, learning_rate='adaptive', shuffle=True, early_stopping=False, max_iter=m)
        mlp.fit(X_train, y_train)
        y_pred=mlp.predict(X_test)
        y_score=accuracy_score(y_test, y_pred)
        if y_score<best_result:
                break
        else:
                best_result=y_score
                m=m+1
best_result=0.000
a=0.0
while True:
        mlp=MLPClassifier(hidden_layer_sizes=(i, i, i, i, i), alpha=a, batch_size=b, verbose=False, learning_rate='adaptive', shuffle=True, early_stopping=False, max_iter=m)
        mlp.fit(X_train, y_train)
        y_pred=mlp.predict(X_test)
        y_score=accuracy_score(y_test, y_pred)
        if y_score<best_result:
                break
        else:
                best_result=y_score
                a=a+0.1
mlp=MLPClassifier(hidden_layer_sizes=(i, ), activation='relu', solver='adam', alpha=a, batch_size=b, verbose=False, learning_rate='adaptive', shuffle=True, early_stopping=False, max_iter=m)
mlp.fit(X_train, y_train)
y_pred=mlp.predict(X_test)
print("MLP Summary")
print(str(accuracy_score(y_test, y_pred)*100)+"%")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
