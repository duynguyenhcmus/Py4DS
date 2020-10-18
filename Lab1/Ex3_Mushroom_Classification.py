## Load libraries ##
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
#import ML models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Import libraries for cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# Import library for label encoder
from sklearn.preprocessing import LabelEncoder

## 1. Loading Data ##
# Load dataset
path = 'https://raw.githubusercontent.com/duynguyenhcmus/Pythonfordatascience/main/Week01/Py4DS_Lab1_Dataset/mushrooms.csv'
df = pd.read_csv(path)
print(df.head())

# 2. Data Preprocessing
y = df['class']
X = df.drop('class', axis=1)
print(y.head())
# Transform data to numerical data
labelencoder = LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])
print(df.dtypes)
print(df.head())
print(df.describe())

## 3. Splitting Data ##
# Split dataset into training set and test set
# 70% training and 30% test
X = df.drop(['class'], axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

## 4. Building Decision Tree Model ##
# Create Decision Tree classifier object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifier
clf = clf.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier corret?
print("Accuracy of decision tree: ", metrics.accuracy_score(y_test, y_pred))
# Evaluate a score by cross validation
scores = cross_val_score(clf, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))

## 5. Building Random Forest Model ##
# Create Random Forest object
rdf = RandomForestClassifier()
# Train Random Forest Classifier
rdf = rdf.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = rdf.predict(X_test)
# Model Accuracy, how often is the classifier corret?
print("Accuracy of random forest: ", metrics.accuracy_score(y_test, y_pred))
# Evaluate a score by cross validation
scores = cross_val_score(rdf, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))

## 6. Building Logistic Regression Model ##
# Create Logistic Regression object
lr = LogisticRegression(max_iter=2000)
# Train Logistic Regression
lr = lr.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = lr.predict(X_test)
# Model Accuracy, how often is the classifier corret?
print("Accuracy of logistic regression: ", metrics.accuracy_score(y_test, y_pred))
# Evaluate a score by cross validation
scores = cross_val_score(lr, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))

## 7. Building Ada Boost Classifier Model ##
# Create Ada Boost Classifier object
abc = AdaBoostClassifier()
# Train Ada Boost Classifier Classifier
abc = abc.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = abc.predict(X_test)
# Model Accuracy, how often is the classifier corret?
print("Accuracy of ada boost classifier: ", metrics.accuracy_score(y_test, y_pred))
# Evaluate a score by cross validation
scores = cross_val_score(abc, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))

## 8. Building SVC Model ##
# Create SVC object
svc = SVC()
# Train SVC
svc = svc.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = svc.predict(X_test)
# Model Accuracy, how often is the classifier corret?
print("Accuracy of SVC: ", metrics.accuracy_score(y_test, y_pred))
# Evaluate a score by cross validation
scores = cross_val_score(svc, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))

## 9. Building MLP classifier Model ##
# Create MLP object
mlp = MLPClassifier()
# Train MLP
mlp = mlp.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = mlp.predict(X_test)
# Model Accuracy, how often is the classifier corret?
print("Accuracy of MLP classifier: ", metrics.accuracy_score(y_test, y_pred))
# Evaluate a score by cross validation
scores = cross_val_score(svc, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))

