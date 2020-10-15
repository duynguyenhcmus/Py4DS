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

y = df['class']
X = df.drop('class', axis=1)
print(y.head())
# Transform data to numeric data
labelencoder = LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])
print(df.dtypes)
print(df.head())
print(df.describe())

#df = df.drop(['veil-type'], axis=1)

## Quick at the characteristics of the data ##
# df_div = pd.melt(df, "class", var_name="Characteristics")
# fig, ax = plt.subplots(figsize=(10,5))
# p = sns.violinplot(ax=ax, x="Characteristics", y="value", hue="class", split=True, data=df_div)
# df_no_class = df.drop(["class"], axis=1)
# p.set_xticklabels(rotation=50, labels=list(df_no_class.columns))
# # Is the data balanced?
# plt.figure()
# pd.Series(df['class']).value_counts().sort_index().plot(kind='bar')
# plt.ylabel('Count')
# plt.xlabel('class')
# plt.title('Number of poisonous/edible mushrooms (0=edible, 1=poisonous)')
# plt.figure(figsize=(14,12))
# sns.heatmap(df.corr(), linewidths=1, cmap='YlGnBu', annot=True)
# plt.yticks(rotation=0)
# plt.show()

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
# Create Decision Tree classifier object
rdf = RandomForestClassifier()
# Train Decision Tree Classifier
rdf = rdf.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = rdf.predict(X_test)
# Model Accuracy, how often is the classifier corret?
print("Accuracy of random forest: ", metrics.accuracy_score(y_test, y_pred))
# Evaluate a score by cross validation
scores = cross_val_score(rdf, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))


## 6. Building Logistic Regression Model ##
# Create Decision Tree classifier object
lr = LogisticRegression(max_iter=2000)
# Train Decision Tree Classifier
lr = lr.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = lr.predict(X_test)

## Evaluating Logistic Regression Model ##
# Model Accuracy, how often is the classifier corret?
print("Accuracy of logistic regression: ", metrics.accuracy_score(y_test, y_pred))
# Evaluate a score by cross validation
scores = cross_val_score(lr, X, y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))

