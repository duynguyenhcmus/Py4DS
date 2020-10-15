## Load libraries ##
import pandas as pd
# Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Calculating and Displaying importance using the eli5 library
import eli5
from eli5.sklearn import PermutationImportance

#from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

## 1. Loading Data ##
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# Load dataset
path = 'https://raw.githubusercontent.com/duynguyenhcmus/Pythonfordatascience/main/Week01/Py4DS_Lab1_Dataset/diabetes.csv'
df = pd.read_csv(path, header=0, names=col_names)
print(df.head())
print(df.info())

## 2. Feature Selection ##
# Split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = df[feature_cols] # Feature
y = df.label # Target variable

## 3. Splitting Data ##
# Split dataset into training set and test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

## 4. Building Decision Tree Model ##
# Create Decision Tree classifier object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifier
clf = clf.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = clf.predict(X_test)
# Return an explanation of estimator parameters (weights). Use this function to show classifier weights
perm = PermutationImportance(clf, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())

## 5. Evaluating Model ##
# Model Accuracy, how often is the classifier corret?
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

## 6. Visualizing Decision Trees ##
# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('diabetes.png')
# Image(graph.create_png())
