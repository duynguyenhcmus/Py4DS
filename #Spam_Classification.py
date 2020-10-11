#Spam_Classification
#Load libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#Pandas Options
pd.set_option('display.max_colwidth',1000,'display.max_rows',None,\
             'display.max_columns',None)

#Plotting options
mpl.style.use('ggplot')
sns.set(style='whitegrid')

#Path of dataset 
path='https://raw.githubusercontent.com/duynguyenhcmus/Pythonfordatascience/main/spam.csv'
dataset_pd=pd.read_csv(path)
dataset_np=np.genfromtxt(path,delimiter=',')

X=dataset_np[:,:-1]
y=dataset_np[:,-1]

from sklearn.model_selection import train_test_split
#Split  dataset #
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)
print(X_train.shape,"\n\n",X_test.shape,"\n\n",y_train.shape,"\n\n",y_test.shape,"\n\n")
#import ML models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

#Import metrics to evaluate the performance of each model
from sklearn import metrics

#Import libraries for cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

clf=DecisionTreeClassifier()
clf=clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("CART (Tree Prediction) Accuracy: {}".format(sum(y_pred==y_test)/len(y_pred)))
print("CART (Tree Prediction) Accuracy by calling metrics: ",metrics.accuracy_score(y_test,y_pred))
#5-fold cross-validation
#Evaluate a score by cross-validation
scores=cross_val_score(clf,X,y,cv=5)
print("scores = {} \n final score = {} \n".format(scores,scores.mean()))
print("\n")