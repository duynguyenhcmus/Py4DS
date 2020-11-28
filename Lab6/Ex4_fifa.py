'''
    Lab 6: Feature Selection vs Dimensionality Reduction
'''

#Load libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#Load sklearn features-selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE

#Load sklearn dimensionality reduction
from sklearn.decomposition import PCA

#Load sklearn models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Import library for label encoder
from sklearn.preprocessing import LabelEncoder

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

#Label Encoder
def label_encoder(data):
    '''
        Purpose: Perform Label Encoding to transform object type
        Input: data - DataFrame
        Output: data - DataFrame - Dataset that has been transformed
    '''
    label = LabelEncoder()
    data_colums = data.dtypes.pipe(lambda X: X[X=='object']).index
    for col in data_colums:
        data[col] = label.fit_transform(data[col])
    return data

#Variance Threshold
def variancethreshold(data):
    '''
        Purpose: Performing Variance Threshold for feature selection
        Input: Data - Dataframe
        Output: data_out - DataFrame - Dataframe has feature-selected,
        features - Series - selected features
    '''
    selection=VarianceThreshold(threshold = 0.99)
    data_out=selection.fit_transform(data)
    features = data.columns[selection.get_support()]
    return data_out,features

#Random Forest Classifier
def random_forest(X_train,y_train,X_test,y_test):
    '''
        Purpose: Perform Random Forest Classifier
        Input: X_train,y_train,X_test,y_test - DataFrame
        Output: the accuracy score of prediction
    '''
    rdf=RandomForestClassifier(random_state=1)
    rdf.fit(X_train,y_train)
    y_pred=rdf.predict(X_test)
    return metrics.accuracy_score(y_pred,y_test)

#Logistic Regression
def logistic_regression(x_train, x_test, y_train, y_test):
    '''
        Purpose: Perform Logistic Regression
        Input: X_train,y_train,X_test,y_test - DataFrame
        Output: the accuracy score of prediction
    '''
    clf = LogisticRegression(solver='liblinear')
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return metrics.accuracy_score(y_test, y_pred)

def main():
    #Load dataset
    path_data='https://raw.githubusercontent.com/duynguyenhcmus/Py4DS/main/\
Lab6/Py4DS_Lab6_Dataset/data.csv'
    df=pd.read_csv(path_data)

    #Pandas Options
    pd.set_option("display.max_columns", 100,"display.max_rows",100)

    #Print the head of df
    print('The head of dataset')
    print(df.head())
    print('='*80)

    #Print the shape of dataset
    print('The shape of dataset: ',df.shape)
    print('='*80)
    '''
        As we can notice, there are 89 columns in the dataset. That's quite
        big. So we'll try to reduce them later.
    '''
    
    #Print the type of dataset
    print('The information of types in dataset:')
    print(df.dtypes.value_counts())
    print('='*80)

    #Explore null values in the dataset
    print('Explore the null value in the dataset: ')
    print(df.isnull().sum().sort_values(ascending=False))
    '''
        As we can notice, there are a lot of null values in Loaned From features.
        So we'll drop this columns. There are an equal number of null values on some
        features. So we'll drop their rows.
    '''

    #Drop null values
    df=df.drop(['Loaned From'],axis=1)
    df=df.dropna()
    print('='*80)

    #Checking for null values
    print('Checking for null values: ',df.isnull().sum().sum())
    print('The shape of dataset after drop null values: ',df.shape)
    print('='*80)

    #Drop columns that are irrelevant
    print('The head of dataset: ')
    print(df.head(10))
    '''
        we'll drop Unnamed: 0, ID, Photo, Flag, Club Logo as these features
        are irrelevant
    '''
    df=df.drop(['Unnamed: 0','ID','Photo','Flag','Club Logo'],axis=1)
    print('='*80)

    #Split data into train and test dataset
    x=df.drop(['Position'],axis=1)
    y=df['Position']
    x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.2,random_state=1,shuffle=True)

    #Preprocessing data
    #With Value features
    x_train['Value']=x_train['Value'].apply(lambda x:x[1:-1])
    x_test['Value']=x_test['Value'].apply(lambda x:x[1:-1])

    # #With Wage features
    # x_train['Wage']=x_train['Wage'].apply(lambda x:x[1:-1])
    # x_train['Wage']=x_train['Wage'].apply(lambda x:x[1:-1])
    
    # #With Joined features
    # x_train['Joined']=x_train['Joined'].apply(lambda x:x[-4:])
    # x_test['Joined']=x_test['Joined'].apply(lambda x:x[-4:])

    # #With Height features
    # x_train['Height']=x_train['Height'].apply(lambda x:(int(x[0])*12+int(x[2:])))
    # x_test['Height']=x_test['Height'].apply(lambda x:(int(x[0])*12+int(x[2:])))

    # #With Weight features
    # x_train['Weight']=x_train['Weight'].apply(lambda x:x[:-3])
    # x_test['Weight']=x_test['Weight'].apply(lambda x:x[:-3])

    # #With Release Clause
    # x_train['Release Clause']=x_train['Release Clause'].apply(lambda x:x[1:-1])
    # x_test['Release Clause']=x_test['Release Clause'].apply(lambda x:x[1:-1])

    # #Label Encoding
    # x_train=label_encoder(x_train)
    # x_test=label_encoder(x_test)
    # y_train=label_encoder(y_train)
    # y_test=label_encoder(y_train)

    # '''
    #     We'll use some features-selection and dimensionality reduction to reduce
    #     features for building models
    # '''

    # ##1. Variance Threshold
    # print('Perform Variance Threshold')
    # train_variance,features_variance=variancethreshold(x_train)
    # print('Shape of dataset before variance threshold: ',x_train.shape)
    # print('Shape of dataset after variance threshold: ',train_variance.shape)

    # #Processing data for Models
    # x_train_var=train_variance
    # y_train_var=y_train
    # x_test_var=x_test[features_variance]
    # y_test_var=y_test

    # #Build Model
    # accuracy_var_rdf=random_forest(x_train_var,y_train_var,x_test_var,y_test_var)
    # accuracy_var_lr=logistic_regression(x_train_var,y_train_var,x_test_var,y_test_var)
    # print('Accuracy of Random Forest after Variance Threshold: ',accuracy_var_rdf)
    # print('Accuracy of Logistic Regression after Variance Threshold: ',accuracy_var_lr)
    # print('='*80)




    

    





if __name__=='__main__':
    main()