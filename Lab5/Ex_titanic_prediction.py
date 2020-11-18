'''
    Lab 5:Perform an Exploratory Data Analysis (EDA), Data cleaning, 
    Building models for prediction, Presenting resultsusing on the following datasets
'''

#Load libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#import ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

#Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

#Label Encoder
def label_encoder(data):
    '''
        Purpose: Using Label Encoder to transform the categorical data
        Param: data - DataFrame
        Output: the encoded DataFrame 
    '''
    label = LabelEncoder()
    data_colums = data.dtypes.pipe(lambda X: X[X=='object']).index
    for col in data_colums:
        data[col] = label.fit_transform(data[col])
    return data

#Exploratory Data Analysis
def eda(data,label):
    '''
        Purpose: Analyze the dataframe, Detect Outliers
        Input: data - Dataframe
        Output: The figure that represent some information
    '''
    #Boxplot to detect outliers
    plt.figure(figsize=(15,7))
    sns.boxplot(data=data)
    plt.savefig('Boxplot_titanic.jpg')
    #Heatmap to see the correlation between features
    corr = data.corr()
    corr = corr.filter(items = [label])
    sns.heatmap(corr, annot=True)
    plt.tight_layout()
    plt.savefig('Heatmap_titanic.jpg')

#Removing Outlier
def remove_outlier(data):
    '''
        Purpose: helping remove outliers of the dataset.
        Param: data - DataFrame
        Output: DataFrame that is outlier-removed
    '''
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data_out = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    print('Shape of data before handling outlier values: ', data.shape)
    print('Shape of data after handling outlier values: ', data_out.shape)
    return data_out

#Logistic Regression
def logistic_regression(X_train,y_train,X_test,y_test):
    '''
        Purpose: Perform Logistic Regression
        Input: X_train,y_train,X_test,y_test - DataFrame
        Output: The accuracy score of logistic regression
    '''
    # Create Logistic Regression object
    lr = LogisticRegression(max_iter=2000)
    # Train Logistic Regression
    lr = lr.fit(X_train, y_train)
    # Predict the response for test dataset
    y_pred = lr.predict(X_test)
    # Model Accuracy, how often is the classifier corret?
    return metrics.accuracy_score(y_test, y_pred)

#AdaBoostClassifier
def adaboost(X_train,y_train,X_test,y_test):
    '''
        Purpose: Perform AdaBoostClassifier
        Input: X_train,y_train,X_test,y_test - DataFrame
        Output: The accuracy score of AdaBoostClassifier
    '''
    abc = AdaBoostClassifier()
    abc = abc.fit(X_train, y_train)
    y_pred_abc = abc.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred_abc)

#Main function
def main():
    ##1. Load the dataset
    path='https://raw.githubusercontent.com/duynguyenhcmus/Py4DS/main/Lab5/\
Py4DS_Lab5_Dataset/titanic_train.csv'
    df=pd.read_csv(path)
    #Print the head of dataframe
    print(df.head())
    print('='*80)
    '''
        As we can see, there are 12 columns. So we have 11 features and 1 feature
        for classification which is Survived
    '''

    #Print the information of dataframe
    print(df.info())
    print('='*80)
    '''
        According to the info of dataframe, there are 3 features which have null
        value: Age, Cabin and Embarked
    '''

    #Describe the dataframe
    print(df.describe())
    print('='*80)
    '''
        We can notice outliers in Fare features as the max is way too high comparing
        with Q3.
    '''

    ##2. Handling Missing values
    #Fill in the missing values of Age feature with mean
    df['Age']=df['Age'].fillna(round(df['Age'].mean()))

    #Print Cabin value_counts
    print(df['Cabin'].value_counts())
    print('='*80)

    #Print Embarked value_counts
    print(df['Embarked'].value_counts())
    print('='*80)

    #Fill in the missing values of Embarked feature with mode
    df['Embarked']=df['Embarked'].fillna('S')

    #Drop Cabin columns
    df=df.drop(['Cabin'],axis=1)

    #Check for missing values
    print(df.info())
    print('='*80)

    #Label-encoding
    df=label_encoder(df)

    ##3. Data Cleaning
    #Drop Duplicates
    df.drop_duplicates(subset = df.columns.values[:-1], keep = 'first', inplace = True)
    print('Shape of dataframe after drop duplicates: ',df.shape)
    print('='*80)

    ##4. Exploratory Data Analysis
    eda(df,'Survived')
    '''
        As we expected, there are lots of outlier in Age and Fare features

        With heatmap, the correlation between survived feature and PassengerID
        is too low (0.005). So we drop this columns.

        We drop column Name also. Because all the values in Name feature are all
        unique. So the information will be irrelevant.
    '''
    #Drop PassengerId columns
    df=df.drop(['PassengerId','Name'],axis=1)

    #Remove Outlier
    df=remove_outlier(df)
    print('='*80)

    ##5. Splitting Data
    X = df.drop(['Survived'], axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=91)
 
    ##6. Build Models
    #Logistic Regression
    accuracy_logistic=logistic_regression(X_train,y_train,X_test,y_test)
    print("Accuracy of Logistic Regression: ",accuracy_logistic)

    #Ada Boost Classifier
    accuracy_ada=adaboost(X_train,y_train,X_test,y_test)
    print("Accuracy of AdaBoostClassifier: ", accuracy_ada)
    '''
        Summary:
            Accuracy of Logistic Regression:  0.8275862068965517
            Accuracy of AdaBoostClassifier:  0.896551724137931
    '''

if __name__ == '__main__':
    main()