'''
    Lab 4: Clustering
    Ex1: Using spam.csv dataset to perform an Exploratory Data Analysis (EDA),
    Data Cleaning, Clustering Models for prediction
'''

#Load Libraries:
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#Import Scikit - Learn for model selection
from sklearn.model_selection import train_test_split
#Import Scikit - Learn Models
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
#Import Scikit - Learn Metrics
from sklearn import metrics
#Import scikit-learn RobustScaler for scaling data
from sklearn.preprocessing import RobustScaler

#Cleaning Data
def cleaning_data(data):
    '''
        Purpose: Remove duplicate values and missing values
        Param: Data - dataframe
        Output: Cleaned Dataframe - dataframe
    '''
    #Remove duplicates values
    data.drop_duplicates(subset = data.columns.values[:-1], keep = 'first', inplace = True)
    #Remove missing values
    data.dropna(inplace=True)
    return data

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
    print('> Shape of data before handling outlier values: ', data.shape)
    print('> Shape of data after handling outlier values: ', data_out.shape)
    return data_out

#Plot figure of EDA
def eda_plot(data):
    '''
        Purpose: Create plot of spam or non-spam
        Param: data - dataframe
        Output: the figure of countplot and heatmap -jpg
    '''
    #Countplot
    plt.figure(figsize=(15,15))
    sns.countplot(x="spam",data=data,linewidth=2,edgecolor=sns.color_palette("dark"))
    plt.title("The countplot of Spam versus Non-Spam Email")
    plt.xlabel("Spam or Non-Spam Email")
    plt.ylabel("The number of Spam versus Non-Spam")
    plt.savefig("Countplot-Spam.jpg")
    #Heatmap
    plt.figure(figsize=(15,15))
    corr = data.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,\
         linewidths=.1, cmap="Reds", annot=False)
    plt.tight_layout()
    plt.savefig("Heatmap-Spam.jpg")

def robust_scaler(x_train, x_test):
    '''
        - This function helps scale data without label from train, test data by Robust Scaler Model.
        - Parameters:
            + x_train : DataFrame
                DataFrame is used to train.
            + x_test: DataFrame
                DataFrame is used to test.
        Returns: DataFrames x_train_scaled and x_test_scaled
            Return scaled train, test data.
    '''
    x_scale = RobustScaler()
    x_scale.fit(x_train)
    x_train_scaled = x_scale.transform(x_train)
    x_scale.fit(x_test)
    x_test_scaled = x_scale.transform(x_test)
    return x_train_scaled, x_test_scaled

#K-Means Algorithm
def kmeans_algorithm(X_train, X_test, y_train, y_test):
    '''
        Purpose: Perform KMeans Clustering Algorithm for Classification
        Param: X_train, X_test, y_train, y_test - ndarray
        Output: The Accuracy of the KMeans - float64
    '''
    kmeans=KMeans(n_clusters=2, random_state=8)
    kmeans.fit(X_train)
    kmeans_pred=kmeans.predict(X_test)
    return metrics.accuracy_score(y_test, kmeans_pred)

#Agglomerative Algorithm
def agglomerative_algorithm(X_train, X_test, y_train, y_test):
    '''
        Purpose: Perform Agglomerative Clustering Algorithm for Classification
        Param: X_train, X_test, y_train, y_test - ndarray
        Output: The Accuracy of the Agglomerative - float64
    '''
    agglo=AgglomerativeClustering(n_clusters=2, affinity='l2', linkage='average')
    agglo.fit(X_train)
    agglo_pred=agglo.fit_predict(X_test)
    return metrics.accuracy_score(y_test,agglo_pred)

#MiniBatchKMeans Algorithm
def MiniBatchKMeans_algorithm(X_train, X_test, y_train, y_test):
    '''
        Purpose: Perform MiniBatchKMeans Clustering Algorithm for Classification
        Param: X_train, X_test, y_train, y_test - ndarray
        Output: The Accuracy of the MiniBatchKMeans - float64
    '''
    mini=MiniBatchKMeans(n_clusters=2, random_state=8)
    mini.fit(X_train)
    mini_pred=mini.predict(X_test)
    return metrics.accuracy_score(y_test,mini_pred)

def main():
    '''
        Purpose: Processing all the steps with the dataset
        Param: None
        Output: The clustering models for prediction
    '''
    ##1. Loading dataset##
    print("-"*80)
    path='https://raw.githubusercontent.com/duynguyenhcmus/Py4DS/main/Lab4/Py4DS_Lab4_Dataset/spam_original.csv'
    df=pd.read_csv(path)
    pd.set_option("display.max_columns", 100)
    df.columns = df.columns.str.replace(' ', '')

    ##2. Print the number of null value
    print("-"*80)
    print(df.isnull().sum().sort_values(ascending = False))
    '''
        As we can see, there are no null values on the dataset.
    '''

    ##3. Cleaning Data
    print("-"*80)
    data_cleaned=cleaning_data(df)
    data_cleaned=remove_outlier(data_cleaned)

    ##4. Exploratory Data Analysis
    print("-"*80)
    eda_plot(df)
    '''
        As we can see from the countplot, the number of non-spam emails is half
        as many again as the number of spam emails.
    '''
    ##5. Splitting Data
    X=data_cleaned.drop(['spam'],axis=1)
    y=data_cleaned['spam']

    #The size of test set: 25% of the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 1)

    ##6. Scaling data
    X_train, X_test = robust_scaler(X_train, X_test)

    ##7. Build Birch Algorithm
    accuracy_kmeans=kmeans_algorithm(X_train,X_test,y_train,y_test)
    print("The Accuracy of KMeans: {}".format(accuracy_kmeans)) 

    ##8. Build Agglomerative Algorithm
    accuracy_agglomerative=agglomerative_algorithm(X_train,X_test,y_train,y_test)
    print("The Accuracy of Agglomerative: {}".format(accuracy_agglomerative))

    ##9. Build MiniBatchKMeans Algorithm
    accuracy_mini=MiniBatchKMeans_algorithm(X_train,X_test,y_train,y_test)
    print("The Accuracy of MiniBatchKMeans: {}".format(accuracy_mini))
    '''
        Summary:
            The Accuracy of KMeans: 0.8620689655172413
            The Accuracy of Agglomerative: 0.896551724137931
            The Accuracy of MiniBatchKMeans: 0.8620689655172413
    '''

if __name__ == '__main__':
    main()






