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
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch

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

#Plot figure of EDA
def eda_plot(data):
    '''
        Purpose: Create plot of spam or non-spam
        Param: data - dataframe
        Output: the figure of countplot and heatmap -jpg
    '''
    #Countplot
    plt.figure(figsize=(15,15))
    sns.countplot(x="1",data=data,linewidth=2,edgecolor=sns.color_palette("dark"))
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

#K-Means Algorithm
def kmeans_algorithm(X_train, X_test, y_train, y_test):
    '''
        Purpose: Perform K-means Clustering Algorithm for Classification
        Param: X_train, X_test, y_train, y_test - ndarray 
        Output: The Accuracy of the K-means Clustering - float64
    '''
    kmeans=KMeans(n_clusters=2,max_iter=2000,random_state=1)
    kmeans.fit(X_train,y_train)
    y_pred=kmeans.fit_predict(X_test)
    return sum(y_pred==y_test)/len(y_pred)

#Agglomerative Algorithm
def agglomerative_algorithm(X_train, X_test, y_train, y_test):
    '''
        Purpose: Perform Agglomerative Clustering Algorithm for Classification
        Param: X_train, X_test, y_train, y_test - ndarray
        Output: The Accuracy of the Agglomerative - float64
    '''
    agglo=AgglomerativeClustering(n_clusters=2)
    agglo.fit(X_train,y_train)
    agglo_pred=agglo.fit_predict(X_test)
    return sum(agglo_pred==y_test)/len(agglo_pred)

#Birch Algorithm
def birch_algorithm(X_train, X_test, y_train, y_test):
    '''
        Purpose: Perform Birch Clustering Algorithm for Classification
        Param: X_train, X_test, y_train, y_test - ndarray
        Output: The Accuracy of the Birch - float64
    '''
    birch=Birch(n_clusters=2)
    birch.fit(X_train,y_train)
    birch_pred=birch.predict(X_test)
    return sum(birch_pred==y_test)/len(birch_pred)

def main():
    '''
        Purpose: Processing all the steps with the dataset
        Param: None
        Output: The clustering models for prediction
    '''
    ##1. Loading dataset##
    path='https://raw.githubusercontent.com/duynguyenhcmus/Py4DS/main/Lab4/\
Py4DS_Lab4_Dataset/spam.csv'
    df=pd.read_csv(path)
    pd.set_option("display.max_columns", 100)
    #Print the number of null value
    print(df.isnull().sum().sort_values(ascending = False))
    '''
        As we can see, there are no null values on the dataset.
    '''

    ##2. Cleaning Data
    data_cleaned=cleaning_data(df)

    ##3. Exploratory Data Analysis
    #eda_plot(df)
    '''
        As we can see from the countplot, the number of non-spam emails is half
        as many again as the number of spam emails.

        As we can see from the heatmap, there is a strong correlation between 
        features in the middle the heatmap from 0.16 to 0.27.
    '''
    ##4. Splitting Data
    X=data_cleaned.drop(['1'],axis=1)
    y=data_cleaned['1']

    #The size of test set: 30% of the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    ##5. Build KMeans Clustering
    accuracy_kmeans=kmeans_algorithm(X_train,X_test,y_train,y_test)
    print("The Accuracy of KMeans: {}".format(accuracy_kmeans))

    ##6. Build Agglomerative Algorithm
    accuracy_agglomerative=agglomerative_algorithm(X_train,X_test,y_train,y_test)
    print("The Accuracy of Agglomerative: {}".format(accuracy_agglomerative))

    ##7. Build Birch Algorithm
    accuracy_birch=birch_algorithm(X_train,X_test,y_train,y_test)
    print("The Accuracy of Birch: {}".format(accuracy_birch))

if __name__ == '__main__':
    main()







