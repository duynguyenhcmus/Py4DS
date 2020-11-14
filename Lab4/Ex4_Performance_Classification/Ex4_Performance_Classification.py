'''
    Lab 4: Clustering
    Ex4: Using xAPI-Edu-Data.csv dataset to perform an Exploratory Data Analysis (EDA),
    Data Cleaning, Clustering Models for prediction
'''
#Import Libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#Import library for label encoder
from sklearn.preprocessing import LabelEncoder
#Import Scikit - Learn for model selection
from sklearn.model_selection import train_test_split
#Import Scikit - Learn Models
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
#Import Scikit - Learn Metrics
from sklearn import metrics

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

#Remove outlier
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

#Data Cleaning
def cleaning_data(data):
    '''
        Purpose: Dropping duplicates and remove missing values
        Param: data - DataFrame
        Output: The cleaned DataFrame
    '''
    # Remove duplicate values
    data.drop_duplicates(subset = data.columns.values[:-1], keep = 'first', inplace = True)
    # Remove missing values
    data.dropna()
    return data

#Exploratory Data Analysis
def eda_plot(data):
    '''
        Purpose: Exploratory Data Analysis
        Param: data - DataFrame
        Output: The figure that illustrate the dataframe
    '''
    #Heatmap
    plt.figure(figsize=(15,15))
    corr = data.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,\
         linewidths=.1, cmap="Reds", annot=False)
    plt.tight_layout()
    plt.savefig("Heatmap-Performance.jpg")
    #Countplot
    plt.figure(figsize=(15,15))
    sns.countplot(x="Class",data=data,linewidth=2,edgecolor=sns.color_palette("dark"))
    plt.title("The countplot of Class")
    plt.xlabel("Class")
    plt.ylabel("The number of student")
    plt.savefig("Countplot-Performance.jpg")

#KMeans Algorithm
def kmeans_algorithm(X_train, X_test, y_train, y_test):
    '''
        Purpose: Perform K-means Clustering Algorithm for Classification
        Param: X_train, X_test, y_train, y_test - ndarray 
        Output: The Accuracy of the K-means Clustering - float64
    '''
    kmeans=KMeans(n_clusters=3 ,random_state=6)
    kmeans.fit(X_train)
    y_pred=kmeans.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

#Agglomerative Algorithm
def Agglomerative_algorithm(X_train, X_test, y_train, y_test):
    '''
        Purpose: Perform Agglomerative Clustering Algorithm for Classification
        Param: X_train, X_test, y_train, y_test - ndarray
        Output: The Accuracy of the Agglomerative - float64
    '''
    Agglomerative=AgglomerativeClustering(n_clusters=3, linkage="average", affinity='l1')
    Agglomerative.fit(X_train)
    Agglomerative_pred=Agglomerative.fit_predict(X_test)
    return metrics.accuracy_score(y_test, Agglomerative_pred)

#MiniBatchKMeans Algorithm
def minibatchkmeans_algorithm(X_train, X_test, y_train, y_test):
    '''
        Purpose: Perform MiniBatchKMeans Clustering for Classification
        Param: X_train, X_test, y_train, y_test - ndarray
        Output: The Accuracy of the MiniBatchKMeans - float64
    '''
    mini=MiniBatchKMeans(n_clusters=3, random_state=6)
    mini.fit(X_train)
    y_pred=mini.predict(X_test)
    return metrics.accuracy_score(y_test,y_pred)

def main():
    '''
        Purpose: Processing all the steps with the dataset
        Param: None
        Output: The clustering models for prediction
    '''
    ##1. Loading Dataset
    print("-"*80)
    path='https://raw.githubusercontent.com/duynguyenhcmus/Py4DS/main/Lab4/\
Py4DS_Lab4_Dataset/xAPI-Edu-Data.csv'
    df=pd.read_csv(path)
    print(df.head())

    ##2. Label Encoding
    print("-"*80)
    df=label_encoder(df)

    ##Print the number of null value
    print(df.isnull().sum().sort_values(ascending = False))
    '''
        As we can see, there are no null values on the dataset.
    '''
    ##3. Cleaning Data
    data_cleaned=cleaning_data(df)

    ##4. Handle Outlier
    print("-"*80)
    data=remove_outlier(data_cleaned)

    ##5. Plot the boxplot to detect outliers
    plt.figure(figsize=(15,15))
    sns.boxplot(data=data)
    plt.xticks(rotation=90)
    plt.savefig("Boxplot-Performance.jpg")
    '''
        As we can see from the boxplot, the NationalITy and PlaceofBirth has
        a lot of outliers.
    '''

    ##6. Exploratory Data Analysis
    eda_plot(data)
    '''
        As we can see from the heatmap, there is a strong inverse correlation
        between 2 features: StageID and GradeID

        We can notice from the countplot that the number of each class is quite
        balance.
    '''

    ##7. Splitting Training and Test set
    #Splitting Data
    X = data.drop(['Class'], axis=1)
    y = data['Class']

    #Set the size of the test set is 20% of the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)
    print("-"*80)

    ##8. Build KMeans Clustering Algorithm
    accuracy_kmeans=kmeans_algorithm(X_train,X_test,y_train,y_test)
    print("The Accuracy of KMeans: {}".format(accuracy_kmeans))

    ##9. Build Agglomerative Algorithm
    accuracy_Agglomerative=Agglomerative_algorithm(X_train,X_test,y_train,y_test)
    print("The Accuracy of Agglomerative: {}".format(accuracy_Agglomerative)) 

    ##10. Build MiniBatchKMeans Algorithm
    accuracy_mini=minibatchkmeans_algorithm(X_train,X_test,y_train,y_test)
    print("The Accuracy of MiniBatchKMeans: {}".format(accuracy_mini))
    '''
        Summary:
            The Accuracy of KMeans: 0.6805555555555556
            The Accuracy of Agglomerative: 0.6666666666666666
            The Accuracy of MiniBatchKMeans: 0.6666666666666666
    '''

if __name__ == '__main__':
    main()