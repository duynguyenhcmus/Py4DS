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

# Import library for label encoder
from sklearn.preprocessing import LabelEncoder
#Import Scikit - Learn for model selection
from sklearn.model_selection import train_test_split
##Import Scikit - Learn Models
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch

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
    kmeans=KMeans(n_clusters=3,max_iter=2000,random_state=1)
    kmeans.fit(X_train,y_train)
    y_pred=kmeans.fit_predict(X_test)
    return sum(y_pred==y_test)/len(y_pred)

#Birch Algorithm
def birch_algorithm(X_train, X_test, y_train, y_test):
    '''
        Purpose: Perform Birch Clustering Algorithm for Classification
        Param: X_train, X_test, y_train, y_test - ndarray
        Output: The Accuracy of the Birch - float64
    '''
    birch=Birch(n_clusters=3)
    birch.fit(X_train,y_train)
    birch_pred=birch.predict(X_test)
    return sum(birch_pred==y_test)/len(birch_pred)

#Agglomerative Algorithm
def agglomerative_algorithm(X_train, X_test, y_train, y_test):
    '''
        Purpose: Perform Agglomerative Clustering Algorithm for Classification
        Param: X_train, X_test, y_train, y_test - ndarray
        Output: The Accuracy of the Agglomerative - float64
    '''
    agglo=AgglomerativeClustering(n_clusters=3)
    agglo.fit(X_train,y_train)
    agglo_pred=agglo.fit_predict(X_test)
    return sum(agglo_pred==y_test)/len(agglo_pred)

def main():
    '''
        Purpose: Processing all the steps with the dataset
        Param: None
        Output: The clustering models for prediction
    '''
    ##1. Loading Dataset
    path='https://raw.githubusercontent.com/duynguyenhcmus/Py4DS/main/Lab4/\
Py4DS_Lab4_Dataset/xAPI-Edu-Data.csv'
    df=pd.read_csv(path)

    ##Print the number of null value
    print(df.isnull().sum().sort_values(ascending = False))
    '''
        As we can see, there are no null values on the dataset.
    '''
    ##2. Cleaning Data
    data_cleaned=cleaning_data(df)

    ##3. Label Encoding Data
    data_encoded=label_encoder(data_cleaned)

    #Plot the boxplot to detect outliers
    plt.figure(figsize=(15,15))
    sns.boxplot(data=data_encoded)
    plt.xticks(rotation=90)
    plt.show()
    '''
        As we can see from the boxplot, the NationalITy and PlaceofBirth has
        a lot of outliers. So we drop these features
    '''
    #Remove NationalITy and PlaceofBirth features
    data=data_encoded.drop(['NationalITy','PlaceofBirth'],axis=1)

    ##4. Exploratory Data Analysis
    eda_plot(data)
    '''
        As we can see from the heatmap, there is a strong inverse correlation
        between 2 features: StageID and GradeID

        We can notice from the countplot that the number of each class is quite
        balance.
    '''

    ##5. Splitting Training and Test set
    #Splitting Data
    X = data.drop(['Class'], axis=1)
    y = data['Class']
    #Set the size of the test set is 30% of the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    ##6. Build KMeans Clustering Algorithm
    accuracy_kmeans=kmeans_algorithm(X_train,X_test,y_train,y_test)
    print("The Accuracy of KMeans: {}".format(accuracy_kmeans))

    ##7. Build Birch Algorithm
    accuracy_birch=birch_algorithm(X_train,X_test,y_train,y_test)
    print("The Accuracy of Birch: {}".format(accuracy_birch)) 

    ##8. Build Agglomerative Algorithm
    accuracy_agglomerative=agglomerative_algorithm(X_train,X_test,y_train,y_test)
    print("The Accuracy of Agglomerative: {}".format(accuracy_agglomerative))

if __name__ == '__main__':
    main()