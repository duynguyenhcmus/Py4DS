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

#Import Scikit - Learn Models
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering 
#Cleaning Data
def cleaning_data(data):
    '''
        Purpose: Remove duplicate values and missing values
        Param: Data - dataframe
        Output: Cleaned Dataframe
    '''
    #Remove duplicates values
    data.drop_duplicates(subset = data.columns.values[:-1], keep = 'first', inplace = True)
    #Remove missing values
    data.dropna(inplace=True)
    return data

def main():
    '''
        Purpose: Processing all the steps with the dataset
        Param: None
        Output: The clustering models for prediction
    '''
    ##1. Loading dataset##
    path='https://raw.githubusercontent.com/duynguyenhcmus/Py4DS/main/Lab4/Py4DS_Lab4_Dataset/spam.csv'
    df=pd.read_csv(path)
    pd.set_option("display.max_columns", 100)
    print(df['1'].value_counts())
    ##2. Cleaning Data
    data_cleaned=cleaning_data(df)
    ##3. Splitting Data
    
    X=data_cleaned.drop(['1'],axis=1)
    y=data_cleaned['1']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    ##4. KMeans Clustering
    kmeans=KMeans(n_clusters=2,max_iter=2000,random_state=1,algorithm='elkan')
    kmeans.fit(X_train,y_train)
    y_pred=kmeans.fit_predict(X_test)
    print("Accuracy of elkan algorithm: ",sum(y_pred==y_test)/len(y_pred))

    ##5. KMeans Clustering with StandardScaler
    x_scale = StandardScaler()
    x_scale.fit(X_train)
    x_train_scaled = x_scale.transform(X_train)
    x_scale.fit(X_test)
    x_test_scaled = x_scale.transform(X_test)
    kmeans_scale=KMeans(n_clusters=2,max_iter=2000,random_state=1,algorithm='elkan')
    kmeans_scale.fit(x_train_scaled,y_train)
    y_pred_scale=kmeans.fit_predict(x_test_scaled)
    print("Accuracy of elkan algorithm after scale: ",sum(y_pred_scale==y_test)/len(y_pred_scale))

    
    ##6. AffinityPropagation Algorithm
    affinity=AffinityPropagation(random_state=0,max_iter=2000,preference=500)
    affinity.fit(X_train,y_train)
    y_pred_affinity=affinity.fit_predict(X_test)
    print("Accuracy of Affinity: ",sum(y_pred_affinity==y_test)/len(y_pred_affinity))

    ##7. MeanShift Algorithm
    meanshift=MeanShift(bandwidth=1,max_iter=2000)
    meanshift.fit(X_train,y_train)
    y_pred_meanshift=meanshift.fit_predict(X_test)
    print("Accuracy of MeanShift: ",sum(y_pred_meanshift==y_test)/len(y_pred_meanshift))
    
    ##8. Agglomerative Algorithm
    agglo=AgglomerativeClustering(n_clusters=2)
    agglo.fit(X_train,y_train)
    agglo_pred=agglo.fit_predict(X_test)
    print("Accuracy of Agglomerative: ",sum(agglo_pred==y_test)/len(agglo_pred))



    
    


if __name__ == '__main__':
    main()







