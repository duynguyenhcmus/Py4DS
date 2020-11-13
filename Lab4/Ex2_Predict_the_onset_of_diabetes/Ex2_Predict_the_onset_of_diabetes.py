## Load libraries ##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics # Import scikit-learn metrics module for accuracy calculation
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch # import scikit-learn KMeans, MiniBatchKMeans and Birch clustering module for clustering data
from sklearn.preprocessing import RobustScaler # import scikit-learn RobustScaler for scaling data

def remove_outlier(data):
    '''
        - This function helps remove outliers of the dataset.
        - Parameters:
            + data :  DataFrame
                The dataset that you want to detect outliers.
        - Returns: DataFrame
            Return the dataframe is removed outliers.
    '''
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data_out = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    print('> Shape of data before handling outlier values: ', data.shape)
    print('> Shape of data after handling outlier values: ', data_out.shape)
    return data_out

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

def clustering_accuracy(x_train, x_test, y_test, model='MiniBatchKMeans', n_clusters=2, random_state=17):
    '''
        - This function helps calculate accuracy between y_pred and y_test by using model
        - Parameters:
            + x_train : DataFrame
                DataFrame is used to train
            + x_test : DataFrame
                DataFrame is used to test
            + y_test : DataFrame
                DataFrame is used to compare with the predicted label
            + model : string, {'MiniBatchKMeans', 'KMeans', 'Birch'}, default = 'MiniBatchKMeans'
                Use model in scikit-learn cluster
            + n_cluster : int, default = 2
                The dimension of the projection subspace.
            + random_state : int, default = 17
                Determines random number generation for centroid initialization. Use an int to make the randomness deterministic. 
        - Returns: float
            This function returns the accuracy between y_test and y_pred
    '''
    clustering = {'MiniBatchKMeans': MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state),
                  'Birch': Birch(n_clusters=n_clusters),
                  'KMeans': KMeans(n_clusters=n_clusters, random_state=random_state)}
    clustering[model].fit(x_train)
    y_pred = clustering[model].predict(x_test)
    return metrics.accuracy_score(y_test, y_pred)

def main():
    # ======================================================================= #
    ''' Load dataset '''
    # ======================================================================= #
    path = 'https://raw.githubusercontent.com/duynguyenhcmus/Py4DS/main/Lab4/Py4DS_Lab4_Dataset/diabetes.csv'
    df = pd.read_csv(path)
    print("*"*80)
    print(">> Dataset")
    print(df.head())

    # ======================================================================= #
    ''' Preprocessing '''
    # ======================================================================= #
    # Show information of dataset
    print("*"*80)
    print(">> Information of dataset")
    print(df.info())
    # Describe dataset
    print("*"*80)
    print(">> Describe dataset")
    print(df.describe())
    # % Missing values
    print("*"*80)
    print(">> Percent of missing values in dataset")
    missing_values = df.isnull().sum().sort_values(ascending = False)
    percentage_missing_values = (missing_values / len(df)) * 100
    missing_values = pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing'])
    print(missing_values)
    # Select duplicate rows except first occurrence based on all columns
    print("*"*80)
    print(">> Duplicate Rows except first occurrence based on all columns are:")
    print(df[df.duplicated()])

    # ======================================================================= #
    ''' EDA '''
    # ======================================================================= #
    # Handle ouliers
    print("*"*80)
    print(">> Handle outliers:")
    # Box plot of dataframe before remove outliers
    boxplot = df[[col for col in df.columns]]
    f, ax = plt.subplots(ncols = 4, nrows = int(len(boxplot.columns)/4), figsize=(10,len(boxplot.columns)/3))
    for i, c in zip(ax.flatten(), boxplot.columns):
        sns.boxplot(boxplot[c], ax = i)
    f.tight_layout()
    plt.savefig("Boxplot_before_remove_outliers.jpg")
    # Remove outliers
    data_cleaned = remove_outlier(df)
    # Box plot of dataframe after remove outliers
    boxplot = data_cleaned[[col for col in data_cleaned.columns]]
    f, ax = plt.subplots(ncols = 4, nrows = int(len(boxplot.columns)/4), figsize=(10,len(boxplot.columns)/3))
    for i, c in zip(ax.flatten(), boxplot.columns):
        sns.boxplot(boxplot[c], ax = i)
    f.tight_layout()
    plt.savefig("Boxplot_after_remove_outliers.jpg")

    # ======================================================================= #
    ''' Train model '''
    # ======================================================================= #
    # Split dataset into training set and test set
    # 75% training and 25% test
    X = data_cleaned.drop(['Outcome'], axis=1)
    y = data_cleaned['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    # Train KMeans model
    print("*"*80)
    print(">> Train model:")
    ## Scaling data
    X_train, X_test = robust_scaler(X_train, X_test)
    ## Calculate the accuracy
    n_clusters = len(np.unique(y_train))
    acc_minibatchkmeans = clustering_accuracy(X_train, X_test, y_test, model='MiniBatchKMeans', n_clusters=n_clusters)
    acc_kmeans = clustering_accuracy(X_train, X_test, y_test, model='KMeans', n_clusters=n_clusters)
    acc_birch = clustering_accuracy(X_train, X_test, y_test, model='Birch', n_clusters=n_clusters)
    print(f"- The accuracy of kmeans clustering algorithm:\t\t{acc_kmeans}")
    print(f"- The accuracy of minibatch kmeans clustering algorithm:{acc_minibatchkmeans}")
    print(f"- The accuracy of birch clustering algorithm:\t\t{acc_birch}")

    # ======================================================================= #
    ''' Sumary '''
    # ======================================================================= #
    '''
        - The accuracy of kmeans clustering algorithm:          0.74375
        - The accuracy of minibatch kmeans clustering algorithm:0.76875
        - The accuracy of birch clustering algorithm:           0.75625
    '''

if __name__ == '__main__':
    main()