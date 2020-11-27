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

#Logistic regression
def logistic_regression(X_train,y_train,X_test):
    '''
        Purpose: Perform Logistic Regression
        Input: X_train,y_train,X_test - DataFrame
        Output: the prediction of X_test - Dataframe
    '''
    lr = LogisticRegression(max_iter=2000,random_state=1)
    lr = lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    return y_pred

#Random Forest Classifier
def random_forest(X_train,y_train,X_test):
    '''
        Purpose: Perform Random Forest Classifier
        Input: X_train,y_train,X_test,y_test - DataFrame
        Output: the prediction of X_test - Dataframe
    '''
    rdf=RandomForestClassifier(random_state=1)
    rdf.fit(X_train,y_train)
    y_pred=rdf.predict(X_test)
    return y_pred

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

#Mutual Information
def mutual(data):
    '''
        Purpose: Perform mutual information for feature selection
        Input: Data - Dataframe
        Output: data_out-Dataframe-Transformed DataFrame, 
        features-Series-selected features
    '''
    x_train=data.drop(['id','target'],axis=1)
    y_train=data['target']
    numerical_x_train = x_train[x_train.select_dtypes([np.number]).columns]
    selection = SelectKBest(mutual_info_classif, k=10)
    data_out=selection.fit_transform(numerical_x_train, y_train)
    features = x_train.columns[selection.get_support()]
    return data_out,features

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
    data_out = data[~((data<(Q1-1.5*IQR))|(data>(Q3+1.5*IQR))).any(axis=1)]
    print('> Shape of data before handling outlier values: ', data.shape)
    print('> Shape of data after handling outlier values: ', data_out.shape)
    return data_out

#Heatmap correlation plot
def correlation_plot(data):
    '''
        Purpose: Plot the heatmap to see the correlation between features
        Input: data - Dataframe
        Output: the heatmap plot for correlation
    '''
    plt.figure(figsize=(15,15))
    corr = data.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,\
         linewidths=.1, cmap="Reds", annot=False)
    plt.tight_layout()
    plt.savefig("Heatmap-dontoverfit.jpg")
#Principal Component Analysis
def pca(data):
    '''
        Purpose: Perform Principal Component Analysis
        Input: data - Dataframe
        Output: x_pca - Dataframe - has been dimensionality-reduced
    '''
    pca = PCA(n_components=2)
    pca.fit(data)
    x_pca = pca.transform(data)
    return x_pca

#Select From Model
def selectfrommodel(X_train,y_train):
    '''
        Purpose: Perform SelectFromModel for features selection
        Input: X_train, y_train - Dataframe
        Output: model_features - Dataframe-transformed DataFrame,
         features - Series - selected features
    '''
    rfc = RandomForestClassifier(n_estimators=100)
    select_model =SelectFromModel(estimator=rfc)
    fit = select_model.fit(X_train, y_train)
    model_features = fit.transform(X_train)
    features =X_train.columns[select_model.get_support()]
    return model_features,features

#Recursive Feature Elimination
def recursive(x_train,y_train,x_test):
    '''
        Purpose: Perform Recursive Feature Elimination
        Input: x_train, y_train, x_test - DataFrame
        Output: y_pred - Series - Prediction of RFE
    '''
    rfc = RandomForestClassifier(n_estimators=100)
    rfe = RFE(estimator=rfc, n_features_to_select=3)
    rfe.fit(x_train, y_train)
    y_pred=rfe.predict(x_test)
    return y_pred

#Main function
def main():
    #Load Dataset
    path_train='https://raw.githubusercontent.com/duynguyenhcmus/Py4DS/main/\
Lab6/Py4DS_Lab6_Dataset/train.csv'
    df_train=pd.read_csv(path_train)

    path_test='https://raw.githubusercontent.com/duynguyenhcmus/Py4DS/main/\
Lab6/Py4DS_Lab6_Dataset/test.csv'
    df_test=pd.read_csv(path_test)

    #Pandas Options
    pd.set_option("display.max_columns", 100)

    #Print the head of df_train
    print('The train dataset')
    print(df_train.head())
    print('='*80)

    #Print the head of df_test
    print('the test dataset')
    print(df_test.head())
    print('='*80)

    #Print the shape of train and test dataset
    print('The shape of 2 dataset')
    print('The shape of train dataset: ',df_train.shape)
    print('The shape of test dataset: ',df_test.shape)
    print('='*80)

    #Cheking for null value in the dataset
    print('Checking for null value in the dataset')

    #Checking for train dataset
    if df_train.isnull().sum().sum()==0:
        print('There is no null value in the train dataset')
    else:
        print('There are null values in the train dataset')
        print(df_train[df_train.isnull().sum()!=0])

    #Checking for test dataset
    if df_test.isnull().sum().sum()==0:
        print('There is no null value in the test dataset')
    else:
        print('There are null values in the test dataset')
        print(df_test[df_test.isnull().sum()!=0])
    print('='*80)

    '''
        As we can notice, there are over 300 columns. So we will do feature
        selection and Dimensionality Reduction to reduce the number of features
    '''

    #We plot the heatmap of features of train dataset to see the corr()
    print('Plot the heatmap to see the correlation between features')
    #correlation_plot(df_train)
    print('='*80)
    '''
        It seems like all the features is uncorrelatec with others. So we cann't
        remove any features here
    '''

    #Removing Outliers
    print('Removing Outliers')
    df_train=remove_outlier(df_train)
    print('='*80)

    #Split data
    x_train=df_train.drop(['id','target'],axis=1)
    y_train=df_train['target']
    x_test=df_test.drop(['id'],axis=1)

    #Now, we will use some dimensionality reduction and feature selection
    ##1. Perform Variance Threshold
    print('Perform Variance Threshold')
    train_variance,features_variance=variancethreshold(x_train)
    print('Shape of dataset before variance threshold: ',x_train.shape)
    print('Shape of dataset after variance threshold: ',train_variance.shape)

    #Processing data for Models
    x_train_var=train_variance
    y_train_var=y_train
    x_test_var=df_test[features_variance]

    #Build Model
    y_pred_variance=random_forest(x_train_var,y_train_var,x_test_var)
    print('The result of Variance Threshold: ')
    data_pred_variance=pd.DataFrame(y_pred_variance)
    result_variance=pd.concat([df_test[['id']],data_pred_variance],axis=1)
    result_variance.columns=['id','target']
    print(result_variance)
    print('='*80)

    ##2. Perform Mutual Information - SelectKBest
    print('Perform Mutual Information - SelectKBest')
    train_mutual,features_mutual=mutual(df_train)
    print('Shape of dataset before mutual: ',df_train.shape)
    print('Shape of dataset after mutual: ',train_mutual.shape)

    #Processing data for models
    x_train_mutual=train_mutual
    y_train_mutual=df_train['target']
    x_test_mutual=df_test[features_mutual]

    #Build Model
    y_pred_mutual=random_forest(x_train_mutual,y_train_mutual,x_test_mutual)
    print('The result of Mutual Information: ')
    data_pred_mutual=pd.DataFrame(y_pred_mutual)
    result_mutual=pd.concat([df_test[['id']],data_pred_mutual],axis=1)
    result_mutual.columns=['id','target']
    print(result_mutual)
    print('='*80)

    ##3. Perform Principal Component Analysis
    print('Perform Principal Component Analysis')
    x_pca_train=pca(x_train)
    x_pca_test=pca(x_test)
    print('Shape of dataset before PCA: ',x_train.shape)
    print('Shape of dataset after PCA: ',x_pca_train.shape)

    #Build model
    y_pred_pca=random_forest(x_pca_train,y_train,x_pca_test)
    print('The result of Principal Component Analysis: ')
    data_pred_pca=pd.DataFrame(y_pred_pca)
    result_pca=pd.concat([df_test[['id']],data_pred_pca],axis=1)
    result_pca.columns=['id','target']
    print(result_pca)
    print('='*80)

    ##4. Perform Select From Model
    print('Perform Select From Model')
    x_train_select,features_select=selectfrommodel(x_train,y_train)
    print('Shape of dataset before select from model: ',x_train.shape)
    print('Shape of dataset after select from model: ',x_train_select.shape)

    #Processing data for building model
    x_test_select=df_test[features_select]

    #Build Model
    y_pred_select=logistic_regression(x_train_select,y_train,x_test_select)
    print('The result of Select From Model: ')
    data_pred_select=pd.DataFrame(y_pred_select)
    result_select=pd.concat([df_test[['id']],data_pred_select],axis=1)
    result_select.columns=['id','target']
    print(result_select)
    print('='*80)

    ##5. Recursive Feature Elimination
    print('Perform Recursive Feature Elimination')
    y_pred_recursive=recursive(x_train,y_train,x_test)

    #Print the result
    print('The result of Recursive Feature Elimination: ')
    data_pred_recursive=pd.DataFrame(y_pred_recursive)
    result_recursive=pd.concat([df_test[['id']],data_pred_recursive],axis=1)
    result_recursive.columns=['id','target']
    print(result_recursive)
    print('='*80)

if __name__=='__main__':
    main()