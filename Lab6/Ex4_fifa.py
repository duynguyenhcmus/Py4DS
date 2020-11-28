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
    selection=VarianceThreshold(threshold = 100)
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
    rdf=RandomForestClassifier(random_state=10)
    rdf.fit(X_train,y_train.ravel())
    y_pred=rdf.predict(X_test)
    return metrics.accuracy_score(pd.DataFrame(y_pred),y_test)

#Logistic Regression
def logistic_regression(x_train, x_test, y_train, y_test):
    '''
        Purpose: Perform Logistic Regression
        Input: X_train,y_train,X_test,y_test - DataFrame
        Output: the accuracy score of prediction
    '''
    clf = LogisticRegression(solver='liblinear',random_state=10)
    clf = clf.fit(x_train, y_train.ravel())
    y_pred = clf.predict(x_test)
    return metrics.accuracy_score(pd.DataFrame(y_pred),y_test)

#Mutual Information
def mutual(x_train,y_train):
    '''
        Purpose: Perform mutual information for feature selection
        Input: x_train,y_train - Dataframe
        Output: data_out-Dataframe-Transformed DataFrame, 
        features-Series-selected features
    '''
    numerical_x_train = x_train[x_train.select_dtypes([np.number]).columns]
    selection = SelectKBest(mutual_info_classif, k=10)
    data_out=selection.fit_transform(numerical_x_train, y_train.ravel())
    features = x_train.columns[selection.get_support()]
    return data_out,features

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
    rfc = RandomForestClassifier(n_estimators=100,random_state=10)
    select_model =SelectFromModel(estimator=rfc)
    fit = select_model.fit(X_train, y_train.ravel())
    model_features = fit.transform(X_train)
    features =X_train.columns[select_model.get_support()]
    return model_features,features

#Recursive Feature Elimination
def recursive(x_train,y_train,x_test):
    '''
        Purpose: Perform Recursive Feature Elimination
        Input: x_train, y_train, x_test - DataFrame
        Output: y_pred_rfc - Series - Prediction of RFE
    '''
    rfc = RandomForestClassifier(n_estimators=100,random_state=10)
    rfe = RFE(estimator=rfc, n_features_to_select=3)
    rfe.fit(x_train, y_train.ravel())
    y_pred_rfc=rfe.predict(x_test)
    return y_pred_rfc

#Main Function
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
    df=df.drop(['Unnamed: 0','Photo','ID','Flag','Club Logo'],axis=1)
    print('='*80)

    #Split data into train and test dataset
    x=df.drop(['Position'],axis=1)
    y=df['Position']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1,shuffle=True)

    #Convert into DataFrame for Preprocessing data
    x_train=pd.DataFrame(x_train)
    y_train=pd.DataFrame(y_train)
    x_test=pd.DataFrame(x_test)
    y_test=pd.DataFrame(y_test)

    #Preprocessing data
    #With Value features
    x_train['Value']=x_train['Value'].apply(lambda x:x[1:-1]).astype('float64')
    x_test['Value']=x_test['Value'].apply(lambda x:x[1:-1]).astype('float64')

    #With Wage features
    x_train['Wage']=x_train['Wage'].apply(lambda x:x[1:-1])
    x_train['Wage']=x_train['Wage'].apply(lambda x:x[1:-1])
    
    #With Joined features
    x_train['Joined']=x_train['Joined'].apply(lambda x:x[-4:]).astype('float64')
    x_test['Joined']=x_test['Joined'].apply(lambda x:x[-4:]).astype('float64')

    #With Height features
    x_train['Height']=x_train['Height'].apply(lambda x:(int(x[0])*12+int(x[2:]))).astype('float64')
    x_test['Height']=x_test['Height'].apply(lambda x:(int(x[0])*12+int(x[2:]))).astype('float64')

    #With Weight features
    x_train['Weight']=x_train['Weight'].apply(lambda x:x[:-3]).astype('float64')
    x_test['Weight']=x_test['Weight'].apply(lambda x:x[:-3]).astype('float64')

    #With Release Clause
    x_train['Release Clause']=x_train['Release Clause'].apply(lambda x:x[1:-1]).astype('float64')
    x_test['Release Clause']=x_test['Release Clause'].apply(lambda x:x[1:-1]).astype('float64')

    #Label Encoding
    x_train=label_encoder(x_train)
    x_test=label_encoder(x_test)
    y_train=label_encoder(y_train)
    y_test=label_encoder(y_test)

    #Converting to ndarray for Random-Forest and Logistic-Regression
    y_train=np.array(y_train)
    y_test=np.array(y_test)

    '''
        We'll use some features-selection and dimensionality reduction to reduce
        features for building models
    '''

    ##1. Variance Threshold
    print('Perform Variance Threshold')
    train_variance,features_variance=variancethreshold(x_train)
    print('Shape of dataset before variance threshold: ',x_train.shape)
    print('Shape of dataset after variance threshold: ',train_variance.shape)

    #Processing data for Models
    x_train_var=train_variance
    y_train_var=y_train
    x_test_var=x_test[features_variance]
    y_test_var=y_test

    #Build Model
    accuracy_var_rdf=random_forest(x_train_var,y_train_var,x_test_var,y_test_var)
    accuracy_var_lr=logistic_regression(x_train_var,x_test_var,y_train_var,y_test_var)
    print('Accuracy of Random Forest after Variance Threshold: ',accuracy_var_rdf)
    print('Accuracy of Logistic Regression after Variance Threshold: ',accuracy_var_lr)
    print('='*80)
    '''
        Accuracy of Random Forest after Variance Threshold:  0.4482875551034249
        Accuracy of Logistic Regression after Variance Threshold:  0.3669040352661919
    '''

    ##2. Mutual Information
    print('Perform Mutual Information - SelectKBest')
    train_mutual,features_mutual=mutual(x_train,y_train)
    print('Shape of dataset before mutual: ',x_train.shape)
    print('Shape of dataset after mutual: ',train_mutual.shape)

    #Processing data for models
    x_train_mutual=train_mutual
    y_train_mutual=y_train
    x_test_mutual=x_test[features_mutual]
    y_test_mutual=y_test

    #Build Model
    accuracy_mutual_rdf=random_forest(x_train_mutual,y_train_mutual,x_test_mutual,y_test_mutual)
    accuracy_mutual_lr=logistic_regression(x_train_mutual,x_test_mutual,y_train_mutual,y_test_mutual)
    print('Accuracy of Random Forest after Mutual Information: ',accuracy_mutual_rdf)
    print('Accuracy of Logistic Regression after Mutual Information: ',accuracy_mutual_lr)
    print('='*80)
    '''
        Accuracy of Random Forest after Mutual Information:  0.3553747032892506
        Accuracy of Logistic Regression after Mutual Information:  0.340793489318413
    '''

    ##3. Principal Component Analysis
    print('Perform Principal Component Analysis')
    x_pca_train=pca(x_train)
    x_pca_test=pca(x_test)
    print('Shape of dataset before PCA: ',x_train.shape)
    print('Shape of dataset after PCA: ',x_pca_train.shape)

    #Build model
    accuracy_pca_rdf=random_forest(x_pca_train,y_train,x_pca_test,y_test)
    accuracy_pca_lr=logistic_regression(x_pca_train,x_pca_test,y_train,y_test)
    print('Accuracy of Random Forest after PCA: ',accuracy_pca_rdf)
    print('Accuracy of Logistic Regression after PCA: ',accuracy_pca_lr)
    print('='*80)
    '''
        Accuracy of Random Forest after PCA:  0.1017293997965412
        Accuracy of Logistic Regression after PCA:  0.1271617497456765
    '''

    ##4. Select From Model
    print('Perform Select From Model')
    train_select,features_select=selectfrommodel(x_train,y_train)
    print('Shape of dataset before select from model: ',x_train.shape)
    print('Shape of dataset after select from model: ',train_select.shape)

    #Processing data for building model
    x_train_select=train_select
    y_train_select=y_train
    x_test_select=x_test[features_select]
    y_test_select=y_test

    #Build Model
    accuracy_select_rdf=random_forest(x_train_select,y_train_select,x_test_select,y_test_select)
    accuracy_select_lr=logistic_regression(x_train_select,x_test_select,y_train_select,y_test_select)
    print('Accuracy of Random Forest after Select From Model: ',accuracy_select_rdf)
    print('Accuracy of Logistic Regression after Select From Model: ',accuracy_select_lr)
    print('='*80)
    '''
        Accuracy of Random Forest after Select From Model:  0.5079688029840624
        Accuracy of Logistic Regression after Select From Model:  0.46863343506273314
    '''

    ##5. Recursive Feature Elimination
    print('Perform Recursive Feature Elimination')
    pred_recursive_rdf=recursive(x_train,y_train,x_test)
    accuracy_recursive_rdf=metrics.accuracy_score(pd.DataFrame(pred_recursive_rdf),y_test)
    print('Accuracy of Random Forest after RFE: ',accuracy_recursive_rdf)
    print('='*80)
    '''
        Accuracy of Random Forest after RFE:  0.32451678535096645
    '''

if __name__=='__main__':
    main()