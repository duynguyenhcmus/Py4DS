'''
    Lab 8: Model Selection

    Using model selection technical to discover a well-performing model 
    configuration for the sonar dataset.

    The sonar dataset is a standard machine learning dataset comprising 208 
    rows of data with 60 numerical input variables and a target variable with
    two class values, e.g. binary classification. 

    The dataset involves predicting whether sonar returns indicate a rock or 
    simulated mine.
''' 
#Import necessary library
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#Principal Component Analysis
def pca(data):
    '''
        Purpose: Perform Principal Component Analysis
        Input: data - Dataframe
        Output: x_pca - Dataframe - has been dimensionality-reduced
    '''
    pca = PCA(n_components=10)
    pca.fit(data)
    x_pca = pca.transform(data)
    return x_pca

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
    print('Shape of data before handling outlier values: ', data.shape)
    print('Shape of data after handling outlier values: ', data_out.shape)
    return data_out

#Main function
def main():
    #Load the Dataset
    url_data='https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
    data=pd.read_csv(url_data,header=None)
    print(data.head())
    print('='*100)

    #Pandas Options
    pd.set_option("display.max_columns", 100,"display.max_rows",100)

    #Explore the dataset
    #Print the shape of the dataset
    print('Shape of the dataset: ',data.shape)
    print(data.iloc[:,60].value_counts())
    print('='*100)
    '''
        As we can see, there are 208 rows and 61 columns. Respectively, there
        are 60 features with 111 rows classified as Mine and 97 rows for Rock.

        There are 60 features. So we'll use feature-selection to reduce them.
    '''

    #Cheking for null value in the dataset
    print('Checking for null value in the dataset')
    if data.isnull().sum().sum()==0:
        print('There is no null value in the dataset')
    else:
        print('There are null values in the dataset')
        print(data[data.isnull().sum()!=0])
    print('='*100)

    #Perform Label-encoding
    data=label_encoder(data)

    #Perform outlier-removing
    data=remove_outlier(data)
    print('='*100)

    #Split Data
    X=data.iloc[:,:-1]
    y=data.iloc[:,-1]

    #Do Principal Component Analysis as pca
    print('Perform Principal Component Analysis')
    X_pca=pca(X)
    print('Shape of dataset before PCA: ',X.shape)
    print('Shape of dataset after PCA: ',X_pca.shape)
    print('='*100)

    #Perform GridSearchCV
    print('Perform GridSearchCV')

    #Define model
    model = Ridge()

    #Define evaluation
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    #Define search space
    space = dict()
    space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
    space['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    space['fit_intercept'] = [True, False]
    space['normalize'] = [True, False]

    #Define search
    search = GridSearchCV(model, space, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv)
    
    #Execute search
    result = search.fit(X_pca, y)

    #Summarize result
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    print('='*100)
    '''
        Summary:
            Best Score: -0.1618722594173935
            Best Hyperparameters: {'alpha': 0.1, 'fit_intercept': True, 'normalize': True, 'solver': 'lsqr'}
    '''

if __name__=='__main__':
    main()