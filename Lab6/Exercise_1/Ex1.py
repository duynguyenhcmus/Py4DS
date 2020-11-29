#Load libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

def constant_features(x_train, x_test, threshold=0):
    """
    Removing Constant Features using Variance Threshold
    Input: threshold parameter to identify the variable as constant
            train data (pd.Dataframe) 
            test data (pd.Dataframe)
    Output: train data, test data after applying filter methods
    """
    # import and create the VarianceThreshold object.
    from sklearn.feature_selection import VarianceThreshold
    vs_constant = VarianceThreshold(threshold)

    # select the numerical columns only.
    numerical_x_train = x_train[x_train.select_dtypes([np.number]).columns]

    # fit the object to our data.
    vs_constant.fit(numerical_x_train)

    # get the constant colum names.
    constant_columns = [column for column in numerical_x_train.columns
                        if column not in numerical_x_train.columns[vs_constant.get_support()]]

    # detect constant categorical variables.
    constant_cat_columns = [column for column in x_train.columns 
                            if (x_train[column].dtype == "O" and len(x_train[column].unique())  == 1 )]

    # concatenating the two lists.
    all_constant_columns = constant_cat_columns + constant_columns

    print(">> All constant features in data: ")        
    print(all_constant_columns)

    # drop the constant columns
    x_train = x_train.drop(labels=all_constant_columns, axis=1)
    x_test = x_test.drop(labels=all_constant_columns, axis=1)
    return x_train, x_test

def quasi_constant_features(x_train, x_test, threshold=0):
    """
    Removing Quasi-Constant Features
    Input:  threshold parameter to identify the variable as constant
            train data (pd.Dataframe) 
            test data (pd.Dataframe)
    Output: train data, test data after applying filter methods
    """
    # create empty list
    quasi_constant_feature = []

    # loop over all the columns
    for feature in x_train.columns:
        # calculate the ratio.
        predominant = (x_train[feature].value_counts() / np.float(len(x_train)))\
                      .sort_values(ascending=False).values[0]
        # append the column name if it is bigger than the threshold
        if predominant >= threshold:
            quasi_constant_feature.append(feature)

    print(">> All quasi-constant features in data: ")        
    print(quasi_constant_feature)

    # drop the quasi constant columns
    x_train = x_train.drop(labels=quasi_constant_feature, axis=1)
    x_test = x_test.drop(labels=quasi_constant_feature, axis=1)
    return x_train, x_test

def duplicated_feature(x_train, x_test):
    """
    Removing Duplicated Features
    Input:  train data (pd.Dataframe) 
            test data (pd.Dataframe)
    Output: train data, test data after applying filter methods
    """
    # transpose the feature matrice
    train_features_T = x_train.T

    # print the number of duplicated features
    print(train_features_T.duplicated().sum())

    # select the duplicated features columns names
    duplicated_columns = train_features_T[train_features_T.duplicated()].index.values

    # drop those columns
    x_train = x_train.drop(labels=duplicated_columns, axis=1)
    x_test= x_test.drop(labels=duplicated_columns, axis=1)
    return x_train, x_test

def correlation_filter(x_train, x_test, method='pearson'):
    """
    Removing Features that have high correlation
    Input:  method : {'pearson', 'kendall', 'spearman'} or callable
                Method of correlation:
                    - pearson : standard correlation coefficient
                    - kendall : Kendall Tau correlation coefficient
                    - spearman : Spearman rank correlation
                    - callable: callable with input two 1d ndarrays
                        and returning a float. Note that the returned matrix from 
                        corr will have 1 along the diagonals and will be symmetric
                        regardless of the callable’s behavior.
            train data (pd.Dataframe) 
            test data (pd.Dataframe)
    Output: train data, test data after applying filter methods
    """
    # creating set to hold the correlated features
    corr_features = set()

    # create the correlation matrix (default to pearson)
    corr_matrix = x_train.corr(method=method)

    # optional: display a heatmap of the correlation matrix
    print("Display a heatmap of the correlation matrix.")
    plt.figure(figsize=(11,11))
    sns.heatmap(corr_matrix)
    plt.savefig("Heatmap_correlation.jpg")

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                colname = corr_matrix.columns[i]
                corr_features.add(colname)
                
    x_train = x_train.drop(labels=corr_features, axis=1)
    x_test = x_test.drop(labels=corr_features, axis=1)
    return x_train, x_test

def statistical_ranking_filter(number_feature, x_train, x_test, y_train, y_test, method='MI'):
    """
    Perform statistical and ranking filter method
    Input:  method : {'MI', 'chi2'} - default: 'MI'
                Method of statistical & ranking filter:
                    - MI : mutual information filter
                    - chi2 : Chi-Square score
            train data (pd.Dataframe) 
            test data (pd.Dataframe)
    Output: train data, test data after applying filter methods
    """
    from sklearn.feature_selection import SelectKBest

    if method == 'MI':
        # import the required functions and object.
        from sklearn.feature_selection import mutual_info_classif

        # get only the numerical features.
        numerical_x_train = x_train[x_train.select_dtypes([np.number]).columns]
        numerical_x_test = x_test[x_test.select_dtypes([np.number]).columns]

        # create the SelectKBest with the mutual info strategy.
        selection_train = SelectKBest(mutual_info_classif, k=number_feature).fit(numerical_x_train, y_train)
        selection_test = SelectKBest(mutual_info_classif, k=number_feature).fit(numerical_x_test, y_test)

    elif method == 'chi2':
        # import the required functions and object.
        from sklearn.feature_selection import chi2

        # apply the chi2 score on the data and target (target should be binary).  
        selection_train = SelectKBest(chi2, k=number_feature).fit(x_train, y_train)
        selection_test = SelectKBest(chi2, k=number_feature).fit(x_test, y_test)
    
    return selection_train.transform(x_train), selection_test.transform(x_test)

def selection_from_model(model, x_train, x_test, y_Train):
    """
    Using Select From Model 
    Input:  model (object):  model from sklearn
            train data (pd.Dataframe) 
            test data (pd.Dataframe)
    Output: train data, test data after applying filter methods
    """
    from sklearn.feature_selection import SelectFromModel
    # feature extraction
    select_model = SelectFromModel(model)
    # fit on train set
    fit = select_model.fit(x_train, y_train)
    # transform train set
    return fit.transform(x_train), fit.transform(x_test)

def PCA(x_train, x_test, n_components=2):
    """
    Perform Principal Component Analysis (PCA)
    Input:  n_components (int/float/None/str):  number of components to keep
            train data (pd.Dataframe) 
            test data (pd.Dataframe)
    Output: train data, test data after applying filter methods
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    pca.fit(x_train)
    return pca.transform(x_train), pca.transform(x_test)

def RFE(method, x_train, x_test, y_train, y_test):
    """
    Perform recursive feature elimination (RFE)
    Input:  model (object):  model from sklearn
            train data (pd.Dataframe) 
            test data (pd.Dataframe)
    Output: train data, test data after applying filter methods
    """
    from sklearn.feature_selection import RFE
    # define model
    rfe = RFE(estimator=method, n_features_to_select=3)
    # fit the model
    rfe.fit(x_train, y_train)
    # transform the data
    return rfe.transform(x_train), rfe.transform(x_test)

def main():
    # ======================= Test constant features filter =======================
    print("="*80)
    # Create data to test
    data = pd.DataFrame({'a' : [0,1,2,3], 'b' : [0,0,0,0], 'c' : [4,3,2,1], 'd' : [0,1,0,1]})
    print("Testing data: ")
    print(data)
    print('='*80)
    print(">> Information of data: ")
    print(data.info())

    # Split data
    X, y = data.drop('d', axis = 1), data['d']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    # Display data before removing constant features
    print('='*80)
    print('>> Before removing constant features:')
    print('>> X_train:')
    print(X_train)
    print('>> X_test:')
    print(X_test)
    print("="*80)

    # Remove constant features
    x_train, x_test = constant_features(x_train=X_train, x_test=X_test)

    # Display data after applying method
    print('>> After removing constant features:')
    print('>> X_train:')
    print(x_train)
    print('>> X_test:')
    print(x_test)

    # ======================= Test quasi-constant features filter =======================
    print("="*80)
    # Create data to test
    data = pd.DataFrame({'a' : [0,1,2,3], 'b' : [0,0,1,0], 'c' : [4,3,2,1], 'd' : [0,1,0,1]})
    print("Testing data: ")
    print(data)
    print('='*80)
    print(">> Information of data: ")
    print(data.info())

    # Split data
    X, y = data.drop('d', axis = 1), data['d']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Display data before removing quasi-constant features
    print('='*80)
    print('>> Before removing quasi-constant features:')
    print('>> X_train:')
    print(X_train)
    print('>> X_test:')
    print(X_test)
    print("="*80)

    # Remove constant features
    x_train, x_test = quasi_constant_features(x_train=X_train, x_test=X_test, threshold=0.8)

    # Display data after applying method
    print('>> After removing quasi-constant features:')
    print('>> X_train:')
    print(x_train)
    print('>> X_test:')
    print(x_test)

    # ======================= Test duplicated_features function =======================
    print("="*80)
    # Create data to test
    data = pd.DataFrame({'a' : [1,1,2,3], 'b' : [1,1,0,0], 'c' : [1,1,2,3], 'd' : [0,1,0,1]})
    print("Testing data: ")
    print(data)
    print('='*80)
    print(">> Information of data: ")
    print(data.info())

    # Split data
    X, y = data.drop('d', axis = 1), data['d']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Display data before removing duplicated features
    print('='*80)
    print('>> Before removing duplicated features:')
    print('>> X_train:')
    print(X_train)
    print('>> X_test:')
    print(X_test)
    print("="*80)

    # Remove constant features
    x_train, x_test = duplicated_feature(x_train=X_train, x_test=X_test)

    # Display data after applying method
    print('>> After removing duplicated features:')
    print('>> X_train:')
    print(x_train)
    print('>> X_test:')
    print(x_test)    

    # ======================= Test correlation_filter function =======================
    # Set random state for random data
    rds = np.random.RandomState(0)
    # Create data to test
    data = pd.DataFrame({'a' : rds.randn(5), 'b' : rds.randn(5),\
                            'd' : rds.binomial(1, 0.3, size = 5)})
    # Append 2 columns correlation with columns a and b
    data['c'] = 10*data['a']
    data['e'] = 10*data['a'] + data['b']
    print("Testing data: ")
    print(data)
    print('='*80)
    print(">> Information of data: ")
    print(data.info())

    # Split data
    X, y = data.drop('d', axis = 1), data['d']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Display data before removing correlated features
    print('='*80)
    print('>> Before removing correlated features:')
    print('>> X_train:')
    print(X_train)
    print('>> X_test:')
    print(X_test)
    print("="*80)

    # Remove constant features
    x_train, x_test = correlation_filter(x_train=X_train, x_test=X_test)

    # Display data after applying method
    print('>> After removing correlated features:')
    print('>> X_train:')
    print(x_train)
    print('>> X_test:')
    print(x_test)    

    # ======================= Test mutual information method =======================
    # Create data to test
    data = pd.DataFrame({'a' : rds.randn(10), 'b' : rds.randn(10),\
                            'c' : rds.randn(10), 'e' : rds.randn(10),\
                            'd' : rds.binomial(1, 0.3, size = 10)})
    data['f'] = 10*data['a']
    data['g'] = 10*data['a'] + data['b']
    print("Testing data: ")
    print(data)
    print('='*80)
    print(">> Information of data: ")
    print(data.info())

    # Split data
    X, y = data.drop('d', axis = 1), data['d']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Display data before removing correlated features
    print('='*80)
    print('>> Before applying mutual information method:')
    print('>> X_train:')
    print(X_train)
    print('>> X_test:')
    print(X_test)
    print("="*80)

    # Perform mutual information method
    x_train, x_test = statistical_ranking_filter(4, X_train, X_test, y_train, y_test)

    # Display data after applying method
    print('>> After applying mutual information method:')
    print('>> X_train:')
    print(x_train)
    print('>> X_test:')
    print(x_test)    


if __name__ == "__main__":
    main()