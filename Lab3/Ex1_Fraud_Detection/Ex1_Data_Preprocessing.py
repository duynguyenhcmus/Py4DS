'''
    Describe
    ! Note: comment to strange lib
'''

## Load libraries ##
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
#import ML models
from sklearn.linear_model import LogisticRegression
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Import library for label encoder
from sklearn.preprocessing import LabelEncoder
# Import libraries nomalizer
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, MinMaxScaler

def plotting_correlation(data):
    '''
        Purpose:
        Param / Input:
            + Type?
            + To do?
            + Describe
        Output / Return:
            + Type?
            + Meaning?
    '''
    print(data.corr())
    #Plotting correlation of Features
    plt.figure(figsize = (14,14))
    plt.title('Correlation plot of Credit Card Transactions features using Pearson plot')
    corr = data.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.1, cmap="Reds")
    plt.savefig("heatmap-correlation.jpg")

def cleaning_data(data):
    print("> Shape of data before cleaning: ", data.shape)
    # Remove duplicate values
    data.drop_duplicates(subset = data.columns.values[:-1], keep = 'first', inplace = True)
    print("> Shape of data after drop duplicate values: ", data.shape)
    # Remove missing values
    data.dropna()
    print("> Shape of data after drop missing values: ", data.shape)
    return data

def remove_outlier(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data_out = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    print('> Shape of data after handling outlier values: ', data_out.shape)
    return data_out

def logistic_regression(x_train, x_test, y_train, y_test):
    clf = LogisticRegression(solver='liblinear')
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return metrics.accuracy_score(y_test, y_pred)

def scaling_data(x_train, x_test, method='None'):
    if method == 'Normalizer':
        x_scale = Normalizer() 
    elif method == 'StandardScaler':
        x_scale = StandardScaler()
    elif method == 'MinMaxScaler':
        x_scale = MinMaxScaler()
    elif method == 'RobustScaler':
        x_scale = RobustScaler()
    x_scale.fit(x_train)
    x_train_scaled = x_scale.transform(x_train)
    x_scale.fit(x_test)
    x_test_scaled = x_scale.transform(x_test)
    return x_train_scaled, x_test_scaled

def main():
    ## 1. Loading Data ##
    # Load dataset
    path = '/home/dinh_anh_huy/GitHub/2020-2021/semester-1/python-for-data-science/Lab2/creditcard.csv'
    df = pd.read_csv(path)
    pd.set_option("display.max_columns", 100)
    print(df.head())

    ## 2. EDA ##
    # Plotting correlation of Features
    plotting_correlation(df)
    # ========================================================= #
    '''
    There is no notable correlation between features V1-V28. 
    There are certain correlations between some of these features and Time (inverse correlation with V3) and Amount (direct correlation with V7 and V20, inverse correlation with V1 and V5).
    '''
    # ========================================================= #
    
    ## 3. Cleaning data ##
    data_cleaned = cleaning_data(df)

    ## 4. Splitting Data ##
    # Split dataset into training set and test set
    # 70% training and 30% test
    X = data_cleaned.drop(['Class'], axis=1)
    y = data_cleaned['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    ## 4. Build Logistic Regression Model ##
    accuracy = logistic_regression(X_train, X_test, y_train, y_test)
    print("The accuracy of logistic regression algorithm before scaling data: ", accuracy)

    ## 6. Scaling data ##
    # Use Normalizer to scale
    x_train_normal, x_test_normal = scaling_data(X_train, X_test, method='Normalizer')
    # Use Standard Scaler to scale
    x_train_standard, x_test_standard = scaling_data(X_train, X_test, method='StandardScaler')
    # Use Robust Scaler to scale
    x_train_robust, x_test_robust = scaling_data(X_train, X_test, method='RobustScaler')
    # Use Min Max Scaler to scale
    x_train_minmax, x_test_minmax = scaling_data(X_train, X_test, method='MinMaxScaler')

    ## 7. Build Logistic Regression Model after scaling data ##
    accuracy_normal = logistic_regression(x_train_normal, x_test_normal, y_train, y_test)
    accuracy_standard = logistic_regression(x_train_standard, x_test_standard, y_train, y_test)
    accuracy_robust = logistic_regression(x_train_robust, x_test_robust, y_train, y_test)
    accuracy_minmax = logistic_regression(x_train_minmax, x_test_minmax, y_train, y_test)
    print("The accuracy of logistic regression algorithm after using normalizer: ", accuracy_normal)
    print("The accuracy of logistic regression algorithm after using standard scaler: ", accuracy_standard)
    print("The accuracy of logistic regression algorithm after using robust scaler: ", accuracy_robust)
    print("The accuracy of logistic regression algorithm after using min max scaler: ", accuracy_minmax)
    # ========================================================= #
    '''
    With random_state = 1, 
        The accuracy of logistic regression algorithm before scaling data: 0.999071876688832
        The accuracy of logistic regression algorithm after using normalizer:  0.9985314504570126
        The accuracy of logistic regression algorithm after using standard scaler:  0.9992715994266782
        The accuracy of logistic regression algorithm after using robust scaler:  0.9992598510303343
        The accuracy of logistic regression algorithm after using min max scaler:  0.9991541154632393
    '''
    # ========================================================= #

if __name__ == '__main__':
    main()
