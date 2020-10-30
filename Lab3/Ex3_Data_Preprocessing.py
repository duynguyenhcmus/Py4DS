## Load libraries ##
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
#import ML models
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Import libraries for cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# Import library for label encoder
from sklearn.preprocessing import LabelEncoder
# Import libraries nomalizer
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, MinMaxScaler

def plotting_correlation(data):
    #Plotting correlation of Features
    plt.figure(figsize = (14,14))
    plt.title('Correlation plot using Pearson plot')
    corr = data.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.1, cmap="Reds", annot=True)
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(15,8))
    corr = corr.filter(items = ['Man of the Match'])
    sns.heatmap(corr, annot=True)
    plt.tight_layout()
    plt.show()

def label_endcoder(data):
    label = LabelEncoder()
    data_colums = data.dtypes.pipe(lambda X: X[X=='object']).index
    for col in data_colums:
        data[col] = label.fit_transform(data[col])
    return data

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

def random_forest_classifier(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier(random_state=1)
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
    path = 'https://raw.githubusercontent.com/duynguyenhcmus/Py4DS/main/Lab3/Py4DS_Lab3_Dataset/FIFA2018Statistics.csv'
    df = pd.read_csv(path)
    print(df.head())

    ## 2. EDA ##
    # % Missing values
    missing_values = df.isnull().sum().sort_values(ascending = False)
    percentage_missing_values = (missing_values/len(df))*100
    missing_values = pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing'])
    print(missing_values)
    # Encode target variable 'Man of the match' into binary format
    df['Man of the Match'] = df['Man of the Match'].map({'Yes': 1, 'No': 0})
    # Plotting correlation of Features
    plotting_correlation(df)
    # ========================================================= #
    '''
    - As 'Own goal Time' and 'Own goals' are having > 90% missing values, filling them with any combination will lead predictive model to false direction. 
    So, dropping them is the best option.
    - '1st Goal' is negligebly correlated with 'Man of the Match', hence, dropping this should not have any impact.
    - Dropping 'Corners', 'Fouls Committed' and 'On-Targets' will remove high correlated elements and remove chances of multi-collinearity. 
    These features are selected based on their low collinearity with 'Man of the Match' and high collinearity with other features.
    - Dropping 'Date' as it should definately not impact a player formance.
    '''
    # ========================================================= #
    cols = ['Own goal Time', 'Own goals', '1st Goal', 'Corners', 'Fouls Committed', 'On-Target', 'Date']
    df.drop(cols, axis = 1, inplace=True)
    print(df.columns)
    data = label_endcoder(df)
    print(data.head())

    ## 3. Cleaning data ##
    data_cleaned = cleaning_data(df)
    # Box plot each columns to determine the outlier values
    boxplot = data_cleaned[[col for col in data_cleaned.columns]]
    f, ax = plt.subplots(ncols = 4, nrows = int(len(boxplot.columns)/4), figsize=(10,len(boxplot.columns)/3))
    for i, c in zip(ax.flatten(), boxplot.columns):
        sns.boxplot(boxplot[c], ax = i)
    f.tight_layout()
    plt.show()
    # ========================================================= #
    '''
        As we see on boxplots, we can detect outliers. We need to remove them to avoid leading false direction.
    '''
    # ========================================================= #
    data_cleaned = remove_outlier(data_cleaned)
    data_cleaned = pd.get_dummies(data_cleaned)

    ## 4. Splitting Data ##
    # Split dataset into training set and test set
    # 70% training and 30% test
    X = data_cleaned.drop(['Man of the Match'], axis=1)
    y = data_cleaned['Man of the Match']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    ## 5. Build Logistic Regression Model ##
    accuracy = random_forest_classifier(X_train, X_test, y_train, y_test)
    print("The accuracy of random forest classifier algorithm before scaling data: ", accuracy)

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
    accuracy_normal = random_forest_classifier(x_train_normal, x_test_normal, y_train, y_test)
    accuracy_standard = random_forest_classifier(x_train_standard, x_test_standard, y_train, y_test)
    accuracy_robust = random_forest_classifier(x_train_robust, x_test_robust, y_train, y_test)
    accuracy_minmax = random_forest_classifier(x_train_minmax, x_test_minmax, y_train, y_test)
    print("The accuracy of random forest classifier algorithm after using normalizer: ", accuracy_normal)
    print("The accuracy of random forest classifier algorithm after using standard scaler: ", accuracy_standard)
    print("The accuracy of random forest classifier algorithm after using robust scaler: ", accuracy_robust)
    print("The accuracy of random forest classifier algorithm after using min max scaler: ", accuracy_minmax)
    # ========================================================= #
    '''
    With random_state = 1, 
        The accuracy of random forest classifier algorithm before scaling data:  0.6875
        The accuracy of random forest classifier algorithm after using normalizer:  0.75
        The accuracy of random forest classifier algorithm after using standard scaler:  0.75
        The accuracy of random forest classifier algorithm after using robust scaler:  0.75
        The accuracy of random forest classifier algorithm after using min max scaler:  0.6875
    '''
    # ========================================================= #

if __name__ == '__main__':
    main()
