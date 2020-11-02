## Load libraries ##
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
#import ML model
from sklearn.linear_model import LogisticRegression, LinearRegression
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
    plt.title('Correlation Between Different Variables\n')
    corr = data.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.1, cmap="Reds", annot=True)
    plt.tight_layout()
    plt.savefig("heatmap-correlation.jpg")

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

def linear_regression(x_train, x_test, y_train, y_test):
    clf = LinearRegression()
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))

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
    path = 'https://raw.githubusercontent.com/duynguyenhcmus/Py4DS/main/Lab3/Py4DS_Lab3_Dataset/AB_NYC_2019.csv'
    df = pd.read_csv(path)
    pd.set_option("display.max_columns", 100)
    print(df.head())

    ## 2. EDA ##
    # % Missing values
    missing_values = df.isnull().sum().sort_values(ascending = False)
    percentage_missing_values = (missing_values/len(df))*100
    missing_values = pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing'])
    print(missing_values)
    # ========================================================= #
    '''
        - In our case, missing data that is observed does not need too much special treatment. 
        - Columns "id", "name" and "host_name" are irrelevant and insignificant to our data analysis.
        - Columns "last_review" and "review_per_month" need very simple handling. 
        - To elaborate, "last_review" is date; if there were no reviews for the listing - date simply will not exist. 
        In our case, this column is irrelevant and insignificant therefore appending those values is not needed. 
        - For "review_per_month" column we can simply append it with 0.0 for missing values; we can see that in "number_of_review" that column will have a 0, therefore following this logic with 0 total reviews there will be 0.0 rate of reviews per month. 
        - Therefore, let's proceed with removing columns that are not important and handling of missing data.
    '''
    # ========================================================= #
    cols = ['id', 'name', 'host_name', 'last_review']
    df.drop(cols, axis = 1, inplace=True)
    print(df.columns)
    # Replacing all missing values in “review_per_month" with 0
    df.reviews_per_month.fillna(0, inplace=True)
    print(df.isnull().any())
    # Plotting correlation of Features
    plotting_correlation(df)
    # ========================================================= #
    '''
        From the graph above, we know that there is not a strong correlation except review_per_month and number_of_review.
    '''
    # ========================================================= #
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
    plt.savefig("boxplot-outliers.jpg")
    # ========================================================= #
    '''
        As we see on boxplots, we can detect outliers. We need to remove them to avoid leading false direction.
    '''
    # ========================================================= #
    data_cleaned = remove_outlier(data_cleaned)

    ## 4. Splitting Data ##
    # Split dataset into training set and test set
    # 80% training and 20% test
    X = data_cleaned.drop(['price'], axis=1)
    y = data_cleaned['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    ## 5. Build Logistic Regression Model ##
    accuracy = linear_regression(X_train, X_test, y_train, y_test)
    print("The mean squared error of linear regression algorithm before scaling data: ", accuracy)

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
    accuracy_normal = linear_regression(x_train_normal, x_test_normal, y_train, y_test)
    accuracy_standard = linear_regression(x_train_standard, x_test_standard, y_train, y_test)
    accuracy_robust = linear_regression(x_train_robust, x_test_robust, y_train, y_test)
    accuracy_minmax = linear_regression(x_train_minmax, x_test_minmax, y_train, y_test)
    print("The mean squared error of linear regression algorithm after using normalizer: ", accuracy_normal)
    print("The mean squared error of linear regression algorithm after using standard scaler: ", accuracy_standard)
    print("The mean squared error of linear regression algorithm after using robust scaler: ", accuracy_robust)
    print("The mean squared error of linear regression algorithm after using min max scaler: ", accuracy_minmax)
    # ========================================================= #
    '''
    With random_state = 1, 
        The mean squared error of linear regression algorithm before scaling data:  49.23607050334739
        The mean squared error of linear regression algorithm after using normalizer:  68.4111733165353
        The mean squared error of linear regression algorithm after using standard scaler:  49.246496857803805
        The mean squared error of linear regression algorithm after using robust scaler:  86.38395049787577
        The mean squared error of linear regression algorithm after using min max scaler:  49.214451762216825
    '''
    # ========================================================= #

if __name__ == '__main__':
    main()
