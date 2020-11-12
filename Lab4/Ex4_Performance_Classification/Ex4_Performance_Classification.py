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

#Import Scikit - Learn for model selection
from sklearn.model_selection import train_test_split

#

def main():
    '''
        Purpose: Processing all the steps with the dataset
        Param: None
        Output: The clustering models for prediction
    '''
    ##1. Loading Dataset
    path='https://raw.githubusercontent.com/duynguyenhcmus/Py4DS/main/Lab4/Py4DS_Lab4_Dataset/xAPI-Edu-Data.csv'
    df=pd.read_csv(path)
    print(df.head())