#Exercise 4. Exploratory Data Analysis

#Load Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

#Load dataset
path = 'https://raw.githubusercontent.com/duynguyenhcmus/Py4DS/main/Lab2/Py4DS_Lab2_Dataset/HappinessReport2020.csv'
df = pd.read_csv(path)
#print(df.head())

#Print the number of value in each columns
#for i in range(1,17):
    #print(df.iloc[:,i].value_counts())
    #print("*"*20)

print(df.columns)


