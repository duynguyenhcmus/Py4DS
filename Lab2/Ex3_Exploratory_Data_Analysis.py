# Exercise 3. Exploratory Data Analysis
# Load Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

# Load dataset
path = '/home/dinh_anh_huy/Py4DS/Lab2/Py4DS_Lab2_Dataset/creditcard.csv'
df = pd.read_csv(path)
print(df.head())
print(df.info())