#Spam_Classification
#Load libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#Pandas Options
pd.set_option('display.max_colwidth',1000,'display.max_rows',None,\
             'display.max_columns',None)

#Plotting options
mpl.style.use('ggplot')
sns.set(style='whitegrid')

#Path of dataset 
path='https://raw.githubusercontent.com/duynguyenhcmus/Pythonfordatascience/main/spam.csv'
dataset_pd=pd.read_csv(path)
dataset_np=np.genfromtxt(path,delimiter=',')
print(dataset_pd.head())

