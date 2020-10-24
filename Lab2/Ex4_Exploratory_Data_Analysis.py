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

#print(df.columns)

#Rename columns of the dataset
df.rename(columns={'Logged GDP per capita':'Logged GDP','Healthy life expectancy':'life expectancy',
'Freedom to make life choices':'Freedom','Perceptions of corruption':'corruption'},inplace=True)

#Filter factors that contribute to the happiness score
feature_column=['Logged GDP', 'Social support', 'life expectancy',
       'Freedom', 'Generosity',
       'corruption']
df_feature=df[feature_column]
df_region=df[['Regional indicator','Logged GDP', 'Social support', 'life expectancy',
       'Freedom', 'Generosity',
       'corruption']]

def count_plot():
    #Create a countplot with seaborn
    plt.figure(figsize=(10,7))
    countplot=sns.countplot(x="Regional indicator",data=df,linewidth=2,edgecolor=sns.color_palette("dark"))
    countplot.set_xticklabels(countplot.get_xticklabels(),rotation = 90)
    countplot.set_title("Countplot by Region")
    plt.show()

def heatmap():
    #Create a heatmap to illustrate the correlation
    plt.figure(figsize=(14,12))
    sns.heatmap(df_feature.corr(),linewidth=.1,cmap="YlGnBu",annot=True)
    plt.yticks(rotation=0)
    plt.show()

def pairplot():
    #Create a pairplot with seaborn
    sns.pairplot(df_region,hue='Regional indicator')
    plt.show()


#Output:
count_plot() #Create a countplot with seaborn
#Ta thay phan bo giua cac region khong dong deu voi so luong lon nhat la cua Sub-Saharan
# Africa, nhieu hon khoang 8 lan so voi vung co it countries nhat la North America

heatmap() #Create a heatmap to illustrate the correlation
#Bieu do tren cho thay moi tuong quan giua cac yeu to dong gop vao chi so hanh phuc
#Co 8 cap feature co he so tuong quan la duong va 8 cap feature co he so tuong quan
#la am. Trong co feature tuoi tho trung binh va GDP tren moi capita co he so tuong quan
#cao nhat.

pairplot()


