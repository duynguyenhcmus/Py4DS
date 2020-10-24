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
df.rename(columns={'Regional indicator':'Region','Logged GDP per capita':'Logged GDP','Healthy life expectancy':'life expectancy',
'Freedom to make life choices':'Freedom','Perceptions of corruption':'corruption'},inplace=True)

#Filter factors that contribute to the happiness score
feature_column=['Logged GDP', 'Social support', 'life expectancy',
       'Freedom', 'Generosity',
       'corruption']
df_feature=df[feature_column]
df_region=df[['Region','Logged GDP', 'Social support', 'life expectancy',
       'Freedom', 'Generosity',
       'corruption']]

def count_plot():
    #Create a countplot with seaborn
    plt.figure(figsize=(10,7))
    countplot=sns.countplot(x="Region",data=df,linewidth=2,edgecolor=sns.color_palette("dark"))
    countplot.set_xticklabels(countplot.get_xticklabels(),rotation = 90)
    countplot.set_title("Countplot by Region")
    plt.show()

def piechart():
    #Create a Pie Chart
    labels=df.Region.value_counts()
    sizes=df.Region.value_counts().values
    plt.figure(figsize=(7,7))
    plt.pie(sizes,labels=labels,autopct='%1.1f%%')
    plt.title("Pie chart for Region",fontsize=15)
    plt.show()

def heatmap():
    #Create a heatmap to illustrate the correlation
    plt.figure(figsize=(14,12))
    sns.heatmap(df_feature.corr(),linewidth=.1,cmap="YlGnBu",annot=True)
    plt.yticks(rotation=0)
    plt.show()

def pairplot():
    #Create a pairplot with seaborn
    sns.pairplot(df_region,hue='Region')
    plt.show()

def Facetgrid_life():
    #Facetgrid with seaborn for life expectancy feature
    Facetgrid=sns.FacetGrid(df_region,hue="Region",height=6)
    Facetgrid.map(sns.kdeplot,"life expectancy",shade=True)
    Facetgrid.set(xlim=(0,df['life expectancy'].max()))
    Facetgrid.add_legend()
    plt.show()

def histogram_life():
    #Histogram of life expectancy feature
    df['life expectancy'].plot(kind='hist',bins=100,figsize=(20,10),grid=True)
    plt.xlabel("life expectancy")
    plt.legend(loc="upper right")
    plt.title("life expectancy Histogram")
    plt.show()

def boxplot_life():
    #Boxplot of life expectancy feature
    sns.boxplot(x="Region",y="life expectancy",data=df_region)
    plt.xticks(rotation=90)
    plt.show()

#Output:
count_plot() #Create a countplot with seaborn
#Ta thay phan bo giua cac region khong dong deu voi so luong lon nhat la cua Sub-Saharan
#Africa, nhieu hon khoang 8 lan so voi vung co it countries nhat la North America

heatmap() #Create a heatmap to illustrate the correlation
#Bieu do tren cho thay moi tuong quan giua cac yeu to dong gop vao chi so hanh phuc
#Co 8 cap feature co he so tuong quan la duong va 8 cap feature co he so tuong quan
#la am. Trong co feature tuoi tho trung binh va GDP tren moi capita co he so tuong quan
#cao nhat.

pairplot() #Create a pairplot with seaborn
#Ta co the thay tu pairplot la chung ta kho phan biet duoc nhung su khac nhau ve muc do
#hanh phuc dua tren cac chi so giua khu vuc nay voi khu vuc kia. Vi du nhu yeu to Generosity
#voi cac yeu to khac co cac gia tri theo tung khu vuc gom lai thanh mot cum.

piechart() #Create a Pie Chart
#Bieu do the hien ti le phan tram so value cua tung region trong do Sub-Saharan African
#chiem nhieu phan tram nhat voi khoang 25%

Facetgrid_life() #Facetgrid with seaborn for life expectancy feature
#Da so cac region co muc life expectancy cao tu 60 - 75 tuoi
#trong do life expectancy cua vung mau xanh cao nhat
#--> vung nay co dan so gia.

histogram_life() #Histogram of life expectancy feature
#Bieu do histogram the hien phan phoi life expectancy tren the gioi. Tuoi tho trung binh
#cua the gioi da so nam o khoang 67 tuoi.

boxplot_life() #Boxplot of life expectancy feature
#Bieu do cho thay phan bo cua yeu to life expectancy cua tung khu vuc.
#Cac khu vuc nhu Sub-Saharan Africa hay la Asia qua boxplot cho thay du lieu
#phan bo bi lech trai va lech phai.
