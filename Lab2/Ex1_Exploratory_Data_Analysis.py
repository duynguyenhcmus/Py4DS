#Exercise 1. Exploratory Data Analysis
#Load Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

#Load dataset
path = 'https://raw.githubusercontent.com/duynguyenhcmus/Py4DS/main/Lab2/Py4DS_Lab2_Dataset/xAPI-Edu-Data.csv'
df = pd.read_csv(path)
#print(df.head())

#Print the number of value in each columns
#for i in range(1,17):
    #print(df.iloc[:,i].value_counts())
    #print("*"*20)

df.rename(index=str,columns={"gender":"Gender","NationalITy":"Nationality",
"raisedhands":"RaisedHands","VisITedResources":"VisitedResources"},inplace=True)
#Check whether dataframe has been renamed or not
#print(df.columns)

#Create a pairplot with seaborn
#sns.pairplot(df,hue='Class')
#plt.show()

#Create a countplot with seaborn
#sns.countplot(x="Class",data=df,linewidth=2,edgecolor=sns.color_palette("dark"))
#plt.show()

#Create a bar plot which the value is normalized
#df.Class.value_counts(normalize=True).plot(kind="bar")
#plt.show()

#Create a heatmap to illustrate the correlation
#plt.figure(figsize=(14,12))
#sns.heatmap(df.corr(),linewidth=.1,cmap="YlGnBu",annot=True)
#plt.yticks(rotation=0)
#plt.show()

#Exploring RaisedHands feature
#Barplot of RaisedHands feature
#plt.subplots(figsize=(20,8))
#df["RaisedHands"].value_counts().sort_index().plot.bar()
#plt.title("No. of times", fontsize=18)
#plt.xlabel("No. of times, studen raised their hand",fontsize=14)
#plt.ylabel("No. of student, on particular times", fontsize=14)
#plt.show()

#Histogram of RaisedHands feature
#df.RaisedHands.plot(kind='hist',bins=100,figsize=(20,10),grid=True)
#plt.xlabel("RaisedHands")
#plt.legend(loc="upper right")
#plt.title("Raisedhand Histogram")
#plt.show()

#Boxplot of RaisedHands feature
#Raised_hand=sns.boxplot(x="Class",y="RaisedHands",data=df)
#plt.show()

#Facetgrid with seaborn for RaisedHand feature
#Facetgrid=sns.FacetGrid(df,hue="Class",height=6)
#Facetgrid.map(sns.kdeplot,"RaisedHands",shade=True)
#Facetgrid.set(xlim=(0,df['RaisedHands'].max()))
#Facetgrid.add_legend()
#plt.show()

#Exploring 'ParentSchoolSatisfaction' feature use 
#sns.countplot,df.groupby and pd.crosstab functions

#Using df.groupby
#print(df.groupby(['ParentschoolSatisfaction'])['Class'].value_counts())

#Using pd.crosstab function
#print(pd.crosstab(df['Class'],df['ParentschoolSatisfaction']))

#Using countplot with seaborn library
sns.countplot(x='ParentschoolSatisfaction',data=df,hue='Class',palette='bright')
plt.show()

#Create a Pie Chart
labels=df.ParentschoolSatisfaction.value_counts()
colors=["blue","green"]
explode=[0,0]
sizes=df.ParentschoolSatisfaction.value_counts().values

plt.figure(figsize=(7,7))
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')
plt.title("Parent school Satisfaction in Data",fontsize=15)
plt.show()