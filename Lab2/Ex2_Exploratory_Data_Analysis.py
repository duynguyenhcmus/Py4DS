# Exercise 2. Exploratory Data Analysis
# Load Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

# Load dataset
path = 'https://raw.githubusercontent.com/duynguyenhcmus/Py4DS/main/Lab2/Py4DS_Lab2_Dataset/xAPI-Edu-Data.csv'
df = pd.read_csv(path)
print(df.head())
print(df.info())

# Print the number of value in each columns
for i in range(1, 17):
    print(df.iloc[:, i].value_counts())
    print("*" * 20)

df.rename(index=str, columns={"gender" : "Gender", "NationalITy" : "Nationality", "raisedhands" : "RaisedHands", "VisITedResources" : "VisitedResources"}, inplace=True)
# Check whether dataframe has been renamed or not
print(df.columns)

# Create a pairplot with seaborn
sns.pairplot(df, hue='Class')
plt.show()

# Create a countplot with seaborn
sns.countplot(x = "Class", data = df, linewidth = 2, edgecolor = sns.color_palette("dark"))
plt.show()

# Create a bar plot which the value is normalized
df.Class.value_counts(normalize = True).plot(kind = "bar")
plt.show()

# Create a heatmap to illustrate the correlation
plt.figure(figsize = (14, 12))
sns.heatmap(df.corr(), linewidth = .1, cmap = "YlGnBu", annot = True)
plt.yticks(rotation = 0)
plt.show()

# Exploring Discussion feature
# Barplot of Discussion feature
plt.subplots(figsize = (20, 8))
df["Discussion"].value_counts().sort_index().plot.bar()
plt.title("No. of times", fontsize = 18)
plt.xlabel("No. of times, student discuss", fontsize = 14)
plt.ylabel("No. of student, on particular times", fontsize = 14)
plt.show()

# Histogram of Discussion feature
df.Discussion.plot(kind = 'hist', bins = 100, figsize = (20, 10), grid = True)
plt.xlabel("Discussion")
plt.legend(loc = "upper right")
plt.title("Discussion Histogram")
plt.show()

# Boxplot of Discussion feature
Discuss = sns.boxplot(x = "Class", y = "Discussion", data = df)
plt.show()

# Facetgrid with seaborn for Discussion feature
Facetgrid = sns.FacetGrid(df, hue = "Class", height = 6)
Facetgrid.map(sns.kdeplot, "Discussion", shade = True)
Facetgrid.set(xlim = (0, df['Discussion'].max()))
Facetgrid.add_legend()
plt.show()

# Exploring 'StudentAbsenceDays' feature use 
# sns.countplot,df.groupby and pd.crosstab functions

# Using df.groupby
print(df.groupby(['StudentAbsenceDays'])['Class'].value_counts())

# Using pd.crosstab function
print(pd.crosstab(df['Class'],df['StudentAbsenceDays']))

# Using countplot with seaborn library
sns.countplot(x = 'StudentAbsenceDays', data = df, hue = 'Class', palette = 'bright')
plt.show()

# Create a Pie Chart
labels = df.StudentAbsenceDays.value_counts()
colors = ["blue","green"]
explode = [0, 0]
sizes = df.StudentAbsenceDays.value_counts().values

plt.figure(figsize = (7, 7))
plt.pie(sizes, explode = explode, labels = labels, colors = colors, autopct = '%1.1f%%')
plt.title("Student Absence Days in Data", fontsize = 15)
plt.show()