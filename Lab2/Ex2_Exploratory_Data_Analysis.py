# Exercise 2. Exploratory Data Analysis
# Load Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

def main():
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

    # Exploring Discussion feature
    # Barplot of Discussion feature
    plt.subplots(figsize=(20,8))
    df["Discussion"].value_counts().sort_index().plot.bar()
    plt.title("No. of times vs no. of students discuss particular time", fontsize=18)
    plt.xlabel("No. of times, students discuss", fontsize=14)
    plt.ylabel("No. of student, on particilar times", fontsize=14)
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
    # ============================================================== #
    '''
    Ta thấy rằng độ biến động của cả 3 class có độ cân bằng, tuy nhiên class H lại cao hơn một ít.
    --> Các học sinh level High có tần suất thảo luận với nhau nhiều nhất, kế tiếp là lever Medium và cuối cùng là Low
    '''
    # ============================================================== #
    # Facetgrid with seaborn for Discussion feature
    Facetgrid = sns.FacetGrid(df, hue = "Class", height = 6)
    Facetgrid.map(sns.kdeplot, "Discussion", shade = True)
    Facetgrid.set(xlim = (0, df['Discussion'].max()))
    Facetgrid.add_legend()
    plt.show()
    # ============================================================== #
    '''
    - Class L có phần đỉnh cao ở khoảng 15 và bị lệch phải.
    --> Các học sinh level Low có xu hướng rất ít tham gia thảo luận
    - Class M cân bằng hơn so với class L, tuy nhiên vẫn còn lệch phải ít.
    --> Các học sinh level Medium tham gia thảo luận nhiều hơn so với level Low
    - Class H khá cân bằng và tại vị trí 80 lần thảo luận thì class H lớn hơn nhiều so với 2 class còn lại.
    --> Các học sinh level High tham gia thảo luận nhiệt tình hơn và có số lần thảo luận cao hơn rất nhiều so với 2 level còn lại.
    '''
    # ============================================================== #
    # Exploring 'StudentAbsenceDays' feature use 
    # sns.countplot,df.groupby and pd.crosstab functions
    # Using df.groupby
    print(df.groupby(['StudentAbsenceDays'])['Class'].value_counts())
    # Using pd.crosstab function
    print(pd.crosstab(df['Class'],df['StudentAbsenceDays']))
    # Using countplot with seaborn library
    sns.countplot(x = 'StudentAbsenceDays', data = df, hue = 'Class', palette = 'bright')
    plt.show()
    # ============================================================== #
    '''
    - Các học sinh level Low có xu hướng nghỉ học nhiều và có số ngày vắng mặt trên 7 cao nhất.
    - Các học sinh level High lại có xu hướng đi học đầy đủ hơn, số học sinh có số ngày nghỉ dưới 7 lại rất cao so với level Low.
    - Các học sinh level Medium thì lại cân bằng hơn khi số học sinh vắng mặt dưới 7 ngày lại cách không quá xa so với số học sinh nghỉ học trên 7 ngày. 
    '''
    # ============================================================== #
    # Create a Pie Chart
    labels = df.StudentAbsenceDays.value_counts()
    colors = ["blue","green"]
    explode = [0, 0]
    sizes = df.StudentAbsenceDays.value_counts().values
    plt.figure(figsize = (7, 7))
    plt.pie(sizes, explode = explode, labels = labels, colors = colors, autopct = '%1.1f%%')
    plt.title("Student Absence Days in Data", fontsize = 15)
    plt.show()
    # ============================================================== #
    '''
    Dựa vào pie chart, ta thấy rằng số học sinh có số ngày vắng mặt dưới 7 chiếm 39,8% trong khi số học sinh vắng mặt trên 7 ngày lại chiếm đến 60.2%
    '''
    # ============================================================== #

if __name__ == '__main__':
    main()
