# Exercise 3. Exploratory Data Analysis
# Load Libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load dataframe
    path = '/home/dinh_anh_huy/GitHub/2020-2021/semester-1/python-for-data-science/Lab2/creditcard.csv'
    df = pd.read_csv(path)
    # ============================================================== #
    ''' 
    - Time column = Số giây trôi qua giữa giao dịch hiện tại với giao dịch đầu trong bộ dữ liệu
    - Amount column = Số tiền giao dịch
    - Class = 0 => Not Fraud , 1 => Fraud
    - V columns (column 1 to 29) => undisclosed (private)
    '''
    # ============================================================== #
    print('Shape of data: ', df.shape)
    feature_size = len(df.columns)
    label = feature_size - 1
    print('Feature size: ', feature_size)
    print('Target index: ', label)
    print(df.head())
    print(df.describe())
    # ============================================================== #
    ''' Lọc dữ liệu lấy ra 3 cột 'Time', 'Amount' và 'Class' '''
    # ============================================================== #
    cols = ['Time', 'Amount', 'Class']
    data_df = df[cols]
    print(data_df.head())

    Num_of_Fraud = round(data_df['Class'].value_counts()[1]/len(data_df)*100, 3)
    Num_of_NonFraud = round(data_df['Class'].value_counts()[0]/len(data_df)*100, 3)
    print("Number of Fraud Values: ",data_df['Class'].value_counts()[1])
    print("Number of Non Fraud Values: ",data_df['Class'].value_counts()[0])
    print("Percentage of Fraud transactions: ", Num_of_Fraud)
    print("Percentage of Normal (Non-Fraud) transactions: ",Num_of_NonFraud)

    sns.countplot(x="Class", data=data_df, linewidth=2, edgecolor=sns.color_palette("dark"))
    plt.title("Class Count", fontsize=18)
    plt.xlabel("Fraud", fontsize=15)
    plt.ylabel("Count", fontsize=15)
    plt.show()
    # ============================================================== #
    '''
    Dựa vào đồ thị ta thấy rằng số lượng giao dịch fraud có số lượng rất nhỏ so với số lượng giao dịch bình thường (nhỏ hơn khoảng 577 lần)
    '''
    # ============================================================== #
    # Pie chart
    fig, ax = plt.subplots(1, 1)
    ax.pie(data_df.Class.value_counts(),autopct='%1.3f%%', labels=['Normal','Fraud'], colors = ['yellowgreen', 'red'])
    plt.axis('equal')
    plt.ylabel('')
    plt.show()
    # ============================================================== #
    '''
    Dựa vào pie chart ta thấy rằng tỷ lệ số giao dịch fraud chiếm rất ít trên tổng số giao dịch (chỉ chiếm khoảng 0.173%)
    '''
    # ============================================================== #
    fig, (axis_1, axis_2) = plt.subplots(2, 1, sharex = True, figsize = (12, 4))
    bins = 50
    axis_1.hist(data_df.Time[data_df.Class == 1], bins = bins)
    axis_1.set_title('Fraud')
    axis_2.hist(data_df.Time[df.Class == 0], bins = bins)
    axis_2.set_title('Normal (Non-fraud)')
    plt.xlabel('Time (in seconds)')
    plt.ylabel('Number of transactions')
    plt.show()
    # ============================================================== #
    '''
    Dựa vào histogram ở trên, ta thấy rằng các giao dịch bình thường có dạng phân phối đều trong khi các giao dịch 'fraud' thì không
    '''
    # ============================================================== #

    fig, (axis_1, axis_2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))
    bins = 30
    axis_1.hist(data_df.Amount[data_df.Class == 1], bins = bins)
    axis_1.set_title('Fraud')
    axis_2.hist(data_df.Amount[data_df.Class == 0], bins = bins)
    axis_2.set_title('Normal (Non-Fraud)')
    plt.xlabel('Amount')
    plt.ylabel('Number of Transactions')
    plt.yscale("log")
    plt.show()
    # ============================================================== #
    '''
    Dựa vào histogram ta thấy rằng:
    - Các giao dịch fraud có amount nhỏ hơn
    - Các giao dịch bình thường cũng có nhiều giao dịch có amount nhỏ
    Hai điều trên cho thấy rằng nếu amount nhỏ thì nó có thể là amount của 1 giao dịch bình thường hoặc 1 giao dịch fraud.
    Do đó không thể sử dụng amount để phân biệt các giao dịch bình thường và fraud. 
    '''
    # ============================================================== #
    ax = sns.boxplot(x ="Class", y="Amount", data=data_df)
    ax.set(ylim=(0,300))
    ax.set_title("Class x Amount", fontsize=18)
    ax.set_xlabel("Class", fontsize=14)
    ax.set_ylabel("Amount", fontsize = 14)
    plt.show()
    # ============================================================== #
    '''
    Theo đồ thị trên, ta thấy rằng:
    - Độ biến động của 'fraud transaction' nhiều hơn so với 'normal transaction'
    - Median của class 1 lệch khá nhiều về phía Q1 --> Dữ liệu bị lệch phải
    '''
    # ============================================================== #
    ax = sns.barplot(data_df['Class'], data_df['Amount'], dodge = False)
    plt.show()
    # ============================================================== #
    '''
    Ta thấy rằng 'fraud transactions' có lượng amount cao hơn 'normal transaction'.
    --> Điều đó cho thấy rằng 'fraud transaction' thường giao dịch với số tiền lớn đột biến so với bình thường.
    '''
    # ============================================================== #
    fraud_df = data_df['Time'].loc[data_df['Class']==1]
    non_fraud_df = data_df['Time'].loc[data_df['Class']==0]
    print(fraud_df.describe())
    print('*'*30)
    print(non_fraud_df.describe())

    plt.figure(figsize=(7,5))
    sns.kdeplot(fraud_df, color = 'red', shade = True, label = 'fraud transaction')
    sns.kdeplot(non_fraud_df, color = 'darkgreen', shade = True, label = 'non-fraud transaction')
    plt.title('Distribution of Time', fontsize = 18)
    plt.xlabel('Time', fontsize = 14)
    plt.show()
    # ============================================================== #
    '''
    Từ đồ thị ta thấy:
    + Sau khoảng 100000 giây (khoảng 28 giờ) kể từ khi bắt đầu khảo sát, 'fraud transaction' tăng đột biến so với 'normal transaction'
    --> Cho thấy rằng các giao dịch bất thường thường diễn ra vào khoảng 4 giờ sáng ngày thứ 2 (sau khoảng 28 giờ) kể từ khi bắt đầu khảo sát.
    + Sau khoảng 130000 giây (khoảng 36 giờ) từ khi bắt đầu khảo sát, 'normal transaction' tăng trở lại và 'fraud transaction' giảm đáng kể.
    --> Cho thấy rằng các giao dịch bất thường bắt đầu giảm dần và các giao dịch bình thường diễn ra nhiều hơn vào 12 giờ ngày thứ hai kể từ khi bắt đầu khảo sát.
    '''
    # ============================================================== #

if __name__ == "__main__":
    main()