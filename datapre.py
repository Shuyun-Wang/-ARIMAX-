import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

def datdapro():
    df=pd.read_csv("data.csv",encoding='gbk')
    df_c=df.interpolate()
    print(df_c)
    df_c.to_csv('data1.csv',encoding='utf_8_sig')

def mapout():
    plt.rcParams['font.sans-serif'] = 'SimHei'
    df=pd.read_csv("data1.csv",encoding='utf_8_sig')
    df.index = df['时间']

    df['猪肉价格'].plot()
    plt.legend()
    plt.show()

    df['仔猪价格'].plot()
    plt.legend()
    plt.show()

    df['玉米市场价格'].plot()
    plt.legend()
    plt.show()

    df['通货膨胀率CPI'].plot()
    plt.legend()
    plt.show()

    df['鸡肉价格'].plot()
    plt.legend()
    plt.show()

    df['存栏量'].plot()
    plt.legend()
    plt.show()

def del1():
    data=[]
    with open('data1.csv','r',encoding='utf_8_sig') as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        birth_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            row[1]=row[1][:4]+' '+row[1][5:-1]
            data.append(row[1:])
    print(data)

    f = open('data2.csv','w',encoding='utf-8_sig',newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["时间","猪肉价格","仔猪价格","玉米市场价格","通货膨胀率CPI","鸡肉价格","存栏量"])
    for i in range(len(data)):
        csv_writer.writerow(data[i])
    f.close()


if __name__ == "__main__":
    # del1()