import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import matplotlib.pyplot as plt
from matplotlib.pylab import style
import statsmodels.api as sm
from itertools import product
import pyflux as pf
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

listlabel=['猪肉价格','仔猪价格','玉米市场价格','通货膨胀率CPI','鸡肉价格','存栏量']

def datacf():
    df=pd.read_csv("data2.csv",encoding='gbk') 
    df['时间'] = pd.to_datetime(df['时间'])
    df.index = df['时间']
    df = df.resample('3M').mean()
    ## 1:单位根检验检验序列的平稳性,ADF 检验
    for i in listlabel:
        df[i]=df[i].diff()
        df[i]=df[i].dropna()
    df["存栏量"]=df["存栏量"].diff()
    df["存栏量"]=df["存栏量"].dropna()

def cfmapandsave():
    for i in listlabel:
        df[i].plot()
        plt.legend()
        plt.show()
    df.to_csv('datacf.csv',encoding='gbk')

def chafenacfpacf():
    for i in listlabel:
        diff=df[i]
        diff=diff.diff()
        diff=diff.dropna()
        dftest = sm.tsa.adfuller(diff,autolag='BIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
        print(i,"一阶差分 检验结果：")
        print(dfoutput)
        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(diff, lags=20, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(diff, lags=20, ax=ax2)
        plt.subplots_adjust(hspace = 0.3)
        plt.show()
        diff.to_csv('datacf.csv',encoding='gbk')

    diff=diff.diff()
    diff=diff.dropna()
    dftest = sm.tsa.adfuller(diff,autolag='BIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
    print("存栏量二阶差分 检验结果：")
    print(dfoutput)
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(diff, lags=20, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(diff, lags=20, ax=ax2)
    plt.subplots_adjust(hspace = 0.3)
    plt.show()

def p_valuetest():
    dftest = sm.tsa.adfuller(df['仔猪价格'],autolag='BIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
    print("仔猪价格 检验结果：")
    print(dfoutput)
    dftest = sm.tsa.adfuller(df['玉米市场价格'],autolag='BIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
    print("玉米市场价格 检验结果：")
    print(dfoutput)
    dftest = sm.tsa.adfuller(df['通货膨胀率CPI'],autolag='BIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
    print("通货膨胀率CPI 检验结果：")
    print(dfoutput)
    dftest = sm.tsa.adfuller(df['鸡肉价格'],autolag='BIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
    print("鸡肉价格 检验结果：")
    print(dfoutput)
    dftest = sm.tsa.adfuller(df['存栏量'],autolag='BIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
    print("存栏量 检验结果：")
    print(dfoutput)



org=pd.read_csv("data2.csv",encoding='gbk') 
org['时间'] = pd.to_datetime(org['时间'])
org.index =org['时间']
org = org.resample('3M').mean()

    
df=pd.read_csv("datacf.csv",encoding='gbk') 
df['时间'] = pd.to_datetime(df['时间'])
df.index = df['时间']
df = df.resample('3M').mean()
def pidchoose():
    print('最优模型筛选')
    num=3
    best_aic = float('inf')
    results = []
    for i in range(num):
        for j in range(num):
            model = pf.ARIMAX(data=df,ar=i, ma=j,formula='猪肉价格~仔猪价格+玉米市场价格+鸡肉价格+存栏量',family=pf.Normal()).fit()
            aic = model.aic
            if aic < best_aic:
                best_model = model
                best_aic = model.aic
            results.append([i,j, model.aic])
    results_table = pd.DataFrame(results)
    results_table.columns = ['ar','ma','aic']
    best_model.summary()


model = pf.ARIMAX(data=df,ar=0, ma=2,formula='猪肉价格~仔猪价格+玉米市场价格+鸡肉价格+存栏量',family=pf.Normal())
x=model.fit()
print(x)
x.summary()
model.plot_fit(figsize=(15,10))
model.plot_predict(h=5, oos_data=df.iloc[-5:], past_values=73, figsize=(15,5))
# a=model.predict(h=5, oos_data=df.iloc[-5:])