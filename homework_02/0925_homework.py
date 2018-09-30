# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 12:25:11 2018

@author: LilJun
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from sklearn.linear_model import LinearRegression



plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  #使用中文字體


csv_file = "ks-projects-201801.csv"  # 讀入 csv 檔案
ksr = pd.read_csv(csv_file)

#1:對資料做描述性統計
print("1:對資料做描述性統計")
      
data=pd.DataFrame(ksr)
print(data.info()) #顯示原始資料属性的基本訊息
X =pd.DataFrame(ksr,columns=["main_category"])
print(X.head(10))        #列出前十筆資料募資專案的主要類型
print("backers:",ksr["backers"].describe())    #計算"贊助人數"的統計值
print("usd_pledged_real:",ksr["usd_pledged_real"].describe())  #計算"實際募資到的金額"的統計值
print("usd_goal_real",ksr["usd_goal_real"].describe())     #計算"專案目標金額"的統計值
#print(ksr.describe())

print("---------------------------------------------------------------------------------")

#2:對想要預測資料畫散步圖
print("2:對想要預測資料畫散佈圖")

ksrfailed = ksr[ksr['state'] == 'failed']
ksrfailed.plot(kind = 'scatter', x = 'usd_pledged_real', y = 'usd_goal_real')
plt.title("專案失敗，實際募資到的金額與需要目標金額的關係")
plt.xlabel("收到的募資金額") 
plt.ylabel("專案目標金額") 
plt.show()   # 印出過去專案失敗，實際募資到的金額與需要目標金額的關係


ksrfailed = ksr[ksr['state'] == 'successful']
ksrfailed.plot(kind = 'scatter', x = 'usd_pledged_real', y = 'usd_goal_real')
plt.title("專案成功，實際募資到的金額與需要目標金額的關係")
plt.xlabel("收到的募資金額") 
plt.ylabel("專案目標金額")
plt.show() 

ksrfailed = ksr[ksr['state'] =='successful' ]
ksrfailed.plot(kind = 'scatter', x = 'usd_pledged_real', y = 'backers')
plt.title("專案成功，實際募資到的金額與贊助者人數的關係")
plt.xlabel("收到的募資金額") 
plt.ylabel("贊助者人數")
plt.show() 

ksrfailed = ksr[ksr['state'] =='failed' ]
ksrfailed.plot(kind = 'scatter', x = 'usd_pledged_real', y = 'backers')
plt.title("專案失敗，實際募資到的金額與贊助者人數的關係")
plt.xlabel("收到的募資金額") 
plt.ylabel("贊助者人數")
plt.show() 

ksrfailed = ksr[ksr['deadline'] >='2016/01/01' ]
ksrfailed.plot(kind = 'scatter', x = 'usd_pledged_real', y = 'usd_goal_real')
plt.title("2016年後結束募資，實際募資到的金額與需要目標金額的關係")
plt.xlabel("收到的募資金額") 
plt.ylabel("專案目標金額")
plt.show() 

ksrfailed = ksr[ksr['deadline'] >='2016/01/01' ]
ksrfailed.plot(kind = 'scatter', x = 'usd_pledged_real', y = 'backers')
plt.title("2016年後結束募資，實際募資到的金額與贊助者人數的關係")
plt.xlabel("收到的募資金額") 
plt.ylabel("贊助者人數")    
plt.show() 

print("---------------------------------------------------------------------------------")

#3:計算相關係數
print("3:計算相關係數")

corr=ksr["usd_pledged_real"].corr(ksr['backers'])   
print("實際募到金額與贊助人數相關係數:",corr)
corr=ksr["usd_goal_real"].corr(ksr['backers'])   
print("目標金額與贊助人數相關係數:",corr)
corr=ksr["usd_pledged_real"].corr(ksr['usd_goal_real'])   
print("實際募到金額與目標金額相關係數:",corr)

print("---------------------------------------------------------------------------------")
#4-1. 資料正規化(Z-score)
print("4-1. 資料正規化(Z-score)")

df = pd.DataFrame({"usd_pledged_real" : ksr['usd_pledged_real'],
                   "backers" : ksr['backers']})
scaler = preprocessing.StandardScaler()
np_std = scaler.fit_transform(df)
df_std = pd.DataFrame(np_std, columns=["usd_pledged_real", "backers"])
print(df_std.head())

df_std.plot(kind="scatter", x="usd_pledged_real", y="backers")

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))

ax1.set_title('Before Scaling')
sns.kdeplot(df['usd_pledged_real'], ax=ax1)
sns.kdeplot(df['backers'], ax=ax1)
ax2.set_title('After Standard Scaler')
sns.kdeplot(df_std['usd_pledged_real'], ax=ax2)
sns.kdeplot(df_std['backers'], ax=ax2)
plt.show()
print("---------------------------------------------------------------------------------")

#4-2. 資料正規化(最大最小值)
print("4-2. 資料正規化(最大最小值)")

df_scaled = pd.DataFrame(preprocessing.scale(df), 
                         columns=["usd_pledged_real", "backers"])
print(df_scaled.head())
df_scaled.plot(kind="scatter", x="usd_pledged_real", y="backers")

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
np_minmax = scaler.fit_transform(df)
df_minmax = pd.DataFrame(np_minmax, columns=["usd_pledged_real", "backers"])
print(df_minmax.head())
df_minmax.plot(kind="scatter", x="usd_pledged_real", y="backers")


scaler = preprocessing.MinMaxScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=["usd_pledged_real", "backers"])

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
ax1.set_title('Before Scaling')
sns.kdeplot(df['usd_pledged_real'], ax=ax1)
sns.kdeplot(df['backers'], ax=ax1)

ax2.set_title('After Min-Max Scaling')
sns.kdeplot(df_minmax['usd_pledged_real'], ax=ax2)
sns.kdeplot(df_minmax['backers'], ax=ax2)

plt.show()

print("第4點總結:從圖表來看，Z-score 、最大最小值並無太大差別")
print("---------------------------------------------------------------------------------")

#5.檢查資料有無遺漏值
print("5.檢查資料有無遺漏值")

print(data.info()) #顯示原始資料属性的基本訊息，從各欄位count 觀察是否有遺漏值?

#issue1:從資料中可看到name少了4筆資料，但name並不是一個標準的參考值，也就是說可能任何兩格專案或多個專案可能有相同募資名稱，相對來說專案的"ID"才具有
#唯一性，而在分析時，也不太可能真正從這378661筆資料中去了解某個專案名稱是甚麼，所以這邊缺少的4筆name值將以'U0'補上
#issue2:use pledged 明顯也缺少資料，但以這個資料急解釋:usd_pledged只是pledged欄位換算成美元的值，所以在EXCEL轉換過程可能出了錯誤，但以pledged換算
#成usd_pledged_real真實值是正確的，所以可以刪除usd_pledged欄位，並不會影響整體分析過程
#整合兩點論述，此資料集整體來說，在分析使用上是不會出現有遺漏值欄位問題
ironmen_df_na_filled = data.fillna("No name") # 有遺失值的觀測值填補 0
 #print(ironmen_df_na_filled)
 #print("---") # 分隔線
print("第5點總結:此dataset並無遺漏值，詳細論述在code註記內")
print("---------------------------------------------------------------------------------")

#6.分類值
print("6.分類值 (對專案所屬城市名稱轉換成數字)")
#將各個募資專案所屬城市名稱轉成数字，以便做訓練模型用。

label_encoder = preprocessing.LabelEncoder()
data["country"] = label_encoder.fit_transform(data["category"])
print(data["country"])
print("---------------------------------------------------------------------------------")

#7.線性回歸
print("7.線性回歸")
#根據專案實際收到的募資金額和募資人數做線性回歸

x1 = pd.DataFrame(ksr, columns=["usd_pledged_real"])
target = pd.DataFrame(ksr, columns=["backers"])
y1 = target["backers"]

lm = LinearRegression()
lm.fit(x1, y1)
print("迴歸係數:", lm.coef_)
print("截距:", lm.intercept_ )

plt.scatter(x1, y1)
plt.xlabel("usd_pledged_real")
plt.ylabel("backers")
plt.title("Relationship between usd_pledged_real and backers")
plt.show()

print("---------------------------------------------------------------------------------")

x1 = data[["usd_pledged_real"]].values
y1 = data["backers"].values

slr = LinearRegression()
slr.fit(x1, y1)

print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)
def lin_regplot(x1, y1, model):
    plt.scatter(x1, y1, c='lightblue')
    plt.plot(x1, model.predict(x1), color='red', linewidth=2)    
    return 

lin_regplot(x1, y1, slr)
plt.xlabel('usd_pledged_real')
plt.ylabel('backers')
plt.tight_layout()
plt.show()

print("---------------------------------------------------------------------------------")

#8.計算MSE
print("8.計算MSE")

pledged =data[["usd_pledged_real"]].values
goal = data["usd_goal_real"].values

error = []
for i in range(len(data)):   
    error.append(goal[i] - pledged[i]) 
    
#print("Errors: ", error)   

squaredError = []
absError = []
    
for val in error:    
   squaredError.append(val * val)#goal之差平方     
   absError.append(abs(val))#誤差絕對值  
   
#print("Square Error: ", squaredError)
#print("Absolute Value of Error: ", absError)  
print("實際募資金額與目標專案金額的均方誤差")
print("MSE = ", sum(squaredError) / len(squaredError))  #均方誤差MSE

print("---------------------------------------------------------------------------------")