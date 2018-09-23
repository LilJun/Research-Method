# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 12:25:11 2018

@author: LilJun
"""
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  #使用中文字體


csv_file = "ks-projects-201801.csv"  # 讀入 csv 檔案
ksr = pd.read_csv(csv_file)
#print(type(ksr))
#ksr.head()
ksr.info()   # 顯示 csv 檔案的欄位資訊

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

