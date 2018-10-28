# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 16:12:30 2018

@author: LilJun
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

csv_file = "ks-projects-201801_1.csv"  # 讀入 csv 檔案

file_encoding = 'utf8'        # set file_encoding to the file encoding (utf8, latin1, etc.)
input_fd = open(csv_file, encoding=file_encoding, errors = 'backslashreplace')
df=pd.read_csv(input_fd)

de=df.dropna()  #將NaN空值欄位進行刪除動作

df_x = pd.DataFrame([de["usd pledged"],de["usd_pledged_real"],de["backers"]]).T  #df_x為變數值
#df_x = pd.DataFrame(de, columns = ['usd pledged', 'usd_pledged_real', 'backers']) 


#將state字串欄位轉為分類值
label_encoder = preprocessing.LabelEncoder()
encoded_state= label_encoder.fit_transform(de["state"]) #state為預測值

# 切分訓練與測試資料
train_X, test_X, train_y, test_y = train_test_split(df_x, encoded_state, test_size = 0.3,random_state=0)


scaler = preprocessing.StandardScaler();
scaler.fit(train_X)
train_X_std = scaler.transform(train_X)

scaler.fit(test_X)
test_X_std = scaler.transform(test_X)

#用scikit learn的PCA和回歸分析


print("用scikit learn的PCA圖示")

sklearn_pca =PCA()
Y_sklearn = sklearn_pca.fit_transform(train_X_std)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(train_y), colors, markers):
    plt.scatter(Y_sklearn[train_y == l, 0] *-1,
                Y_sklearn[train_y == l, 1] ,
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower center')
plt.tight_layout()
plt.show()



print("PCA的回歸分析")
for i in range(3):
    a=i+1
    esor=PCA(n_components=a)
    pca_x_train=esor.fit_transform(train_X_std)
    pca_x_test=esor.fit_transform(test_X_std)
    lm=LinearRegression()
    lm.fit(pca_x_train,train_y)   


print('Slope: %.3f' % lm.coef_[0])
print('Intercept: %.3f' % lm.intercept_)

  


    
    