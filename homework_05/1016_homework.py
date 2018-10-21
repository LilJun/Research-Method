# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 21:32:52 2018

@author: LilJun
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


"""
sc=StandardScaler()
train_X_std=sc.fit_transform(train_X)
test_X_std=sc.fit_trandform(test_X)
"""

#沒用sklearn
cov_mat=np.cov(train_X_std.T)
eigen_vals,eigen_vecs=np.linalg.eig(cov_mat)

print('\nEigenvalues \n%s' % eigen_vals)

tot=sum(eigen_vals)
var_exp=[(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp=np.cumsum(var_exp)


plt.bar(range(1,4),var_exp,alpha=0.5,align='center',label='individual explained variance')
plt.step(range(1,4),cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend()
plt.tight_layout()
plt.show()


#PCA  有用sklearn

pca=PCA()
train_X_pca=pca.fit_transform(train_X_std)
print(pca.explained_variance_ratio_)

plt.bar(range(1,4),pca.explained_variance_ratio_,alpha=0.5,align='center')
plt.step(range(1,4),np.cumsum(pca.explained_variance_ratio_), where='mid')

plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()

