# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 09:35:49 2018

@author: LilJun
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

#1:載入資料集
print("1:載入資料集")

csv_file = "ks-projects-201801_1.csv"  # 讀入 csv 檔案

file_encoding = 'utf8'        # set file_encoding to the file encoding (utf8, latin1, etc.)
input_fd = open(csv_file, encoding=file_encoding, errors = 'backslashreplace')
preksr=pd.read_csv(input_fd)

ksr=preksr.dropna()  #將NaN空值欄位進行刪除動作

#2:對資料做描述性統計
print("2:對資料做描述性統計")
      
data=pd.DataFrame(ksr)
print(data.info()) #顯示原始資料属性的基本訊息
X =pd.DataFrame(ksr,columns=["main_category"])
print(X.head(10))        #列出前十筆資料募資專案的主要類型
print("backers:",ksr["backers"].describe())    #計算"贊助人數"的統計值
print("usd_pledged_real:",ksr["usd_pledged_real"].describe())  #計算"實際募資到的金額"的統計值
print("usd_goal_real",ksr["usd_goal_real"].describe())     #計算"專案目標金額"的統計值
#print(ksr.describe())

print("---------------------------------------------------------------------------------")

#3:散佈圖(兩兩變數)
print("3:散佈圖(兩兩變數)")

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
plt.show()  # 印出過去專案成功，實際募資到的金額與需要目標金額的關係
print("---------------------------------------------------------------------------------")

#4:相關矩陣
print("4:相關矩陣")
"""
real = pd.DataFrame(ksr, columns = ['usd_pledged_real']) 
backers = pd.DataFrame(ksr, columns = ['backers']) 
y=[real,backers]
b=np.corrcoef(y)i
"""
"""
logistic_model = logistic_reg(df_x, encoded_state) 
print("係數:",logistic_model.coef_)
corr=ksr["usd_pledged_real"].corr(ksr['backers'])   
print("實際募到金額與贊助人數相關係數:",corr)
"""

real= pd.DataFrame(ksr, columns = [ 'usd_pledged_real', 'backers']) 
print("實際募到金額與贊助人數相關矩陣:\n",real.corr())
print("---------------------------------------------------------------------------------")

#5:共變異數矩陣
print("5:共變異數矩陣")

df_x = pd.DataFrame([ksr["usd pledged"],ksr["usd_pledged_real"],ksr["backers"]]).T  #df_x為變數值

#將state字串欄位轉為分類值
label_encoder = preprocessing.LabelEncoder()
encoded_state= label_encoder.fit_transform(ksr["state"]) #state為預測值

# 切分訓練與測試資料
train_X, test_X, train_y, test_y = train_test_split(df_x, encoded_state, test_size = 0.3,random_state=0)
#標準化
scaler = preprocessing.StandardScaler();
scaler.fit(train_X)
train_X_std = scaler.transform(train_X)

scaler.fit(test_X)
test_X_std = scaler.transform(test_X)

mean_vec = np.mean(train_X_std, axis=0)
cov_mat = (train_X_std - mean_vec).T.dot((train_X_std - mean_vec)) / (train_X_std.shape[0]-1)

#real= pd.DataFrame(ksr, columns = [ 'usd_pledged_real', 'backers']) 
#print("實際募到金額與贊助人數共變異數矩陣:\n",real.cov())

cov_mat=np.cov(train_X_std.T)
print('Covariance matrix \n%s' %cov_mat)

print("---------------------------------------------------------------------------------")
#6:eigenvalue分解
print("6:eigenvalue分解")
eigen_vals,eigen_vecs=np.linalg.eig(cov_mat)

tot=sum(eigen_vals)
var_exp=[(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp=np.cumsum(var_exp)
print("\n解釋量:",cum_var_exp)
print('\nEigenvalues \n%s' % eigen_vals)

plt.bar(range(1,4),var_exp,alpha=0.5,align='center',label='individual explained variance')
plt.step(range(1,4),cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend()
plt.tight_layout()
plt.show()


print("---------------------------------------------------------------------------------")

#7:PCA矩陣
print("7:PCA矩陣")

eigen_pairs =[(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))] #把特徵值和對應的特徵向量組成對
eigen_pairs.sort(reverse=True) #用特徵值排序
"""
first = eigen_pairs[0][1]
second = eigen_pairs[1][1]
first = first[:,np.newaxis]
second = second[:,np.newaxis]
W = np.hstack((first,second))
print("Matrix w:\n",W)
"""

w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
              eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)

print("---------------------------------------------------------------------------------")

#8:選出主成分比較不同數目主成分的MSE
print("8:選出主成分比較不同數目主成分的MSE")
for i in range(3):
    a=i+1
    esor=PCA(n_components=a)
    pca_x_train=esor.fit_transform(train_X_std)
    pca_x_test=esor.fit_transform(test_X_std)
    lm=LinearRegression()
    lm.fit(pca_x_train,train_y)
    pca_y_predict=lm.predict(pca_x_test)
    mse=np.mean((pca_y_predict-test_y)**2)
    print("MSE:",a,":",mse,"\n")

print("回歸分析")
print('Slope: %.3f' % lm.coef_[0])
print('Intercept: %.3f' % lm.intercept_,"\n")

    
    
    
    
print("沒用scikit learn PCA圖示")


def pca(X,k):#k is the components you want
  #mean of each feature
  n_samples, n_features = X.shape
  mean=np.array([np.mean(X[:,i]) for i in range(n_features)])
  #normalization
  norm_X=X-mean
  #scatter matrix
  scatter_matrix=np.dot(np.transpose(norm_X),norm_X)
  #Calculate the eigenvectors and eigenvalues
  eig_val, eig_vec = np.linalg.eig(scatter_matrix)
  eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
  # sort eig_vec based on eig_val from highest to lowest
  eig_pairs.sort(reverse=True)
  # select the top k eig_vec
  feature=np.array([ele[1] for ele in eig_pairs[:k]])
  #get new data
  data=np.dot(norm_X,np.transpose(feature))

  return data


X_train_pca= pca(train_X_std,2)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(train_y), colors, markers):
    plt.scatter(X_train_pca[train_y == l, 0] ,
                X_train_pca[train_y == l, 1] ,
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower center')
plt.tight_layout()
plt.show()










