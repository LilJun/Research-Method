# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 17:05:23 2018

@author: LilJun
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from statsmodels.stats.contingency_tables import mcnemar

csv_file = "ks-projects-201801_1.csv"  # 讀入 csv 檔案

file_encoding = 'utf8'        # set file_encoding to the file encoding (utf8, latin1, etc.)
input_fd = open(csv_file, encoding=file_encoding, errors = 'backslashreplace')
df=pd.read_csv(input_fd)

de=df.dropna()  #將NaN空值欄位進行刪除動作

df_x = pd.DataFrame([de["usd pledged"],de["usd_pledged_real"],de["backers"]]).T  #df_x為變數值
#df_x = pd.DataFrame(de, columns = ['usd pledged', 'usd_pledged_real', 'backers']) 
df_y = pd.DataFrame(de, columns = ['state']) #df_y為預測值

#將state字串欄位轉為分類值
label_encoder = preprocessing.LabelEncoder()
de["state"] = label_encoder.fit_transform(df_y["state"])
y = de["state"]


train_ID3_X, test_ID3_X, train_ID3_y, test_ID3_y = train_test_split(df_x, y, test_size = 0.25)
train_CART_X, test_CART_X, train_CART_y, test_CART_y = train_test_split(df_x, y, test_size = 0.25)

#ID3 function
def id3_tree(features, target):
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(features, target)
    return clf

#CART function
def cart_tree(features, target):
    clf = tree.DecisionTreeClassifier(criterion='gini')
    clf.fit(features, target)
    return clf


decision_model = id3_tree(train_ID3_X, train_ID3_y)  #call function 
id3_predictions = decision_model.predict(test_ID3_X)
 #print ("Test Accuracy  : ", accuracy_score(test_ID3_y, decision_predictions))

decision_model = cart_tree(train_CART_X, train_CART_y)  #call function 
cart_predictions = decision_model.predict(test_CART_X)
 #print ("Test Accuracy  : ", accuracy_score(test_ID3_y, decision_predictions))


#比較表
voters=pd.DataFrame({"ID3":id3_predictions,"CART":cart_predictions})
voter_tab=pd.crosstab(voters.ID3,voters.CART,margins=True)
observed=voter_tab.iloc[0:3,0:3]
print(observed,"\n")

#McNemar檢驗
result=mcnemar(observed,exact=True)
print('statistic=%.3f,p-value=%.3f' % (result.statistic,result.pvalue))

alpha=0.05

if result.pvalue > alpha:
    print('兩種分類器沒有顯著差異')
else:
    print('兩種分類器有顯著差異')

