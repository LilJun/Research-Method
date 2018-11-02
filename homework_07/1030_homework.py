# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 21:54:48 2018

@author: LilJun
"""


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

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
train_X, test_X, train_y, test_y = train_test_split(df_x, encoded_state, test_size = 0.3,random_state=0)


#特徵縮放
scaler = preprocessing.StandardScaler();
scaler.fit(train_X)
train_X_std = scaler.transform(train_X)

scaler.fit(test_X)
test_X_std = scaler.transform(test_X)

"""
#測試
lda = LinearDiscriminantAnalysis(n_components = 2)
X_train_lda = lda.fit_transform(train_X_std, train_y)
X_test_lda = lda.transform(test_X_std)
        
        
ld = lda.fit(X_train_lda,train_y).predict(X_test_lda)
accuracy = accuracy_score(test_y,ld)

print(accuracy)
"""



def random_forest(features, target):
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf

random_model = random_forest(train_X, train_y) #call function
#print ("Trained model : ", random_model)

predictions = random_model.predict(test_X)

#print ("Train Accuracy : ", accuracy_score(train_y,random_model.predict(train_X)))
print ("沒用LDA RandomForestClassifier Test Accuracy  : ", accuracy_score(test_y, predictions))


def decision_tree(features, target):
    clf = tree.DecisionTreeClassifier(max_depth = 3)
    clf.fit(features, target)
    return clf


decision_model = decision_tree(train_X, train_y)  #call function 
#print ("Trained model : ", decision_model)
decision_predictions = decision_model.predict(test_X)
#print ("Train Accuracy : ", accuracy_score(train_y,decision_model.predict(train_X)))
print ("沒用LDA DecisionTreeClassifier Test Accuracy  : ", accuracy_score(test_y, decision_predictions))

def lr(features, target):
    clf = LogisticRegression()
    clf.fit(features, target)
    return clf


lr_model = lr(train_X, train_y)  #call function 
#print ("Trained model : ", lr_model)
lr_predictions = lr_model.predict(test_X)
#print ("Train Accuracy : ", accuracy_score(train_y,decision_model.predict(train_X)))
print ("沒用LDA LogisticRegression Test Accuracy  : ", accuracy_score(test_y, lr_predictions))


#利用分割好的訓練集進行模型訓練並對測試集進行預測
lda = LinearDiscriminantAnalysis(n_components=3)


for n in range(0, 3):
    
   # X_train_lda, X_test_lda = [], []
    if (n == 0):
        X_train_lda = train_X_std 
        X_test_lda = test_X_std 
    else:
        lda = LinearDiscriminantAnalysis(n_components = n)
        X_train_lda = lda.fit_transform(train_X_std, train_y)
        X_test_lda = lda.transform(test_X_std)
        
        

    ld = lda.fit(X_train_lda,train_y).predict(X_test_lda)
    accuracy = accuracy_score(test_y,ld)
    print('%s components of LDA, \'s lda accuracy is %s' % (n, accuracy))

    # RandomForestClassifier
    classifier = RandomForestClassifier(random_state=0)
    classifier.fit(X_train_lda, train_y)  
    cl_predict = classifier.predict(X_test_lda)
    accuracy = accuracy_score(test_y, cl_predict)
    print('%s components of LDA, RandomForestClassifier\'s accuracy is %s' % (n, accuracy))
    
    # DecisionTreeClassifier
    dtc = tree.DecisionTreeClassifier()
    dtc.fit(X_train_lda, train_y)
    y_pred = dtc.predict(X_test_lda)
    accuracy = accuracy_score(test_y, y_pred)
    print('%s components of LDA, DecisionTreeClassifier\'s accuracy is %s' % (n, accuracy))
    
    # LogisticRegression
    lr = LogisticRegression()
    lr.fit(X_train_lda, train_y)
    lr_predict = lr.predict(X_test_lda)
    accuracy = accuracy_score(test_y, lr_predict)
    print('%s components of LDA, LogisticRegression\'s accuracy is %s' % (n, accuracy))
   
