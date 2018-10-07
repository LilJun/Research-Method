# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:53:25 2018

@author: LilJun
"""


import pandas as pd
import math 
from scipy import stats

def RandomSampling(data,number):
    try:
         slice = data.sample(number,random_state=123)    
         return slice
    except:
         print ('sample larger than population')
         

if __name__=='__main__':

 N=100
 csv_file = "ks-projects-201801_1.csv"  # 讀入 csv 檔案

 file_encoding = 'utf8'        # set file_encoding to the file encoding (utf8, latin1, etc.)
 input_fd = open(csv_file, encoding=file_encoding, errors = 'backslashreplace')
 ksr=pd.read_csv(input_fd)
 data=pd.DataFrame(ksr,columns = ['usd_pledged_real', 'usd_goal_real', 'backers'])
# train_data=data.sample(N,random_state=123)
 train_data=RandomSampling(data,N)
 population_mean=[]
 name=0
 X=0
for i in data:
  mean=train_data[i].mean()
  std=train_data[i].std()
  SE=std/math.sqrt(N-1)
  label= data.columns[name]
  print(label,"mean:",mean,"standard error:",SE)
  name=name+1
  print ('99% confidence interval:',stats.norm.interval(0.99,loc=mean,scale=SE))
  print ('95% confidence interval:',stats.norm.interval(0.95,loc=mean,scale=SE))
  print ('80% confidence interval:',stats.norm.interval(0.80,loc=mean,scale=SE))
 # print (data[0:N]
 
  population_mean.append(data[i].mean())
  t_obtained=(mean-population_mean[X])/SE
  print("檢定統計量",t_obtained)
  print(stats.ttest_1samp(a=train_data[i],popmean=population_mean[X])) 
  X=X+1





  
