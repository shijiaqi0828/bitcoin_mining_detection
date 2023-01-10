# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 13:13:59 2022

@author: Administrator
"""

import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics.scorer import make_scorer
from sklearn.ensemble import GradientBoostingClassifier

train_profile = pd.read_csv(r"D:\竞赛\新建文件夹\train_profile.csv")
test_profile = pd.read_csv(r"D:\竞赛\新建文件夹\test_profile.csv")
submit_version=pd.read_csv(r"D:\竞赛\submit_example.csv")

train_profile=train_profile.dropna(axis=0,how='any')
#train_profile=train_profile[~(train_profile['ID']==329833811)]
X_train=train_profile.iloc[:,2:]

del X_train['IS_FLAG']
col_name=X_train.columns.tolist()                   # 将数据框的列名全部提取出来存放在列表里
print(col_name)

col_name.insert(len(col_name),'ratio_1') 
col_name.insert(len(col_name)+1,'ratio_2') 
                  
X_train=X_train.reindex(columns=col_name) 
X_train['ratio_1']=X_train.loc[:, 'mean_day_consumption'].div(X_train.loc[:, 'RUN_CAP'].values, axis=0)  
X_train['ratio_2']=X_train.loc[:, 'mean_month_consumption'].div(X_train.loc[:, 'RUN_CAP'].values, axis=0)

# del X_train['ELEC_TYPE_NAME']
# del X_train['VOLT_NAME']
# del X_train['PRC_NAME']
# del X_train['CHK_CYCLE']
# del X_train['daycount_5']
# del X_train['daycount_4']
# del X_train['daycount_3']
# del X_train['daycount_2']
# del X_train['monthcount_5']
# del X_train['monthcount_4']


X_train=X_train.values
#X_train[np.isnan(X_train)]=0
##X_train=X_train[~np.isnan(X_train).any(axis=1)]
######归一化#######
scaler1= MinMaxScaler()  # 默认(0,1)
X_train[:,5:6]=scaler1.fit_transform(X_train[:,5].reshape(-1, 1))

scaler2= MinMaxScaler()  # 默认(0,1)
X_train[:,6:7]=scaler1.fit_transform(X_train[:,6].reshape(-1, 1))

scaler3= MinMaxScaler()  # 默认(0,1)
X_train[:,7:8]=scaler1.fit_transform(X_train[:,7].reshape(-1, 1))
####特征选择###
#X_train=X_train[:,3:]
y_train = train_profile['IS_FLAG'].values


# from collections import Counter
# from imblearn.over_sampling import SMOTE, ADASYN

# # from imblearn.over_sampling import BorderlineSMOTE
# from imblearn.over_sampling import SVMSMOTE, KMeansSMOTE
# X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
# # X_train, y_train = BorderlineSMOTE(random_state=42).fit_resample(X_train, y_train)
# # #X_train, y_train = SVMSMOTE().fit_resample(X_train, y_train)
# print(Counter(y_train))


from mlxtend.classifier import StackingClassifier 
from sklearn.linear_model import LogisticRegression  
from sklearn.neighbors import KNeighborsClassifier     
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#clf= XGBClassifier().fit(X_train, y_train)
clf= XGBClassifier(max_depth=4,
                      learning_rate=0.085,
                      n_estimators=350,
                      subsample=0.85,
                      gamma=0).fit(X_train, y_train)
feature_contribution=clf.feature_importances_
#clf = KNeighborsClassifier(n_neighbors=20).fit(X_train, y_train)  
#clf = SVC(C=1000, kernel='rbf', gamma=1e-07).fit(X_train, y_train)  
# clf = RandomForestClassifier().fit(X_train, y_train)  

# clf1 = XGBClassifier(max_depth=4,
#                       learning_rate=0.075,
#                       n_estimators=400,
#                       subsample=0.85,
#                       gamma=0)
# clf2 = SVC(C=1000, kernel='rbf', gamma=1e-05) 
# clf3 = KNeighborsClassifier()
# clf4 = RandomForestClassifier()
  
# lr = LogisticRegression()  
# clf = StackingClassifier(classifiers=[clf1,clf4],   
#                             meta_classifier=lr).fit(X_train, y_train)  

X_test=test_profile.iloc[:,2:]

# del X_test['ELEC_TYPE_NAME']
# del X_test['VOLT_NAME']
# del X_test['PRC_NAME']
# del X_test['CHK_CYCLE']
# del X_test['daycount_5']
# del X_test['daycount_4']
# del X_test['daycount_3']
# del X_test['daycount_2']
# del X_test['monthcount_5']
# del X_test['monthcount_4']

#X_test=X_test.loc[test_profile["ELEC_TYPE_NAME"]==0]
col_name=X_test.columns.tolist()                   # 将数据框的列名全部提取出来存放在列表里
print(col_name)

col_name.insert(len(col_name),'ratio_1') 
col_name.insert(len(col_name)+1,'ratio_2') 
                  
X_test=X_test.reindex(columns=col_name) 
X_test['ratio_1']=X_test.loc[:, 'mean_day_consumption'].div(X_test.loc[:, 'RUN_CAP'].values, axis=0)  
X_test['ratio_2']=X_test.loc[:, 'mean_month_consumption'].div(X_test.loc[:, 'RUN_CAP'].values, axis=0)

X_test=X_test.values

X_test[:,5:6]=scaler1.fit_transform(X_test[:,5].reshape(-1, 1))

X_test[:,6:7]=scaler1.fit_transform(X_test[:,6].reshape(-1, 1))

X_test[:,7:8]=scaler1.fit_transform(X_test[:,7].reshape(-1, 1))
X_test[np.isnan(X_test)]=0
X_test[np.isinf(X_test)]=0



####特征选择###
#X_test=X_test[:,3:]

# X_test=pd.DataFrame(X_test)
# X_test.to_csv(r'D:/竞赛/新建文件夹/1.csv')
y_test=clf.predict(X_test)


col_name=test_profile.columns.tolist()  
col_name.insert(len(col_name),'label') 
test_profile=test_profile.reindex(columns=col_name) 
test_profile['label']=y_test.reshape(-1,1)

test_profile.to_csv(r'D:/竞赛/新建文件夹/label_visual.csv')


#test_ID=test_profile['ID'][test_profile["ELEC_TYPE_NAME"]==0]
test_ID=test_profile['ID']
test_ID=np.array(test_ID,dtype=np.int64).reshape(-1,1)
e=np.concatenate((test_ID,y_test.reshape(-1,1)),axis=1)
df=pd.DataFrame(e,columns=['id','label'])
label_data=submit_version.merge(df,how='left',on='id')
label_data=label_data.iloc[:,-1].values
a=sum(label_data==1)
#label_data[np.isnan(label_data)]=0
