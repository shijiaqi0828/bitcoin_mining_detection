# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 21:14:59 2022

@author: Administrator
"""
import time
import numpy as np
from coinfunction import month_data_info
from coinfunction import day_data_info
import pandas as pd
import warnings

start =time.time() 

warnings.filterwarnings('ignore')

##############Dataset loading##############
train_daydata = pd.read_csv(r"D:\1.paper\2022\挖矿\12.31\bitcoin_mining_detection\Train_dataset_daily_power_consumption.csv")
test_daydata= pd.read_csv(r"D:\1.paper\2022\挖矿\12.31\bitcoin_mining_detection\Test_dataset_daily_power_consumption.csv")

daydata=pd.concat([train_daydata , test_daydata], axis=0)

train_monthdata = pd.read_csv(r"D:\1.paper\2022\挖矿\12.31\bitcoin_mining_detection\Train_dataset_monthly_power_consumption.csv")
test_monthdata= pd.read_csv(r"D:\1.paper\2022\挖矿\12.31\bitcoin_mining_detection\Test_dataset_monthly_power_consumption.csv")

monthdata=pd.concat([train_monthdata , test_monthdata], axis=0)


data_profile=pd.read_csv(r"D:\1.paper\2022\挖矿\12.31\bitcoin_mining_detection\user_profile.csv")

data_profile["ELEC_TYPE_NAME"] = data_profile["ELEC_TYPE_NAME"].map({"Urban residential":0, 
                                                                     "Rural residential":1,
                                                                     "Non residential lighting":2,
                                                                     "Campus":3,
                                                                     "Industry":4,
                                                                     "Residential":5,
                                                                     "Non industry":6,
                                                                     "Commercial use":7,
                                                                     "Large industry":8,
                                                                     "Agricultural production":9})
data_profile["VOLT_NAME"] = data_profile["VOLT_NAME"].map({"AC 220V":0, 
                                                           "AC 380V":1,
                                                           "AC 10kV":2,
                                                           "AC 6kV":3})
data_profile["PRC_NAME"] = data_profile["PRC_NAME"].map({"<1kV":0,
                                                         "<10kV,>1kV":1})

del data_profile['SHIFT_NO']
del data_profile['CANCEL_DATE']
del data_profile['TMP_NAME']
del data_profile['TMP_DATE']



train_daydata=daydata
train_monthdata=monthdata
train_profile=data_profile
###############################################################################################
##########################Traingdata preprocessing(daily power consumption)####################
train_daydata= train_daydata[~(train_daydata['rq'] == '2021-01-05 00:00:00')]#The daily power consumption information on January 5 was deleted due to the loss of all users' information
number_id=train_daydata['id'].value_counts()
number_normal_training=number_id.loc[number_id.values==107]
training_sample_number=number_normal_training.shape[0]
train_daydata_1=train_daydata.loc[train_daydata['id'].isin(number_normal_training.index)]
train_daydata_1['kwh'].fillna(method='backfill', inplace=True,limit=3)
del train_daydata_1['kwh_rap']
del train_daydata_1['kwh_pap_r1']
del train_daydata_1['rq']

train_day_ID=train_daydata_1.iloc[0::107,0]
train_day_ID=np.array(train_day_ID,dtype=np.int64).reshape(-1,1)
daydata=np.array(train_daydata_1,dtype=np.float32)
#########Characterizing user daily power consumption##########
#########We select 12 holiday periods to compare the daily power consumption. 
#########Then we compute the daily statistics feature that meets the following rules in 12 holiday periods:
########(i) The average holiday power consumption and average workday power consumption exceed 90 kWh.
########(ii)The daily fluctuation coefficient is in the threshold interval ([lower_bound, upper_bound]) which is set as [0.7,1.3], [0.75,1.25], [0.8,1.2], [0.85,1.15], [0.9,1.1], [0.95,1.05].
rate,holiday_workday_difference_5=day_data_info(daydata,1.3,0.7,90)
rate,holiday_workday_difference_4=day_data_info(daydata,1.25,0.75,90)
rate,holiday_workday_difference_3=day_data_info(daydata,1.2,0.8,90)
rate,holiday_workday_difference_2=day_data_info(daydata,1.15,0.85,90)
rate,holiday_workday_difference_1=day_data_info(daydata,1.1,0.9,90)
rate,holiday_workday_difference_0=day_data_info(daydata,1.05,0.95,90)

rate[np.isnan(rate)]=0

day_label_count_5=np.zeros((training_sample_number,1), dtype=np.int)
day_label_count_4=np.zeros((training_sample_number,1), dtype=np.int)
day_label_count_3=np.zeros((training_sample_number,1), dtype=np.int)
day_label_count_2=np.zeros((training_sample_number,1), dtype=np.int)
day_label_count_1=np.zeros((training_sample_number,1), dtype=np.int)
day_label_count_0=np.zeros((training_sample_number,1), dtype=np.int)
mean_day_consumption=np.zeros((training_sample_number,1), dtype=np.float16)

for i in range(int(training_sample_number)):
    day_label_count_5[i]=sum(holiday_workday_difference_5[i,:])
    day_label_count_4[i]=sum(holiday_workday_difference_4[i,:])
    day_label_count_3[i]=sum(holiday_workday_difference_3[i,:])
    day_label_count_2[i]=sum(holiday_workday_difference_2[i,:])
    day_label_count_1[i]=sum(holiday_workday_difference_1[i,:])
    day_label_count_0[i]=sum(holiday_workday_difference_0[i,:])

mean_day_consumption=np.zeros([training_sample_number,1])        
test_day_no_missing=train_daydata_1['kwh'].fillna(method='backfill', inplace=False)
test_day_no_missing=np.array(test_day_no_missing,dtype=np.float16).reshape(-1,1)
for i in range(int(training_sample_number)):
    mean_day_consumption[i,:]=np.mean(test_day_no_missing[107*i:107*i+107,:])
       

sum_train_daydata=np.concatenate((train_day_ID, day_label_count_5.reshape(-1,1),day_label_count_4.reshape(-1,1),
                        day_label_count_3.reshape(-1,1),day_label_count_2.reshape(-1,1),day_label_count_1.reshape(-1,1),
                        day_label_count_0.reshape(-1,1),mean_day_consumption.reshape(-1,1)),axis=1)
label_day_data=pd.DataFrame(sum_train_daydata,columns=['ID','daycount_5','daycount_4','daycount_3','daycount_2','daycount_1','daycount_0','mean_day_consumption',])
train_profile=train_profile.merge(label_day_data,how='left',on='ID')



################################################################################################
##############################Traingdata preprocessing(monthly power consumption)###############

train_monthdata_ID=train_monthdata.iloc[0::22,0]
train_monthdata=np.array(train_monthdata,dtype=np.float32)
#########Characterizing user monthly power consumption##########

#########We select 22 month periods to compare the power consumption. 
#########Then we compute the statistics feature that meets the following rules in 22 month periods.
########(i) Monthly power consumption exceed 3000 kwh.
########(ii)The monthly fluctuation coefficient is in the threshold interval ([lower_bound, upper_bound]) which is set as [0.7,1.3], [0.75,1.25], [0.8,1.2], [0.85,1.15], [0.9,1.1], [0.95,1.05].
mean_month_consumption, label_test_5=month_data_info(train_monthdata,1.3,0.7,3000)
mean_month_consumption, label_test_4=month_data_info(train_monthdata,1.25,0.75,3000)   
mean_month_consumption, label_test_3=month_data_info(train_monthdata,1.2,0.8,3000)       
mean_month_consumption, label_test_2=month_data_info(train_monthdata,1.15,0.85,3000)      
mean_month_consumption, label_test_1=month_data_info(train_monthdata,1.1,0.9,3000)    
mean_month_consumption, label_test_0=month_data_info(train_monthdata,1.05,0.95,3000)    

train_monthdata_ID=np.array(train_monthdata_ID,dtype=np.int64).reshape(-1,1)
sum_train_monthdata=np.concatenate((train_monthdata_ID, label_test_5,label_test_4,label_test_3,label_test_2,
                                  label_test_1,label_test_0,mean_month_consumption.reshape(-1,1)),axis=1)
label_month_data=pd.DataFrame(sum_train_monthdata,columns=['ID','monthcount_5','monthcount_4','monthcount_3','monthcount_2','monthcount_1','monthcount_0','mean_month_consumption'])
train_profile=train_profile.merge(label_month_data,how='left',on='ID')



#####################Training module#############
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_profile= train_profile.fillna(0)
X_train=train_profile.iloc[:,1:]
y=X_train['IS_FLAG']
y=y.values

del X_train['IS_FLAG']
col_name=X_train.columns.tolist()
col_name.insert(len(col_name),'ratio_1') 
col_name.insert(len(col_name)+1,'ratio_2') 
                  
X_train=X_train.reindex(columns=col_name)
###########Daily power consumption divided by contract capacity######### 
X_train['ratio_1']=X_train.loc[:, 'mean_day_consumption'].div(X_train.loc[:, 'RUN_CAP'].values, axis=0)  
###########Monthly power consumption divided by contract capacity######### 
X_train['ratio_2']=X_train.loc[:, 'mean_month_consumption'].div(X_train.loc[:, 'RUN_CAP'].values, axis=0)

X_train= X_train.fillna(0)
X_train=X_train.replace(np.inf, 0)
# # del X_train['ELEC_TYPE_NAME']
# # del X_train['VOLT_NAME']
# # del X_train['PRC_NAME']
# # del X_train['CHK_CYCLE']
# # del X_train['daycount_5']
# # del X_train['daycount_4']
# # del X_train['daycount_3']
# # del X_train['daycount_2']
# # del X_train['monthcount_5']
# # del X_train['monthcount_4']
X_train=X_train.values
#x_train, x_test, y_train,y_test = train_test_split(X_train, y, test_size=0.25, random_state=100)
#########Data normalization#########
scaler1= MinMaxScaler()  # User account building year
X_train[:,5:6]=scaler1.fit_transform(X_train[:,5].reshape(-1, 1))
scaler2= MinMaxScaler()   # Inspection cycle
X_train[:,6:7]=scaler2.fit_transform(X_train[:,6].reshape(-1, 1))
scaler3= MinMaxScaler()   # 上Last checking year
X_train[:,7:8]=scaler3.fit_transform(X_train[:,7].reshape(-1, 1))
#########Dataset_split#########
x_train=X_train[:8840,:]
x_test=X_train[8840:,:]
y_train=y[:8840]
y_test=y[8840:]
###
from memory_profiler import profile
@profile
def train_model (x,y):
         clf= XGBClassifier(max_depth=4,
                       learning_rate=0.085, 
                       n_estimators=350,
                       subsample=0.85,
                       gamma=0).fit(x, y) 
         return clf


########Imbalanced class by SMOTE##############
#from imblearn.over_sampling import SMOTE, ADASYN

from imblearn.over_sampling import BorderlineSMOTE,SMOTE
from imblearn.over_sampling import SVMSMOTE, KMeansSMOTE
#x_train, y_train = SMOTE(random_state=42).fit_resample(x_train, y_train)
x_train, y_train = BorderlineSMOTE().fit_resample(x_train, y_train)
#x_train, y_train = SVMSMOTE().fit_resample(x_train, y_train)

#######################Training module############
#start =time.time() 
clf=train_model(x_train,y_train)
y_pred=clf.predict(x_test)
f1= f1_score(y_pred, y_test, average='macro')
end = time.time()
print('Running time: %s Seconds'%(end-start))
print('F1: %s '%(f1))
