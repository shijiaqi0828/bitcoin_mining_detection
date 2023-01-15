# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 21:15:15 2022

@author: Administrator
"""
import numpy as np

#########Characterizing user daily power consumption##########
#########We select 12 holiday periods to compare the daily power consumption. 
#########Then we compute the daily statistics feature that meets the following rules in 12 holiday periods:
########(i) The average holiday power consumption and average workday power consumption exceed 90 kWh.
########(ii)The daily fluctuation coefficient is in the threshold interval ([lower_bound, upper_bound]) which is set as [0.7,1.3], [0.75,1.25], [0.8,1.2], [0.85,1.15], [0.9,1.1], [0.95,1.05].
def day_data_info(test_daydata_workday,upper_bound_day,lower_bound_day,electri_bound):
    number=test_daydata_workday.shape[0]
    holiday_workday_difference=np.zeros([number,12])
    holiday_workday_rate=np.zeros([number,12])
    holiday_mean=np.zeros([number,12])
    workday_mean=np.zeros([number,12])
    # upper_bound_day=1.2
    # lower_bound_day=0.8
    # electri_bound=100
    for i in range(int(number/107)):
        for j in range(12):
            ############Too much missing data is directly computed as 0###########
            if len(test_daydata_workday[0+i*107:107+107*i][np.isnan(test_daydata_workday[0+i*107:107+107*i])])>=28:
                holiday_workday_rate[i,j:j+12]=0
                holiday_workday_difference[i,j:j+12]=0
                break
           #####Spring Festival in 2020####
            elif j==0:
               workday_mean[i,j]=sum(test_daydata_workday[0+i*107:2+i*107,1]+test_daydata_workday[12+i*107:14+i*107,1])/4
               holiday_mean[i,j]=sum(test_daydata_workday[2+i*107:12+i*107,1]/10)
               holiday_workday_rate[i,j]=workday_mean[i,j]/holiday_mean[i,j]
               if(holiday_workday_rate[i,j]>lower_bound_day and holiday_workday_rate[i,j]<upper_bound_day and workday_mean[i,j]>electri_bound and holiday_mean[i,j]>electri_bound):
                  holiday_workday_difference[i,j]=1
               else:
                  holiday_workday_difference[i,j]=0
           #####Qingming in 2020####
            elif j==1:
                workday_mean[i,j]=sum(test_daydata_workday[14+i*107:16+i*107,1]+test_daydata_workday[19+i*107:21+i*107,1])/4
                holiday_mean[i,j]=sum(test_daydata_workday[16+i*107:19+i*107,1]/3)            
                holiday_workday_rate[i,j]=workday_mean[i,j]/holiday_mean[i,j]
                if(holiday_workday_rate[i,j]>lower_bound_day and holiday_workday_rate[i,j]<upper_bound_day and workday_mean[i,j]>electri_bound and holiday_mean[i,j]>electri_bound):
                  holiday_workday_difference[i,j]=1
                else:
                  holiday_workday_difference[i,j]=0
            #####May Day in 2020###   
            elif j==2:
                workday_mean[i,j]=sum(test_daydata_workday[21+i*107:23+i*107,1]+test_daydata_workday[28+i*107:30+i*107,1])/4
                holiday_mean[i,j]=sum(test_daydata_workday[23+i*107:28+i*107,1]/5)
                holiday_workday_rate[i,j]=workday_mean[i,j]/holiday_mean[i,j]
                if(holiday_workday_rate[i,j]>lower_bound_day and holiday_workday_rate[i,j]<upper_bound_day and workday_mean[i,j]>electri_bound and holiday_mean[i,j]>electri_bound):
                  holiday_workday_difference[i,j]=1
                else:
                  holiday_workday_difference[i,j]=0
           #####Dragon Boat Festival in 2020###   
            elif j==3:
                workday_mean[i,j]=sum(test_daydata_workday[30+i*107:32+i*107,1]+test_daydata_workday[35+i*107:37+i*107,1])/4
                holiday_mean[i,j]=sum(test_daydata_workday[32+i*107:35+i*107,1]/3)
                holiday_workday_rate[i,j]=workday_mean[i,j]/holiday_mean[i,j]
                if(holiday_workday_rate[i,j]>lower_bound_day and holiday_workday_rate[i,j]<upper_bound_day and workday_mean[i,j]>electri_bound and holiday_mean[i,j]>electri_bound):
                  holiday_workday_difference[i,j]=1
                else:
                  holiday_workday_difference[i,j]=0
           #####National Day in 2020###   
            elif j==4:
                workday_mean[i,j]=sum(test_daydata_workday[37+i*107:39+i*107,1]+test_daydata_workday[47+i*107:49+i*107,1])/4
                holiday_mean[i,j]=sum(test_daydata_workday[39+i*107:47+i*107,1]/8)
                holiday_workday_rate[i,j]=workday_mean[i,j]/holiday_mean[i,j]
    
                if(holiday_workday_rate[i,j]>lower_bound_day and holiday_workday_rate[i,j]<upper_bound_day and workday_mean[i,j]>electri_bound and holiday_mean[i,j]>electri_bound):
                 holiday_workday_difference[i,j]=1
                else:
                 holiday_workday_difference[i,j]=0
           #####New Year's Day in 2021##
            elif j==5:
                workday_mean[i,j]=sum(test_daydata_workday[49+i*107:51+i*107,1],test_daydata_workday[54+i*107,1])/3
                holiday_mean[i,j]=sum(test_daydata_workday[51+i*107:54+i*107,1]/3)
                holiday_workday_rate[i,j]=workday_mean[i,j]/holiday_mean[i,j]
    
                if(holiday_workday_rate[i,j]>lower_bound_day and holiday_workday_rate[i,j]<upper_bound_day and workday_mean[i,j]>electri_bound and holiday_mean[i,j]>electri_bound):
                   holiday_workday_difference[i,j]=1
                else:
                   holiday_workday_difference[i,j]=0
            #####Spring Festival in 2021###   
            elif j==6:
              workday_mean[i,j]=sum(test_daydata_workday[55+i*107:57+i*107,1]+test_daydata_workday[64+i*107:66+i*107,1])/4
              holiday_mean[i,j]=sum(test_daydata_workday[57+i*107:64+i*107,1]/7)
              holiday_workday_rate[i,j]=workday_mean[i,j]/holiday_mean[i,j]
    
              if(holiday_workday_rate[i,j]>lower_bound_day and holiday_workday_rate[i,j]<upper_bound_day and workday_mean[i,j]>electri_bound and holiday_mean[i,j]>electri_bound):
                 holiday_workday_difference[i,j]=1
              else:
                 holiday_workday_difference[i,j]=0
           #####Qingming in 2020###   
            elif j==7:
                workday_mean[i,j]=sum(test_daydata_workday[66+i*107:68+i*107,1]+test_daydata_workday[71+i*107:73+i*107,1])/4
                holiday_mean[i,j]=sum(test_daydata_workday[68+i*107:71+i*107,1]/3)
                holiday_workday_rate[i,j]=workday_mean[i,j]/holiday_mean[i,j]
    
                if(holiday_workday_rate[i,j]>lower_bound_day and holiday_workday_rate[i,j]<upper_bound_day and workday_mean[i,j]>electri_bound and holiday_mean[i,j]>electri_bound):
                   holiday_workday_difference[i,j]=1
                else:
                   holiday_workday_difference[i,j]=0
           #####May Day in 2021###   
            elif j==8:
                workday_mean[i,j]=sum(test_daydata_workday[73+i*107:75+i*107,1]+test_daydata_workday[80+i*107:82+i*107,1])/4
                holiday_mean[i,j]=sum(test_daydata_workday[75+i*107:80+i*107,1]/5)
                holiday_workday_rate[i,j]=workday_mean[i,j]/holiday_mean[i,j]
    
                if(holiday_workday_rate[i,j]>lower_bound_day and holiday_workday_rate[i,j]<upper_bound_day and workday_mean[i,j]>electri_bound and holiday_mean[i,j]>electri_bound):
                   holiday_workday_difference[i,j]=1
                else:
                   holiday_workday_difference[i,j]=0
           #####Dragon Boat Festival in 2021###   
            elif j==9:
                workday_mean[i,j]=sum(test_daydata_workday[82+i*107:84+i*107,1]+test_daydata_workday[87+i*107:89+i*107,1])/4
                holiday_mean[i,j]=sum(test_daydata_workday[84+i*107:87+i*107,1]/3)
                holiday_workday_rate[i,j]=workday_mean[i,j]/holiday_mean[i,j]
    
                if(holiday_workday_rate[i,j]>lower_bound_day and holiday_workday_rate[i,j]<upper_bound_day and workday_mean[i,j]>electri_bound and holiday_mean[i,j]>electri_bound):
                   holiday_workday_difference[i,j]=1
                else:
                   holiday_workday_difference[i,j]=0
           #####Mid Autumn Festival in 2021###   
            elif j==10:
                workday_mean[i,j]=sum(test_daydata_workday[89+i*107:91+i*107,1]+test_daydata_workday[94+i*107:96+i*107,1])/4
                holiday_mean[i,j]=sum(test_daydata_workday[91+i*107:94+i*107,1]/3)
                holiday_workday_rate[i,j]=workday_mean[i,j]/holiday_mean[i,j]
     
                if(holiday_workday_rate[i,j]>lower_bound_day and holiday_workday_rate[i,j]<upper_bound_day and workday_mean[i,j]>electri_bound and holiday_mean[i,j]>electri_bound):
                   holiday_workday_difference[i,j]=1
                else:
                   holiday_workday_difference[i,j]=0               
           #####National Day in 2021###  
            else:
                workday_mean[i,j]=sum(test_daydata_workday[96+i*107:98+i*107,1]+test_daydata_workday[105+i*107:107+i*107,1])/4
                holiday_mean[i,j]=sum(test_daydata_workday[98+i*107:105+i*107,1]/7)
                holiday_workday_rate[i,j]=workday_mean[i,j]/holiday_mean[i,j]
    
                if(holiday_workday_rate[i,j]>lower_bound_day and holiday_workday_rate[i,j]<upper_bound_day and workday_mean[i,j]>electri_bound and holiday_mean[i,j]>electri_bound):
                   holiday_workday_difference[i,j]=1
                else:
                   holiday_workday_difference[i,j]=0 
    return holiday_workday_rate,holiday_workday_difference



#########Characterizing user monthly power consumption##########

#########We select 22 month periods to compare the power consumption. 
#########Then we compute the statistics feature that meets the following rules in 22 month periods.
########(i) Monthly power consumption exceed 3000 kwh.
########(ii)The monthly fluctuation coefficient is in the threshold interval ([lower_bound, upper_bound]) which is set as [0.7,1.3], [0.75,1.25], [0.8,1.2], [0.85,1.15], [0.9,1.1], [0.95,1.05].



def month_data_info(monthdata,upper_bound_month,lower_bound_month,electri_bound):
    test_monthdata=np.zeros(monthdata.shape)
    for i in range (test_monthdata.shape[0]):
        test_monthdata[i,2:5]=monthdata[i,2:5]/(monthdata[i,5]/3)
        if ((test_monthdata[i,2]>lower_bound_month and test_monthdata[i,2]<upper_bound_month) 
            and (test_monthdata[i,3]>lower_bound_month and test_monthdata[i,3]<upper_bound_month) 
            and (test_monthdata[i,4]>lower_bound_month and test_monthdata[i,4]<upper_bound_month) 
            and (monthdata[i,5]>electri_bound)): 
            test_monthdata[i,5]=1
        else:
            test_monthdata[i,5]=0
    
    month_number=test_monthdata.shape[0]/22
    month_number=int(month_number) 
    mean_month_consumption=np.zeros([month_number,1], dtype=np.float32) 
    label_test=np.zeros((month_number,1), dtype=np.int)
    for i in range(month_number):
        mean_month_consumption[i,:]=np.mean(monthdata[i*22:i*22+22][:,-1])
        label_test[i,:]=sum(test_monthdata[22*i:22*i+22,5])
    return mean_month_consumption, label_test

