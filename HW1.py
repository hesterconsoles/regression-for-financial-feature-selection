import pandas as pd
import numpy as np
from operator import itemgetter
pd.set_option('display.max_columns', None)

#Merging FF,HXZ,PS for lag factors
df_FF=pd.read_csv('Fama French 5 Factors.CSV',sep=',',header=2,engine='python',)[0:628] #196307-201510
df_HXZ= pd.read_excel('HXZ q-Factors (monthly 1967 to 2014).xlsx') #install openpysl #196701-201412
df_PS=pd.read_csv('Pastor Stambaugh Factors.csv') #196307-201510
#df_merge=df_FF[42:628]
df_merge=df_FF[42:618].reset_index(drop=True)
df_merge=pd.concat([df_merge,df_HXZ[['ME',"I/A","ROE"]]],axis=1)
df_merge=pd.concat([df_merge,df_PS[53:].reset_index()[['PS_LEVEL',"PS_INNOV","PS_VWF"]]],axis=1)
#df_merge['PS_LEVEL']=np.array(df_merge['PS_LEVEL'])*100
#df_merge['PS_INNOV']=np.array(df_merge['PS_INNOV'])*100
#df_merge['PS_VWF']=np.array(df_merge['PS_VWF'])*100

#print(df_merge)
writer=pd.ExcelWriter('Merging data.xlsx',engine='xlsxwriter')
df_merge.to_excel(writer)
writer.save()


#Question 1 the lagged factors with 1,2,3,4,5,6 months of lag of the original factors.
for i in range(6):
    i=i+1
    df_merge['%d lag Mkt-RF'%(i)]= df_merge['Mkt-RF'].shift(i)
    df_merge['%d lag SMB'%(i)]= df_merge['SMB'].shift(i)
    df_merge['%d lag HML'%(i)]= df_merge['HML'].shift(i)
    df_merge['%d lag RMW'%(i)]= df_merge['RMW'].shift(i)
    df_merge['%d lag CMA'%(i)]= df_merge['CMA'].shift(i)
    df_merge['%d lag RF'%(i)]= df_merge['RF'].shift(i)
    df_merge['%d lag ME'%(i)]= df_merge['ME'].shift(i)
    df_merge['%d lag I/A'%(i)]= df_merge['I/A'].shift(i)
    df_merge['%d lag ROE'%(i)]= df_merge['ROE'].shift(i)
    df_merge['%d lag PS_LEVEL'%(i)]= df_merge['PS_LEVEL'].shift(i)
    df_merge['%d lag PS_INNOV'%(i)]= df_merge['PS_INNOV'].shift(i)
    df_merge['%d lag PS_VWF'%(i)]= df_merge['PS_VWF'].shift(i)

writer=pd.ExcelWriter('Merging data_ %dM lag.xlsx'%(i),engine='xlsxwriter')
df_merge.to_excel(writer)
writer.save()


#Question 2. Lasso regression for Fama-French regression
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df_stock=pd.read_csv('cleaned_data.csv')
df_Lasso_train=df_merge[264:493].drop(['Unnamed: 0','1 lag RF','2 lag RF','3 lag RF','4 lag RF','5 lag RF','6 lag RF','RF'],axis=1)
df_Lasso_test=df_merge[493:552].drop(['Unnamed: 0','1 lag RF','2 lag RF','3 lag RF','4 lag RF','5 lag RF','6 lag RF','RF'],axis=1)
rf_train=[float(i) for i in df_merge[264:493]['RF']]
rf_test=[float(i) for i in df_merge[493:552]['RF']]
names=df_Lasso_train.columns
#print(df_Lasso_train)

#find the start date and end date
split_index=[]
start_date=df_stock['time_stamp'][0]
for i in range(len(df_stock['time_stamp'])):
    if df_stock['time_stamp'][i]>start_date:
        end_date=df_stock['time_stamp'][i]
#print(end_date)
for i in range(len(df_stock['time_stamp'])):
    if df_stock['time_stamp'][i]==end_date:
        split_index.append(i)
#print(split_index)

#Lasso regression for each stock
modify_index=[i+1 for i in split_index]
modify_index.insert(0,0)
#print(modify_index)
train_mse=[]
test_mse=[]
train_score=[]
test_score=[]
factor_freq=[0 for i in range(77)] #There are 78 factors
for i in range(len(modify_index)-1):
    df_stock_asset=df_stock[modify_index[i]:modify_index[i+1]]
    #separate the data into training set and testing set
    for j in range(len(df_stock_asset['time_stamp'])):
         if df_stock_asset['time_stamp'].tolist()[j]==20080131:
             train_index=j+1
             break
    train_data=df_stock_asset[:train_index]
    test_data=df_stock_asset[train_index:]

    train_x,train_y=df_Lasso_train.values,np.array(np.array(train_data["return"].tolist())-np.array(rf_train))
    model=LassoCV().fit(train_x,train_y) #cv=5
    coef=model.coef_
    train_y_predict=model.predict(train_x) #training
    train_mse.append(mean_squared_error(train_y,train_y_predict)) #mse of training
    test_x,test_y=df_Lasso_test.values,np.array(np.array(test_data["return"].tolist())-np.array(rf_test))
    test_y_predict = model.predict(test_x) #testing
    test_mse.append(mean_squared_error(test_y,test_y_predict)) #mse of testing

    factor=[1 if i!=0 else 0 for i in coef]
    factor_freq=np.array(factor_freq)+np.array(factor)

    train_score.append(model.score(train_x,train_y))
    test_score.append(model.score(test_x,test_y))

'''
    plt.plot(range(len(coef)),coef,label="Stock ID %d"%(train_data["stock_id"][i]))
    plt.xticks(range(len(coef)),names,rotation='vertical')
    plt.ylabel("Feature Importance")
    plt.legend()

plt.show()'''

# Question 3. call train_mse, test_mse, and find the average
stock_id=[]
for i in split_index:
    stock_id.append(df_stock["stock_id"][i])
#print("stock_id:", stock_id)
#print("train MSE:", train_mse)
#print("test MSE:", test_mse)
print("average train MSE:",np.array(train_mse).mean())
print("average test MSE:",np.array(test_mse).mean())
print('avg train r2:',np.array(train_score).mean())
print('avg test r2:',np.array(test_score).mean())
#Question 4. The Collection of coefficients

stock_num=len(stock_id)
print("Number of stock",stock_num)
print("The factors:",names)
print("The factor selection frequency:",factor_freq)

percentage=[i for i in (np.array(factor_freq)/np.array(stock_num)*100)]
percentage= ['{}%'. format(round(i)) for i in percentage]
print("Collection Perentage of each feature in %:",percentage)


factorNfreq=[]
factorNfreq=[[i,j] for i, j in zip(names,percentage)]
factorNfreq.sort(key=itemgetter(1),reverse=True)
print("Factors in ranking:",factorNfreq)

plt.bar(names,factor_freq,width=1,align='center',edgecolor="black",linewidth=0.8)
plt.title("Factors Selection Frequency")
plt.xticks(range(len(coef)),names,rotation='vertical',fontsize="xx-small")
plt.gca().margins(x=0)
for i,j,k in zip(range(len(coef)),factor_freq,percentage):
    plt.text(i,j+1,k,ha='center',va="bottom",fontsize="xx-small")

plt.show()



