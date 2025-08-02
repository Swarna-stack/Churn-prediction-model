# Churn-prediction-model
import pandas as pd
import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
df=pd.read_csv("customer_churn.csv")
df.drop('customerID',axis='columns',inplace=True)#to drop unnecessary columns
df1=df[df.TotalCharges!='']#ro drop those columns which are null
df1.TotalCharges=pd.to_numeric(df1.TotalCharges,errors='coerce')#coerce to ignore any null 
tenure_churn_no=df1[df1.Churn=='No'].tenure
tenure_churn_yes=df1[df1.Churn=='Yes'].tenure
df1.replace('No internet service','No',inplace=True)
df1.replace('No phone service','No',inplace=True)
yes_no_column=['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for column in yes_no_column:
    replaced_series=df1[column].replace({'Yes':1,'No':0})
    df1[column]=pd.to_numeric(replaced_series,downcast='integer')
df1['gender']=pd.to_numeric(df['gender'].replace({'Female':1,'Male':0}),downcast='integer')
df2=pd.get_dummies(data=df1,columns=['InternetService','Contract','PaymentMethod'])
df2['TotalCharges']=pd.to_numeric(df2['TotalCharges'],errors='coerce')
df2.dropna(inplace=True)
cols_to_scale=['tenure','MonthlyCharges','TotalCharges']
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df2[cols_to_scale]=scaler.fit_transform(df2[cols_to_scale])
from sklearn.model_selection import train_test_split
X=df2.drop('Churn',axis='columns')
y=df2['Churn']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=keras.Sequential([keras.layers.Dense(20,input_shape=(26,),activation='relu'),keras.layers.Dense(15,activation='relu'),keras.layers.Dense(1,activation='sigmoid')])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=100)
model.evaluate(X_test,y_test)
y_predicted=model.predict(X_test)
y_pred=np.round(y_predicted)
print(y_pred[:5])
print(y_test[:5])










 

















