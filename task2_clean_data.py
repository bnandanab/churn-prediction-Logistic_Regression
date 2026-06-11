import pandas as pd
df=pd.read_csv("churnguard_data.csv")
df=df.drop("customerID",axis=1)
df = df.drop_duplicates()
df["gender"]=df["gender"].str.strip()
df["PaymentMethod"] = df["PaymentMethod"].str.strip()
df["Churn"]=df["Churn"].str.strip().str.title()
df["PhoneService"]=df["PhoneService"].str.strip().str.title()
df["PaperlessBilling"]=df["PaperlessBilling"].str.strip().str.title()
df["Contract"]=df["Contract"].str.lower().str.strip()
print(df["Contract"].unique().tolist())
df["Contract"]=df["Contract"].replace({"month-to-month":"Month-to-month","month to month":"Month-to-month","monthly":"Month-to-month",'one year':"One year",'1 year':"One year",'two year':"Two year",'1 year':"Two year"})
print(df["Contract"].isna().sum())
#print(df.loc[df["Contract"].str.startswith("Mon"),"Contract"]="Monthly" #df.loc used to select rows, filter and modify like this )
print(df["InternetService"].value_counts().index.tolist()) #DSL, Fiber optic, No
df["InternetService"]=df["InternetService"].str.strip().replace({'fiber optic':'Fiber optic','FiberOptic':'Fiber optic','dsl':'DSL','Dsl':'DSL','DSl':'DSL','NO':'No','NO':'No'})
df["InternetService"]=df["InternetService"].fillna(df["InternetService"].mode()[0])
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
df=df[df['tenure']>0]
df=df[(df["MonthlyCharges"]>10) | (df["MonthlyCharges"]<200)]
'''Fill missing values:
MonthlyCharges → column mean
TotalCharges → column mean
tenure → column median (use integer rounding)'''
df["MonthlyCharges"]=df["MonthlyCharges"].fillna(df["MonthlyCharges"].mean())
df['TotalCharges']=df['TotalCharges'].fillna(df['TotalCharges'].mean())
df['tenure']=df['tenure'].fillna(df['tenure'].median())
print(df.shape)
print(df.isna().sum())
print(df[df["Contract"].isna()])