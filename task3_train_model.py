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
df["Contract"]=df["Contract"].replace({"month-to-month":"Month-to-month","month to month":"Month-to-month","monthly":"Month-to-month",'one year':"One year",'1 year':"One year",'two year':"Two year",'1 year':"Two year"})
#print(df.loc[df["Contract"].str.startswith("Mon"),"Contract"]="Monthly" #df.loc used to select rows, filter and modify like this )
#print(df["InternetService"].value_counts().index.tolist()) #DSL, Fiber optic, No
df["InternetService"]=df["InternetService"].str.strip().replace({'fiber optic':'Fiber optic','FiberOptic':'Fiber optic','dsl':'DSL','Dsl':'DSL','DSl':'DSL','NO':'No','NO':'No'})
df["InternetService"]=df["InternetService"].fillna(df["InternetService"].mode()[0])
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
df=df[df['tenure']>0]
df=df[(df["MonthlyCharges"]>10) | (df["MonthlyCharges"]<200)]
df["MonthlyCharges"]=df["MonthlyCharges"].fillna(df["MonthlyCharges"].mean())
df['TotalCharges']=df['TotalCharges'].fillna(df['TotalCharges'].mean())
df['tenure']=df['tenure'].fillna(df['tenure'].median())



#TASK 3
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
le = LabelEncoder()
df["Churn"] = le.fit_transform(df["Churn"])
df=pd.get_dummies(df,columns=["gender", "PhoneService", "InternetService", "Contract", "PaperlessBilling", "PaymentMethod"],drop_first=True)
X=df.drop("Churn", axis=1)
y=df["Churn"]
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=42)
scaler=StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled,y_train)
y_pred_test = lr_model.predict(X_test_scaled)
print("Accuracy: ",accuracy_score(y_test,y_pred_test))
print("Classification report:\n",classification_report(y_test,y_pred_test,target_names=['Stay', 'Churn']))

import joblib

joblib.dump(lr_model,'churn_model.pkl')
print("dumped model means saved the model")


