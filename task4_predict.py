#Task-4

import joblib
import pandas as pd

model = joblib.load('churn_model.pkl')
import pandas as pd

userdf = pd.DataFrame([{
    "tenure": 24,
    "MonthlyCharges": 65.50,
    "TotalCharges": 1500.00,
    "SeniorCitizen": 0,
    "Contract": 1,
    "gender": 1,
    "Partner": 0,
    "Dependents": 0,
    "PhoneService": 1,
    "PaperlessBilling": 1,
    "InternetService": 2,
    "OnlineSecurity": 0,
    "OnlineBackup": 1,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 1,
    "StreamingMovies": 1
}])

prediction = model.predict(userdf)

if prediction == 1:
    print("Prediction: This customer is likely to CHURN.")
else:
    print("Prediction: This customer is likely to STAY.")
