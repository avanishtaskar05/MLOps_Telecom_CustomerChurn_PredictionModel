import pandas as pd
def data_ingestion():
    '''
    * Independent Columns [X]:
       'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges'

    * Dependent Columns [Y]:
       'Churn'

    * Problem Statement:
    1. To predict the churn rate
    2. Use Clustering approach to predict and classify the cluster pattern such as {l1,l2,l3}
    '''
    df = pd.read_csv(r'C:\MLOps_Telecom_CustomerChurn_PredictionModel\data\raw\churn - churn.csv')
    return df