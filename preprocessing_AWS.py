import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def convert_credit_history_age(x):
    try:
        years = int(str(x).split()[0])
        months = int(str(x).split()[3])
        return years * 12 + months
    except:
        return np.nan

if __name__ == "__main__":
    if os.path.exists("/opt/ml/processing"):
        input_dir = "/opt/ml/processing/ingested"
        out_train = "/opt/ml/processing/train"
        out_test = "/opt/ml/processing/test"
    else:
        input_dir = "./ingested"
        out_train = "./train"
        out_test = "./test"

    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_test, exist_ok=True)

    df = pd.read_csv(os.path.join(input_dir, "data_A.csv"))

    drop_cols = ["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN", "Month", "Occupation", "Type_of_Loan", "Credit_Mix", "Payment_Behaviour"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    numeric_object_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_of_Loan', 'Outstanding_Debt', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance', 'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Credit_Utilization_Ratio', 'Interest_Rate', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Delay_from_due_date']

    for col in numeric_object_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^0-9\.-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Num_of_Delayed_Payment' in df.columns:
        df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].astype(str).str.replace(r'[^0-9]', '', regex=True)
        df['Num_of_Delayed_Payment'] = pd.to_numeric(df['Num_of_Delayed_Payment'], errors='coerce')

    if 'Credit_History_Age' in df.columns:
        df['Credit_History_Age'] = df['Credit_History_Age'].apply(convert_credit_history_age)

    if 'Payment_of_Min_Amount' in df.columns:
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace('NM', np.nan)

    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    if 'Payment_of_Min_Amount' in df.columns:
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({'Yes': 1, 'No': 0}).fillna(0)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Credit_Score"])

    train_df.to_csv(os.path.join(out_train, "train.csv"), index=False)
    test_df.to_csv(os.path.join(out_test, "test.csv"), index=False)

    print("✅ Preprocessing complete.")