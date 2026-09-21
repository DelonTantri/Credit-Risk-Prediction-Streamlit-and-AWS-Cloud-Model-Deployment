from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class CreditScorePreprocessor:

    def __init__(self):
        self.artifact_dir = Path("artifacts")
        self.artifact_dir.mkdir(exist_ok=True, parents=True)

    def convert_credit_history_age(self, x):
        try:
            years, months = int(str(x).split()[0]), int(str(x).split()[3])
            return years * 12 + months
        except (ValueError, IndexError):
            return np.nan

    def process(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset tidak ditemukan:\n{path.resolve()}")

        print("\nPREPROCESSING\n")
        print(f"Reading Dataset : {path.resolve()}")

        data = pd.read_csv(path)
        print(f"Original Shape : {data.shape}")

        drop_cols = ["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN", "Month", "Occupation", "Type_of_Loan", "Credit_Mix", "Payment_Behaviour"]
        data.drop(columns=[c for c in drop_cols if c in data.columns], inplace=True)

        numeric_object_cols = ["Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_of_Loan", "Outstanding_Debt", "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance", "Changed_Credit_Limit", "Num_Credit_Inquiries", "Credit_Utilization_Ratio", "Interest_Rate", "Num_Bank_Accounts", "Num_Credit_Card", "Delay_from_due_date"]

        for col in numeric_object_cols:
            if col in data.columns:
                data[col] = data[col].astype(str).str.replace(r"[^0-9\.-]", "", regex=True)
                data[col] = pd.to_numeric(data[col], errors="coerce")

        if "Num_of_Delayed_Payment" in data.columns:
            data["Num_of_Delayed_Payment"] = data["Num_of_Delayed_Payment"].astype(str).str.replace(r"[^0-9]", "", regex=True)
            data["Num_of_Delayed_Payment"] = pd.to_numeric(data["Num_of_Delayed_Payment"], errors="coerce")

        replace_map = {"Changed_Credit_Limit": "_", "Payment_of_Min_Amount": "NM"}
        for col, value in replace_map.items():
            if col in data.columns:
                data[col] = data[col].replace(value, np.nan)

        if "Credit_History_Age" in data.columns:
            data["Credit_History_Age"] = data["Credit_History_Age"].apply(self.convert_credit_history_age)

        if "Payment_of_Min_Amount" in data.columns:
            data["Payment_of_Min_Amount"] = data["Payment_of_Min_Amount"].map({"Yes": 1, "No": 0})

        print(f"Missing Before : {data.isnull().sum().sum()}")
        for col in data.columns:
            if data[col].dtype in ["int64", "float64"]:
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = data[col].fillna(data[col].mode()[0])
        print(f"Missing After  : {data.isnull().sum().sum()}")

        numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
        for col in numeric_cols:
            data[col] = np.where(data[col] < 0, data[col].median(), data[col])

        for col in numeric_cols:
            q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - (1.5 * iqr), q3 + (1.5 * iqr)
            data[col] = np.where(data[col] < lower, lower, data[col])
            data[col] = np.where(data[col] > upper, upper, data[col])

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(data["Credit_Score"])
        X = data.drop(columns=["Credit_Score"])
        feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        joblib.dump(X_train, self.artifact_dir / "X_train.pkl")
        joblib.dump(X_test, self.artifact_dir / "X_test.pkl")
        joblib.dump(y_train, self.artifact_dir / "y_train.pkl")
        joblib.dump(y_test, self.artifact_dir / "y_test.pkl")
        joblib.dump(label_encoder, self.artifact_dir / "label_encoder.pkl")
        joblib.dump(feature_names, self.artifact_dir / "feature_names.pkl")

        print("\nArtifacts Saved")
        print("\nPreprocessing Completed Successfully")

        return X_train, X_test, y_train, y_test, label_encoder, feature_names

if __name__ == "__main__":
    dataset_path = Path("ingested") / "data_A.csv"
    preprocessor = CreditScorePreprocessor()
    X_train, X_test, y_train, y_test, _, feature_names = preprocessor.process(dataset_path)

    print("\nPREPROCESSING SUCCESS\n")
    print(f"Train Shape : {X_train.shape}\nTest Shape  : {X_test.shape}\nFeatures    : {len(feature_names)}")