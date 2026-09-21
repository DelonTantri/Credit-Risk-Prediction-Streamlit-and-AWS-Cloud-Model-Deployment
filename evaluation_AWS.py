import os
import json
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    if os.path.exists("/opt/ml/processing"):
        model_path = "/opt/ml/processing/model/model_credit_score.joblib"
        encoder_path = "/opt/ml/processing/model/label_encoder.joblib"
        test_path = "/opt/ml/processing/test/test.csv"
        output_dir = "/opt/ml/processing/evaluation"
    else:
        model_path = "./model/model_credit_score.joblib"
        encoder_path = "./model/label_encoder.joblib"
        test_path = "./test/test.csv"
        output_dir = "./eval"

    os.makedirs(output_dir, exist_ok=True)

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    test_df = pd.read_csv(test_path)

    X_test = test_df.drop("Credit_Score", axis=1)
    y_test = encoder.transform(test_df["Credit_Score"])

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    report_dict = {"classification_metrics": {"accuracy": {"value": float(acc)}}}

    with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
        json.dump(report_dict, f)

    print(f"✅ Accuracy: {acc}")