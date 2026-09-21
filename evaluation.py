import json
import joblib
import mlflow
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

class CreditScoreEvaluator:

    def __init__(self):
        self.artifact_dir = Path("artifacts")
        self.eval_dir = Path("eval")
        self.eval_dir.mkdir(exist_ok=True, parents=True)

    def run(self):
        model = joblib.load(self.artifact_dir / "model.pkl")
        X_test = joblib.load(self.artifact_dir / "X_test.pkl")
        y_test = joblib.load(self.artifact_dir / "y_test.pkl")

        with open(self.artifact_dir / "run_id.txt", "r") as file:
            run_id = file.read().strip()

        print("MODEL EVALUATION")

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted")
        recall = recall_score(y_test, predictions, average="weighted")
        f1 = f1_score(y_test, predictions, average="weighted")
        report = classification_report(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)

        report_path = self.eval_dir / "classification_report.txt"
        with open(report_path, "w") as file:
            file.write(report)

        cm_path = self.eval_dir / "confusion_matrix.csv"
        pd.DataFrame(cm).to_csv(cm_path, index=False)

        metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
        metrics_path = self.eval_dir / "metrics.json"
        with open(metrics_path, "w") as file:
            json.dump(metrics, file, indent=4)

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision_weighted", precision)
            mlflow.log_metric("recall_weighted", recall)
            mlflow.log_metric("f1_weighted", f1)
            mlflow.log_artifact(report_path)
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(metrics_path)

        print(report)
        print("\nConfusion Matrix\n", cm)
        print(f"\nEvaluation Metrics\nAccuracy  : {accuracy:.4f}\nPrecision : {precision:.4f}\nRecall    : {recall:.4f}\nF1 Score  : {f1:.4f}")
        print("\nEvaluation files saved in /eval")

        return metrics

if __name__ == "__main__":
    evaluator = CreditScoreEvaluator()
    evaluator.run()