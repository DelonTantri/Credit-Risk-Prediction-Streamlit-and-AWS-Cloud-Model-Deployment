import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

class CreditScoreTrainer:

    def __init__(self):
        self.artifact_dir = Path("artifacts")
        self.artifact_dir.mkdir(exist_ok=True, parents=True)

    def run(self):
        X_train = joblib.load(self.artifact_dir / "X_train.pkl")
        X_test = joblib.load(self.artifact_dir / "X_test.pkl")
        y_train = joblib.load(self.artifact_dir / "y_train.pkl")
        feature_names = joblib.load(self.artifact_dir / "feature_names.pkl")

        print("\nMODEL TRAINING\n")
        print(f"Training Data : {X_train.shape}\nTesting Data  : {X_test.shape}")

        mlflow.set_experiment("Credit Score Classification")

        with mlflow.start_run() as run:
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            mlflow.log_param("model", "RandomForest")
            mlflow.log_param("random_state", 42)
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))

            mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
            joblib.dump(model, self.artifact_dir / "model.pkl")

            with open(self.artifact_dir / "run_id.txt", "w") as file:
                file.write(run.info.run_id)

            print(f"\nModel Training Completed\nRun ID : {run.info.run_id}\nModel saved to artifacts.")
            return run.info.run_id

if __name__ == "__main__":
    trainer = CreditScoreTrainer()
    trainer.run()