import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "./train")
    model_dir = os.environ.get("SM_MODEL_DIR", "./model")

    os.makedirs(model_dir, exist_ok=True)

    train_file = os.path.join(train_dir, "train.csv")
    df = pd.read_csv(train_file)

    print("TRAIN FILE:", train_file)
    
    df = pd.read_csv(train_file)
    
    print("COLUMNS:")
    print(df.columns.tolist())

    X_train = df.drop("Credit_Score", axis=1)
    y_train = df["Credit_Score"]

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)

    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[("num", SimpleImputer(strategy="median"), numeric_features)], remainder="drop")

    model_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1))])

    model_pipeline.fit(X_train, y_train)

    # Simpan nama fitur
    feature_names = X_train.columns.tolist()
    
    joblib.dump(feature_names, os.path.join(model_dir, "feature_names.joblib"))
    
    # Simpan model
    joblib.dump(model_pipeline, os.path.join(model_dir, "model_credit_score.joblib"))
    
    # Simpan label encoder
    joblib.dump(encoder, os.path.join(model_dir, "label_encoder.joblib"))

    print("✅ Training complete.")