import joblib
import pandas as pd

class CreditScoreInference:

    def __init__(self):
        self.model = joblib.load("artifacts/model.pkl")
        self.encoder = joblib.load("artifacts/label_encoder.pkl")
        self.feature_names = joblib.load("artifacts/feature_names.pkl")

    def predict(self, data_dict):
        # Convert input dictionary to DataFrame
        df = pd.DataFrame([data_dict])

        # Reorder columns to match training data
        df = df.reindex(columns=self.feature_names, fill_value=0)

        # Make prediction
        prediction = self.model.predict(df)
        label = self.encoder.inverse_transform(prediction)[0]

        return label