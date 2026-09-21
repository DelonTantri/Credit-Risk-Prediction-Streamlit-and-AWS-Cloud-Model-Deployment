import json
import os
import joblib
import pandas as pd

JSON_CONTENT_TYPE = "application/json"

MODEL_FILE = "model_credit_score.joblib"
ENCODER_FILE = "label_encoder.joblib"
FEATURE_FILE = "feature_names.joblib"

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, MODEL_FILE))
    encoder = joblib.load(os.path.join(model_dir, ENCODER_FILE))
    feature_names = joblib.load(os.path.join(model_dir, FEATURE_FILE))
    return {"model": model, "encoder": encoder, "feature_names": feature_names}

def input_fn(request_body, request_content_type):
    if request_content_type == JSON_CONTENT_TYPE:
        payload = json.loads(request_body)
        instances = payload["instances"]
        return instances
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, artifacts):
    model = artifacts["model"]
    encoder = artifacts["encoder"]
    feature_names = artifacts["feature_names"]

    df = pd.DataFrame(input_data, columns=feature_names)
    prediction = model.predict(df)
    labels = encoder.inverse_transform(prediction)
    return {"predictions": prediction.tolist(), "labels": labels.tolist()}

def output_fn(prediction, accept):
    if accept == JSON_CONTENT_TYPE:
        return (json.dumps(prediction), JSON_CONTENT_TYPE)
    raise ValueError(f"Unsupported accept type: {accept}")