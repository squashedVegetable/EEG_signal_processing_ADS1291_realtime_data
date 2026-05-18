import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

clf = joblib.load("blink_model.pkl")
scaler = joblib.load("scaler.pkl")

print("Model loaded:", type(clf))
print("Scaler loaded:", type(scaler))

pipeline = Pipeline([
    ("scaler", scaler),
    ("classifier", clf)
])

n_features = scaler.mean_.shape[0]

initial_type = [
    ("input", FloatTensorType([None, n_features]))
]

print("Converting to ONNX...")

onnx_model = convert_sklearn(
    pipeline,
    initial_types=initial_type
)

output_file = "blink_model.onnx"

with open(output_file, "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Saved ONNX model to:", output_file)
print("\nTesting ONNX export consistency...")

import onnxruntime as ort

sess = ort.InferenceSession(output_file)

# fake test input (replace with real sample if you want)
test_input = np.random.randn(1, n_features).astype(np.float32)

skl_pred = pipeline.predict(test_input)
onnx_pred = sess.run(None, {"input": test_input.astype(np.float32)})[0]

print("sklearn prediction:", skl_pred)
print("onnx prediction   :", onnx_pred)

print("If these match → export is correct 🎯")