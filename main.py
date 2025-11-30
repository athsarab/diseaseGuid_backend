from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf   # <--- use tensorflow

IMG_SIZE = (224, 224)
CONF_THRESHOLD = 0.6  # below this => "not sure"

def load_labels(path):
    with open(path, "r") as f:
        return [l.strip() for l in f]

disease_labels = load_labels("disease_labels.txt")
severity_labels = load_labels("severity_labels.txt")

# ---- Load TFLite models using tf.lite.Interpreter ----
disease_interpreter = tf.lite.Interpreter(
    model_path="disease_model.tflite"
)
disease_interpreter.allocate_tensors()
d_in = disease_interpreter.get_input_details()[0]["index"]
d_out = disease_interpreter.get_output_details()[0]["index"]

severity_interpreter = tf.lite.Interpreter(
    model_path="severity_model.tflite"
)
severity_interpreter.allocate_tensors()
s_in = severity_interpreter.get_input_details()[0]["index"]
s_out = severity_interpreter.get_output_details()[0]["index"]

def preprocess_image(data: bytes):
    img = Image.open(BytesIO(data)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)  # 0-255
    arr = np.expand_dims(arr, axis=0)      # (1, 224, 224, 3)
    return arr

def run_tflite(interpreter, input_index, output_index, img_array):
    interpreter.set_tensor(input_index, img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)[0]
    probs = preds.astype(float)
    idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    return idx, conf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; lock down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "PapayaPulse API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img_array = preprocess_image(data)

    d_idx, d_conf = run_tflite(disease_interpreter, d_in, d_out, img_array)
    disease = disease_labels[d_idx]

    s_idx, s_conf = run_tflite(severity_interpreter, s_in, s_out, img_array)
    severity = severity_labels[s_idx]

    if disease == "NotPapaya" or d_conf < CONF_THRESHOLD:
        disease = "NotPapaya"
        severity = "unknown"

    return {
        "disease": disease,
        "disease_confidence": d_conf,
        "severity": severity,
        "severity_confidence": s_conf,
    }
