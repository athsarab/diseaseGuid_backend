from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from pathlib import Path
import numpy as np
import traceback
import sys
import logging

# Try to import TensorFlow; fallback to tflite_runtime if TF not available
try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    logging.info("Using tensorflow for TFLite Interpreter")
except Exception:
    try:
        from tflite_runtime.interpreter import Interpreter
        logging.info("Using tflite_runtime Interpreter")
    except Exception as e:
        logging.exception("No tflite interpreter available. Install tensorflow or tflite-runtime.")
        raise

BASE_DIR = Path(__file__).resolve().parent

IMG_SIZE = (224, 224)   # fallback default
CONF_THRESHOLD = 0.6

def load_labels(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]

# Paths (make absolute so working dir won't break)
d_labels_path = BASE_DIR / "disease_labels.txt"
s_labels_path = BASE_DIR / "severity_labels.txt"
d_model_path = BASE_DIR / "disease_model.tflite"
s_model_path = BASE_DIR / "severity_model.tflite"

disease_labels = load_labels(d_labels_path)
severity_labels = load_labels(s_labels_path)

# Load interpreters
def make_interpreter(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    interp = Interpreter(model_path=str(path))
    interp.allocate_tensors()
    return interp

disease_interpreter = make_interpreter(d_model_path)
severity_interpreter = make_interpreter(s_model_path)

# get details for each interpreter (we will refer to details at runtime)
d_input_details = disease_interpreter.get_input_details()[0]
d_output_details = disease_interpreter.get_output_details()[0]

s_input_details = severity_interpreter.get_input_details()[0]
s_output_details = severity_interpreter.get_output_details()[0]

def preprocess_image_for_interpreter(data: bytes, input_details):
    """Resize/convert and return array matching interpreter input dtype and shape."""
    img = Image.open(BytesIO(data)).convert("RGB")
    # determine required shape
    shape = input_details["shape"]  # e.g. [1,224,224,3]
    # pick height,width from shape
    _, h, w, c = shape
    img = img.resize((w, h))
    arr = np.array(img)
    # cast to required dtype
    dtype = input_details["dtype"]
    # If interpreter expects float, normalize to 0-1 (common for float models). If it expects uint8, keep 0-255.
    if np.issubdtype(dtype, np.floating):
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(dtype)
    # add batch dimension if needed
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)
    # ensure final shape matches exactly
    arr = arr.reshape(shape).astype(dtype)
    return arr

def run_tflite(interpreter, input_details, output_details, img_array):
    # set tensor using the input index from details
    interpreter.set_tensor(input_details["index"], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details["index"])[0]
    probs = preds.astype(float)
    idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    return idx, conf

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "PapayaPulse API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # preprocess for each interpreter separately (in case they expect different sizes/dtypes)
        d_img = preprocess_image_for_interpreter(data, d_input_details)
        d_idx, d_conf = run_tflite(disease_interpreter, d_input_details, d_output_details, d_img)
        disease = disease_labels[d_idx] if d_idx < len(disease_labels) else "unknown"

        s_img = preprocess_image_for_interpreter(data, s_input_details)
        s_idx, s_conf = run_tflite(severity_interpreter, s_input_details, s_output_details, s_img)
        severity = severity_labels[s_idx] if s_idx < len(severity_labels) else "unknown"

        if disease == "NotPapaya" or d_conf < CONF_THRESHOLD:
            disease = "NotPapaya"
            severity = "unknown"

        return {
            "disease": disease,
            "disease_confidence": d_conf,
            "severity": severity,
            "severity_confidence": s_conf,
        }
    except HTTPException:
        # re-raise FastAPI HTTP errors
        raise
    except Exception as e:
        # log traceback to server console
        tb = traceback.format_exc()
        logging.error("Prediction error:\n" + tb)
        # return a controlled 500 with some clue for debugging (avoid leaking sensitive info in prod)
        raise HTTPException(status_code=500, detail=f"Internal Server Error. See server logs for traceback.")
