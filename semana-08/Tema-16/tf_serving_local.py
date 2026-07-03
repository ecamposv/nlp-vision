"""Mini-servidor que emula la API REST de TensorFlow Serving (sin Docker).

Sirve el SavedModel de MobileNet en:
    http://localhost:8501/v1/models/modelo:predict

Uso:
    python tf_serving_local.py

Con esto, la celda de Flask del Tema 16 (que hace POST a localhost:8501) funciona
tal cual, sin necesitar Docker ni el binario oficial de TensorFlow Serving.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silencia logs de TF

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tf_serving", "modelo", "1")

print(f"Cargando SavedModel desde: {MODEL_DIR}")
model = tf.saved_model.load(MODEL_DIR)
infer = model.signatures["serving_default"]
input_key = list(infer.structured_input_signature[1].keys())[0]
output_key = list(infer.structured_outputs.keys())[0]
print(f"Modelo listo (input='{input_key}', output='{output_key}').")

app = Flask(__name__)


@app.route("/v1/models/<name>:predict", methods=["POST"])
def predict(name):
    body = request.get_json(force=True)
    if "instances" in body:
        arr = np.array(body["instances"], dtype=np.float32)
    elif "inputs" in body:
        arr = np.array(body["inputs"], dtype=np.float32)
    else:
        return jsonify({"error": "Falta 'instances' o 'inputs' en el cuerpo"}), 400

    out = infer(**{input_key: tf.constant(arr)})
    preds = out[output_key].numpy().tolist()
    return jsonify({"predictions": preds})


@app.route("/v1/models/<name>", methods=["GET"])
def status(name):
    return jsonify({"model_version_status": [{"version": "1", "state": "AVAILABLE"}]})


if __name__ == "__main__":
    print("Sirviendo en http://localhost:8501 (Ctrl+C para detener)")
    app.run(host="0.0.0.0", port=8501)
