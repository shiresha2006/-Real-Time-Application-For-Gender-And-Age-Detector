import base64
import io
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from detector import detect

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect", methods=["POST"])
def api_detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    out, meta = detect(img)
    ok, buf = cv2.imencode(".jpg", out)
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    return jsonify({
        "image": f"data:image/jpeg;base64,{b64}",
        "car_count": meta["car_count"],
        "people_count": meta["people_count"],
        "cars": meta["cars"],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
