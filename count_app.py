import base64
import json
import os
import re
from pathlib import Path

import anthropic
from flask import Flask, jsonify, render_template, request

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB limit

client = anthropic.Anthropic()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_bytes: bytes) -> str:
    return base64.standard_b64encode(image_bytes).decode("utf-8")


def get_media_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[1].lower()
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/jpeg")


def count_objects(image_bytes: bytes, filename: str, target: str) -> dict:
    image_data = image_to_base64(image_bytes)
    media_type = get_media_type(filename)

    prompt = (
        f"画像の中に「{target}」は何個ありますか？\n\n"
        "以下のJSON形式だけで答えてください（説明文は不要）：\n"
        '{"count": <整数>, "description": "<簡潔な説明>"}'
    )

    message = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    response_text = message.content[0].text.strip()
    # Extract JSON even if surrounded by markdown code fences
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {"count": 0, "description": response_text}


@app.route("/")
def index():
    return render_template("counter.html")


@app.route("/count", methods=["POST"])
def count():
    if "image" not in request.files:
        return jsonify({"error": "画像ファイルが必要です"}), 400

    file = request.files["image"]
    target = request.form.get("target", "").strip()

    if not file.filename:
        return jsonify({"error": "ファイルが選択されていません"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "対応形式: PNG, JPG, GIF, WebP"}), 400
    if not target:
        return jsonify({"error": "数える対象を入力してください"}), 400

    image_bytes = file.read()
    result = count_objects(image_bytes, file.filename, target)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
