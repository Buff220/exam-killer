import cv2
from flask import Flask, Response, render_template, jsonify
import requests
import base64

# =====================
# CONFIG
# =====================
CAM_URL = "rtsp://192.168.100.1:8080/?action=stream"
GROQ_API_KEY = open("api.txt").read().strip()
MODEL = "llama-3.2-11b-vision-preview"
# =====================

app = Flask(__name__)
cap = cv2.VideoCapture(CAM_URL)

paused = False
last_frame = None


def ask_groq(frame):
    # Encode frame → PNG → Base64
    _, buffer = cv2.imencode('.png', frame)
    img_b64 = base64.b64encode(buffer.tobytes()).decode()

    prompt = (
        "Extract the correct answers from this image. "
        "Format ONLY like:\n1-a\n2-c\n3-d\n"
        "Do NOT add explanations."
    )

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json=data,
        headers=headers
    )

    return r.json()["choices"][0]["message"]["content"]


def gen_frames():
    global paused, last_frame

    while True:
        success, frame = cap.read()
        if not success:
            continue

        if not paused:
            last_frame = frame
            display = frame
        else:
            # frozen grayscale frame when paused
            display = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

        _, buffer = cv2.imencode('.jpg', display)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')


@app.route('/pause')
def pause():
    global paused
    paused = True
    return ('', 204)


@app.route('/resume')
def resume():
    global paused
    paused = False
    return ('', 204)


@app.route('/process')
def process():
    global last_frame
    if last_frame is None:
        return jsonify({"error": "no_frame"}), 400

    ai_text = ask_groq(last_frame)
    return jsonify({"ai": ai_text})


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
    app.run("0.0.0.0", 5000, debug=True)
