from flask import Flask, Blueprint, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import time
from pykalman import KalmanFilter
from scipy.signal import savgol_filter, butter, filtfilt
from sklearn.metrics import mean_squared_error
import math

# Create Flask app
app = Flask(__name__)

# Create a Blueprint for the bicep curl
bicepcurl_app = Blueprint('bicepcurl_app', __name__)

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# Global variables
counter = 0
stage = "down"
high_score = 0
angles_over_time = []
timestamps = []

# Kalman Filter initialization
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

# Function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# Smoothing filters with latency calculations
def smooth_angles_kalman(angles):
    start_time = time.time()
    if len(angles) < 2:
        return angles, 0.0
    measurements = np.array(angles).reshape(-1, 1)
    smoothed_data, _ = kf.smooth(measurements)
    latency = time.time() - start_time
    return smoothed_data.flatten().tolist(), latency

def smooth_angles_savgol(angles, window_size=9, polyorder=3):
    start_time = time.time()
    if len(angles) < window_size:
        return angles, 0.0
    smoothed_data = savgol_filter(angles, window_size, polyorder).tolist()
    latency = time.time() - start_time
    return smoothed_data, latency

def smooth_angles_butterworth(angles, cutoff=0.1, fs=30, order=4):
    start_time = time.time()
    if len(angles) < 2:
        return angles, 0.0
    b, a = butter(order, cutoff, btype="low", fs=fs)
    smoothed_data = filtfilt(b, a, angles).tolist()
    latency = time.time() - start_time
    return smoothed_data, latency

# Benchmark metrics
def calculate_rmse(raw, filtered):
    return math.sqrt(mean_squared_error(raw, filtered))

def calculate_snr(raw, filtered):
    signal_power = np.mean(np.square(filtered))
    noise_power = np.mean(np.square(np.array(raw) - np.array(filtered)))
    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")

# Video generation for live feed
def gen_frames():
    global counter, stage, high_score, angles_over_time, timestamps
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5),
                mp_drawing.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10),
            )

            try:
                landmarks = results.pose_landmarks.landmark
                shoulder = [
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                ]
                elbow = [
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                ]
                wrist = [
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                ]

                angle = calculate_angle(shoulder, elbow, wrist)
                angles_over_time.append(angle)
                timestamps.append(time.time() - start_time)

                smoothed_kalman, _ = smooth_angles_kalman(angles_over_time)

                # Rep counting logic
                if smoothed_kalman[-1] > 160:
                    if stage == "up":
                        counter += 1
                        stage = "down"
                        if counter > high_score:
                            high_score = counter
                if smoothed_kalman[-1] < 40:
                    if stage == "down":
                        stage = "up"
            except Exception as e:
                print(f"Error: {e}")

        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        frame = buffer.tobytes()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

    cap.release()

@bicepcurl_app.route("/bicepcurl")
def bicepcurl():
    return render_template("model_page.html", model_name="Bicep Curl")

@bicepcurl_app.route("/video_feed_bicepcurl")
def video_feed_bicepcurl():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@bicepcurl_app.route("/update_graph_data")
def update_graph_data():
    global angles_over_time, timestamps
    kalman_data, _ = smooth_angles_kalman(angles_over_time)
    savgol_data, _ = smooth_angles_savgol(angles_over_time)
    butterworth_data, _ = smooth_angles_butterworth(angles_over_time)
    return jsonify(
        kalman=kalman_data,
        savgol=savgol_data,
        butterworth=butterworth_data,
        timestamps=timestamps,
    )

@bicepcurl_app.route("/benchmark_metrics")
def benchmark_metrics():
    global angles_over_time
    if len(angles_over_time) < 10:  # Ensure there are enough data points
        return jsonify(
            metrics={
                "Kalman Filter": {"RMSE": 0.0, "SNR": 0.0, "Latency": 0.0},
                "Savitzky-Golay Filter": {"RMSE": 0.0, "SNR": 0.0, "Latency": 0.0},
                "Butterworth Filter": {"RMSE": 0.0, "SNR": 0.0, "Latency": 0.0},
            }
        )

    kalman_data, kalman_latency = smooth_angles_kalman(angles_over_time)
    savgol_data, savgol_latency = smooth_angles_savgol(angles_over_time)
    butterworth_data, butterworth_latency = smooth_angles_butterworth(angles_over_time)

    return jsonify(
        metrics={
            "Kalman Filter": {
                "RMSE": calculate_rmse(angles_over_time, kalman_data),
                "SNR": calculate_snr(angles_over_time, kalman_data),
                "Latency": kalman_latency,
            },
            "Savitzky-Golay Filter": {
                "RMSE": calculate_rmse(angles_over_time, savgol_data),
                "SNR": calculate_snr(angles_over_time, savgol_data),
                "Latency": round(savgol_latency, 6),
            },
            "Butterworth Filter": {
                "RMSE": calculate_rmse(angles_over_time, butterworth_data),
                "SNR": calculate_snr(angles_over_time, butterworth_data),
                "Latency": round(butterworth_latency, 6),
            },
        }
    )

@bicepcurl_app.route("/update_data_bicepcurl")
def update_data_bicepcurl():
    global counter, high_score, stage
    return jsonify(stage=stage, counter=counter, high_score=high_score)

@bicepcurl_app.route("/reset_counter_bicepcurl", methods=["POST"])
def reset_counter_bicepcurl():
    global counter, stage, angles_over_time, timestamps
    counter = 0
    stage = "down"
    angles_over_time = []
    timestamps = []
    return jsonify(success=True)

# Register the Blueprint with the Flask app
app.register_blueprint(bicepcurl_app)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8081)
