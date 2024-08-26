from flask import Flask, Blueprint, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import time

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

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def smooth_angles(angles, window_size=5):
    if len(angles) < window_size:
        return angles
    return np.convolve(angles, np.ones(window_size)/window_size, mode='valid')

def gen_frames():
    global counter, stage, high_score, angles_over_time, timestamps
    cap = cv2.VideoCapture(0)  # Open the camera
    previous_angle = None
    hold_start_time = None  # To track the hold time at the "up" position
    frames_in_position = 0  # To ensure the user holds the position
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.resize(frame, (640, 480))  # Resize the frame for display
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5),
                                      mp_drawing.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10))

            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates for the left shoulder, elbow, and wrist
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angles
                angle = calculate_angle(shoulder, elbow, wrist)

                # Smooth angle using exponential moving average (optional)
                if previous_angle is not None:
                    smoothed_angle = 0.8 * previous_angle + 0.2 * angle
                else:
                    smoothed_angle = angle

                previous_angle = smoothed_angle

                # Save the angle and timestamp
                current_time = time.time() - start_time
                angles_over_time.append(smoothed_angle)
                timestamps.append(current_time)

                # Calculate the horizontal distance between the shoulder and wrist to ensure proper arm alignment
                shoulder_wrist_dist = np.abs(shoulder[0] - wrist[0])

                # Check for arm alignment, velocity, and angle thresholds
                if shoulder_wrist_dist > 0.1 and smoothed_angle > 160:
                    stage = "down"
                    hold_start_time = None  # Reset hold time when the arm is fully extended
                    frames_in_position = 0  # Reset frame counter

                if shoulder_wrist_dist > 0.1 and smoothed_angle < 40 and stage == 'down':
                    stage = "up"
                    if hold_start_time is None:
                        hold_start_time = time.time()

                    # Check if the user held the "up" position for at least 0.5 seconds and maintain the position for a few frames
                    hold_time = time.time() - hold_start_time
                    frames_in_position += 1

                    if hold_time > 0.5 and frames_in_position > 5:  # Ensuring the user holds the position for at least 5 frames
                        counter += 1
                        stage = "counted"  # Change stage to a new value to prevent multiple increments
                        frames_in_position = 0  # Reset frame counter
                        if counter > high_score:
                            high_score = counter

                # Reset the stage back to "down" only when the arm is fully extended again
                if shoulder_wrist_dist > 0.1 and smoothed_angle > 160 and stage == "counted":
                    stage = "down"

            except Exception as e:
                print(f"Error: {e}")

        # Render the video feed with the annotations
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@bicepcurl_app.route('/bicepcurl')
def bicepcurl():
    return render_template('model_page.html', model_name='Bicep Curl')

@bicepcurl_app.route('/video_feed_bicepcurl')
def video_feed_bicepcurl():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@bicepcurl_app.route('/update_graph_data')
def update_graph_data():
    global angles_over_time, timestamps
    smoothed_angles = smooth_angles(angles_over_time)
    return jsonify(angles=smoothed_angles.tolist(), timestamps=timestamps[-len(smoothed_angles):])

@bicepcurl_app.route('/update_data_bicepcurl')
def update_data_bicepcurl():
    global counter, high_score, stage
    return jsonify(stage=stage, counter=counter, high_score=high_score)

@bicepcurl_app.route('/reset_counter_bicepcurl', methods=['POST'])
def reset_counter_bicepcurl():
    global counter, stage, angles_over_time, timestamps
    counter = 0  # Reset the reps counter only
    stage = "down"
    angles_over_time = []
    timestamps = []
    return jsonify(success=True)

# Register the Blueprint with the Flask app
app.register_blueprint(bicepcurl_app)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
