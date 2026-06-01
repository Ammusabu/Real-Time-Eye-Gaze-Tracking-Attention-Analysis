import cv2
import os

from modules.face_detection import FaceDetector
from modules.eye_detection import EyeDetector
from modules.gaze_estimation import GazeEstimator
from modules.analytics import AttentionAnalytics
from modules.csv_logger import CSVLogger
from modules.attention_graph import AttentionGraph
from modules.attention_heatmap import AttentionHeatmap

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Webcam
cap = cv2.VideoCapture(0)

# Modules
face_detector = FaceDetector()
eye_detector = EyeDetector()
gaze_estimator = GazeEstimator()

analytics = AttentionAnalytics()
attention_graph = AttentionGraph()
attention_heatmap = AttentionHeatmap()
csv_logger = CSVLogger()


print("Webcam started successfully!")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # -----------------------------------
    # FACE DETECTION
    # -----------------------------------

    faces = face_detector.detect_faces(frame)

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # -----------------------------------
    # EYE TRACKING
    # -----------------------------------

    eye_data = eye_detector.get_eye_data(frame)

    if eye_data:

        # Draw iris centers
        cv2.circle(
            frame,
            eye_data["left_center"],
            3,
            (255, 0, 0),
            -1
        )

        cv2.circle(
            frame,
            eye_data["right_center"],
            3,
            (0, 0, 255),
            -1
        )

        # -----------------------------------
        # GAZE ESTIMATION
        # -----------------------------------

        gaze = gaze_estimator.estimate_gaze(eye_data)

        # -----------------------------------
        # ATTENTION CLASSIFICATION
        # -----------------------------------

        if gaze == "CENTER":
            attention = "ATTENTIVE"

        else:
            attention = "DISTRACTED"

        # -----------------------------------
        # UPDATE ANALYTICS
        # -----------------------------------

        analytics.update(attention)
        attention_heatmap.update(gaze)

        # Update graph
        current_time = time.time()

        if current_time - last_graph_update > 0.3:

            attention_graph.update(attention)

            last_graph_update = current_time
        # CSV logging
        csv_logger.log(gaze, attention)

        # Get stats
        stats = analytics.get_statistics()

        # -----------------------------------
        # DISPLAY TEXT
        # -----------------------------------

        cv2.putText(
            frame,
            f"Gaze: {gaze}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Attention: {attention}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Attention %: {stats['attentive_percent']}%",
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Distracted %: {stats['distracted_percent']}%",
            (30, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Session Time: {stats['session_duration']}s",
            (30, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    # -----------------------------------
    # DRAW LIVE ATTENTION GRAPH
    # -----------------------------------

    graph = attention_graph.draw_graph()

    # Show windows
    cv2.imshow(
        "Eye Gaze Tracking",
        frame
    )

    cv2.imshow(
        "Attention Timeline",
        graph
    )

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------------
# FINAL SESSION SUMMARY
# -----------------------------------

stats = analytics.get_statistics()

print("\n========== SESSION SUMMARY ==========")

print(f"Session Duration: {stats['session_duration']} seconds")

print(f"Attention Percentage: {stats['attentive_percent']}%")

print(f"Distracted Percentage: {stats['distracted_percent']}%")

# Attention score
if stats['attentive_percent'] >= 70:
    score = "GOOD"

elif stats['attentive_percent'] >= 40:
    score = "AVERAGE"

else:
    score = "POOR"

print(f"Attention Score: {score}")

print("=====================================\n")
# Save final graph
attention_graph.save_graph()
attention_heatmap.save_heatmap()

print("Attention heatmap saved.")

print("Attention graph saved in outputs folder.")

# Cleanup
cap.release()
cv2.destroyAllWindows()