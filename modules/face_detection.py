import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection

        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence
        )

        # Drawing utility
        self.mp_draw = mp.solutions.drawing_utils

    def detect_faces(self, frame):

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = self.face_detection.process(rgb_frame)

        detections = []

        if results.detections:

            h, w, _ = frame.shape

            for detection in results.detections:

                bbox = detection.location_data.relative_bounding_box

                # Convert relative coordinates to pixels
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                detections.append((x, y, width, height))

        return detections