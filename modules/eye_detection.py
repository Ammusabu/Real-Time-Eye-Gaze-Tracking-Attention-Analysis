import cv2
import mediapipe as mp
import numpy as np


class EyeDetector:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Iris landmarks
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]

        # Eye corner landmarks
        self.LEFT_EYE_LEFT = 263
        self.LEFT_EYE_RIGHT = 362

        self.RIGHT_EYE_LEFT = 133
        self.RIGHT_EYE_RIGHT = 33

        # Vertical eye landmarks
        self.LEFT_EYE_TOP = 386
        self.LEFT_EYE_BOTTOM = 374

        self.RIGHT_EYE_TOP = 159
        self.RIGHT_EYE_BOTTOM = 145

    def get_eye_data(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb_frame)

        h, w, _ = frame.shape

        if results.multi_face_landmarks:

            face_landmarks = results.multi_face_landmarks[0]

            # ---------- LEFT IRIS ----------
            left_iris = []

            for idx in self.LEFT_IRIS:

                point = face_landmarks.landmark[idx]

                x = int(point.x * w)
                y = int(point.y * h)

                left_iris.append((x, y))

            # RIGHT IRIS ----------
            right_iris = []

            for idx in self.RIGHT_IRIS:

                point = face_landmarks.landmark[idx]

                x = int(point.x * w)
                y = int(point.y * h)

                right_iris.append((x, y))

            # Iris centers
            left_center = np.mean(left_iris, axis=0).astype(int)
            right_center = np.mean(right_iris, axis=0).astype(int)

            # Eye corners
            left_eye_left = face_landmarks.landmark[self.LEFT_EYE_LEFT]
            left_eye_right = face_landmarks.landmark[self.LEFT_EYE_RIGHT]

            right_eye_left = face_landmarks.landmark[self.RIGHT_EYE_LEFT]
            right_eye_right = face_landmarks.landmark[self.RIGHT_EYE_RIGHT]

            left_eye_left = (
                int(left_eye_left.x * w),
                int(left_eye_left.y * h)
            )

            left_eye_right = (
                int(left_eye_right.x * w),
                int(left_eye_right.y * h)
            )

            right_eye_left = (
                int(right_eye_left.x * w),
                int(right_eye_left.y * h)
            )

            right_eye_right = (
                int(right_eye_right.x * w),
                int(right_eye_right.y * h)
            )

            # Vertical landmarks
            left_eye_top = face_landmarks.landmark[self.LEFT_EYE_TOP]
            left_eye_bottom = face_landmarks.landmark[self.LEFT_EYE_BOTTOM]

            right_eye_top = face_landmarks.landmark[self.RIGHT_EYE_TOP]
            right_eye_bottom = face_landmarks.landmark[self.RIGHT_EYE_BOTTOM]

            left_eye_top = (
                int(left_eye_top.x * w),
                int(left_eye_top.y * h)
            )

            left_eye_bottom = (
                int(left_eye_bottom.x * w),
                int(left_eye_bottom.y * h)
            )

            right_eye_top = (
                int(right_eye_top.x * w),
                int(right_eye_top.y * h)
            )

            right_eye_bottom = (
                int(right_eye_bottom.x * w),
                int(right_eye_bottom.y * h)
            )

            return {
                "left_iris": left_iris,
                "right_iris": right_iris,
                "left_center": tuple(left_center),
                "right_center": tuple(right_center),
                "left_eye_left": left_eye_left,
                "left_eye_right": left_eye_right,
                "right_eye_left": right_eye_left,
                "right_eye_right": right_eye_right,

                "left_eye_top": left_eye_top,
                "left_eye_bottom": left_eye_bottom,

                "right_eye_top": right_eye_top,
                "right_eye_bottom": right_eye_bottom
            }

        return None