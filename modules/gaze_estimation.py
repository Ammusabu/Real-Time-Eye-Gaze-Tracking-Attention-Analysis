class GazeEstimator:

    def estimate_gaze(self, eye_data):

        # LEFT EYE
        left_center_x = eye_data["left_center"][0]

        left_eye_left_x = eye_data["left_eye_left"][0]
        left_eye_right_x = eye_data["left_eye_right"][0]

        left_ratio = (
            (left_center_x - left_eye_right_x) /
            (left_eye_left_x - left_eye_right_x)
        )

        # RIGHT EYE
        right_center_x = eye_data["right_center"][0]

        right_eye_left_x = eye_data["right_eye_left"][0]
        right_eye_right_x = eye_data["right_eye_right"][0]

        right_ratio = (
            (right_center_x - right_eye_right_x) /
            (right_eye_left_x - right_eye_right_x)
        )

        # Average
        gaze_ratio = (left_ratio + right_ratio) / 2


        # Temporary thresholds
        if gaze_ratio < 0.45:
            return "LEFT"

        elif gaze_ratio > 0.55:
            return "RIGHT"

        else:
            return "CENTER"