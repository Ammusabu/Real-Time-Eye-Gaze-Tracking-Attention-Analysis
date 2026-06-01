import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import customtkinter as ctk
import cv2

from PIL import Image, ImageTk

from modules.face_detection import FaceDetector
from modules.eye_detection import EyeDetector
from modules.gaze_estimation import GazeEstimator
from modules.analytics import AttentionAnalytics
from modules.attention_heatmap import AttentionHeatmap


class AttentionDashboard:

    def __init__(self):

        # -----------------------------
        # MAIN WINDOW
        # -----------------------------

        self.root = ctk.CTk()

        self.root.title(
            "AI Attention Monitoring System"
        )

        self.root.geometry("1400x850")

        ctk.set_appearance_mode("dark")

        # -----------------------------
        # VARIABLES
        # -----------------------------

        self.running = False

        self.cap = None

        # -----------------------------
        # MODULES
        # -----------------------------

        self.face_detector = FaceDetector()

        self.eye_detector = EyeDetector()

        self.gaze_estimator = GazeEstimator()

        self.analytics = AttentionAnalytics()

        self.attention_heatmap = AttentionHeatmap()

        # -----------------------------
        # TITLE
        # -----------------------------

        title = ctk.CTkLabel(
            self.root,
            text="AI Attention Monitoring System",
            font=("Arial", 36, "bold")
        )

        title.pack(pady=20)

        # -----------------------------
        # SESSION STATUS
        # -----------------------------

        self.status_label = ctk.CTkLabel(
            self.root,
            text="SESSION STATUS: STOPPED",
            font=("Arial", 20, "bold"),
            text_color="red"
        )

        self.status_label.pack(pady=5)

        # -----------------------------
        # MAIN FRAME
        # -----------------------------

        self.main_frame = ctk.CTkFrame(
            self.root,
            width=1300,
            height=650
        )

        self.main_frame.pack(pady=10)

        # -----------------------------
        # WEBCAM FRAME
        # -----------------------------

        self.webcam_frame = ctk.CTkFrame(
            self.main_frame,
            width=820,
            height=570,
            corner_radius=20
        )

        self.webcam_frame.place(
            x=30,
            y=30
        )

        self.webcam_label = ctk.CTkLabel(
            self.webcam_frame,
            text="Camera Feed",
            width=800,
            height=550
        )

        self.webcam_label.pack(
            padx=10,
            pady=10
        )

        # -----------------------------
        # ANALYTICS PANEL
        # -----------------------------

        self.analytics_frame = ctk.CTkFrame(
            self.main_frame,
            width=380,
            height=570,
            corner_radius=20
        )

        self.analytics_frame.place(
            x=900,
            y=30
        )

        analytics_title = ctk.CTkLabel(
            self.analytics_frame,
            text="Live Analytics",
            font=("Arial", 28, "bold")
        )

        analytics_title.pack(pady=25)

        # -----------------------------
        # ANALYTICS LABELS
        # -----------------------------

        self.attention_label = ctk.CTkLabel(
            self.analytics_frame,
            text="Attention: ---",
            font=("Arial", 22)
        )

        self.attention_label.pack(pady=20)

        self.attention_percent = ctk.CTkLabel(
            self.analytics_frame,
            text="Attention %: 0%",
            font=("Arial", 22)
        )

        self.attention_percent.pack(pady=20)

        self.distracted_percent = ctk.CTkLabel(
            self.analytics_frame,
            text="Distracted %: 0%",
            font=("Arial", 22)
        )

        self.distracted_percent.pack(pady=20)

        self.session_time = ctk.CTkLabel(
            self.analytics_frame,
            text="Session Time: 0s",
            font=("Arial", 22)
        )

        self.session_time.pack(pady=20)

        # -----------------------------
        # BUTTONS
        # -----------------------------

        self.button_frame = ctk.CTkFrame(
            self.root,
            fg_color="transparent"
        )

        self.button_frame.pack(pady=20)

        self.start_button = ctk.CTkButton(
            self.button_frame,
            text="Start Session",
            width=220,
            height=60,
            font=("Arial", 22),
            command=self.start_session
        )

        self.start_button.grid(
            row=0,
            column=0,
            padx=80
        )

        self.end_button = ctk.CTkButton(
            self.button_frame,
            text="End Session",
            width=220,
            height=60,
            font=("Arial", 22),
            fg_color="red",
            command=self.end_session
        )

        self.end_button.grid(
            row=0,
            column=1,
            padx=80
        )

        # Disable initially
        self.end_button.configure(
            state="disabled"
        )

    # --------------------------------
    # START SESSION
    # --------------------------------

    def start_session(self):

        if self.running:
            return

        # Reset everything
        self.analytics = AttentionAnalytics()

        self.attention_heatmap = AttentionHeatmap()

        self.cap = cv2.VideoCapture(0)

        self.running = True

        self.start_button.configure(
            state="disabled"
        )

        self.end_button.configure(
            state="normal"
        )

        self.status_label.configure(
            text="SESSION STATUS: LIVE",
            text_color="lime"
        )

        self.update_video()

    # --------------------------------
    # UPDATE VIDEO
    # --------------------------------

    def update_video(self):

        if not self.running:
            return

        ret, frame = self.cap.read()

        if ret:

            frame = cv2.flip(frame, 1)

            # FACE DETECTION

            faces = self.face_detector.detect_faces(frame)

            for (x, y, w, h) in faces:

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

            # EYE TRACKING

            eye_data = self.eye_detector.get_eye_data(frame)

            if eye_data:

                gaze = self.gaze_estimator.estimate_gaze(
                    eye_data
                )

                self.attention_heatmap.update(gaze)

                if gaze == "CENTER":
                    attention = "ATTENTIVE"

                else:
                    attention = "DISTRACTED"

                self.analytics.update(attention)

                stats = self.analytics.get_statistics()

                # UPDATE LABELS

                self.attention_label.configure(
                    text=f"Attention: {attention}"
                )

                self.attention_percent.configure(
                    text=f"Attention %: {stats['attentive_percent']}%"
                )

                self.distracted_percent.configure(
                    text=f"Distracted %: {stats['distracted_percent']}%"
                )

                self.session_time.configure(
                    text=f"Session Time: {stats['session_duration']}s"
                )

                # DRAW TEXT

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

            # SHOW FRAME

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(rgb)

            image = image.resize((800, 550))

            photo = ImageTk.PhotoImage(image=image)

            self.webcam_label.configure(
                image=photo,
                text=""
            )

            self.webcam_label.image = photo

        self.root.after(
            10,
            self.update_video
        )

    # --------------------------------
    # END SESSION
    # --------------------------------

    def end_session(self):

        if not self.running:
            return

        self.running = False

        if self.cap:
            self.cap.release()

        self.start_button.configure(
            state="normal"
        )

        self.end_button.configure(
            state="disabled"
        )

        self.status_label.configure(
            text="SESSION STATUS: STOPPED",
            text_color="red"
        )

        self.webcam_label.configure(
            image=None,
            text="Session Ended"
        )

        # SAVE HEATMAP

        self.attention_heatmap.save_heatmap()

        # SESSION STATS

        stats = self.analytics.get_statistics()

        attentive_percent = stats["attentive_percent"]

        distracted_percent = stats["distracted_percent"]

        session_duration = stats["session_duration"]

        # SCORE

        if attentive_percent >= 70:
            score = "GOOD"

        elif attentive_percent >= 40:
            score = "AVERAGE"

        else:
            score = "POOR"

        # OVERALL STATE

        if attentive_percent > distracted_percent:
            overall = "MOSTLY ATTENTIVE"

        else:
            overall = "MOSTLY DISTRACTED"

        # REPORT WINDOW

        report = ctk.CTkToplevel(self.root)

        report.title("Session Report")

        report.geometry("900x850")

        title = ctk.CTkLabel(
            report,
            text="SESSION ANALYTICS REPORT",
            font=("Arial", 30, "bold")
        )

        title.pack(pady=20)

        report_text = f"""

Attention Score: {score}

Overall State: {overall}

Attention Percentage: {attentive_percent}%

Distracted Percentage: {distracted_percent}%

Session Duration: {session_duration}s

"""

        summary = ctk.CTkLabel(
            report,
            text=report_text,
            font=("Arial", 22),
            justify="left"
        )

        summary.pack(pady=20)

        # LOAD HEATMAP

        try:

            heatmap_image = Image.open(
                "outputs/attention_heatmap.png"
            )

            heatmap_image = heatmap_image.resize(
                (500, 500)
            )

            heatmap_photo = ImageTk.PhotoImage(
                heatmap_image
            )

            heatmap_label = ctk.CTkLabel(
                report,
                image=heatmap_photo,
                text=""
            )

            heatmap_label.image = heatmap_photo

            heatmap_label.pack(pady=20)

        except:

            error_label = ctk.CTkLabel(
                report,
                text="Heatmap image not found.",
                font=("Arial", 20)
            )

            error_label.pack(pady=20)

    # --------------------------------
    # RUN APP
    # --------------------------------

    def run(self):

        self.root.mainloop()


if __name__ == "__main__":

    app = AttentionDashboard()

    app.run()