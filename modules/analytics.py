import time


class AttentionAnalytics:

    def __init__(self):

        self.start_time = time.time()

        self.attentive_frames = 0
        self.distracted_frames = 0

    def update(self, attention_state):

        if attention_state == "ATTENTIVE":
            self.attentive_frames += 1

        else:
            self.distracted_frames += 1

    def get_statistics(self):

        total_frames = (
            self.attentive_frames +
            self.distracted_frames
        )

        if total_frames == 0:
            attentive_percent = 0
            distracted_percent = 0

        else:
            attentive_percent = (
                self.attentive_frames / total_frames
            ) * 100

            distracted_percent = (
                self.distracted_frames / total_frames
            ) * 100

        session_duration = (
            time.time() - self.start_time
        )

        return {
            "attentive_percent": round(attentive_percent, 2),
            "distracted_percent": round(distracted_percent, 2),
            "session_duration": round(session_duration, 1)
        }