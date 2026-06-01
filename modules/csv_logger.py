import csv
import os
import time


class CSVLogger:

    def __init__(self):

        # Create logs folder if not exists
        os.makedirs("data/gaze_logs", exist_ok=True)

        # File name using timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        self.filename = (
            f"data/gaze_logs/session_{timestamp}.csv"
        )

        # Create CSV file
        with open(self.filename, mode='w', newline='') as file:

            writer = csv.writer(file)

            writer.writerow([
                "Timestamp",
                "Gaze Direction",
                "Attention State"
            ])

    def log(self, gaze, attention):

        current_time = time.strftime("%H:%M:%S")

        with open(self.filename, mode='a', newline='') as file:

            writer = csv.writer(file)

            writer.writerow([
                current_time,
                gaze,
                attention
            ])