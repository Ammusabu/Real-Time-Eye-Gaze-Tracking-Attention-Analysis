import cv2
import numpy as np


class HeatmapGenerator:

    def __init__(self, width=600, height=200):

        self.width = width
        self.height = height

        self.heatmap = np.zeros(
            (height, width),
            dtype=np.float32
        )

    def add_attention(self, attention_state):

        if attention_state == "ATTENTIVE":
            base_x = int(self.width * 0.5)

        else:
            base_x = np.random.choice([
                int(self.width * 0.2),
                int(self.width * 0.8)
            ])

        base_y = int(self.height * 0.5)

        x = base_x + np.random.randint(-40, 40)
        y = base_y + np.random.randint(-40, 40)

        cv2.circle(
            self.heatmap,
            (x, y),
            25,
            2,
            -1
        )

    def get_heatmap(self):

        blurred = cv2.GaussianBlur(
            self.heatmap,
            (81, 81),
            0
        )

        normalized = cv2.normalize(
            blurred,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        normalized = normalized.astype(np.uint8)

        colored = cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_JET
        )

        return colored

    def save_heatmap(self, filename="outputs/final_heatmap.png"):

        heatmap = self.get_heatmap()

        cv2.imwrite(filename, heatmap)