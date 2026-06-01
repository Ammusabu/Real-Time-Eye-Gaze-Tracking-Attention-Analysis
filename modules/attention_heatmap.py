import numpy as np
import matplotlib.pyplot as plt
import os


class AttentionHeatmap:

    def __init__(self):

        # STORE GAZE POINTS
        self.points = []

        # SCREEN SIZE
        self.width = 800
        self.height = 400

    # UPDATE HEATMAP

    def update(self, gaze):

        # CENTER POSITION
        center_x = self.width * 0.5
        center_y = self.height * 0.5

        # CENTER
        if gaze == "CENTER":

            x = np.random.normal(
                center_x,
                70
            )

            y = np.random.normal(
                center_y,
                55
            )

        # LEFT
        elif gaze == "LEFT":

            x = np.random.normal(
                self.width * 0.3,
                65
            )

            y = np.random.normal(
                center_y,
                55
            )

        # RIGHT
        elif gaze == "RIGHT":

            x = np.random.normal(
                self.width * 0.7,
                65
            )

            y = np.random.normal(
                center_y,
                55
            )

        # UP
        elif gaze == "UP":

            x = np.random.normal(
                center_x,
                70
            )

            y = np.random.normal(
                self.height * 0.3,
                50
            )

        # DOWN
        else:

            x = np.random.normal(
                center_x,
                70
            )

            y = np.random.normal(
                self.height * 0.7,
                50
            )

        # EXTRA MICRO MOVEMENTS
        x += np.random.randint(-20, 20)
        y += np.random.randint(-20, 20)

        # KEEP INSIDE SCREEN
        x = np.clip(
            x,
            0,
            self.width - 1
        )

        y = np.clip(
            y,
            0,
            self.height - 1
        )

        # STORE POINT
        self.points.append((x, y))

    
    # SAVE HEATMAP

    def save_heatmap(
        self,
        filename="outputs/final_heatmap.png"
    ):

        # MINIMUM POINTS
        if len(self.points) < 20:

            print("Not enough points for heatmap")
            return

        # CREATE OUTPUT FOLDER
        os.makedirs(
            "outputs",
            exist_ok=True
        )

        # GRID SIZE
        rows = 8
        cols = 12

        # EMPTY GRID
        heatmap = np.zeros(
            (rows, cols)
        )

        # MAP GAZE POINTS TO GRID
        for point in self.points:

            x = point[0]
            y = point[1]

            col = int(
                (x / self.width) * cols
            )

            row = int(
                (y / self.height) * rows
            )

            col = min(
                col,
                cols - 1
            )

            row = min(
                row,
                rows - 1
            )

            heatmap[row, col] += 1

        # NORMALIZE
        max_value = np.max(heatmap)

        if max_value > 0:

            heatmap = heatmap / max_value

        # STYLE
        plt.style.use("dark_background")

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        # SHOW HEATMAP
        im = ax.imshow(
            heatmap,
            cmap="magma",
            interpolation="nearest",
            aspect="auto"
        )

        # VALUES INSIDE CELLS
        for i in range(rows):

            for j in range(cols):

                value = round(
                    heatmap[i, j],
                    2
                )

                ax.text(
                    j,
                    i,
                    value,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9
                )

        # COLORBAR
        cbar = fig.colorbar(im)

        cbar.set_label(
            "Attention Density",
            color="white",
            fontsize=12
        )

        cbar.ax.yaxis.set_tick_params(
            color='white'
        )

        plt.setp(
            plt.getp(
                cbar.ax.axes,
                'yticklabels'
            ),
            color='white'
        )

        # TITLE
        ax.set_title(
            "AI Attention Heatmap",
            fontsize=24,
            color="white",
            pad=20
        )

        # LABELS
        ax.set_xlabel(
            "Screen Regions",
            fontsize=14,
            color="white"
        )

        ax.set_ylabel(
            "Attention Zones",
            fontsize=14,
            color="white"
        )

        # TICKS
        ax.tick_params(
            colors='white'
        )

        # GRID LINES
        ax.set_xticks(
            np.arange(-0.5, cols, 1),
            minor=True
        )

        ax.set_yticks(
            np.arange(-0.5, rows, 1),
            minor=True
        )

        ax.grid(
            which='minor',
            color='white',
            linestyle='-',
            linewidth=0.5,
            alpha=0.2
        )

        # SAVE IMAGE
        plt.tight_layout()

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )

        print("GRID HEATMAP SAVED")

        plt.close()