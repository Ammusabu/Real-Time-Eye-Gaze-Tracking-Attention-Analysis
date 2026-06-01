
import matplotlib.pyplot as plt
import os


class AttentionGraph:

    def __init__(self):

        self.timeline = []

   
    # UPDATE DATA
   

    def update(self, attention):

        if attention == "ATTENTIVE":

            self.timeline.append(1)

        else:

            self.timeline.append(0)

    # SAVE GRAPH
    

    def save_graph(
        self,
        filename="outputs/attention_graph.png"
    ):

        if len(self.timeline) == 0:
            return

        os.makedirs(
            "outputs",
            exist_ok=True
        )

        plt.style.use("dark_background")

        fig, ax = plt.subplots(
            figsize=(14, 4)
        )

        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        # DRAW LINE
        ax.plot(
            self.timeline,
            color="#00FF66",
            linewidth=2
        )

        # LIMITS
        ax.set_ylim(-0.2, 1.2)

        # REMOVE EXTRA SPINES
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # AXIS COLORS
        ax.spines['left'].set_color("white")
        ax.spines['bottom'].set_color("white")

        # LABELS
        ax.set_xlabel(
            "Time",
            fontsize=16,
            color="white"
        )

        ax.set_ylabel(
            "Attention",
            fontsize=16,
            color="white"
        )

        # TICKS
        ax.set_yticks([0, 1])

        ax.set_yticklabels(
            ["DISTRACTED", "ATTENTIVE"],
            fontsize=14
        )

        ax.tick_params(
            axis='x',
            colors='white'
        )

        ax.tick_params(
            axis='y',
            colors='white'
        )

        # TITLE
        ax.set_title(
            "Attention Timeline",
            fontsize=22,
            color="white",
            pad=20
        )

        # GRID
        ax.grid(
            alpha=0.2,
            color="white"
        )

        plt.tight_layout()

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor()
        )

        plt.close()

