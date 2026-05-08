import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def main():
    wrapped_positions = np.load("trajectory_samples.npy")
    box_size = np.load("run_history.npz")["box_size"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, box_size[0])
    ax.set_ylim(0, box_size[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("LJ Trajectory (xy projection)")
    scatter = ax.scatter([], [], s=8, alpha=0.8)

    def update(frame_idx):
        xy = wrapped_positions[frame_idx, :, :2]
        scatter.set_offsets(xy)
        ax.set_title(f"LJ Trajectory (frame {frame_idx})")
        return (scatter,)

    ani = FuncAnimation(fig, update, frames=wrapped_positions.shape[0], interval=80, blit=True)
    ani.save("md_trajectory.gif", writer="pillow", fps=15)
    plt.close(fig)
    print("saved animation: md_trajectory.gif")


if __name__ == "__main__":
    main()
