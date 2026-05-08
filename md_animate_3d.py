import numpy as np
import plotly.graph_objects as go


def main():
    wrapped_positions = np.load("trajectory_samples.npy")
    history_data = np.load("run_history.npz")
    box_size = history_data["box_size"]

    frame_count = wrapped_positions.shape[0]
    start_xyz = wrapped_positions[0]

    frames = []
    for frame_idx in range(frame_count):
        xyz = wrapped_positions[frame_idx]
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=xyz[:, 0],
                        y=xyz[:, 1],
                        z=xyz[:, 2],
                        mode="markers",
                        marker=dict(size=3, color=xyz[:, 2], colorscale="Viridis", opacity=0.9),
                    )
                ],
                name=str(frame_idx),
            )
        )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=start_xyz[:, 0],
                y=start_xyz[:, 1],
                z=start_xyz[:, 2],
                mode="markers",
                marker=dict(size=3, color=start_xyz[:, 2], colorscale="Viridis", opacity=0.9),
            )
        ],
        frames=frames,
    )

    fig.update_layout(
        title="Lennard-Jones MD 3D Trajectory",
        scene=dict(
            xaxis=dict(range=[0, box_size[0]], title="x"),
            yaxis=dict(range=[0, box_size[1]], title="y"),
            zaxis=dict(range=[0, box_size[2]], title="z"),
            aspectmode="cube",
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(frame=dict(duration=50, redraw=True), fromcurrent=True, transition=dict(duration=0)),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ],
    )

    fig.write_html("md_trajectory_3d.html", auto_play=False)
    print("saved interactive 3d animation: md_trajectory_3d.html")


if __name__ == "__main__":
    main()
