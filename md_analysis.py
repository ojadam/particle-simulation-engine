import numpy as np
import matplotlib.pyplot as plt


def min_image(delta, box):
    return delta - box * np.round(delta / box)


def unwrap_positions(wrapped_pos, box):
    pos_unwrapped = np.zeros_like(wrapped_pos)
    pos_unwrapped[0] = wrapped_pos[0]
    for frame_idx in range(1, wrapped_pos.shape[0]):
        step = wrapped_pos[frame_idx] - wrapped_pos[frame_idx - 1]
        step = min_image(step, box)
        pos_unwrapped[frame_idx] = pos_unwrapped[frame_idx - 1] + step
    return pos_unwrapped


def compute_rdf(positions, box, n_bins=120, r_max=None):
    n_particles = positions.shape[0]
    number_density = n_particles / np.prod(box)
    if r_max is None:
        r_max = 0.5 * np.min(box)

    dr = r_max / n_bins
    edges = np.linspace(0.0, r_max, n_bins + 1)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    counts = np.zeros(n_bins)

    for i in range(n_particles - 1):
        for j in range(i + 1, n_particles):
            delta = positions[i] - positions[j]
            delta = min_image(delta, box)
            r = np.linalg.norm(delta)
            if r < r_max:
                bin_idx = int(r / dr)
                counts[bin_idx] += 2.0

    shell_volumes = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    ideal_counts = number_density * shell_volumes * n_particles
    rdf = counts / ideal_counts
    return r_centers, rdf


def compute_msd(unwrapped_pos):
    disp = unwrapped_pos - unwrapped_pos[0]
    return np.mean(np.sum(disp**2, axis=2), axis=1)


def main():
    hist = np.load("run_history.npz")
    wrapped_pos = np.load("trajectory_samples.npy")

    box = hist["box_size"]
    dt = float(hist["dt"])
    sample_stride = int(hist["sample_every"])
    time = hist["step"] * dt

    unwrapped_pos = unwrap_positions(wrapped_pos, box)
    msd = compute_msd(unwrapped_pos)
    msd_time = np.arange(wrapped_pos.shape[0]) * dt * sample_stride

    rdf_r, rdf_g = compute_rdf(wrapped_pos[-1], box)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(time, hist["total"], label="Total")
    axes[0, 0].plot(time, hist["kinetic"], label="Kinetic")
    axes[0, 0].plot(time, hist["potential"], label="Potential")
    axes[0, 0].set_title("Energy vs Time")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].legend()

    axes[0, 1].plot(time, hist["temperature"], color="tab:red")
    axes[0, 1].set_title("Temperature vs Time")
    axes[0, 1].set_xlabel("Time")

    axes[1, 0].plot(rdf_r, rdf_g, color="tab:green")
    axes[1, 0].set_title("Radial Distribution Function g(r)")
    axes[1, 0].set_xlabel("r")

    axes[1, 1].plot(msd_time, msd, color="tab:purple")
    axes[1, 1].set_title("MSD vs Time")
    axes[1, 1].set_xlabel("Time")

    fig.tight_layout()
    fig.savefig("md_analysis_plots.png", dpi=150)
    plt.close(fig)

    dim = int(hist["n_dims"])
    fit_start = max(1, int(0.5 * len(msd)))
    coeffs = np.polyfit(msd_time[fit_start:], msd[fit_start:], 1)
    diffusion = coeffs[0] / (2.0 * dim)
    print("saved plot: md_analysis_plots.png")
    print("estimated diffusion coefficient:", diffusion)


if __name__ == "__main__":
    main()
