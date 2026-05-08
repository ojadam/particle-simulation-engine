import csv
import time
import numpy as np
import matplotlib.pyplot as plt

from md_lj import compute_lj_forces


def time_force_call(n_particles, density=0.8, repeats=3, seed=42):
    n_dims = 3
    cutoff = 2.5
    box_length = (n_particles / density) ** (1.0 / n_dims)
    box_size = np.array([box_length, box_length, box_length])
    rng = np.random.default_rng(seed)
    positions = rng.random((n_particles, n_dims)) * box_size

    times_by_method = {}
    for method in ("bruteforce", "cell_list"):
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            compute_lj_forces(positions, box_size, cutoff, method=method)
            timings.append(time.perf_counter() - start)
        times_by_method[method] = float(np.mean(timings))
    return times_by_method


def main():
    particle_counts = [128, 256, 512]
    rows = []

    for n_particles in particle_counts:
        out = time_force_call(n_particles)
        speedup = out["bruteforce"] / max(out["cell_list"], 1e-12)
        rows.append(
            {
                "n_particles": n_particles,
                "bruteforce_s": out["bruteforce"],
                "cell_list_s": out["cell_list"],
                "speedup": speedup,
            }
        )
        print(
            f"N={n_particles} bruteforce={out['bruteforce']:.5f}s "
            f"cell_list={out['cell_list']:.5f}s speedup={speedup:.2f}x"
        )

    with open("benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["n_particles", "bruteforce_s", "cell_list_s", "speedup"])
        writer.writeheader()
        writer.writerows(rows)

    particle_counts_arr = np.array([r["n_particles"] for r in rows])
    brute = np.array([r["bruteforce_s"] for r in rows])
    cell = np.array([r["cell_list_s"] for r in rows])
    speedup = np.array([r["speedup"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(particle_counts_arr, brute, marker="o", label="Brute force")
    axes[0].plot(particle_counts_arr, cell, marker="o", label="Cell list")
    axes[0].set_xlabel("Particle count")
    axes[0].set_ylabel("Force call time (s)")
    axes[0].set_title("Force Routine Runtime")
    axes[0].legend()

    axes[1].plot(particle_counts_arr, speedup, marker="o", color="tab:green")
    axes[1].set_xlabel("Particle count")
    axes[1].set_ylabel("Speedup (brute/cell)")
    axes[1].set_title("Cell-List Speedup")

    fig.tight_layout()
    fig.savefig("benchmark_summary.png", dpi=150)
    plt.close(fig)
    print("saved benchmark artifacts: benchmark_results.csv, benchmark_summary.png")


if __name__ == "__main__":
    main()
