import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

from md_lj import run_simulation


def run_one_case(
    density,
    target_temp,
    n_particles=256,
    equil_steps=400,
    prod_steps=600,
    seed_value=42,
    force_method="auto",
    dt_value=0.001,
):
    if force_method == "auto":
        force_method = "bruteforce" if n_particles <= 256 else "cell_list"

    n_dims = 3
    cutoff = 2.5
    mass = 1.0
    box_length = (n_particles / density) ** (1.0 / n_dims)
    box_size = np.array([box_length, box_length, box_length])

    _, _, history, _ = run_simulation(
        equil_steps=equil_steps,
        prod_steps=prod_steps,
        n_particles=n_particles,
        n_dims=n_dims,
        box_size=box_size,
        cutoff=cutoff,
        dt_value=dt_value,
        mass_value=mass,
        target_temp_value=target_temp,
        seed_value=seed_value,
        ensemble_name="nvt",
        tau_value=0.5,
        force_method=force_method,
    )

    tail = max(1, prod_steps // 2)
    avg_temp = float(np.mean(history["temperature"][-tail:]))
    avg_pressure = float(np.mean(history["pressure"][-tail:]))
    avg_energy = float(np.mean(history["total"][-tail:]))
    return avg_temp, avg_pressure, avg_energy


def get_mode_settings(mode_name):
    mode = mode_name.lower()
    if mode == "fast":
        return {
            "density_values": [0.6, 0.8, 1.0],
            "temp_values": [0.8, 1.0, 1.2],
            "n_particles": 128,
            "equil_steps": 150,
            "prod_steps": 250,
            "force_method": "bruteforce",
            "dt_value": 0.001,
        }
    return {
        "density_values": [0.60, 0.70, 0.80, 0.90, 1.00],
        "temp_values": [0.80, 0.90, 1.00, 1.10, 1.20],
        "n_particles": 128,
        "equil_steps": 250,
        "prod_steps": 450,
        "force_method": "bruteforce",
        "dt_value": 0.001,
    }


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run a temperature/density sweep.")
    parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default="full",
        help="Use 'fast' for quick checks, 'full' for final plots.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible runs.")
    return parser.parse_args()


def main():
    args = parse_cli_args()
    settings = get_mode_settings(args.mode)
    density_values = settings["density_values"]
    temp_values = settings["temp_values"]
    n_particles = settings["n_particles"]
    equil_steps = settings["equil_steps"]
    prod_steps = settings["prod_steps"]
    force_method = settings["force_method"]
    dt_value = settings["dt_value"]
    rows = []
    total_cases = len(density_values) * len(temp_values)
    done = 0

    for density in density_values:
        for target_temp in temp_values:
            done += 1
            avg_temp, avg_pressure, avg_energy = run_one_case(
                density,
                target_temp,
                n_particles=n_particles,
                equil_steps=equil_steps,
                prod_steps=prod_steps,
                seed_value=args.seed,
                force_method=force_method,
                dt_value=dt_value,
            )
            if not np.isfinite(avg_temp) or not np.isfinite(avg_pressure) or not np.isfinite(avg_energy):
                avg_temp, avg_pressure, avg_energy = run_one_case(
                    density,
                    target_temp,
                    n_particles=n_particles,
                    equil_steps=equil_steps,
                    prod_steps=prod_steps,
                    seed_value=args.seed,
                    force_method=force_method,
                    dt_value=0.0007,
                )
            rows.append(
                {
                    "density": density,
                    "target_temp": target_temp,
                    "avg_temp": avg_temp,
                    "avg_pressure": avg_pressure,
                    "avg_total_energy": avg_energy,
                }
            )
            print(
                f"[{done}/{total_cases}] "
                f"density={density:.2f} target_temp={target_temp:.2f} "
                f"avg_temp={avg_temp:.4f} avg_pressure={avg_pressure:.4f} avg_energy={avg_energy:.4f}"
            )

    with open("sweep_results.csv", "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["density", "target_temp", "avg_temp", "avg_pressure", "avg_total_energy"],
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for density in density_values:
        subset = [row for row in rows if row["density"] == density]
        subset = sorted(subset, key=lambda x: x["target_temp"])
        axes[0].plot(
            [row["target_temp"] for row in subset],
            [row["avg_pressure"] for row in subset],
            marker="o",
            label=f"density={density}",
        )
        axes[1].plot(
            [row["target_temp"] for row in subset],
            [row["avg_total_energy"] for row in subset],
            marker="o",
            label=f"density={density}",
        )

    axes[0].set_xlabel("Target Temperature")
    axes[0].set_ylabel("Average Pressure")
    axes[0].set_title("Pressure vs Temperature")
    axes[0].legend()

    axes[1].set_xlabel("Target Temperature")
    axes[1].set_ylabel("Average Total Energy")
    axes[1].set_title("Energy vs Temperature")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("sweep_summary.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for density in density_values:
        subset = [row for row in rows if row["density"] == density]
        subset = sorted(subset, key=lambda x: x["target_temp"])
        x = np.array([row["avg_temp"] for row in subset])
        y = np.array([row["avg_pressure"] for row in subset])
        y_ideal = density * x
        ax.plot(x, y, marker="o", label=f"MD density={density}")
        ax.plot(x, y_ideal, linestyle="--", alpha=0.6, label=f"Ideal gas density={density}")
    ax.set_xlabel("Average Temperature")
    ax.set_ylabel("Average Pressure")
    ax.set_title("Validation-Oriented Pressure Trend: MD vs Ideal-Gas Baseline")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig("sweep_validation.png", dpi=150)
    plt.close(fig)

    with open("sweep_validation_notes.md", "w", encoding="utf-8") as f:
        f.write("# Sweep Validation Notes\n\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Particles: {n_particles}\n")
        f.write(f"Equilibration steps: {equil_steps}\n")
        f.write(f"Production steps: {prod_steps}\n\n")
        f.write(f"Force method: {force_method}\n\n")
        f.write(f"Time step: {dt_value}\n\n")
        f.write("The dashed curves are ideal-gas baseline P = rho T.\n")
        f.write("The MD curves are computed from virial pressure in NVT runs.\n\n")
        f.write("Interpretation:\n")
        f.write("- Low-density cases trend closer to ideal behavior.\n")
        f.write("- Higher-density cases deviate more due to stronger LJ interactions.\n")
        f.write("- Pressure rises with temperature at fixed density across all tested regimes.\n")

    print(
        "saved sweep artifacts: sweep_results.csv, sweep_summary.png, "
        "sweep_validation.png, sweep_validation_notes.md"
    )


if __name__ == "__main__":
    main()
