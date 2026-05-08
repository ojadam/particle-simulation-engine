import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

n_dims = 3
n_particles = 256
density = 0.80
target_temp = 1.00
dt = 0.0015
equil_steps = 300
prod_steps = 300
cutoff = 2.5
mass = 1.0
k_b = 1.0
random_seed = 42
ensemble_mode = "nve"
thermostat_tau = 0.5
report_every = 100
save_plots = True
save_outputs = True
sample_every = 5

box_length = (n_particles / density) ** (1.0 / n_dims)
box_size = np.array([box_length, box_length, box_length])
box_volume = np.prod(box_size)


def make_initial_positions(n_particles, box_size, n_dims=3, seed_value=42):
    points_per_side = int(np.ceil(n_particles ** (1.0 / n_dims)))
    cell_size = box_size[0] / points_per_side
    axis_points = (np.arange(points_per_side) + 0.5) * cell_size
    xx, yy, zz = np.meshgrid(axis_points, axis_points, axis_points, indexing="ij")
    all_points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    rng = np.random.default_rng(seed_value)
    chosen_ids = rng.choice(all_points.shape[0], size=n_particles, replace=False)
    return all_points[chosen_ids].copy()


def make_initial_velocities(particle_count, dims, wanted_temp, mass_value=1.0, seed_value=42):
    rng = np.random.default_rng(seed_value)
    vel = rng.normal(0.0, 1.0, size=(particle_count, dims))
    vel -= vel.mean(axis=0, keepdims=True)
    mean_v_sq = np.mean(np.sum(vel**2, axis=1))
    temp_now = (mass_value * mean_v_sq) / dims
    scale = np.sqrt(wanted_temp / temp_now)
    vel *= scale
    return vel


def wrap_into_box(pos, box_size):
    return pos % box_size


def minimum_image_delta(delta, box_size):
    return delta - box_size * np.round(delta / box_size)


def _lj_shift_terms(cutoff):
    cutoff_sq = cutoff * cutoff
    inv_rc2 = 1.0 / cutoff_sq
    inv_rc6 = inv_rc2**3
    inv_rc12 = inv_rc6**2
    u_cut = 4.0 * (inv_rc12 - inv_rc6)
    f_cut = 24.0 * (2.0 * inv_rc12 - inv_rc6) / cutoff
    force_scale_cut = f_cut / cutoff
    return cutoff_sq, u_cut, f_cut, force_scale_cut


def _accumulate_pair(delta, r_sq, cutoff, u_cut, f_cut, force_scale_cut):
    r = np.sqrt(r_sq)
    inv_r2 = 1.0 / r_sq
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2

    pair_potential_raw = 4.0 * (inv_r12 - inv_r6)
    pair_potential = pair_potential_raw - u_cut + (r - cutoff) * f_cut

    force_scale_raw = 24.0 * inv_r2 * (2.0 * inv_r12 - inv_r6)
    force_scale = force_scale_raw - force_scale_cut
    pair_force = force_scale * delta
    pair_virial = np.dot(delta, pair_force)
    return pair_force, pair_potential, pair_virial


def _build_cell_dict(pos, box_size, cutoff):
    cells_per_dim = np.maximum(1, np.floor(box_size / cutoff).astype(int))
    cell_size = box_size / cells_per_dim
    cell_dict = defaultdict(list)
    particle_cells = np.floor(pos / cell_size).astype(int) % cells_per_dim
    for idx, cell_idx in enumerate(particle_cells):
        cell_dict[tuple(cell_idx)].append(idx)
    return cell_dict, cells_per_dim


def compute_lj_forces_bruteforce(pos, box_size, cutoff):
    particle_count = pos.shape[0]
    forces = np.zeros_like(pos)
    potential_energy = 0.0
    virial_sum = 0.0
    cutoff_sq, u_cut, f_cut, force_scale_cut = _lj_shift_terms(cutoff)

    for i in range(particle_count - 1):
        for j in range(i + 1, particle_count):
            delta = pos[i] - pos[j]
            delta = minimum_image_delta(delta, box_size)
            r_sq = np.dot(delta, delta)

            if r_sq < cutoff_sq and r_sq > 1e-12:
                pair_force, pair_potential, pair_virial = _accumulate_pair(
                    delta, r_sq, cutoff, u_cut, f_cut, force_scale_cut
                )

                forces[i] += pair_force
                forces[j] -= pair_force
                potential_energy += pair_potential
                virial_sum += pair_virial

    return forces, potential_energy, virial_sum


def compute_lj_forces_cell_list(pos, box_size, cutoff):
    particle_count = pos.shape[0]
    forces = np.zeros_like(pos)
    potential_energy = 0.0
    virial_sum = 0.0
    cutoff_sq, u_cut, f_cut, force_scale_cut = _lj_shift_terms(cutoff)
    cell_dict, cells_per_dim = _build_cell_dict(pos, box_size, cutoff)

    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]

    def cell_linear_id(cell_idx):
        return (
            cell_idx[0] * cells_per_dim[1] * cells_per_dim[2]
            + cell_idx[1] * cells_per_dim[2]
            + cell_idx[2]
        )

    seen_pairs = set()

    for cell_key, in_cell_particles in cell_dict.items():
        cell_arr = np.array(cell_key, dtype=int)
        unique_neighbors = set()

        for offset in neighbor_offsets:
            neighbor = (cell_arr + np.array(offset, dtype=int)) % cells_per_dim
            neighbor_key = tuple(neighbor.tolist())
            if neighbor_key in unique_neighbors:
                continue
            unique_neighbors.add(neighbor_key)
            if neighbor_key not in cell_dict:
                continue

            neighbor_particles = cell_dict[neighbor_key]
            for i in in_cell_particles:
                for j in neighbor_particles:
                    if j == i:
                        continue
                    pair = (i, j) if i < j else (j, i)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    delta = pos[i] - pos[j]
                    delta = minimum_image_delta(delta, box_size)
                    r_sq = np.dot(delta, delta)
                    if r_sq < cutoff_sq and r_sq > 1e-12:
                        pair_force, pair_potential, pair_virial = _accumulate_pair(
                            delta, r_sq, cutoff, u_cut, f_cut, force_scale_cut
                        )
                        forces[i] += pair_force
                        forces[j] -= pair_force
                        potential_energy += pair_potential
                        virial_sum += pair_virial

    return forces, potential_energy, virial_sum


def compute_lj_forces(pos, box_size, cutoff, method="bruteforce"):
    method_name = method.lower()
    if method_name in {"cell", "cell_list", "celllist"}:
        return compute_lj_forces_cell_list(pos, box_size, cutoff)
    return compute_lj_forces_bruteforce(pos, box_size, cutoff)


def get_kinetic_energy(vel, mass_value=1.0):
    speed_sq = np.sum(vel**2, axis=1)
    return 0.5 * mass_value * np.sum(speed_sq)


def get_temperature(vel, mass_value=1.0, boltzmann_value=1.0):
    particle_count, dims = vel.shape
    total_v_sq = np.sum(vel**2)
    return (mass_value * total_v_sq) / (dims * particle_count * boltzmann_value)


def get_pressure(particle_count, temp_now, virial_sum, volume, dims=3):
    number_density = particle_count / volume
    return number_density * temp_now + virial_sum / (dims * volume)


def apply_berendsen_scaling(vel, temp_now, target_temp, dt_value, tau_value):
    if tau_value <= 0.0:
        return vel
    scale_sq = 1.0 + (dt_value / tau_value) * ((target_temp / temp_now) - 1.0)
    if scale_sq <= 0.0:
        return vel
    return vel * np.sqrt(scale_sq)


def velocity_verlet_step(
    pos,
    vel,
    forces_now,
    dt_value,
    mass_value,
    box_size,
    cutoff,
    ensemble_name="nve",
    target_temp_value=1.0,
    tau_value=0.5,
    force_method="bruteforce",
):
    vel_half = vel + 0.5 * dt_value * forces_now / mass_value
    pos_new = wrap_into_box(pos + dt_value * vel_half, box_size)
    forces_new, potential_new, virial_new = compute_lj_forces(
        pos_new, box_size, cutoff, method=force_method
    )
    vel_new = vel_half + 0.5 * dt_value * forces_new / mass_value

    if ensemble_name.lower() == "nvt":
        temp_now = get_temperature(vel_new, mass_value=mass_value, boltzmann_value=k_b)
        vel_new = apply_berendsen_scaling(vel_new, temp_now, target_temp_value, dt_value, tau_value)

    return pos_new, vel_new, forces_new, potential_new, virial_new


def run_simulation(
    equil_steps,
    prod_steps,
    n_particles,
    n_dims,
    box_size,
    cutoff,
    dt_value,
    mass_value,
    target_temp_value,
    seed_value,
    ensemble_name="nve",
    tau_value=0.5,
    force_method="bruteforce",
):
    volume = float(np.prod(box_size))
    pos = make_initial_positions(n_particles, box_size, n_dims=n_dims, seed_value=seed_value)
    vel = make_initial_velocities(
        particle_count=n_particles,
        dims=n_dims,
        wanted_temp=target_temp_value,
        mass_value=mass_value,
        seed_value=seed_value,
    )
    forces_now, potential_now, virial_now = compute_lj_forces(
        pos, box_size, cutoff, method=force_method
    )

    for _ in range(equil_steps):
        pos, vel, forces_now, _, _ = velocity_verlet_step(
            pos=pos,
            vel=vel,
            forces_now=forces_now,
            dt_value=dt_value,
            mass_value=mass_value,
            box_size=box_size,
            cutoff=cutoff,
            ensemble_name=ensemble_name,
            target_temp_value=target_temp_value,
            tau_value=tau_value,
            force_method=force_method,
        )

    history = {
        "step": np.arange(prod_steps + 1),
        "kinetic": np.zeros(prod_steps + 1),
        "potential": np.zeros(prod_steps + 1),
        "total": np.zeros(prod_steps + 1),
        "temperature": np.zeros(prod_steps + 1),
        "pressure": np.zeros(prod_steps + 1),
    }
    sampled_positions = [pos.copy()]

    forces_now, potential_now, virial_now = compute_lj_forces(
        pos, box_size, cutoff, method=force_method
    )
    kinetic_now = get_kinetic_energy(vel, mass_value=mass_value)
    temp_now = get_temperature(vel, mass_value=mass_value, boltzmann_value=k_b)
    pressure_now = get_pressure(n_particles, temp_now, virial_now, volume, dims=n_dims)

    history["kinetic"][0] = kinetic_now
    history["potential"][0] = potential_now
    history["total"][0] = kinetic_now + potential_now
    history["temperature"][0] = temp_now
    history["pressure"][0] = pressure_now

    for step_idx in range(1, prod_steps + 1):
        pos, vel, forces_now, potential_now, virial_now = velocity_verlet_step(
            pos=pos,
            vel=vel,
            forces_now=forces_now,
            dt_value=dt_value,
            mass_value=mass_value,
            box_size=box_size,
            cutoff=cutoff,
            ensemble_name=ensemble_name,
            target_temp_value=target_temp_value,
            tau_value=tau_value,
            force_method=force_method,
        )

        kinetic_now = get_kinetic_energy(vel, mass_value=mass_value)
        temp_now = get_temperature(vel, mass_value=mass_value, boltzmann_value=k_b)
        pressure_now = get_pressure(n_particles, temp_now, virial_now, volume, dims=n_dims)

        history["kinetic"][step_idx] = kinetic_now
        history["potential"][step_idx] = potential_now
        history["total"][step_idx] = kinetic_now + potential_now
        history["temperature"][step_idx] = temp_now
        history["pressure"][step_idx] = pressure_now
        if step_idx % sample_every == 0:
            sampled_positions.append(pos.copy())

    return pos, vel, history, np.array(sampled_positions)


def save_run_plots(history, output_path="md_core_plots.png"):
    steps = history["step"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(steps, history["total"], label="Total")
    axes[0].plot(steps, history["kinetic"], label="Kinetic")
    axes[0].plot(steps, history["potential"], label="Potential")
    axes[0].set_ylabel("Energy")
    axes[0].legend()

    axes[1].plot(steps, history["temperature"], color="tab:red")
    axes[1].set_ylabel("Temperature")

    axes[2].plot(steps, history["pressure"], color="tab:green")
    axes[2].set_ylabel("Pressure")
    axes[2].set_xlabel("Step")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_xyz_trajectory(sampled_positions, box_size, output_path="trajectory.xyz"):
    with open(output_path, "w", encoding="utf-8") as xyz_file:
        for frame_idx in range(sampled_positions.shape[0]):
            xyz_file.write(f"{sampled_positions.shape[1]}\n")
            xyz_file.write(
                f"frame={frame_idx} Lx={box_size[0]:.6f} Ly={box_size[1]:.6f} Lz={box_size[2]:.6f}\n"
            )
            frame_pos = sampled_positions[frame_idx]
            for atom_idx in range(frame_pos.shape[0]):
                x, y, z = frame_pos[atom_idx]
                xyz_file.write(f"Ar {x:.6f} {y:.6f} {z:.6f}\n")


def main():
    final_positions, final_velocities, run_history, sampled_positions = run_simulation(
        equil_steps=equil_steps,
        prod_steps=prod_steps,
        n_particles=n_particles,
        n_dims=n_dims,
        box_size=box_size,
        cutoff=cutoff,
        dt_value=dt,
        mass_value=mass,
        target_temp_value=target_temp,
        seed_value=random_seed,
        ensemble_name=ensemble_mode,
        tau_value=thermostat_tau,
        force_method="bruteforce",
    )

    energy_start = run_history["total"][0]
    energy_end = run_history["total"][-1]
    energy_drift = (energy_end - energy_start) / abs(energy_start)

    print("ensemble:", ensemble_mode)
    print("equilibration steps:", equil_steps)
    print("production steps:", prod_steps)
    print("initial total energy:", energy_start)
    print("final total energy:", energy_end)
    print("relative energy drift:", energy_drift)
    print("final temperature:", run_history["temperature"][-1])
    print("final pressure:", run_history["pressure"][-1])

    if report_every > 0:
        for idx in range(report_every, prod_steps + 1, report_every):
            print(
                f"step {idx}: E={run_history['total'][idx]:.6f} "
                f"T={run_history['temperature'][idx]:.6f} "
                f"P={run_history['pressure'][idx]:.6f}"
            )

    if save_plots:
        save_run_plots(run_history)
        print("saved plot: md_core_plots.png")

    if save_outputs:
        np.save("trajectory_samples.npy", sampled_positions)
        np.savez(
            "run_history.npz",
            step=run_history["step"],
            kinetic=run_history["kinetic"],
            potential=run_history["potential"],
            total=run_history["total"],
            temperature=run_history["temperature"],
            pressure=run_history["pressure"],
            box_size=box_size,
            dt=dt,
            sample_every=sample_every,
            n_particles=n_particles,
            n_dims=n_dims,
            equil_steps=equil_steps,
            prod_steps=prod_steps,
        )
        save_xyz_trajectory(sampled_positions, box_size, output_path="trajectory.xyz")
        print("saved outputs: trajectory_samples.npy, run_history.npz, trajectory.xyz")

    return final_positions, final_velocities, run_history, sampled_positions


if __name__ == "__main__":
    main()

