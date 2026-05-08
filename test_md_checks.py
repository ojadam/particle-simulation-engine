import numpy as np

from md_lj import (
    minimum_image_delta,
    wrap_into_box,
    compute_lj_forces_bruteforce,
    compute_lj_forces_cell_list,
    run_simulation,
)


def test_wrap_into_box():
    box = np.array([10.0, 10.0, 10.0])
    pos = np.array([[11.2, -0.5, 9.8], [20.0, 15.1, -3.2]])
    wrapped = wrap_into_box(pos, box)
    assert np.all(wrapped >= 0.0)
    assert np.all(wrapped < box)


def test_minimum_image():
    box = np.array([10.0, 10.0, 10.0])
    delta = np.array([6.0, -6.5, 4.9])
    out = minimum_image_delta(delta, box)
    assert np.all(np.abs(out) <= box / 2.0 + 1e-12)


def test_force_consistency():
    box = np.array([8.0, 8.0, 8.0])
    cutoff = 2.5
    rng = np.random.default_rng(123)
    pos = rng.random((64, 3)) * box

    f1, u1, v1 = compute_lj_forces_bruteforce(pos, box, cutoff)
    f2, u2, v2 = compute_lj_forces_cell_list(pos, box, cutoff)
    assert np.allclose(f1, f2, atol=1e-6)
    assert abs(u1 - u2) < 1e-6
    assert abs(v1 - v2) < 1e-6
    assert np.allclose(np.sum(f1, axis=0), 0.0, atol=1e-6)


def test_short_nve_energy_drift():
    n_particles = 128
    density = 0.8
    n_dims = 3
    box_length = (n_particles / density) ** (1.0 / n_dims)
    box_size = np.array([box_length, box_length, box_length])
    _, _, history, _ = run_simulation(
        equil_steps=100,
        prod_steps=200,
        n_particles=n_particles,
        n_dims=n_dims,
        box_size=box_size,
        cutoff=2.5,
        dt_value=0.001,
        mass_value=1.0,
        target_temp_value=1.0,
        seed_value=42,
        ensemble_name="nve",
        tau_value=0.5,
        force_method="cell_list",
    )
    e0 = history["total"][0]
    e1 = history["total"][-1]
    rel_drift = abs((e1 - e0) / max(abs(e0), 1e-12))
    assert rel_drift < 0.05


def main():
    test_wrap_into_box()
    test_minimum_image()
    test_force_consistency()
    test_short_nve_energy_drift()
    print("all checks passed")


if __name__ == "__main__":
    main()
