"""Integration tests for the quantum harmonic oscillator."""
import logging

import jax
import jax.numpy as jnp
import numpy as np

import vmcnet.examples.harmonic_oscillator as qho
import vmcnet.mcmc as mcmc
import vmcnet.train as train
import vmcnet.updates as updates
import vmcnet.utils as utils


def _make_initial_params_and_data(model_omega, nchains):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    random_particle_positions = jax.random.normal(subkey, shape=(nchains, 5, 1))

    # because there are 5 particles total, the spin split is (3, 2)
    log_psi_model = qho.make_harmonic_oscillator_spin_half_model(2, model_omega)

    key, subkey = jax.random.split(key)
    params = log_psi_model.init(subkey, random_particle_positions)
    amplitudes = log_psi_model.apply(params, random_particle_positions)
    return log_psi_model, params, random_particle_positions, amplitudes, key


def test_five_particle_ground_state_harmonic_oscillator():
    """Test five non-interacting harmonic oscillators with two spins."""
    omega = 2.0
    (
        log_psi_model,
        params,
        random_particle_positions,
        _,
        _,
    ) = _make_initial_params_and_data(omega, 4)

    local_energy_fn = qho.make_harmonic_oscillator_local_energy(
        omega, log_psi_model.apply
    )
    local_energies = local_energy_fn(params, random_particle_positions)

    np.testing.assert_allclose(
        local_energies, omega * (0.5 + 1.5 + 0.5 + 1.5 + 2.5) * jnp.ones(4), rtol=1e-5
    )


def test_harmonic_oscillator_vmc(caplog):
    """Test that the trainable sqrt(omega) converges to the true sqrt(spring constant).

    Integration test for the overall API, to make sure it comes together correctly and
    can optimize a simple 1 parameter model rapidly.
    """
    # Problem parameters
    model_omega = 2.5
    spring_constant = 1.5

    # Training hyperparameters
    nchains = 100 * jax.local_device_count()
    nburn = 100
    nepochs = 50
    nsteps_per_param_update = 5
    std_move = 0.25
    learning_rate = 1e-4

    # Initialize model and chains of walkers
    (
        log_psi_model,
        params,
        random_particle_positions,
        amplitudes,
        key,
    ) = _make_initial_params_and_data(model_omega, nchains)
    data = updates.data.PositionAmplitudeData(random_particle_positions, amplitudes)

    # Setup metropolis step
    metrop_step_fn = mcmc.metropolis.make_position_amplitude_gaussian_metropolis_step(
        std_move, log_psi_model.apply
    )

    # Setup parameter updates
    local_energy_fn = qho.make_harmonic_oscillator_local_energy(
        spring_constant, log_psi_model.apply
    )

    def sgd_apply(grad, params, learning_rate):
        return (
            jax.tree_map(lambda a, b: a - learning_rate * b, params, grad),
            learning_rate,
        )

    update_param_fn = updates.params.create_position_amplitude_data_update_param_fn(
        log_psi_model.apply, local_energy_fn, nchains, sgd_apply
    )

    # Distribute everything via jax.pmap
    (
        data,
        params,
        optimizer_state,
        key,
    ) = utils.distribute.distribute_data_params_optstate_and_key(
        data, params, learning_rate, key
    )

    # Train!
    with caplog.at_level(logging.INFO):
        params, _, _ = train.vmc.vmc_loop(
            params,
            optimizer_state,
            data,
            nchains,
            nburn,
            nepochs,
            nsteps_per_param_update,
            metrop_step_fn,
            update_param_fn,
            key,
        )

    # Grab the one parameter and make sure it converged to sqrt(spring constant)
    np.testing.assert_allclose(
        jax.tree_leaves(params)[0], jnp.sqrt(spring_constant), rtol=1e-6
    )