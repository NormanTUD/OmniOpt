import numpy as np
import math
import copy

from hyperopt import hp, fmin, Trials, pyll
from hyperopt.base import miscs_update_idxs_vals
from hyperopt.pyll.base import dfs, as_apply
from hyperopt.pyll.stochastic import implicit_stochastic_symbols


class PSOError(Exception):
    pass


def validate_space_pso(space):
    supported_stochastic_symbols = ['uniform', 'quniform', 'loguniform', 'qloguniform', 'normal', 'qnormal', 'lognormal', 'qlognormal']
    for node in dfs(as_apply(space)):
        if node.name in implicit_stochastic_symbols:
            if node.name not in supported_stochastic_symbols:
                raise PSOError('PSO is only possible with the following stochastic symbols: ' + ', '.join(supported_stochastic_symbols))

def pso(objective, bounds, n_particles, n_iterations, c1, c2, w):

    # Initialize the particles
    particles = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(n_particles, len(bounds)))

    # Initialize the velocities
    velocities = np.zeros((n_particles, len(bounds)))

    # Initialize the personal best positions
    personal_bests = particles.copy()

    # Evaluate the objective function at the initial positions
    particle_objectives = np.array([objective(p) for p in particles])

    # Initialize the global best position
    global_best = particles[np.argmin(particle_objectives)]

    # Iterate for the specified number of iterations
    for i in range(n_iterations):

        # Update the velocities
        for j in range(n_particles):
            r1 = np.random.random(len(bounds))
            r2 = np.random.random(len(bounds))
            velocities[j] = w * velocities[j] + c1 * r1 * (personal_bests[j] - particles[j]) + c2 * r2 * (global_best - particles[j])

            # Clip the velocities to the specified bounds
            for k in range(len(bounds)):
                velocities[j][k] = np.clip(velocities[j][k], -abs(bounds[k][1] - bounds[k][0]), abs(bounds[k][1] - bounds[k][0]))

        # Update the particle positions
        for j in range(n_particles):
            particles[j] += velocities[j]

            # Clip the positions to the specified bounds
            for k in range(len(bounds)):
                particles[j][k] = np.clip(particles[j][k], bounds[k][0], bounds[k][1])

        # Evaluate the objective function at the new positions
        new_particle_objectives = np.array([objective(p) for p in particles])

        # Update the personal best positions
        personal_best_indices = new_particle_objectives < particle_objectives
        personal_bests[personal_best_indices] = particles[personal_best_indices]
        particle_objectives[personal_best_indices] = new_particle_objectives[personal_best_indices]

        # Update the global best position
        global_best_index = np.argmin(particle_objectives)
        global_best = particles[global_best_index]

    return global_best, particle_objectives[global_best_index]
