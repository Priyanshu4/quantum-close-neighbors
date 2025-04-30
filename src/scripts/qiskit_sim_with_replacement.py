""" Use Qiskit to simulate close_neighbors_1 (with replacement). """

import math
import argparse
import json
import datetime
import time
from multiprocessing import cpu_count
import logging
from typing import Tuple

from quantum_close_neighbors.close_neighbors import (
    close_neighbors,
    CloseNeighborsHyperparameters,
    DEFAULT_HYPERPARAMETERS,
    CloseNeighborsResult)
from quantum_close_neighbors.grovers import QiskitGroversStaticPhaseOracle

logger = logging.getLogger(__name__)

def run_close_neighbors_with_replacement(
        n_particles: int,
        n_neighbors_per_particle: int,
        desired_success_probability: float,
        hyperparameters:  CloseNeighborsHyperparameters) -> Tuple[CloseNeighborsResult, bool]:
    """ Run close neighbors using Qiskit. """
    
    logger.info(f"Starting run_close_neighbors_with_replacement")
    logger.info(f"n_particles = {n_particles}, n_neighbors_per_particle = {n_neighbors_per_particle}, desired_success_probability = {desired_success_probability}")
    logger.info(f"Hyperparameters: c1 = {hyperparameters.c1}, c2 = {hyperparameters.c2}, lambda_val = {hyperparameters.lambda_val}, m_factor_increase = {hyperparameters.m_factor_increase}")

    nu = math.comb(n_particles, 2)                          # nu is n choose 2
    mu = n_particles * n_neighbors_per_particle // 2        # mu is the number of close neighbors
    alpha = -math.log(1 - desired_success_probability) / math.log(nu)

    logger.info(f"nu = {nu}, mu = {mu}, alpha = {alpha}")

    grovers = QiskitGroversStaticPhaseOracle.init_with_random_marked_elements(nu, mu)
    result = close_neighbors(nu, alpha, hyperparameters, grovers)
    success = (result.n_close_neighbors == mu)

    return result, success


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger('qiskit').setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Run close neighbors with replacement using Qiskit.")
    parser.add_argument("n_particles", type=int, help="Number of particles.")
    parser.add_argument("n_neighbors_per_particle", type=int, help="Number of neighbors per particle.")
    parser.add_argument("output_folder", type=str, help="Output folder")
    parser.add_argument("--desired_success_probability", type=float, default="0.99", help="Desired success probability.")
    parser.add_argument("--c1", type=float, help="Hyperparameter c1.")
    parser.add_argument("--c2", type=float, help="Hyperparameter c2.")

    args = parser.parse_args()
    n_particles = args.n_particles
    n_neighbors_per_particle = args.n_neighbors_per_particle
    desired_success_probability = args.desired_success_probability

    hyperparameters = DEFAULT_HYPERPARAMETERS
    if args.c1 is not None:
        hyperparameters = hyperparameters._replace(c1=args.c1)
    if args.c2 is not None:
        hyperparameters = hyperparameters._replace(c2=args.c2)

    start = time.time()
    result, success = run_close_neighbors_with_replacement(
        n_particles=n_particles, 
        n_neighbors_per_particle=n_neighbors_per_particle, 
        desired_success_probability=desired_success_probability, 
        hyperparameters=hyperparameters)
    end = time.time()
    runtime = end - start
 
    logger.info(f"Close Neighbors Found: {result.n_close_neighbors}")
    logger.info(f"Iterations: {result.iterations}")
    logger.info(f"Grover Measurements: {result.grover_measurements}")
    logger.info(f"Success: {success}")
    logger.info(f"Time taken: {runtime} seconds")

    
    # create a json file to store the results, with filename with timestamp
    output_folder = args.output_folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"close_neighbors_with_replacement_{timestamp}.json"

    out = {
        "n_particles": n_particles,
        "n_neighbors_per_particle": n_neighbors_per_particle,
        "desired_success_probability": desired_success_probability,
        "hyperparameters": {
            "c1": hyperparameters.c1,
            "c2": hyperparameters.c2,
            "lambda_val": hyperparameters.lambda_val,
            "m_factor_increase": hyperparameters.m_factor_increase,
        },
        "result": {
            "n_close_neighbors": result.n_close_neighbors,
            "grover_measurements": result.grover_measurements,
            "iterations": {
                "precritical": result.iterations[0],
                "postcritical": result.iterations[1],
                "final_collection": result.iterations[2]
            },
        },
        "runtime": runtime,
        "num_cores": cpu_count(),
    }
    
    with open(f"{output_folder}/{filename}", "w") as f:
        json.dump(out, f, indent=4)

    logger.info(f"Results saved to {output_folder}/{filename}")

