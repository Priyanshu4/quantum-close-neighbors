import math 
import random
import numpy as np
from collections import namedtuple
from quantum_close_neighbors.grovers import Grovers
import scipy.optimize
import logging

logger = logging.getLogger(__name__)

CloseNeighborsHyperparameters = namedtuple('CloseNeighborsHyperparameters', 
                                ['final_collection_k',      # Function that gives k in the final collection stage
                                 'm_factor_increase',       # Factor to increase m by after the precritical stage
                                 'c1',                      # Constant 450 from Chernoff Bound
                                 'c2',                      # Constant 6 from Chernoff Bound
                                 'lambda_val']              # Constant 6/5
                                 )

CloseNeighborsResult = namedtuple('CloseNeighborsResult',
                       ['n_close_neighbors',                # Number of close neighbors found
                        'iterations',                       # Tuple of iterations in each stage
                        'grover_measurements'])             # Number of samples used (grover measurements)



DEFAULT_HYPERPARAMETERS = CloseNeighborsHyperparameters(
    final_collection_k=lambda n, alpha, A1 : (alpha+1) * 4 * A1 * np.log(n),
    m_factor_increase=1,
    c1=450,
    c2=6,
    lambda_val=6/5
)

def close_neighbors(N: int, alpha: float, hyperparameters: CloseNeighborsHyperparameters, grover: Grovers):
    """

    Args:
    N: Number of pairs
    alpha: The probability parameter for high probability bound (default 1)
    grover: The grover search object.

    Returns:
    A set of all close neighbor pairs
    """
    logging.info(f"Starting close_neighbors with N = {N}, alpha = {alpha}")

    iterations_precritical = 0
    iterations_postcritical = 0
    iterations_final_collection = 0
    grover_measurements = 0

    final_collection_k = hyperparameters.final_collection_k
    m_factor_increase = hyperparameters.m_factor_increase
    c1 = hyperparameters.c1
    c2 = hyperparameters.c2
    lambda_val = hyperparameters.lambda_val

    m = 1
    
    # Step 2-11: Find the critical stage
    logging.info("Starting precritical stage")
    while True:
        n_success = 0
        k = int(c1 * alpha * math.log(N))
        logging.info(f"Sampling k = {k} times with m = {m}")

        for _ in range(k):
            j = random.randint(0, math.ceil(m) - 1)
            element = grover.run(j, mark_result_seen=False)
            grover_measurements += 1
            iterations_precritical += j
            if element.is_marked():
                n_success += 1
   
        logging.info(f"n_success: {n_success}, k: {k}, p_m: {n_success / k}")

        if n_success / k >= 1/5:
            logging.info(f"Breaking at m: {m}")
            m = m * m_factor_increase
            logging.info(f"m increased to {m}")
            break
        
        m = min(lambda_val * m, int(math.sqrt(N)))

    logging.info(f"Precritical stage complete in {iterations_precritical} Grover iterations and {grover_measurements} Grover measurements")

        
    # Step 12-31: Collect all solutions
    logging.info("Starting postcritical stage")
    success = False
    A1 = 0
    while not success:
        n_success = 0
        k = int(c2 * alpha * math.log(N))
        d = 0
        
        logging.info(f"Sampling until k = {k} successful samples with m = {m}")
        while n_success < k:
            j = random.randint(0, math.ceil(m) - 1)

            element = grover.run(j) 
            iterations_postcritical += j
            grover_measurements += 1
            if element.is_marked():
                n_success += 1
                if not element.is_seen():
                    d += 1
                    A1 += 1
        
        logging.info(f"n_success: {n_success}, d: {d}, k: {k}, d/k: {d / k}")
        if d / k < 1/2:
            success = True

    logging.info(f"Postcritical stage complete in {iterations_postcritical} Grover iterations")

    logging.info("Starting final collection stage")
    # Final collection step
    n_success = 0
    k = final_collection_k(N, alpha,  A1) 
    k = int(k)

    logging.info(f"Sampling until k = {k} successful samples with m = {m}")

    while n_success < k:
        j = random.randint(0, math.ceil(m) - 1)
        element = grover.run(j)
        grover_measurements += 1
        iterations_final_collection += j
        if element.is_marked():
            n_success += 1
            if not element.is_seen():
                A1 += 1

   
    return CloseNeighborsResult(
        n_close_neighbors=A1,
        iterations=(iterations_precritical, iterations_postcritical, iterations_final_collection),
        grover_measurements=grover_measurements
    )

def close_neighbors_2(N: int, alpha: float, hyperparameters: CloseNeighborsHyperparameters, grover: Grovers, verbose: bool = False):
    """

    Args:
    N: Number of pairs
    alpha: The probability parameter for high probability bound (default 1)
    grover: The grover search object.

    Returns:
    A set of all close neighbor pairs
    """
    iterations_precritical = 0
    iterations_postcritical = 0
    iterations_final_collection = 0

    c1 = hyperparameters.c1
    lambda_val = hyperparameters.lambda_val

    m = 1
    grover_measurements = 0
    unique_elements_seen = 0

    while True:
        n_success = 0
        k = int(c1 * alpha * math.log(N))
        
        for _ in range(k):
            j = random.randint(0, math.ceil(m) - 1)
            element = grover.run(j, mark_result_seen=True)
            iterations_precritical += j
            grover_measurements += 1
            if element.is_marked():
                n_success += 1
                if not element.is_seen():
                    unique_elements_seen += 1
                    
    
        if n_success / k >= 1/5:
            break
        
        m = max(min(lambda_val * m, int(math.sqrt(N))), 1)
    

    thetas = find_thetas(math.ceil(m), n_success/k, intervals=1)
    if len(thetas) == 0:
        raise ValueError("No thetas found")
    
    if len(thetas) > 1:
        print(f"m: {m}, p_m: {n_success/k}")
        print(f"Thetas: {thetas}")
        raise ValueError("Multiple thetas found")
    

    theta = thetas[0]

    # Calculate the number of marked elements
    # sin(theta)**2 = mu / nu
    initial_mu_estimate = N * np.sin(theta)**2 
   
    # Remove the set of seen marked elements from the oracle
    grover.remove_all_seen_elements_from_oracle()

    running_mu_estimate = (initial_mu_estimate * 1) - unique_elements_seen
    running_mu_estimate = max(running_mu_estimate, 1)

    theta = np.arcsin(np.sqrt(running_mu_estimate / N))
    m = 1 / np.sin(2 * theta)

    while unique_elements_seen < initial_mu_estimate * 1.3:

        # batches of 5% of the running mu estimate
        batch_size = max(int(running_mu_estimate * 0.1), 50)

        # compute the optimal number of iterations
        j = round(0.58278 * np.sqrt(N / running_mu_estimate))

        n_success = 0
        for _ in range(batch_size):
            element = grover.run(j)
            grover_measurements += 1
            iterations_final_collection += j
            if element.is_marked():
                n_success += 1
                unique_elements_seen += 1
                grover.remove_seen_element_from_oracle()
                running_mu_estimate -= 1
                
                if unique_elements_seen % 100 == 0 and verbose:
                    print(f"Unique elements seen: {unique_elements_seen}, Actual mu: {grover.n_marked_elements}, Running mu estimate: {running_mu_estimate}")
                    
                    if running_mu_estimate > 0:
                        print(f"Theory probability: {math.sin((2 * j + 1) * np.sqrt(N / running_mu_estimate)) ** 2}")
                        print(f"Actual probability: {grover.p_success(j)}")
        
        if n_success == 0:
            break


        p_success = n_success / batch_size

        estimated_theta = np.arcsin(np.sqrt(p_success)) / (2*j + 1)
        estimated_mu = N * np.sin(estimated_theta)**2 

        # Take the average of our estimate and the running estimate
        running_mu_estimate = round(running_mu_estimate * 0.8 + estimated_mu * 0.2)

        if running_mu_estimate <= 0:
            running_mu_estimate = 1

        if verbose:
            print(f"Batch size: {batch_size}")
            print(f"p success: {p_success}")
            print(f"estimated theta: {estimated_theta}")
            print(f"estimated mu: {estimated_mu}")
            print(f"Actual mu: {grover.n_marked_elements}")

    return CloseNeighborsResult(unique_elements_seen, (iterations_precritical, iterations_postcritical, iterations_final_collection), grover_measurements)

def find_thetas(m, P_m, intervals=50):
    # Define the equation based on the "Tight Bounds on Quantum Searching" paper
    def equation(theta, m, P_m):
        # this equals 0 when theta is the correct value, and theta != 0 but is in the range [0, pi/2]
        # refer to "Tight Bounds on Quantum Searching" by Boyer et al., 1999
        return (1/2 - P_m) * (4 * m * np.sin(2 * theta)) - np.sin(4 * m * theta) 
    
    # Initialize an empty list to store unique roots
    roots = []
    
    # Calculate the size of each interval
    interval_size = (np.pi / 2) / intervals
    
    # Loop over each interval and find roots
    for i in range(intervals):
        start = i * interval_size
        if i == 0:
            start = 1e-6
        end = (i + 1) * interval_size
        if i == intervals - 1:
            end = np.pi / 2 - 1e-6
        try:
            root = scipy.optimize.brentq(equation, start, end, args=(m, P_m))

            # Add root if it's not already in the list (within tolerance)
            if not any(np.isclose(root, r, atol=1e-6) for r in roots):

                # Make sure the root evaluated is close to 0
                if np.isclose(equation(root, m, P_m), 0, atol=1e-6):
                    roots.append(root)

            
        except ValueError as e:
            # No root found in this interval, skip it
            print(f"No root found in interval {i} with start {start} and end {end}")
            print(f"m: {m}, P_m: {P_m}")
            print(e)
            pass

    return roots