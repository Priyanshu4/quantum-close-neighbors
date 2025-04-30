""" Algorithms proposed in by Combarro et al. (2023) in the paper "Quantum algorithms to compute the neighbour list of N-body simulations
"""


import math 
import random
from quantum_close_neighbors.grovers import Grovers, AbstractGroverResult


def algorithm_1(nu: int, mu: int, error_bound: float, grover: Grovers):
  """
  Implements Algorithm 1 for finding marked elements.

  Args:

    nu: The number of elements in the dataset.
    mu: The known number of marked elements.
    error_bound: The desired error bound probability.
    grover: An instance of the Grovers class.
    
  Returns:
    A set of marked elements.
  """
  total_grovers_iterations = 0
  total_grovers_measurements = 0

  marked_elements = 0
  R = math.ceil(math.log(error_bound / mu) / math.log(1 - 1/(2*mu)))

  for _ in range(R):
    
    # Run Grover's algorithm with pi/4 * sqrt(nu/mu) iterations
    iterations = math.ceil(math.pi / 4 * math.sqrt(nu / mu))
    total_grovers_measurements += 1
    element = grover.run(iterations)
    total_grovers_iterations += iterations

    # Add the marked element to the set if it's not already present
    if element is AbstractGroverResult.MARKED_UNSEEN:
      marked_elements += 1

    # Stop if all marked elements have been found
    if marked_elements == mu:
      break

  return marked_elements, total_grovers_iterations, total_grovers_measurements

def algorithm_2(nu: int, B: int, error_bound: float, grover: Grovers):
    """
    Implements Algorithm 2 
    
    Args:
        nu: The number of elements in the dataset.
        B: Upper bound on the number of marked elements (mu).
        error_bound: The desired error bound probability.
        grover: An instance of the Grovers class.
    """
    total_grovers_iterations = 0
    total_grovers_measurements = 0
    marked_elements = 0

    # Compute R
    R = math.ceil(math.log(1 - (1-error_bound)**(1/B)) / math.log(3/4))

    found = False    
    done = False
    while not done:
        
        for _ in range(R):

            # Choose j uniformly at random from [0, sqrt(nu) - 1]
            j = random.randint(0, int(math.sqrt(nu)) - 1)
            
            # Run Grover's with j iterations and measure the result
            element = grover.run(j)
            total_grovers_iterations += j
            total_grovers_measurements += 1
            
            if element is AbstractGroverResult.MARKED_UNSEEN:
                found = True
                break

        if found:
            found = False  
            element = grover.remove_seen_element_from_oracle()
            marked_elements += 1
        else:
            done = True
            
    return marked_elements, total_grovers_iterations, total_grovers_measurements

def algorithm_3(nu: int, B: int, error_bound: float, grover: Grovers):
    """
    Implements algorithm 3

    Args:
        nu: The number of elements in the dataset.
        B: Upper bound on the number of marked elements (mu).
        error_bound: The desired error bound probability.
        grover: The Grovers object.
    """
    total_grovers_iterations = 0
    total_grovers_measurements = 0

    marked_elements = 0

    m = 1  # Starting number of iterations
    lambda_factor = 6/5  # Growth factor
    R = 1

    found = False
    done = False
    while not done:

        for _ in range(R):

            # Choose j uniformly at random from [0, m - 1]
            j = random.randint(0, math.ceil(m)-1)

            # Run Grover's with j iterations and measure the result
            element = grover.run(j)
            total_grovers_iterations += j 
            total_grovers_measurements += 1
            
            if element is AbstractGroverResult.MARKED_UNSEEN:
                found = True
                break

        if found:
            m = 1
            R = 1
            found = False
            element = grover.remove_seen_element_from_oracle()
            marked_elements += 1
        else:
            if m < math.sqrt(nu):
                m = min(m * lambda_factor, math.sqrt(nu))
                if m >= math.sqrt(nu):
                    R = math.ceil(math.log(1 - (1-error_bound)**(1/B)) / math.log(3/4))
            else:
                done = True
                
    return marked_elements, total_grovers_iterations, total_grovers_measurements