import math
import random
from quantum_close_neighbors.grovers import AbstractGroverResult
from quantum_close_neighbors.grovers import Grovers
    
class ProbabilisticGrovers(Grovers):

    def __init__(self, n_elements, n_marked_elements):
        self.n_elements = n_elements
        self.n_marked_elements = n_marked_elements
        self.seen_marked_elements = 0
    
    def p_success(self, j: int) -> float:
        """ Returns the probability of choosing a marked element after j iterations.
        """

        nu = self.n_elements
        mu = self.n_marked_elements

        theta = math.asin(math.sqrt(mu / nu))
        prob_success = math.sin((2 * j + 1) * theta) ** 2
        return prob_success  
    
    def p_success_rand_iterations(self, m: int) -> float:
        """ Returns the probability of choosing a marked element after j iteratons chosen randomly from set {0, 1, ... m-1}.
        """

        nu = self.n_elements
        mu = self.n_marked_elements

        theta = math.asin(math.sqrt(mu / nu))

        prob_success = 1/2  - math.sin(4 * m * theta)/(4 * m * math.sin(2 * theta))

        return prob_success  
    
    def run(self, num_iterations: int, mark_result_seen: bool = True) -> AbstractGroverResult:
        """ Simulate grover's as a probabilistic draw.
        """
        if self.n_marked_elements == 0:
            return AbstractGroverResult.UNMARKED
                
        if random.random() > self.p_success(num_iterations):
            # Failure
            return AbstractGroverResult.UNMARKED

        p_seen = self.seen_marked_elements / self.n_marked_elements
        if random.random() < p_seen:
            return AbstractGroverResult.MARKED_SEEN
        else:
            if mark_result_seen:
                self.seen_marked_elements += 1
            return AbstractGroverResult.MARKED_UNSEEN
        
    def reset_seen(self):
        self.seen_marked_elements = 0

    def remove_seen_elements_from_oracle(self, num_elements_to_remove: int):
        """ Remove seen elements from the oracle, such that they are longer marked.
            This means that the number of marked elements is reduced by one.
            The number of seen marked elements is also reduced by one.
        """
        self.n_marked_elements -= num_elements_to_remove
        self.seen_marked_elements -= num_elements_to_remove

    def remove_all_seen_elements_from_oracle(self):
        """ Remove all seen elements from the oracle, such that they are no longer marked.
            This means that the number of marked elements is reduced by one.
            The number of seen marked elements is also reduced by one.
        """
        self.remove_seen_elements_from_oracle(self.seen_marked_elements)

    def remove_seen_element_from_oracle(self):
        """ Remove a seen element from the oracle, such that it is no longer marked.
            This means that the number of marked elements is reduced by one.
            The number of seen marked elements is also reduced by one.
        """
        self.remove_seen_elements_from_oracle(1)



                         
