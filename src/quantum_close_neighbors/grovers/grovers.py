from abc import ABC, abstractmethod
from enum import IntEnum


class AbstractGroverResult(IntEnum):
    """ Enum for the possible outcomes of grover's algorithm.
        Either the element returned is unmarked, marked and never seen before, or marked and already seen.
    """
    UNMARKED = 0            # Unmarked element
    MARKED_UNSEEN = 1       # Marked element that has not been seen before
    MARKED_SEEN = 2         # Marked element that has been seen before

    def is_marked(self):
        """ Returns True if the result is either MARKED_UNSEEN or MARKED_SEEN. """
        return self in {self.MARKED_UNSEEN, self.MARKED_SEEN}
    
    def is_seen(self):
        """ Returns True if the result is MARKED_SEEN. """
        return self == self.MARKED_SEEN

class Grovers(ABC):

    @abstractmethod
    def run(self, num_iterations: int, mark_result_seen: bool = True) -> AbstractGroverResult:
        """ Run Grover's algorithm for a given number of iterations.
        """ 
        pass

    @abstractmethod
    def remove_seen_element_from_oracle(self):
        """ Remove the most recently seen element from the oracle.
            This means that the element is no longer marked.
        """
        pass

    @abstractmethod
    def remove_all_seen_elements_from_oracle(self):
        """ Remove all seen elements from the oracle.
            This means that those elements are no longer marked.
        """
        pass